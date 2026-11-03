"""Shared helpers for the TabICL + chemprop ensemble notebooks.

Both tabicl_chemprop_ens_benchmark.ipynb and submission.ipynb import this,
so the two pipelines cannot drift apart. The benchmark measures a config;
the submission must run that same config or the measurement means nothing.

Design notes
------------
A single multi-task member is used: 8 heads, 4 pIC50 at weight 1.0 and 4
log2fc at 0.3. A wider multi-member ensemble was benchmarked and did not
justify itself: the additional members made no material difference, and the
isoform-specific ones only touched CYP2D6.

The de-leak rule is in build_challenge_frame: non-inhibitors are drawn from
the single-concentration screen minus EVERY compound carrying a pIC50
anywhere in the challenge, never minus just the current training subset.
Subtracting only the subset re-injects held-out compounds as "non-inhibitors"
with their log2fc intact, and log2fc vs pIC50 runs |rho| 0.83-0.94.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------- paths ---
BASE = Path(r"C:\Users\e.tiu\cyp_comp\benchmarkingstudies")

# Everything this method writes lives beside the notebooks, under new_method/.
# Defined once here so the benchmark and the submission cannot disagree about
# where blend_weights.json is.
OUT             = BASE / "new_method"
RESULTS         = OUT / "results"
SUBMIT          = OUT / "submit"
BLEND_WEIGHTS   = RESULTS / "blend_weights.json"
LEGACY_RESULTS  = BASE / "results"   # read-only: split_clean.csv from the old benchmark

CV_CSV          = BASE / "train_df_withCVfolds.csv"   # 4,905 rows, 'split' column
TRAIN_CSV       = BASE / "train_df.csv"               # 4,905 labelled compounds
TEST_BLIND_CSV  = BASE / "test_df.csv"                # 750 blind compounds
SINGLE_CONC_CSV = BASE / "train_single_df.csv"        # log2fc, long format
EMAX_CSV        = BASE / "train_emax_df.csv"          # Emax -> sample weight

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
PIC50_COLS   = [f"{c}_pIC50_direct_inhibition" for c in CYP_ISOFORMS]
LOG2FC_COLS  = [f"{c}_log2fc" for c in CYP_ISOFORMS]
EMAX_COLS    = [f"{c}_EmaxVsPosCtrl_direct_inhibition" for c in CYP_ISOFORMS]

# Task weights: the pIC50 heads carry the objective, the log2fc heads shape
# the shared encoder at 0.3.
TASK_WEIGHTS = [1.0] * 4 + [0.3] * 4
TARGET_COLS  = PIC50_COLS + LOG2FC_COLS

FOUNDATION = "CHEMELEON"
SEED       = 0

# ------------------------------------------------------------- execution ---
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
    _GPU_NAME = torch.cuda.get_device_name(0) if _HAS_CUDA else "n/a"
except Exception:
    _HAS_CUDA, _GPU_NAME = False, "n/a"

ACCELERATOR = "gpu" if _HAS_CUDA else "cpu"
DEVICES     = "auto"

# Do NOT call Scripts/chemprop.exe. On Windows a pip console-script shim hard-codes
# the absolute path of the interpreter it was installed against, and
# `conda create --clone` copies the shim without rewriting it. The clone's
# chemprop.exe therefore launches the ORIGINAL env's python, picking up that env's
# torch -- which is how a CUDA-enabled kernel ended up running chemprop on a CPU-only
# torch and failing with "No supported gpu backend found!".
#
# Invoking the entry point through sys.executable guarantees chemprop runs in the
# same interpreter as this notebook, so the torch it sees is the torch reported below.
CHEMPROP_BOOTSTRAP = (
    "import sys; sys.argv[0] = 'chemprop'; "
    "from chemprop.cli.main import main; main()"
)
CHEMPROP_CMD = [sys.executable, "-c", CHEMPROP_BOOTSTRAP]


def environment_report() -> None:
    """Print what this kernel will actually run on. Call before anything else."""
    import importlib.metadata as md
    env = Path(sys.executable).parent.name
    try:
        cpv = md.version("chemprop")
    except Exception:
        cpv = "NOT INSTALLED"
    print(f"env          : {env}")
    print(f"chemprop     : {cpv}")
    print(f"torch        : {getattr(torch, '__version__', 'n/a')}")
    print(f"GPU          : {_HAS_CUDA}  ({_GPU_NAME})  -> {ACCELERATOR.upper()}")

    # Checking this process is not enough. chemprop runs as a subprocess, and the
    # subprocess is what actually has to see the GPU -- ask it directly.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch, chemprop, sys; print(sys.prefix); "
         "print(chemprop.__file__); print(torch.__version__); "
         "print(torch.cuda.is_available())"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if probe.returncode != 0:
        raise RuntimeError("chemprop subprocess probe failed:\n" + (probe.stderr or "")[-2000:])
    prefix, cp_file, sub_torch, sub_cuda = probe.stdout.strip().splitlines()[:4]
    print(f"subprocess   : torch {sub_torch}, cuda {sub_cuda}")
    print(f"chemprop pkg : {cp_file}")

    if Path(prefix).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError(
            f"chemprop subprocess runs in {prefix}, not {sys.prefix}. Refusing to "
            f"train -- the subprocess would use a different torch than reported above.")
    if (sub_cuda == "True") != _HAS_CUDA:
        raise RuntimeError(
            f"this kernel reports cuda={_HAS_CUDA} but the chemprop subprocess reports "
            f"cuda={sub_cuda}. Training would be configured for the wrong device.")
    if not _HAS_CUDA:
        print("\n  WARNING: running on CPU. A full 5-fold run takes ~70 min per fold\n"
              "  instead of ~10. Switch the kernel to the CUDA-enabled clone.")


# ------------------------------------------------------------------ data ---
def standardise_smiles(smi):
    """Canonical SMILES via RDKit, or None if unparseable."""
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
    except Exception:
        return None


def load_cv_frame() -> pd.DataFrame:
    """The 4,905-row labelled set with its 'split' column (folds 1-5 + test)."""
    df = pd.read_csv(CV_CSV)
    missing = [c for c in ["Molecule_Name", "SMILES", "split"] + PIC50_COLS
               if c not in df.columns]
    if missing:
        raise KeyError(f"{CV_CSV.name} is missing columns: {missing}")
    return df


def _log2fc_wide() -> pd.DataFrame:
    """Single-concentration screen, long -> wide, one row per compound."""
    single = pd.read_csv(SINGLE_CONC_CSV)
    wide = (single.pivot_table(index="Molecule_Name", columns="enzyme",
                               values="log2fc_estimate", aggfunc="mean")
                  .reset_index())
    wide.columns.name = None
    wide.rename(columns={c: f"{c}_log2fc" for c in CYP_ISOFORMS}, inplace=True)
    smi = (single[["Molecule_Name", "SMILES"]].drop_duplicates("Molecule_Name")
           if "SMILES" in single.columns else None)
    return wide, smi


def _emax_weights() -> pd.DataFrame:
    emax = pd.read_csv(EMAX_CSV)[["Molecule_Name"] + EMAX_COLS].copy()
    for c in EMAX_COLS:
        emax[c] = emax[c].abs().clip(0, 1)
    emax["emax_weight"] = emax[EMAX_COLS].mean(axis=1, skipna=True).fillna(1.0)
    return emax[["Molecule_Name", "emax_weight"]]


def build_challenge_frame(labelled_df: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    """Training frame for chemprop: labelled compounds + log2fc aux + genuine
    non-inhibitors.

    labelled_df : the compounds whose pIC50 values may be trained on. For a CV
                  fold that is the four training folds; for the submission it is
                  all 4,905.

    THE DE-LEAK. Non-inhibitors are taken from the single-concentration screen
    minus every compound that carries a pIC50 anywhere in the challenge -- read
    from TRAIN_CSV, not from labelled_df. Using labelled_df here would pull the
    held-out fold back in as "non-inhibitors" still carrying their log2fc.
    """
    wide, smi_map = _log2fc_wide()
    emax = _emax_weights()

    base = labelled_df[["Molecule_Name", "SMILES"] + PIC50_COLS].copy()
    base = base.merge(wide, on="Molecule_Name", how="left")
    base = base.merge(emax, on="Molecule_Name", how="left")
    base["emax_weight"] = base["emax_weight"].fillna(1.0)

    all_labelled = set(pd.read_csv(TRAIN_CSV)["Molecule_Name"])
    if smi_map is None:
        smi_map = base[["Molecule_Name", "SMILES"]].drop_duplicates()
    non_inhib = (wide[~wide["Molecule_Name"].isin(all_labelled)]
                 .merge(smi_map, on="Molecule_Name", how="left"))
    for c in PIC50_COLS:
        non_inhib[c] = np.nan
    non_inhib["emax_weight"] = 1.0

    cols = ["Molecule_Name", "SMILES"] + PIC50_COLS + LOG2FC_COLS + ["emax_weight"]
    for df in (base, non_inhib):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

    out = pd.concat([base[cols], non_inhib[cols]], ignore_index=True)
    out = out.dropna(subset=["SMILES"]).reset_index(drop=True)
    out["SMILES"] = out["SMILES"].map(standardise_smiles)
    out = out.dropna(subset=["SMILES"]).reset_index(drop=True)

    n_pic50 = out[PIC50_COLS].notna().any(axis=1).sum()
    print(f"  [{tag}] training frame: {len(out)} compounds "
          f"({n_pic50} with pIC50, {len(out) - n_pic50} log2fc-only), "
          f"{out[LOG2FC_COLS].notna().any(axis=1).sum()} with log2fc aux")
    return out


def assert_no_leak(train_frame: pd.DataFrame, held_out_names, tag: str = "") -> None:
    """Hard stop if any held-out compound appears in a training frame."""
    bad = set(train_frame["Molecule_Name"]) & set(held_out_names)
    if bad:
        raise RuntimeError(
            f"[{tag}] LEAK: {len(bad)} held-out compounds in the training frame, "
            f"e.g. {sorted(bad)[:5]}")
    print(f"  [{tag}] leak check: 0 of {len(set(held_out_names))} held-out "
          f"compounds present  OK")


# -------------------------------------------------------------- chemprop ---
def _chemprop(cmd, log_path=None):
    cmd = [str(x) for x in cmd]
    if cmd[0] == "chemprop":
        cmd = CHEMPROP_CMD + cmd[1:]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if log_path:
        Path(log_path).write_text((res.stdout or "") + "\n--- STDERR ---\n"
                                  + (res.stderr or ""), encoding="utf-8")
    if res.returncode != 0:
        print("STDERR:", (res.stderr or "")[-3000:])
        raise subprocess.CalledProcessError(res.returncode, cmd)
    time.sleep(1)
    return res


def _find_model(model_dir: Path) -> Path:
    for pat in ("best.pt", "model.pt", "*.pt", "*.ckpt"):
        hits = sorted(Path(model_dir).rglob(pat), key=lambda p: p.stat().st_mtime)
        if hits:
            return hits[-1]
    raise RuntimeError(f"No checkpoint under {model_dir}")


def train_chemprop(frame: pd.DataFrame, out_dir: Path, tag: str,
                   epochs: int = 60, patience: int = 15,
                   batch_size: int = 64, seed: int = SEED,
                   dropout: float | None = None) -> Path:
    """One mt_aux fit. Returns the checkpoint path. Skips if already trained."""
    import shutil
    out_dir = Path(out_dir)
    if (out_dir / "TRAIN_DONE").exists():
        print(f"  [{tag}] already trained -> reusing")
        return _find_model(out_dir)
    if out_dir.exists():
        # Leftovers from a run that raised. Clear them, or _find_model can pick up
        # a checkpoint from the failed attempt and quietly predict with it.
        print(f"  [{tag}] clearing an incomplete previous attempt")
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "train.csv"
    frame[["SMILES", "emax_weight"] + TARGET_COLS].to_csv(csv_path, index=False)

    cmd = ["chemprop", "train",
           "-i", csv_path.resolve(),
           "-s", "SMILES",
           "--target-columns", *TARGET_COLS,
           "--task-type", "regression",
           "--from-foundation", FOUNDATION,
           "--split", "SCAFFOLD_BALANCED", "--split-sizes", "0.9", "0.1", "0.0",
           "--metrics", "mae", "r2",
           "--epochs", epochs,
           "--patience", patience,
           "--batch-norm",
           "--batch-size", batch_size,
           "--accelerator", ACCELERATOR,
           "--devices", DEVICES,
           "-o", out_dir.resolve(),
           "--pytorch-seed", seed,
           "--data-seed", seed,
           "-w", "emax_weight",
           "--task-weights", *[str(w) for w in TASK_WEIGHTS]]
    if dropout is not None:
        cmd += ["--dropout", str(dropout)]

    t0 = time.time()
    print(f"  [{tag}] training {len(frame)} compounds, {len(TARGET_COLS)} heads "
          f"on {ACCELERATOR.upper()} ...")
    _chemprop(cmd, log_path=out_dir / "train.log")
    (out_dir / "TRAIN_DONE").touch()
    print(f"  [{tag}] done in {(time.time() - t0) / 60:.1f} min")
    return _find_model(out_dir)


def predict_chemprop(model_pt: Path, frame: pd.DataFrame,
                     out_csv: Path, tag: str) -> pd.DataFrame:
    """Predict pIC50 for `frame`. Returns Molecule_Name + the 4 pIC50 columns."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    in_csv = out_csv.with_name(out_csv.stem + "_in.csv")
    frame[["Molecule_Name", "SMILES"]].to_csv(in_csv, index=False)

    if not out_csv.exists():
        _chemprop(["chemprop", "predict",
                   "-i", in_csv.resolve(),
                   "-s", "SMILES",
                   "--model-paths", Path(model_pt).resolve(),
                   "--accelerator", ACCELERATOR,
                   "--devices", DEVICES,
                   "-o", out_csv.resolve()])

    pred = pd.read_csv(out_csv)
    missing = [c for c in PIC50_COLS if c not in pred.columns]
    if missing:
        raise RuntimeError(f"[{tag}] prediction file lacks {missing}")

    # A silent all-NaN prediction file averages away invisibly inside an
    # ensemble, so fail loudly here rather than ship a dead member.
    dead = [c for c in PIC50_COLS if pred[c].notna().sum() == 0]
    if dead:
        raise RuntimeError(
            f"[{tag}] chemprop returned all-NaN for {dead}. The model did not "
            f"train usefully -- do not average this into an ensemble.")

    return pred[["Molecule_Name"] + PIC50_COLS]


# ---------------------------------------------------------------- TabICL ---
_DESC_CACHE: dict[str, np.ndarray] = {}


def featurise(smiles: list[str], n_morgan_bits: int = 256) -> pd.DataFrame:
    """RDKit 2D descriptors + folded Morgan counts.

    TabICL is a tabular in-context model, so it wants a few hundred informative
    columns, not a 2048-bit sparse fingerprint. Descriptors carry the physchem
    signal; the folded Morgan block carries substructure.
    """
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=n_morgan_bits, countSimulation=False)
    desc_names = [n for n, _ in Descriptors.descList]
    rows = []
    for smi in smiles:
        key = f"{smi}|{n_morgan_bits}"
        if key in _DESC_CACHE:
            rows.append(_DESC_CACHE[key])
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            vec = np.full(len(desc_names) + n_morgan_bits, np.nan)
        else:
            d = []
            for _, fn in Descriptors.descList:
                try:
                    d.append(fn(mol))
                except Exception:
                    d.append(np.nan)
            fp = np.asarray(gen.GetCountFingerprintAsNumPy(mol), dtype=float)
            vec = np.concatenate([np.asarray(d, dtype=float), fp])
        _DESC_CACHE[key] = vec
        rows.append(vec)

    cols = desc_names + [f"mfp_{i}" for i in range(n_morgan_bits)]
    X = pd.DataFrame(np.vstack(rows), columns=cols)
    return X.replace([np.inf, -np.inf], np.nan)


def clean_features(X_train: pd.DataFrame, *others: pd.DataFrame):
    """Drop all-NaN and zero-variance columns, then median-impute.

    Column selection and the imputation medians come from the TRAINING frame
    only, then are applied unchanged to the others. Fitting them on the pooled
    data would let the evaluation set influence its own features.
    """
    keep = [c for c in X_train.columns
            if X_train[c].notna().any() and X_train[c].nunique(dropna=True) > 1]
    med = X_train[keep].median()
    out = [X_train[keep].fillna(med)]
    for o in others:
        out.append(o[keep].fillna(med))
    return (*out, keep)


def tabicl_predict(X_train: pd.DataFrame, y_train: np.ndarray,
                   X_pred: pd.DataFrame, tag: str, seed: int = SEED):
    """Fit one TabICLRegressor and predict. No gradient training -- in-context."""
    from tabicl import TabICLRegressor

    device = "cuda" if _HAS_CUDA else "cpu"
    t0 = time.time()
    model = TabICLRegressor(random_state=seed, device=device)
    model.fit(X_train.to_numpy(dtype=np.float32), np.asarray(y_train, dtype=np.float32))
    pred = model.predict(X_pred.to_numpy(dtype=np.float32))
    print(f"    [{tag}] TabICL  n_train={len(X_train)}  n_pred={len(X_pred)}  "
          f"{time.time() - t0:.0f}s")
    return np.asarray(pred, dtype=float)


# --------------------------------------------------------------- scoring ---
def _per_isoform(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    if len(y) < 3 or np.std(y) == 0:
        return {k: np.nan for k in ("mae", "st_rae", "r2", "spearman", "kendall", "n")}
    mae = np.mean(np.abs(y - p))
    mad = np.mean(np.abs(y - y.mean()))
    return {"mae": mae,
            "st_rae": mae / mad if mad > 0 else np.nan,
            "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
            "spearman": spearmanr(y, p).statistic,
            "kendall": kendalltau(y, p).statistic,
            "n": len(y)}


def score(truth: pd.DataFrame, preds: pd.DataFrame, cols=PIC50_COLS):
    """-> (macro dict, per-isoform frame). Inputs must be row-aligned."""
    per = pd.DataFrame({c: _per_isoform(truth[c].to_numpy(float),
                                        preds[c].to_numpy(float)) for c in cols}).T
    macro = {f"MA_{k}": per[k].mean()
             for k in ("mae", "st_rae", "r2", "spearman", "kendall")}
    return macro, per


def constant_mean_floor(train: pd.DataFrame, test: pd.DataFrame, cols=PIC50_COLS):
    """Predict each isoform's training mean. MA_st_rae ~ 1.0, MA_r2 ~ 0.
    Anything scoring above this floor is worse than predicting the average."""
    preds = pd.DataFrame({c: np.full(len(test), train[c].mean()) for c in cols})
    return score(test, preds, cols)


def fit_blend_weights(truth: pd.DataFrame, pred_a: pd.DataFrame,
                      pred_b: pd.DataFrame, cols=PIC50_COLS,
                      grid=None) -> dict:
    """Per isoform, pick w minimising MAE for  w*A + (1-w)*B.

    Fitted on out-of-fold predictions, which no model saw during its own
    training. w=1.0 means chemprop alone -- a legitimate outcome, and the
    right one wherever TabICL does not help.
    """
    grid = np.arange(0, 1.0001, 0.05) if grid is None else np.asarray(grid)
    weights = {}
    for c in cols:
        y = truth[c].to_numpy(float)
        a, b = pred_a[c].to_numpy(float), pred_b[c].to_numpy(float)
        m = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
        if m.sum() < 10:
            weights[c] = 1.0
            continue
        maes = [np.mean(np.abs(y[m] - (w * a[m] + (1 - w) * b[m]))) for w in grid]
        weights[c] = float(grid[int(np.argmin(maes))])
    return weights


def apply_blend(pred_a: pd.DataFrame, pred_b: pd.DataFrame,
                weights: dict, cols=PIC50_COLS) -> pd.DataFrame:
    out = {}
    for c in cols:
        w = weights[c]
        out[c] = w * pred_a[c].to_numpy(float) + (1 - w) * pred_b[c].to_numpy(float)
    return pd.DataFrame(out)


def report(name: str, truth: pd.DataFrame, preds: pd.DataFrame) -> dict:
    macro, per = score(truth, preds)
    print(f"\n{name}")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in macro.items()))
    print(per.round(4).to_string())
    return macro
