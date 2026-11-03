# CYP inhibition — direct inhibition track

Submission for the [OpenADMET CYP Inhibition Blind Challenge](https://openadmet.org),
direct inhibition track: predicting pIC50 for **CYP1A2, CYP2C9, CYP2D6 and CYP3A4**
across 750 blind compounds.

The entry is a per-isoform blend of two models that make different mistakes — a
message-passing graph network and a tabular foundation model.

## Method

**chemprop (D-MPNN).** A single multi-task network with the
[CheMeleon](https://github.com/JacksonBurns/chemeleon) foundation encoder and
**8 output heads**: the 4 pIC50 targets at loss weight 1.0, plus 4 auxiliary
`log2fc` heads at weight 0.3.

The auxiliary heads are the point. pIC50 labels are sparse — between 26% and 48%
coverage depending on isoform — while single-concentration `log2fc` readings cover
about 90% of compounds and correlate with pIC50 at |ρ| 0.83–0.94. Training them
jointly lets the dense, cheap signal shape the shared encoder for the sparse,
expensive one.

**TabICL.** A tabular in-context foundation model, one regressor per isoform, over
RDKit 2D descriptors concatenated with 256-bit folded Morgan counts (473 features).
No gradient training — it conditions on the training rows at inference time.

**Blend.** A per-isoform convex combination. Weights were fitted by grid search on
**out-of-fold predictions only**, never on the test split:

| isoform | chemprop | TabICL |
| --- | --- | --- |
| CYP1A2 | 0.35 | 0.65 |
| CYP2C9 | 0.40 | 0.60 |
| CYP2D6 | 0.30 | 0.70 |
| CYP3A4 | 0.35 | 0.65 |

## Model selection

Every choice here — the two model families, the auxiliary heads, and the
per-isoform blend weights — was fixed on a held-out internal benchmark before the
blind set was touched: 5-fold cross-validation over the labelled compounds, with a
further split held back and scored only once.

The blend weights come from out-of-fold predictions alone. The test split was never
used to fit anything, so the benchmark's estimate is an honest one rather than a
number the weights were tuned against.

Benchmark scores are not reported here.

## The de-leak rule

Non-inhibitors are drawn from the single-concentration screen **minus every
compound carrying a pIC50 anywhere in the challenge** — not minus the current
training subset.

That distinction matters more than it looks. Subtracting only the current subset
re-injects held-out compounds into training as "non-inhibitors" while they still
carry their `log2fc` values, which correlate with the very labels being predicted.
The result is a leak that inflates cross-validation scores and tells you nothing
about blind performance. The check is enforced in `assert_no_leak`, which fails
loudly rather than warning.

## Files

| file | |
| --- | --- |
| `submission.ipynb` | the pipeline, executed with outputs |
| `cyp_ens_lib.py` | shared helpers: framing, de-leak, training, featurisation, blending, scoring |
| `blend_weights.json` | the fitted weights used for the submission |
| `submission_latest.csv` | the submitted predictions, 750 rows |

## Reproducing

The challenge data files (`train_df.csv`, `test_df.csv`, the single-concentration
screen and the Emax table) are **not redistributed here**, so the notebook will not
run end to end from a fresh clone. It is a record of the method, with its real
outputs, rather than a turnkey pipeline.

If you do have the data, note that it runs across **two conda environments**:

- **`chemprop_gpu`** — CUDA torch, used for chemprop training and the TabICL fits
- **`chemprop_tabpfn`** — CPU torch, used for RDKit featurisation and all scoring

That split is not a design choice. In the GPU environment, RDKit's 2D descriptor
calculation and `scipy.stats.spearmanr` both abort the interpreter outright with a
delay-load DLL failure (`0xC06D007F`) — no Python traceback, no output flushed.
Everything that touches those two code paths runs in the CPU environment, and
cached feature matrices carry the data across. On a correctly built CUDA
environment the whole pipeline would run in one kernel.

Versions used: `chemprop` 2.3.1, `tabicl` 2.2.0, `torch` 2.5.1, `rdkit` 2026.03.5,
`scikit-learn` 1.6.1, `pandas` 2.3.3, `numpy` 2.4.6, `scipy` 1.17.1.
