# Frozen C/AT and AP checks for review

This package contains exact prompts, donor assignments, predictions, analysis,
and portable reproduction commands for the October 22 follow-up. No handles,
readouts, or backbone weights were trained. It uses Hai's uploaded checkpoints
from commit `399c2ca74837fc464c830aa8ab7af14b4d2327be`.

The C/AT pairs are the exact bank used in this local evaluation, reconstructed
earlier from the published task generator. They are **not claimed to be Hai's
exact retained evaluation pairs**. The full 1,600-pair bank and the unchanged
1,476-pair baseline-only mask are both included. The original full-bank
clean-baseline gate remains failed; artifact verification is not full-chain
certification.

## Main findings

| Metric on the 737 clean C-changing pairs | Standard | Gradual |
|---|---:|---:|
| Direct output accuracy | 87.52% | 89.15% |
| Same-base AT replay | 75.58% | 87.65% |
| Direct and replay both correct | 72.59% | 83.99% |

The replay/joint gains persist among the 424 pairs where both clean AT donors
succeed. This is an explanatory conditional analysis; it never changes the
primary population. Broader clean-AT answer preservation remains only 57.84%.

The AP experiment selected 96 existing AP-only pairs, eight per ordered
transition, by fixed hashes before inference. For each, it tests the original
two-row swap and both possible three-row cycles. All three sources share C,
natural AT, and source AP. Cycles separate AP transfer from copying the source
label at the base-correct slot. All 26 block outputs were tested. The same-slot
site depends on the known base AP and is therefore annotation-routed.

Sixty-five pairs have correct natural predictions for the base and all three
sources. At every block 3-15, golden-slot accuracy drops from 64/65 = 98.46% on
two-row swaps to 0/130 on cycles. Cycles follow the label-copy prediction in
93.08-96.92% of cases across those layers. The uploaded learned final-token AP
handle instead scores 47.69% versus 50.00%; it is not explained by that same
simple copying pattern, but remains unreliable here. All attempted cases are
included alongside the common naturally correct subset; none were replaced.

Both C handles preserve output on all 739 clean preservation pairs. Lower AT
readout accuracy is already present before intervention. Two historical cached
prefix differences disappeared under matched physical layouts; those older
records remain in the results. See [the detailed report](results/RESULTS.txt).

## 1. Recompute the saved results on CPU

From this directory, using Python 3.11:

```sh
python -m pip install -r requirements-cpu.txt
python review.py verify
python -m unittest -v test_review.py
python review.py summarize > recomputed_results.json
```

This needs no GPU, model download, authentication, or PyTorch. It checks SHA256
hashes, parses every original pair, verifies targets, recomputes the clean mask
from baseline predictions only, checks strict scoring, recomputes 2,000 paired
question-template bootstrap intervals, and compares the recovered numbers with
the recorded results. Ties and nonletter winners remain failures.

## 2. Rerun the frozen models on GPU

Create a separate Python 3.11 environment and install:

```sh
python -m pip install -r requirements-gpu.txt
python -m unittest -v test_portable_gpu_helpers.py
```

Publication validation: the standalone CPU verifier and target/scoring tests
were run successfully. The portable hook and final-head helpers are tested
against the pinned source on a small synthetic CPU model. The original local
real-model checks passed at `2e-4`. An attempt to test this newly packaged GPU
driver end to end was stopped by the unchanged AC startup guard before model
loading because the laptop was on battery. The portable GPU driver therefore
still requires its first complete real-model validation; the commands below
are provided for that purpose. This does not change the completed original
experiment or the verified saved-prediction analysis.

You need access to `google/gemma-2-2b` under its Hugging Face terms. Model and
tokenizer revision: `c5ebcd40d208330abc697524c919956e692655cf`. Download/cache the
model before the bounded run. No model weights or activation cache are included.
The repository already contains the required frozen C, AT, AP, and AT-readout
checkpoints; their SHA256 values are checked in `data/protocol.json`.

These examples assume the package is at
`reviews/frozen_handle_checks_20261022/` in Hai's repository:

```sh
# Validate real forwards, capture points, projection/replay, and final-head shortcut.
python reproduce.py --task validate --out ../../reruns/check_validate --local-files-only

# AP control: all 26 layers and the frozen learned AP handle.
python reproduce.py --task ap --out ../../reruns/check_ap --local-files-only

# C/AT: original 1,600 pairs, frozen clean mask, and both donor-control banks.
python reproduce.py --task c-at --out ../../reruns/check_c_at --local-files-only

# Or run both scientific checks together and recompute the resulting tables.
python reproduce.py --task all --out ../../reruns/check_all --local-files-only
python review.py summarize --predictions-dir ../../reruns/check_all > fresh_results.json
```

Use a fresh output directory each time. `--budget-minutes` defaults to 15 and
includes model loading, validation, profiling, and evaluation; allowable bounds
are 1-90 minutes. Each scientific stage profiles the remaining work and includes
a 25% allowance before continuing. A safety or budget stop preserves results
and exits; there is no automatic restart or guard reset. Windows laptop runs
require AC. The original 60-second persistent thermal guard is vendored intact.

For a specific local snapshot directory, add:

```sh
--model-cache /path/to/snapshots/c5ebcd40d208330abc697524c919956e692655cf
```

Its model/tokenizer files are verified against the pinned hashes. If the package
is outside the repository, add `--checkpoints-root /path/to/plot_extension`.

Backbone BF16, projectors FP32, SDPA, TF32 disabled, physical batch 32, and
`atol=rtol=2e-4` are fixed. AP uses the recorded global fixed padding width;
C/AT replays the original natural/base batch order and dynamic padding, with
explicit position IDs. Final-head controls retain physical head batch 32.

Fresh runs save per-example predictions, environment and numerical receipts,
thermal/power samples, profiles, and agreement with the shipped predictions.
Different GPUs/libraries or near-tied winners can change answers. Inspect
`completion.json`, `saved_prediction_agreement.json`, and the recorded-mask
baseline status before claiming numerical reproduction. The fresh analysis
retains the original mask and reports whether its clean baseline still passes;
it never replaces cases or filters by intervention outcomes.

## Files and provenance

- `data/c_at_pairs.json`: all 1,600 exact original directed pairs.
- `data/contexts.json`: the exact 2,741-context pool and order.
- `data/c_at_schedule.json`: original base/source layouts, fixed clean indices,
  and the two matched clean AT donors per retained pair.
- `data/at_qualification.json`: all 55,380 clean-AT donor assignments.
- `data/ap_plan.json`: all 384 AP prompts and 288 pair/variant assignments.
- `data/preservation_plan.json`: exact cases for the numerical-layout diagnosis.
- `predictions/`: per-example full-vocabulary winners/ties and results. C/AP
  files also retain label logits/probabilities; donor-control exports retain
  every winner, tie, margin and correctness field needed for these metrics.
- `results/`: original reports, failed baseline evidence, independent audits,
  numerical comparisons, and source hashes.
- `vendor/`: audited evaluator components; AP Engine and Tail are mechanically
  extracted unchanged, with extraction provenance. The portable driver adds
  path/configuration plumbing, safety accounting, and comparison outputs.
- `original_scripts/`: original workspace scripts, preserved for source review.
  Use `review.py`/`reproduce.py` for portable execution.
- `MANIFEST.json`: file hashes, excluding itself and generated rerun outputs.

There is one uploaded checkpoint per method and no fresh multi-seed replication.
The uploaded AT classifier is a teaching/selection readout, not independent
validation. The AP result concerns these particular same-slot full-vector
interventions; it does not establish AP's absence elsewhere in the model.
