# Rank-128 Standard/Gradual C -> AP -> AT -> Y experiment

Exact source snapshot for the completed October 26-27, 2026 matched experiment
and its full mediator/donor evaluation addendum. This directory does not change
the repository's existing experiment drivers.

## Scope

- Gemma-2-2B **base**, pinned model/tokenizer revision
  `c5ebcd40d208330abc697524c919956e692655cf`.
- Three paired training seeds: `1, 0, 2`; rank 128 throughout this comparison.
- C: layer 0 question-color token, one `2304 x 128` basis.
- AP: layer 14 joint four option-row last-color subtokens, one `9216 x 128` basis.
- AT: layer 25 final token, one `2304 x 128` basis, imported and frozen.
- Twelve new AP/C fits, 1,000 updates each, with matched initialization, batches
  and preservation orders between methods. No AT refit or replacement.
- Standard: direct full-vocabulary cross-entropy only.
- Gradual AP: direct + 2 AT replay + 2 AT readout.
- Gradual C: direct + 2 AP replay + 2 AP readout + 2 AT replay + 2 AT readout,
  using frozen Gradual AP and the common AT teacher. No additional three-hop
  training loss.

AT accuracy qualification alone was explicitly disabled. The imperfect imported
teachers were not silently declared qualified. Numerical, memory, power/thermal,
counter, data-integrity, budget and coverage guards were retained.

## Source map

All original module paths are retained under `source/`:

- `experiments/mcqa_apc_128_20261026/`: amended protocol, registration, AP/C
  fitting dispatch, count-aware auditing, original reporting and CPU checks.
- `experiments/mcqa_ap_gradual_chain_20261025/`: original data, model hooks,
  losses, fitting, frozen readouts, evaluation, bootstraps and full verification.
- `diagnostics/apc_128_20261026/`: guarded pause/resume helpers and the complete
  post-lock mediator/donor addendum. `full_transfer_plan.py` defines its 144
  additional artifacts. `finish_transfer.py` and `finish_transfer_report.py`
  contain the final collection/report implementation.
- The remaining modules are byte-identical local dependencies, including the
  original safety/runtime helpers and three upstream Hai source files.

`SOURCE_MANIFEST.json` records every exported file's original relative path,
SHA256, size and whether it appeared in the initially sealed source snapshot.
No scientific sources were edited for publication.

## Evaluation definitions

The original 243-artifact evaluation includes direct IIA, preservation, identity,
restoration, donor/recipient controls and the complete 2x2 C-method-by-AP-evaluator
matrix. All 15 endpoints (three imported AT plus twelve new AP/C) and nine
factual readouts were locked before opening final data.

- AP primary: direct output **and** induced AT replay correct on the same pair.
- C primary: direct C output **and** AP replay **and** subsequent AT replay
  correct, through the common **Standard AP** evaluator.
- The addendum separately reports IIA, frozen readouts, readout/output joints,
  replay into the original clean base (mediator), and replay into different-
  question recipients (donor, same/different intended intermediate value).
- AP-recipient targets use the new recipient label at the intended AP, not the
  original donor letter. AT-recipient targets retain the intended answer letter.
- Two fixed recipients per parent and condition stay together in 2,000 paired
  original-question/template-cluster bootstrap draws per seed.
- Upstream failures remain in denominators. Missing controls are not silently
  dropped. Recipient-option presence/absence is an explicit stratum.

The diagnostic readout/output joint is not the registered replay/output joint.
Post-lock addendum endpoints are diagnostics, not newly registered confirmatory
tests. These experiments do not prove a unique native computation.

## Verify the code snapshot

Python 3.11, standard library only; no GPU/model/data access:

```sh
python -B verify.py
```

With the original Python dependencies installed, synthetic CPU regressions need
no checkpoints, data banks, authentication, model download or GPU:

```sh
python -m pip install -r requirements-cpu.txt
python -B cpu_checks.py
```

Publication validation passed all 39 synthetic CPU checks from the exported
source, with CUDA uninitialized and no local-source fallback to the original
workspace. The test environment used Python 3.11.17, PyTorch 2.9.1+cu128,
NumPy 2.4.6, Transformers 5.9.0, Matplotlib 3.10.9 and psutil 7.2.2.
The requirements file pins the public PyTorch version; publication did not
separately test another PyTorch wheel/backend. On Windows, use a normal short
temporary directory if a deeply nested workspace exceeds path-length limits.

These tests exercise all recipient roles, target/condition checks, missing-case
rejection, a synthetic 243+144-artifact reporting fixture, strict token scoring,
fit/bootstrap seed separation and retained failed-attempt budget charges. They
must not initialize CUDA. Temporary fixtures are outside this snapshot.

## Rerunning the original scientific run

This is a **code-only archival snapshot**, not a standalone portable training
release. Checkpoints, selected banks/proposals, natural-state caches, optimizer
and RNG state, model weights, signed Windows runtime, diagnostic receipts,
reference PDF, predictions and generated results are deliberately not uploaded.

The original entrypoints under `source/` require that original workspace layout
and its sealed external inputs. They include Windows-specific AC/thermal and GPU
counter checks, signed-runtime paths, and immutable process/resume identities.
Do not invoke historical `resume*` helpers against a different run or transplant
their recorded PIDs/checkpoints. Do not replace unavailable guards with zeros.

Porting to Delta or another environment requires an explicit artifact/input
manifest, migration provenance, platform-appropriate safety validation and fresh
full-reference numerical checks. The snapshot does not claim that such a port
or a fresh GPU rerun was validated during publication.

The original allocation was cumulative 12 hours, not a fresh budget per resume.
Evaluation-only additions later permitted a recorded 14-hour cap; every prior
charge and the five-minute reserve were retained. Earlier 12/13-hour constants
remain in historical source files intentionally.

The final reporting-only seed-name collision was repaired by
`report_seed_repair.py`, preserving the raw results and bootstrap values.
`supplement_counter_context_resume.py` preserves a failed startup charge and
uses the native Windows PowerShell module path for child counter calls, without
changing execution policy or weakening the original counter/MemoryWatch guard.
