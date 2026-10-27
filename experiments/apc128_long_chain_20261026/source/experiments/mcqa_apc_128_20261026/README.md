# Matched AP/C Continuation With Ungated Frozen AT

Explicit October 26 amendment: keep existing rank-128 AT teachers at their current
accuracies, remove their accuracy qualification prerequisite, and proceed with
paired Standard/Gradual AP and C subspace learning. This is **not** a new site,
rank, loss-weight or seed search. Preserve the stopped parent and completed rank
sweep unchanged; all new outputs go to a separate root.

AP uses the registered L14 concatenation of four option-row last-color subtokens,
one 9216x128 basis. C uses the L0 question-color token, one 2304x128 basis.
AT stays at L25 final token, 2304x128. Seeds are 1,0,2. AT weights and selected
factual readouts are imported byte-for-byte, shared within each paired seed,
and never retrained or silently marked qualified.

Train AP Standard/Gradual first, then C Standard/Gradual, exactly 1,000 updates
per new fit. The 12 new fits use original paired initialization, batch and
preservation orders, Adam .01, effective32=24change8pres and frozen backbone.
Standard is direct full-vocabulary CE only. Gradual AP is direct+2ATreplay+
2ATreadout. Gradual C is direct+2APreplay+2APreadout+2ATreplay+2ATreadout, using
frozen Gradual AP and common AT; no extra three-hop training loss is added.

Reuse all ten original selected train/calibration banks, original fixed
proposals and natural caches with explicit source/destination hashes. No final
prompts may be selected until all 15 endpoints and 9 factual readouts are locked.
Finals/controls, missingness handling, full2x2 C-by-AP matrix, co-primary joint
metrics, strong-superiority requirements and 2,000 paired bootstrap draws are
unchanged. All243 registered final artifacts remain mandatory.

Only AT **accuracy qualification** is disabled. All actual-model numerical
comparisons at 2e-4, fixed-batch32 memory/capacity checks, worker-scoped counters,
original AC/thermal guards, startup accounting, complete-remaining-work+25%
budget gates and reserve300 remain enforced. The SAME43200-second allocation
inherits8769.316136500012 seconds, including all prior failures, AT fitting and
rank-sweep charges. It is not a new12h budget. The prior profile charge remains
853.1099996000121 seconds; reuse the immutable full-path profile only after
fresh saved-projector, cached-layout and real-teacher loss/gradient validation.

The independent audit handles an existing representation defect by accepting
only exact rational or exact original FP32-mean accuracy for **identical integer
correct/count values**. No scoring discrepancy, bootstrap change, numerical
tolerance or scientific criterion is relaxed. A repeated final audit reuses its
timestamp only if all other fields agree exactly.

Use the signed Python runtime:

```text
python -u -B -m experiments.mcqa_apc_128_20261026 prepare
python -u -B -m experiments.mcqa_apc_128_20261026 run
```

After actual worker exit and closed ledger:

```text
python -u -B -m experiments.mcqa_apc_128_20261026 finalize
```

Report the imperfect AT teachers explicitly. Relative improvement under a
common teacher is not proof of unique endogenous computation, native switching,
or absence of AP. Do not interpret weak AT as a fixed ceiling on induced-state
performance or silently remove primary examples on downstream/control failure.
