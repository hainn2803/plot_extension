from pathlib import Path

ROOT = Path('eval/mcqa_at_rank_sweep_20261026_v1')
PARENT = Path('eval/mcqa_ap_gradual_chain_20261025_v1')
RANKS = (128, 256, 512)
SEEDS = (1, 0, 2)
STEPS = (0, 100, 250, 500, 750, 1000)
BANKS = ('AT_train', 'AT_calibration', 'readout_AT_train', 'readout_AT_calibration')
PROTOCOL = dict(
    name=ROOT.name, amendment_of=str(PARENT), scope='AT teacher calibration rank sweep only; no AP/C or final access',
    authorization='User: Ok, register new protocol and go ahead with this sweep.',
    ranks=list(RANKS), seeds=list(SEEDS), site='L25 final token', width=2304,
    updates=1000, evaluations=list(STEPS), checkpoint_every=25,
    order='ascending rank, seeds 1,0,2; complete all nine cells regardless of qualification',
    reuse='Read-only existing completed rank128 seed1 fit/readout; verify all orders and hashes; eight new fits',
    data='Exact four already selected parent AT/readout train/calibration banks; no filtering or expanded pool',
    bank_sizes=dict(AT_train=1664, AT_calibration=520, readout_AT_train=1300, readout_AT_calibration=520),
    initialization='Original seeded Gaussian QR at each rank, original four CPU threads; not nested across ranks',
    pairing='Identical per-seed update indices and preservation orders across ranks',
    training=dict(method='Standard DAS', loss='direct full-vocabulary CE only', optimizer='Adam', learning_rate=.01,
                  batch=32, changing=24, preservation=8, cpu_threads=4),
    readout=dict(input='natural AT state projected onto fitted rank-r basis', width='r', classes=26,
                 fit=1300, calibration=520, penalties=[.0001, .001, .01], epochs=200,
                 learning_rate=.01, std_floor=1e-6, selection='highest calibration correct count; first tie'),
    gate=dict(changing_each_target_letter=90, preservation_aggregate=90,
              factual_readout_overall=90, factual_readout_each_letter=90,
              arithmetic='100*correct >= 90*count; ties fail; no floating threshold comparisons'),
    selection='Smallest registered rank passing all four criteria in every seed; none if no rank passes',
    qualification_is_not_internal_faithfulness=True, calibration_selection_not_heldout_evidence=True,
    final_access=False, AP_C_training=False, extra_updates=False, extra_seeds=False, rank_expansion=False,
    model='google/gemma-2-2b base', revision='c5ebcd40d208330abc697524c919956e692655cf',
    backbone='frozen BF16 SDPA', projection='FP32 QR', tf32=False, padding_width=37, atol=2e-4, rtol=2e-4,
    numerical_batches=[1, 2, 4, 8, 16, 32],
    safety='Original AC/thermal watchdog, MemoryWatch, worker-scoped counter and startup accounting unchanged',
    total_gpu_seconds=43200, inherited_seconds=2912.7450253000134,
    inherited_profile_seconds=500.02189460000955, reserve_seconds=300, allowance=1.25,
    accounting='Existing 12h allocation, all prior charges retained; NOT a new 12h allocation',
    profile_limit_seconds=1800, no_automatic_restart=True,
)


def label(rank, seed):
    return f'AT_R{rank:04d}_S{seed}'
