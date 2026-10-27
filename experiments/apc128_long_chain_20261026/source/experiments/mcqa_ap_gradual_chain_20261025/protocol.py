from pathlib import Path

from experiments.mcqa_ap_localization_sweep.protocol import COLORS, LETTERS, REVISION, DIAGNOSIS

ROOT = Path('eval/mcqa_ap_gradual_chain_20261025_v1')
PRIOR = Path('eval/mcqa_ap_localization_sweep_20261024_v1')
SALT = ROOT.name
SEEDS = (1, 0, 2)
METHODS = ('standard', 'gradual')
STEPS = (0, 100, 250, 500, 750, 1000)
BANK_SIZES = {
    **{f'{v}_{s}': n for v in ('AP', 'C') for s, n in
       (('train', 1792), ('calibration', 512), ('final', 1792))},
    'AT_train': 1664, 'AT_calibration': 520, 'AT_final': 1664,
    **{f'readout_{v}_{s}': n for v in ('AT', 'AP') for s, n in
       (('train', 1300), ('calibration', 520))},
}
PROTOCOL = dict(
    name=SALT, model='google/gemma-2-2b', revision=REVISION,
    seeds=list(SEEDS), methods=list(METHODS), sizes=BANK_SIZES,
    sites=dict(AT='L25 final token', AP='L14 joint four option-row last-color subtokens', C='L0 question-color token'),
    rank=128, AP_width=9216, AT_C_width=2304, updates=1000,
    order='AT then AP then C; all endpoints/readouts locked before final access',
    learning_rate=.01, optimizer='Adam', batch=32, changing=24, preservation=8,
    checkpoint_every=25, evaluations=list(STEPS), cpu_threads=4,
    backbone='frozen BF16 SDPA', projection='FP32 QR', tf32=False, width=37,
    atol=2e-4, rtol=2e-4, profile_batches=[1, 2, 4, 8, 16, 32],
    standard_loss=dict(direct=1.),
    gradual_AP_loss=dict(direct=1., AT_replay=2., AT_readout=2.),
    gradual_C_loss=dict(direct=1., AP_replay=2., AP_readout=2., AT_replay=2., AT_readout=2.),
    C_auxiliary_states='AP and AT induced directly by C; three-hop chain is evaluation only',
    readout=dict(fit=1300, calibration=520, penalties=[.0001, .001, .01], epochs=200,
                 lr=.01, std_floor=1e-6, selection='highest calibration accuracy; first tie'),
    AT_gate=dict(changing_each_target_letter=.90, preservation_aggregate=.95,
                 factual_readout_overall=.95, factual_readout_each_letter=.90,
                 fail='stop entire run; no replacement/reseed/extension'),
    letters=list(LETTERS), colors=list(COLORS), splits=dict(train=96, calibration=48, final=108),
    alternatives=256, minimum_final_templates_per_transition=32,
    natural_filter='Full vocabulary native BF16 AND FP32 softcap correct; ties fail; fixed matched layout',
    AP_target='recipient label at donor AP, not donor letter',
    AT_target='donor natural answer letter, including letters outside recipient options',
    controls=dict(AP_donors=2, recipients=4, AT_donors=2, conflicts=2,
                  missing='retained in primary denominator; prevents qualification'),
    primary_AP='direct AND induced AT replay, changing pairs',
    primary_C='direct C AND AP replay through common STANDARD AP AND induced AT replay, changing pairs',
    bootstrap=dict(resamples=2000, seed=20261025, clusters='object/template', paired=True),
    superiority=dict(gain=.05, lower_above=0., maximum_direct_or_preservation_loss=.02, every_seed=True),
    total_gpu_seconds=43200, reserve_seconds=300, allowance=1.25, profile_limit_seconds=1800,
    history='Existing object split/vocabulary reused; final prompts fresh and excluded from inspected finals/new fits',
    no_router_training=True, no_site_rank_weight_search=True,
    authorization='User approved full matched AT/AP/C, fresh gated AT, all auxiliary weights 2, laptop 12h, and autonomous implementation',
)


def fit_names():
    return [f'AT_S{s}' for s in SEEDS] + [
        f'{v}_{m}_S{s}' for v in ('AP', 'C') for s in SEEDS for m in METHODS]
