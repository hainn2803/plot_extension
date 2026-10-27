from pathlib import Path

ROOT = Path('eval/mcqa_ap_localization_sweep_20261024_v1')
SALT = ROOT.name
REVISION = 'c5ebcd40d208330abc697524c919956e692655cf'
SEEDS = (1, 0, 2)
LETTERS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
COLORS = tuple('beige black blue brown burgundy crimson cyan gray green indigo magenta maroon navy olive orange pink purple red tan teal turquoise white yellow'.split())
DIRECTIONS = tuple((b, s) for b in range(4) for s in range(4) if b != s)
SIZES = {'train': (64, 64), 'calibration': (16, 32), 'final': (64, 64), 'c_diagnostic': (16, 16)}
FROZEN = Path('.external/hai_review_publication_20261022')
DIAGNOSIS = Path('eval/mcqa_ap_specialization_routing_20261023_v1/engineering/memory_diagnosis/1792797934777491700/receipt.json')
STEPS = (0, 100, 250, 500, 750, 1000)
PROTOCOL = dict(salt=SALT, model='google/gemma-2-2b', revision=REVISION, tokenizer_revision=REVISION,
    seeds=SEEDS, layers=list(range(26)), colors=COLORS, letters=LETTERS, sizes=SIZES,
    splits={'train':96,'calibration':48,'final':108}, alternatives=256,
    single_sites='All end-relative offsets in registered width, plus each of four label/last-color structural tokens',
    missing_site='Identity if either source or recipient token is padding; retain row',
    distributed_sites='All four labels or all four row-final color subtokens, every layer; rank128',
    selection='worst relevant changing/preservation cell, mean changing, mean preservation, lower CE, ascending site ID',
    screen_updates=50, screen_seed=1, screen_rank=128, screen_steps=(0,50),
    final_updates=1000, final_steps=STEPS, final_slots=('shared','edge','middle','distributed','capacity','baseline'),
    train_batch=32, changing_per_batch=24, preservation_per_batch=8, optimizer='Adam',lr=.01,
    checkpoint_every=25, atol=2e-4,rtol=2e-4,batches=(1,2,4,8,16,32),
    backbone='frozen BF16 SDPA',projection='FP32 QR',tf32=False,objective='Direct full-vocabulary CE only',
    total_gpu_seconds=86400,profile_limit_seconds=1800,reserve_seconds=300,allowance=1.25,
    natural_filter='Strict native BF16 and registered FP32 softcap correctness, no ties; matched fixed layout',
    minimum_final_templates_per_transition=32, bootstrap=2000,bootstrap_seed=20261024,
    final_seal='All confirmation endpoints locked before final filtering',
    controls=dict(AP_donors=2,other_recipients=4,AT_donors=2,alternatives=256,
        missing='Keep primary examples, report missing controls, qualification fails'),
    gates=dict(gain=.05,preservation_loss=.02,changing=.90,preservation=.95,cross_question=.90),
    history='Objects held out from new fits, not from all historical work; vocabulary reused',
    frozen_C='Audited standard and gradual2.0, not Hai newer all-auxiliary configuration',
    no_router_training=True,no_automatic_followup=True)

def domain(arm, dest):
    return arm not in ('edge','middle') or dest in ((0,3) if arm=='edge' else (1,2))
