from pathlib import Path

ROOT = Path('eval/mcqa_ap_specialization_routing_20261023_v1')
PARENT = Path('eval/mcqa_ap_specialization_20261021_v3')
SALT = 'mcqa_ap_specialization_routing_20261023_v1'
REVISION = 'c5ebcd40d208330abc697524c919956e692655cf'
SEEDS = (1, 0, 2)
ARMS = {'shared': (128, 2000), 'edge': (128, 1000), 'middle': (128, 1000), 'large': (256, 2000)}
STEPS = (0, 100, 250, 500, 750, 1000, 1500, 2000)
LETTERS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
COLORS = tuple('beige black blue brown burgundy crimson cyan gray green indigo magenta maroon navy olive orange pink purple red tan teal turquoise white yellow'.split())
DIRECTIONS = tuple((b, s) for b in range(4) for s in range(4) if b != s)
SIZES = {'train': (64, 64), 'calibration': (16, 32), 'final': (64, 64), 'c_diagnostic': (16, 16)}
FROZEN = Path('.external/hai_review_publication_20261022')
PROTOCOL = dict(salt=SALT, model='google/gemma-2-2b', revision=REVISION, tokenizer_revision=REVISION,
    seeds=SEEDS, arms=ARMS, layer=17, position='final_real_token', colors=COLORS, letters=LETTERS,
    splits={'train':96, 'calibration':48, 'final':108}, sizes=SIZES, alternatives=256,
    changing_bundles='Matched transposition and three-cycle; third cycle slot alternates between the two remaining slots',
    minimum_final_templates_per_transition=32, final_template_rule='First 32 accepted bundles in each changing transition have distinct templates',
    natural_filter='Native BF16 rank-three head AND registered FP32-softcapped rank-three head strictly correct and untied; never intervention outcomes',
    selection='First eligible candidate in salted hash order; candidate questions/colors/letters vary; no letter quota',
    train_batch=32, changing_per_batch=24, preservation_per_batch=8, optimizer='Adam', lr=.01,
    steps=STEPS, checkpoint_every=25, backbone='frozen BF16 SDPA', projection='FP32 QR', tf32=False,
    atol=2e-4, rtol=2e-4, batches=[1,2,4,8,16,32], objective='Direct full-vocabulary output CE only',
    router=dict(input='unmodified donor block17 final-token state', standardization='training only',
        epochs=200, optimizer='Adam', lr=.01, penalties=[1e-4,1e-3,1e-2], seed=20261023,
        weighting='Equal AP-position weight', selection='Worst edge/middle accuracy, overall accuracy, negative CE, smaller penalty'),
    final_seal='All 12 endpoint handles plus shared step1000 checkpoints and router locked before final natural filtering',
    total_gpu_seconds=21600, new_profile_seconds=900, allowance=1.25, checkpoint_reserve_seconds=300,
    bootstrap=2000, bootstrap_seed=20261023, gates=dict(gain=.05, preservation_loss=.02, transition=.90, preservation=.95, router=.95, routing_gap=.02),
    interpretation='Conditional learned intervention; does not identify native switching circuitry',
    historical_scope='Objects held out from these fits; vocabulary and templates reused historically')

def domain(arm, dest):
    return arm in ('shared','large') or dest in ((0,3) if arm=='edge' else (1,2))
