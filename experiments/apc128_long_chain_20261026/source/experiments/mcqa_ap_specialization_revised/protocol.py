"""Scientific constants; changing these requires a new registered run."""
from pathlib import Path

ROOT = Path('eval/mcqa_ap_specialization_20261021_v2')
PARENT = Path('eval/mcqa_ap_specialization_20261021_v1')
SALT = 'mcqa_ap_specialization_20261021_v1'
REVISION = 'c5ebcd40d208330abc697524c919956e692655cf'
MODEL = Path('.cache_mcqa_c_at/hub/models--google--gemma-2-2b/snapshots') / REVISION
LETTERS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
COLORS = tuple('beige black blue brown burgundy crimson cyan gray green indigo magenta maroon navy olive orange pink purple red tan teal turquoise white yellow'.split())
DIRECTIONS = tuple((b,s) for b in range(4) for s in range(4) if b != s)
ARMS = ('shared', 'edge', 'middle')
SEEDS = (1,0,2)
STEPS = (0,100,250,500,750,1000)
BANKS = {'train_master':(104,52), 'calibration':(26,26), 'final':(104,52)}
PROTOCOL = dict(salt=SALT, pool_revision=2, parent=str(PARENT),
    approved_amendment='Vary factual color across the 256 alternatives; keep color identical within each base/source pair',
    color_schedule='COLORS[(index of original slot color + alternative index) modulo 23]',
    unchanged='Object splits and slot assignments, AP and letter quotas, alternative order, bank sizes, scientific settings, tolerances, safeguards, cumulative GPU caps',
    model='google/gemma-2-2b', model_revision=REVISION,
    tokenizer_revision=REVISION, colors=COLORS, letters=LETTERS, seeds=SEEDS,
    arms=ARMS, layer=17, position='last_real_prompt_token', rank=128, width=2304,
    optimizer='Adam', lr=.01, effective_batch=32, updates=1000, evaluations=STEPS,
    changing_per_batch=24, preservation_per_batch=8, checkpoint_every=25,
    training_count=832, master_count=1456, calibration_count=416, final_count=1456,
    object_splits={'train':96,'calibration':48,'final':108}, alternatives=256,
    shared_blocks=[0,1], bootstrap=2000, bootstrap_seed=20261021,
    atol=2e-4, rtol=2e-4, batch_candidates=[1,2,4,8,16,32],
    backbone='BF16 frozen', projection='FP32 QR', attention='sdpa', tf32=False,
    objective='full-vocabulary counterfactual CE only', auxiliary_losses=False,
    selection='fixed update 1000; no calibration checkpoint selection',
    total_gpu_seconds=21600, initial_profile_seconds=1800, budget_allowance=1.25,
    final_seal='All nine final checkpoints locked before final natural filtering',
    scope='Object-disjoint new fits; historical vocabulary/template reuse disclosed',
    gates={'gain_pp':5,'preservation_loss_pp':2,'transition_accuracy':.9,'preservation_accuracy':.95},
    reference_commit='399c2ca74837fc464c830aa8ab7af14b4d2327be',
    reference_blob='0943f9dd4a35e8dafd9f09207656d35e432ae8e9')

def in_domain(arm, destination):
    return arm == 'shared' or destination in ((0,3) if arm == 'edge' else (1,2))
