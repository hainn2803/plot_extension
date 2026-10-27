from copy import deepcopy
from pathlib import Path
from experiments.mcqa_ap_gradual_chain_20261025.protocol import PROTOCOL as ORIGINAL, SEEDS, METHODS, BANK_SIZES

ROOT = Path('eval/mcqa_apc_128_20261026_v1')
PARENT = Path('eval/mcqa_ap_gradual_chain_20261025_v1')
SWEEP = Path('eval/mcqa_at_rank_sweep_20261026_v1')
PROTOCOL = deepcopy(ORIGINAL)
PROTOCOL.update(
    name=ROOT.name, amendment_of=str(PARENT),
    authorization='User: stay with the 128 dim handles, with current accuracies, dropping AT gates and proceeding with C/AP Standard + Gradual',
    AT_gate=dict(enabled=False, reason='Explicit user amendment; actual AT calibration/readout weaknesses retained and reported'),
    AT_teachers='Reuse frozen completed rank128 seeds1,0,2 and their actual selected factual readouts; do not refit or replace',
    order='Frozen AT teachers imported; AP Standard/Gradual then C Standard/Gradual; lock all endpoints/readouts before final access',
    new_fits=12, imported_fits=3, new_updates=12000, imported_updates=3000,
    localization_scope='Learn rank128 subspaces at the existing registered sites; no new site, rank or weight search',
    same_teacher_within_seed=True,
    proposal_salt='mcqa_ap_gradual_chain_20261025_v1; original frozen proposals/order generation retained exactly',
    selected_nonfinal_banks='All ten original selected banks reused exactly, including original fixed256 filtering evidence',
    budget='Same existing43200s allocation; all8769.316136500012s prior charges inherited, no reset',
    inherited_seconds=8769.316136500012, inherited_profile_seconds=853.1099996000121,
    original_profile_reused=True,
    engineering='Independent audit accepts only exact rational or exact original FP32 mean representation for identical integer counts; no scoring, loss, tolerance or criterion change',
    limitation='Ungated imperfect AT teachers can bias downstream losses and measurements; consistency is not proof of unique endogenous/native computation',
)
