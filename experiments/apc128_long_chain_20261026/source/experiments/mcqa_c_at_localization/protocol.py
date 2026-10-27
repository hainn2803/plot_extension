"""Scientific settings. No settings are inferred from observed performance."""
from hashlib import sha256
from itertools import permutations
from pathlib import Path

NAME = "mcqa_c_at_localization_20261015_v1"
ROOT = Path("eval") / NAME
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
MODEL_PATH = Path(".cache_mcqa_c_at/hub/models--google--gemma-2-2b/snapshots") / MODEL_REVISION
LENS_ID = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
LENS_FILE = "gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt"
COLORS = ("red", "blue", "green", "yellow")
LABELS = ("A", "B", "C", "D")
PERMUTATIONS = tuple(permutations(range(4)))
SUBJECTS = tuple((
    "ball book cup chair table car boat kite box bag hat coat shirt shoe sock scarf "
    "flag door wall roof fence gate lamp vase bowl plate spoon fork bottle basket bench desk "
    "clock phone pen pencil folder notebook envelope ribbon towel blanket pillow curtain carpet bucket truck bicycle "
    "card mug tray cap tape drum cube disk tile bead ring cone bell brush comb glove boot vest dress skirt suit belt buckle sandal "
    "wallet purse sack case crate chest bin jar can pot pan kettle jug glass dish saucer tub sink stool sofa couch cabinet drawer shelf"
).split())
FAMILIES = (
    "C_change_AT_fixed", "labels_only", "contents_only", "both_options_fixed",
    "four_way", "row_order_only", "irrelevant_fact", "requested_object",
)
BANKS = {
    "train": {"groups": 16, "pairs_per_cell": 96},
    "calibration": {"groups": 8, "pairs_per_cell": 48},
    "readout_fit": {"groups": 4, "prompts": 512},
    "readout_validation": {"groups": 4, "prompts": 256},
    "test": {"groups": 16, "pairs_per_cell": 192},
}
PROTOCOL = {
    "name": NAME, "model": MODEL_ID, "model_revision": MODEL_REVISION,
    "lens": LENS_ID, "lens_revision": LENS_REVISION, "lens_file": LENS_FILE,
    "colors": COLORS, "labels": LABELS, "families": FAMILIES, "banks": BANKS,
    "subjects": SUBJECTS, "seeds": [1, 0, 2], "ranks": [8, 32, 128, 512],
    "layers": 26, "width": 2304, "dtype": "bfloat16", "projection_dtype": "float32",
    "attention": "sdpa", "tf32": False, "optimizer": "Adam", "lr": 0.01,
    "effective_batch": 32, "epochs": 20, "loss": "full_vocabulary_output_CE",
    "patch": "base + ((source.float() - base.float()) @ Q) @ Q.T",
    "strength": 1, "atol": 2e-4, "rtol": 2e-4, "max_trainings": 52,
    "candidate_alternatives_per_slot": 128, "localization_pairs_per_cell": 24,
    "plot": {"epsilon": 0.05, "beta": 0.5, "cost": "sum_squared_effect_difference / pairs",
             "signature": "four_label_softmax_swap_minus_base", "abstract_marginal": [0.5, 0.5],
             "neural_marginal": "uniform", "sidedness": "fixed_abstract_relaxed_neural"},
    "shortlist_per_method_variable": 2, "gate": 0.90, "min_direction_cases": 25,
    "bootstrap": 2000, "structural_prompts": 64, "wording_pairs": 192,
    "local_gpu_hours_with_allowance": 12, "runtime_allowance": 1.25,
    "final_filter_requires_all_selection_locks": True, "gradual": False,
    "one_color_requested_object": "preserve_color_while_moving_colored_fact_and_query",
}


def order_key(value):
    return sha256(f"{NAME}:{value}".encode()).hexdigest()


def allocate_groups():
    assert len(SUBJECTS) == len(set(SUBJECTS)) == 96
    ordered = sorted(SUBJECTS, key=order_key)
    groups = [ordered[i:i + 2] for i in range(0, 96, 2)]
    result, start = {}, 0
    for name, spec in BANKS.items():
        result[name] = groups[start:start + spec["groups"]]
        start += spec["groups"]
    assert start == 48
    return result
