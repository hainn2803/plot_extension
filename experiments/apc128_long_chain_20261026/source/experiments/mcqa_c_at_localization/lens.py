"""Published J-lens transport, with explicit block-output and matrix conventions."""
from pathlib import Path
import torch

from .protocol import LENS_REVISION, LENS_FILE
from .storage import file_hash

LENS_PATH = Path(".cache_mcqa_c_at/hub/models--neuronpedia--jacobian-lens/snapshots") / LENS_REVISION / LENS_FILE


def load_lens(path=LENS_PATH):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("d_model") != 2304 or artifact.get("source_layers") != list(range(25)):
        raise ValueError("Published lens architecture/capture points do not match")
    matrices = artifact["J"]
    if set(matrices) != set(range(25)) or any(tuple(j.shape) != (2304, 2304) or not torch.isfinite(j).all() for j in matrices.values()):
        raise ValueError("Invalid published Jacobian tensors")
    return matrices, {"file_sha256": file_hash(path), "revision": LENS_REVISION,
                      "n_prompts": artifact["n_prompts"], "source_layers": artifact["source_layers"],
                      "matrix_convention": "J[out_coordinate,in_coordinate]; hidden @ J.T",
                      "capture": "decoder block output before final RMSNorm", "last_block": "identity",
                      "checkpoint_revision_inside_artifact": "not recorded by publisher"}


def decode(model, hidden, layer, matrices):
    if not 0 <= layer <= 25:
        raise ValueError("Invalid layer")
    if layer != 25:
        hidden = (hidden.float() @ matrices[layer].to(hidden.device, torch.float32).T).to(hidden.dtype)
    logits = model.lm_head(model.model.norm(hidden)).float()
    softcap = model.config.final_logit_softcapping
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def decoding_metrics(logits, targets, candidate_ids):
    targets = targets.to(logits.device)
    label_logits = logits[..., candidate_ids]
    predicted = torch.tensor(candidate_ids, device=logits.device)[label_logits.argmax(-1)]
    true_scores = logits.gather(-1, targets[..., None]).squeeze(-1)
    # Competition rank gives all exact ties the same rank, without arbitrary token ordering.
    ranks = 1 + (logits > true_scores[..., None]).sum(-1)
    return predicted.eq(targets), ranks.float().reciprocal()
