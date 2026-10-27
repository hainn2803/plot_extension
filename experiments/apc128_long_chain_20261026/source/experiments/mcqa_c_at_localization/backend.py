"""Arbitrary fixed-token interventions with the registered FP32 QR arithmetic."""
from dataclasses import dataclass
import torch

from .protocol import MODEL_PATH, PROTOCOL


class LearnedSubspace(torch.nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        init = torch.randn(width, rank, dtype=torch.float32)
        self.raw = torch.nn.Parameter(torch.linalg.qr(init, mode="reduced").Q)

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


def project(base, source, basis):
    base32, source32 = base.float(), source.float()
    if basis.dtype != torch.float32:
        raise ValueError("Projections must use FP32")
    return (base32 + ((source32 - base32) @ basis) @ basis.T).to(base.dtype)


def settings():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    settings()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if model.config.hidden_size != 2304 or model.config.num_hidden_layers != 26:
        raise ValueError("Unexpected model architecture")
    if model.config._attn_implementation != "sdpa":
        raise ValueError("Unexpected attention backend")
    return model, tokenizer


@dataclass
class Patch:
    layer: int
    position: int | torch.Tensor
    source: torch.Tensor
    basis: torch.Tensor | None = None


def forward(model, ids, patches=(), capture=()):
    """Capture block outputs before final RMSNorm; always compute the full-sequence head."""
    from .runtime import guard
    guard()
    if ids.ndim != 2:
        raise ValueError("Expected equal-length, unpadded token batches")
    handles, states = [], {}
    patches_by_layer = {}
    for patch in patches:
        if not 0 <= patch.layer < len(model.model.layers):
            raise ValueError("Invalid block index")
        patches_by_layer.setdefault(patch.layer, []).append(patch)
    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer in patches_by_layer:
                hidden = hidden.clone()
                rows = torch.arange(len(ids), device=hidden.device)
                for patch in patches_by_layer[layer]:
                    pos = patch.position
                    if isinstance(pos, torch.Tensor):
                        pos = pos.to(hidden.device)
                    source = patch.source.to(hidden.device)
                    if patch.basis is None:
                        replacement = source.to(hidden.dtype)
                    else:
                        replacement = project(hidden[rows, pos], source, patch.basis)
                    hidden[rows, pos] = replacement
            if layer in capture:
                states[layer] = hidden
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return hook
    try:
        for layer in sorted(set(capture) | set(patches_by_layer)):
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
        mask = torch.ones_like(ids)
        positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                             use_cache=False, return_dict=True)
        logits = model.lm_head(output.last_hidden_state).float()
        softcap = model.config.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits[:, -1], states
    finally:
        for handle in handles:
            handle.remove()


def tokenize(tokenizer, prompts):
    encoded = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    if len({len(row) for row in encoded}) != 1:
        raise ValueError("Token length mismatch; no automatic padding or template substitution")
    return torch.tensor(encoded, dtype=torch.long)


def summarize_logits(logits, label_ids):
    logits = logits.detach().float()
    restricted = logits[:, label_ids]
    return {"prediction": logits.argmax(-1).cpu().tolist(),
            "label_logits": restricted.cpu().tolist(),
            "label_probabilities": restricted.softmax(-1).cpu().tolist()}


def assert_frozen(model, versions):
    if any(p.requires_grad or p.grad is not None or p._version != versions[name]
           for name, p in model.named_parameters()):
        raise RuntimeError("Frozen backbone was changed or received parameter gradients")


def same_shape_reference(base, source, basis, trusted):
    """Center the trusted equation to match FP32 operation order before BF16 rounding."""
    base32, delta = base.float(), source.float() - base.float()
    return (base32 + trusted(torch.zeros_like(base32), delta, basis)).to(base.dtype)
