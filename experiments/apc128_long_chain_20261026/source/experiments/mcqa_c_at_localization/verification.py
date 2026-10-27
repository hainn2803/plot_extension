"""CPU alignment audit and fail-closed, matched-arithmetic real-model checks."""
from dataclasses import replace
from itertools import product
from pathlib import Path
import importlib.metadata
import platform

from .data import Prompt, examples
from .protocol import COLORS, LABELS, MODEL_PATH, MODEL_REVISION, PROTOCOL, allocate_groups
from .storage import file_hash, write

KNOWN_MODEL_HASHES = {
    "config.json": "56ee4e528fedfe70b2fabb0704eec9f2ef94461de936928220a3d6e7fac5ef98",
    "model-00001-of-00003.safetensors": "1425aa066ec77e3eb79aac14a5bdea3ebcec46aa5c96cd40608c5c1fd70d193d",
    "model-00002-of-00003.safetensors": "96c111d3dcdbde9271595e463b5d9f7fc4810ad8b79e736309c0a1833e6c0d35",
    "model-00003-of-00003.safetensors": "4e08abc64d1767fdacd2c94da7f2ec4b8c65b25b19a53e87d19dc432901b5f02",
    "model.safetensors.index.json": "e5c6f26fbe40fc3712f4df57da9b89c58b033714316a4547d052b37f43217e8e",
    "tokenizer.json": "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": "bb8245b8d39a065fdf869ee4fce48db9cc491473eb7ff1b41d321fb4caf540d1",
}


def asset_hashes(include_weights=True):
    hashes = {}
    for name, expected in KNOWN_MODEL_HASHES.items():
        if not include_weights and name.endswith("safetensors"):
            continue
        actual = file_hash(MODEL_PATH / name)
        if actual != expected:
            raise ValueError("Pinned model file differs: " + name)
        hashes[name] = actual
    return hashes


def environment():
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "huggingface_hub", "numpy", "scipy", "tokenizers", "scikit-learn")},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "torch_cpu_threads": torch.get_num_threads(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def alignment(tokenizer):
    checked = 0
    reference = None
    names = ["BOS", "prefix_question", "prefix_colon", "fact1_article", "fact1_object", "fact1_is", "fact1_value", "fact1_period",
             "fact2_article", "fact2_object", "fact2_is", "fact2_value", "fact2_period", "query_word", "query_color", "query_is", "query_article", "query_object", "query_end", "newline0",
             "option1_label", "option1_period", "option1_color", "newline1", "option2_label", "option2_period", "option2_color", "newline2",
             "option3_label", "option3_period", "option3_color", "newline3", "option4_label", "option4_period", "option4_color", "newline4", "answer_word", "answer_colon"]
    for groups in allocate_groups().values():
        for group, query, order, color, other, wording in product(groups, (0, 1), ((0, 1), (1, 0)), COLORS, (*COLORS, "here"), ("What", "Which")):
            if color == other:
                continue
            facts = [other, other]
            facts[query] = color
            prompt = Prompt(tuple(group), tuple(facts), query, order, COLORS, LABELS, wording)
            encoded = tokenizer(prompt.text, return_offsets_mapping=True, add_special_tokens=True)
            ids = encoded["input_ids"]
            if len(ids) != len(names) or ids[0] != tokenizer.bos_token_id:
                raise ValueError(f"Fixed slot alignment fails for {prompt.text!r}: {len(ids)} tokens")
            text_pieces = [prompt.text[a:b].strip() for a, b in encoded["offset_mapping"]]
            required = {4: prompt.objects[order[0]], 6: facts[order[0]], 9: prompt.objects[order[1]],
                        11: facts[order[1]], 13: wording, 17: prompt.objects[query]}
            required.update({20 + 4*i: label for i, label in enumerate(LABELS)})
            required.update({22 + 4*i: color for i, color in enumerate(COLORS)})
            if any(text_pieces[i] != text for i, text in required.items()):
                raise ValueError("A semantic field crosses its registered fixed token slot")
            checked += 1
            if reference is None:
                reference = {"text": prompt.text, "ids": ids, "pieces": text_pieces}
    from .answers import answer_ids
    label_ids = [[token] for token in answer_ids(tokenizer).values()]
    color_ids = [tokenizer.encode(" " + s, add_special_tokens=False) for s in COLORS]
    if any(len(v) != 1 for v in label_ids + color_ids):
        raise ValueError("Lens labels/colors must be single tokens")
    return {"checked_templates": checked, "tokens_including_bos": len(names),
            "slots": [{"position": i, "name": name} for i, name in enumerate(names) if i],
            "sites": [{"layer": layer, "position": i} for layer in range(26) for i in range(1, len(names))],
            "candidate_count": 26 * (len(names) - 1), "reference": reference,
            "label_ids": [v[0] for v in label_ids], "color_ids": [v[0] for v in color_ids],
            "answer_continuation_prefix": " "}


def numerical(model, tokenizer, batch, root):
    import torch
    from . import backend
    from .lens import load_lens, decode
    from .reference import jonathan_patch
    directory = Path(root) / "engineering"
    attempt = len(list(directory.glob(f"numerical_batch_{batch}*.json"))) + 1
    path = directory / f"numerical_batch_{batch}_attempt_{attempt:02d}.json"
    rows = examples()
    ids = backend.tokenize(tokenizer, [rows[i % 8]["base"]["text"] for i in range(batch)]).to("cuda")
    source_ids = backend.tokenize(tokenizer, [rows[i % 8]["source"]["text"] for i in range(batch)]).to("cuda")
    checks = []
    def check(name, left, right):
        difference = float((left.detach().float() - right.detach().float()).abs().max())
        ok = torch.allclose(left, right, atol=2e-4, rtol=2e-4)
        checks.append({"name": name, "max_abs_difference": difference, "passed": ok})
        write(path, {"batch": batch, "status": "checking" if ok else "failed", "checks": checks})
        if not ok:
            raise RuntimeError(f"Numerical gate failed: {name}; max abs difference {difference}; tolerances unchanged")
    versions = {n: p._version for n, p in model.named_parameters()}
    torch.manual_seed(1)
    with torch.no_grad():
        base, states = backend.forward(model, ids, capture=range(26))
        source, source_states = backend.forward(model, source_ids, capture=range(26))
        repeat, _ = backend.forward(model, ids)
        check("repeated_forward", repeat, base)
        native_head = {}
        hook = model.lm_head.register_forward_hook(lambda _m, _i, output: native_head.update(raw=output.detach()))
        try:
            native = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1].float()
        finally:
            hook.remove()
        cap = model.config.final_logit_softcapping
        raw = native_head["raw"]
        native_reconstructed = (torch.tanh(raw / cap) * cap)[:, -1].float()
        hai_reconstructed = (torch.tanh(raw.float() / cap) * cap)[:, -1]
        check("native_BF16_softcap_reconstruction", native, native_reconstructed)
        check("native_head_with_registered_FP32_softcap", base, hai_reconstructed)
        native_softcap_difference = float((base - native).abs().max())
        del native_head, raw
        matrices, metadata = load_lens()
        check("J_lens_last_block_identity", decode(model, states[25], 25, matrices)[:, -1], base)
        del matrices
        for layer in range(26):
            for pos in (6, ids.shape[1] - 1):
                identity, _ = backend.forward(model, ids, [backend.Patch(layer, pos, states[layer][:, pos])])
                check(f"identity_L{layer}_P{pos}", identity, base)
        no_path, _ = backend.forward(model, ids, [backend.Patch(25, 6, source_states[25][:, 6])])
        check("final_block_earlier_token_no_path", no_path, base)
        last, _ = backend.forward(model, ids, [backend.Patch(25, ids.shape[1]-1, source_states[25][:, -1])])
        check("final_block_full_vector_equals_clean_source", last, source)
    for rank in (8, 32, 128, 512):
        subspace = backend.LearnedSubspace(2304, rank).to("cuda")
        q = subspace.basis()
        b, s = states[0][:, 6].float(), source_states[0][:, 6].float()
        direct = backend.project(b, s, q)
        trusted = jonathan_patch(b, s, q)
        check(f"trusted_fp32_patch_rank{rank}", direct, trusted)
        probe = torch.randn_like(direct)
        gd = torch.autograd.grad((direct * probe).sum(), subspace.raw, retain_graph=True)[0]
        gt = torch.autograd.grad((trusted * probe).sum(), subspace.raw)[0]
        check(f"trusted_fp32_raw_basis_gradient_rank{rank}", gd, gt)
        with torch.no_grad():
            q = subspace.basis()
            patched, _ = backend.forward(model, ids, [backend.Patch(0, 6, source_states[0][:, 6], q)])
            centered = backend.same_shape_reference(states[0][:, 6], source_states[0][:, 6], q, jonathan_patch)
            reference, _ = backend.forward(model, ids, [backend.Patch(0, 6, centered)])
            check(f"matched_order_bf16_logits_rank{rank}", patched, reference)
    backend.assert_frozen(model, versions)
    if any(layer._forward_hooks for layer in model.model.layers):
        raise RuntimeError("Intervention hooks leaked")
    receipt = {"batch": batch, "status": "passed", "checks": checks, "lens": metadata,
               "atol": 2e-4, "rtol": 2e-4, "environment": environment(),
               "reference_arithmetic": "original Jonathan in FP32; centered Jonathan before BF16",
               "softcap_arithmetic": "registered Hai FP32 after BF16 full-sequence head; native HF BF16 reconstructed separately",
               "native_BF16_softcap_max_difference": native_softcap_difference,
               "backend_sha256": file_hash(Path(backend.__file__))}
    write(path, receipt)
    return receipt
