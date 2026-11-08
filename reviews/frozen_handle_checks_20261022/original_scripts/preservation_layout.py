"""Diagnose two saved prefix-rounding discrepancies with matched fixed layouts."""
from dataclasses import asdict
import gc
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.hai_followup_checks_20261022 import run as follow
from diagnostics.hai_new_checkpoints_20261021 import recheck
from diagnostics.hai_frozen_eval_20261020 import run as old
from experiments.hai_c_at_evaluation import audit, schema
from experiments.hai_c_at_full_chain import gpu
from experiments.mcqa_c_at_localization import runtime

ROOT = Path('eval/hai_followup_preservation_layout_20261022_v1')


def main():
    torch.set_num_threads(2)
    contexts, pairs, schedule, _ = recheck.load()
    natural, base, _ = recheck.caches(schedule)
    indices = np.asarray(schedule['original_pair_indices'])
    source = [p.source for p in pairs]
    error = (natural.c[source]-base.c[indices]).abs().amax(1)
    eligible = [i for i, p in enumerate(pairs) if p.family == 'AT_only']
    eligible.sort(key=lambda i: (-float(error[i]), schema.digest(pairs[i].id)))
    selected = eligible[:32]
    plan = dict(selected_pair_indices=selected,
        selection='The largest cached C-prefix differences, then fixed hashes; diagnostic only, never a reporting-bank filter',
        cached_discrepancy_count=int((error[eligible]>0).sum()), cached_C_max_error=float(error[eligible].max()),
        contexts=[asdict(contexts[pairs[i].base]) for i in selected]+[asdict(contexts[pairs[i].source]) for i in selected],
        fixed_batch=32, no_training=True, GPU_limit_seconds=120,
        parent='eval/hai_followup_checks_20261022_v1', source=schema.file_hash(__file__))
    old.write(ROOT/'plan.json', plan)
    del natural, base
    follow.ROOT = ROOT
    model = engine = budget = None
    try:
        runtime.configure(ROOT)
        budget = gpu.Budget(ROOT, seconds=120)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(old.MODEL, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(old.MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        for p in model.parameters():
            p.requires_grad_(False)
        engine = follow.Engine(model, tokenizer, plan, budget)
        before = engine.forward(list(range(32)), capture=True)
        donor = engine.forward(list(range(32,64)), capture=True)
        checks = []
        old.compare('matched_color_prefix_states', before['color'], donor['color'], checks)
        ids, mask, _ = engine.inputs(list(range(32)))
        bank = dict(base_input_ids=ids.cpu(), base_attention_mask=mask.cpu(), base_position_by_id={
            'color': torch.tensor([engine.color_positions[i]-(engine.width-len(engine.encoded[i]['input_ids'])) for i in range(32)]),
            'last_token': torch.tensor([len(engine.encoded[i]['input_ids'])-1 for i in range(32)])})
        ref = audit.selected_functions(old.SHARED/'mcqa_gradual_das.py',['answer_logits'])
        ref = audit.selected_functions(old.SHARED/'mcqa_gradual_das_c.py',['run_c_intervention_and_capture'],ref)
        with torch.no_grad():
            for method in ('standard','gradual'):
                budget.tick()
                q = recheck.checkpoint(method,'C')['basis'].float().cuda()
                logits, _, _ = ref['run_c_intervention_and_capture'](model,bank,donor['color'],q,0,17,25,None,
                    batch_size=32,require_grad=False)
                old.compare(method+'_C_preservation_output', before['logits'],logits,checks)
                budget.tick()
        if any(b._forward_hooks for b in model.model.layers) or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Hook cleanup or backbone freeze failed')
        old.write(ROOT/'verification.json',dict(passed=True,checks=checks,
            matched_prefix_max_error=float((before['color']-donor['color']).abs().max()),
            cached_discrepancies_retained=True,no_training=True,diagnostic_only=True,
            interpretation='The two historical cache-prefix differences disappear with matched physical layouts; both frozen C interventions then reproduce clean logits exactly'))
        print(json.dumps(dict(passed=True,checks=checks)),flush=True)
    finally:
        engine=model=None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if budget is not None:
            budget.close()
        runtime.close()


if __name__ == '__main__':
    main()
