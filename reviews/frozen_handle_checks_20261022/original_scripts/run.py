"""Frozen C/AT readback and AP transposition-versus-cycle control; no training."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import run as old, descriptive as desc
from diagnostics.hai_new_checkpoints_20261021 import recheck
from experiments.hai_c_at_evaluation import core, schema, backends, audit
from experiments.hai_c_at_full_chain import gpu, verification
from experiments.mcqa_c_at_localization import runtime

ROOT = Path('eval/hai_followup_checks_20261022_v1')
AP = Path('.external/hai_followup_399c2ca_20261022/das_ap.pt')
SALT = 'hai_followup_AP_control_20261022_v1'
BATCH = 32


def save(name, payload, immutable=True):
    old.write(ROOT / name, payload, immutable=immutable)


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def position(c):
    return c.contents.index(c.color)


def saved():
    contexts, pairs, plan, _ = recheck.load()
    indices = np.asarray(plan['original_pair_indices'])
    controls = gpu.read_operation(recheck.ROOT, 'pair_controls')
    identity = old.read(recheck.ROOT / 'operations/pair_controls/identity.json')
    control_ok = controls['correct'].reshape(len(pairs), 2)
    targets = [contexts[p.base].label_for(contexts[p.source].color) for p in pairs]
    if identity['targets'] != np.repeat(targets, 2).tolist():
        raise ValueError('Clean donors and induced replay do not share targets')
    changing = np.array([contexts[p.base].color != contexts[p.source].color for p in pairs])
    masks = {'all': np.ones(len(pairs), bool), 'C_changing': changing, 'C_preservation': ~changing}
    masks.update({f: np.array([p.family == f for p in pairs]) for f in old.FAMILIES})
    output = dict(source=str(recheck.ROOT), pairs=len(pairs), no_training=True,
        pair_bank='Locked reconstructed historical bank; not Hai exact retained evaluation selection',
        clean_donor_count=2, targets_verified=True, methods={},
        controls_qualification=old.read(recheck.ROOT / 'results.json')['controls'],
        source_hashes={str(recheck.ROOT / f): schema.file_hash(recheck.ROOT / f)
            for f in ('identity.json', 'plan.json', 'results.json', 'independent_audit.json')},
        conditioning='Control-success subsets describe errors; primary population unchanged')
    for method in ('standard', 'gradual'):
        a = arrays(recheck.ROOT / method / 'predictions.npz')
        if a['target'][indices].tolist() != targets:
            raise ValueError('Historical pair order changed')
        d, r = a['direct_correct'][indices], a['replay_correct'][indices]
        out = {}
        for cell, mask in masks.items():
            both = control_ok.all(1)
            out[cell] = dict(direct=core.rate(d[mask]), replay=core.rate(r[mask]), joint=core.rate((d & r)[mask]),
                clean_donor_each=core.rate(control_ok[mask].reshape(-1)), both_clean_donors=core.rate(both[mask]),
                paired_outcomes={
                    'clean_both_pass_replay_pass': int((mask & both & r).sum()),
                    'clean_both_pass_replay_fail': int((mask & both & ~r).sum()),
                    'clean_not_both_pass_replay_pass': int((mask & ~both & r).sum()),
                    'clean_not_both_pass_replay_fail': int((mask & ~both & ~r).sum())},
                direct_success_replay_fail_despite_both_clean_donors=int((mask & d & ~r & both).sum()),
                conditional_replay_given_both_clean_donors=core.rate(r[mask & both]),
                conditional_joint_given_both_clean_donors=core.rate((d & r)[mask & both]),
                matched_clean=core.rate(a['matched_base_correct'][indices][mask]))
        output['methods'][method] = out
    output['toy'] = verification.toy_check()
    save('saved_C_AT_checks.json', output)
    print(json.dumps(dict(saved_checks=True, pairs=len(pairs), changing={m: output['methods'][m]['C_changing'] for m in output['methods']})), flush=True)


def rearrange(c, order):
    matches = list(schema.CHOICE.finditer(c.prompt))
    if len(matches) != 4:
        raise ValueError('Four option rows required')
    lines = [m.group(0) for m in matches]
    prompt = c.prompt
    for slot in reversed(range(4)):
        m = matches[slot]
        prompt = prompt[:m.start()] + lines[order[slot]] + prompt[m.end():]
    return replace(c, id=schema.digest(prompt), prompt=prompt,
        labels=tuple(c.labels[i] for i in order), contents=tuple(c.contents[i] for i in order))


def prepare():
    contexts, pairs = schema.import_pairs(old.read(desc.ROOT / 'selected_pairs.json'))
    blob = AP.read_bytes()
    if hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest() != '0943f9dd4a35e8dafd9f09207656d35e432ae8e9':
        raise ValueError('Pinned AP Git blob differs')
    ck = torch.load(AP, map_location='cpu', weights_only=True)
    q = backends.checked_basis(ck, 2304, 'AP')
    if ck['layer'] != 17 or ck['token_position'] != 'last_token' or q.shape[1] != 128:
        raise ValueError('Uploaded AP site/rank differs')
    selected = []
    for start in range(4):
        for dest in range(4):
            if start == dest:
                continue
            eligible = [p for p in pairs if p.family == 'AP_only' and position(contexts[p.base]) == start and position(contexts[p.source]) == dest]
            eligible.sort(key=lambda p: schema.digest([SALT, p.id]))
            if len(eligible) < 8:
                raise ValueError('Fewer than eight historical pairs in a transition')
            selected.extend(eligible[:8])
    bank, ids, rows = [], {}, []
    def add(c):
        if c.id not in ids:
            ids[c.id] = len(bank)
            bank.append(asdict(c))
        return ids[c.id]
    for p in selected:
        b, s = contexts[p.base], contexts[p.source]
        start, dest = position(b), position(s)
        order = list(range(4))
        order[start], order[dest] = order[dest], order[start]
        if rearrange(b, order).prompt != s.prompt:
            raise ValueError('Original source is not the registered transposition')
        variants = [('transposition', s)]
        for other in sorted(set(range(4)) - {start, dest}):
            cycle = list(range(4))
            cycle[dest], cycle[start], cycle[other] = start, other, dest
            variants.append(('cycle_' + str(other), rearrange(b, cycle)))
        for name, donor in variants:
            if donor.color != b.color or donor.answer != b.answer or position(donor) != dest:
                raise ValueError('Source abstract state changed')
            if (donor.labels[start] == b.labels[dest]) != (name == 'transposition'):
                raise ValueError('AP and label-copy targets not separated')
            rows.append(dict(pair_id=p.id, base=add(b), source=add(donor), variant=name,
                start=start, dest=dest, target=b.labels[dest], copy_target=donor.labels[start]))
    plan = dict(contexts=bank, rows=rows, original_pairs=96, transitions=12,
        original_bank=str(desc.ROOT / 'selected_pairs.json'),
        subset_rule='Eight per transition, SHA256 order before inference; no outcome-based refilling',
        sources='Original transposition and both three-row cycles', layers=list(range(26)),
        patch_site='Source and base at base-correct structural label slot; annotation-routed',
        learned_AP=dict(path=str(AP), sha256=schema.file_hash(AP), layer=17, slot='last_token', rank=128),
        settings=dict(model_revision=old.MODEL.name, backbone='BF16', patch='FP32', attention='sdpa',
            TF32=False, physical_batch=32, padding='Global fixed width and explicit position IDs', atol=2e-4, rtol=2e-4),
        reporting='All attempts and shared base/all-three-donor natural-correct subset; no replacements',
        no_training=True, GPU_limit_seconds=900,
        authorization='User authorized local frozen-handle checks and sensible recorded implementation choices')
    save('plan.json', plan)
    sources = [Path(__file__), Path(runtime.__file__), Path(gpu.__file__), Path(core.__file__), Path('diagnostics/ap_thermal_20261021/resume_ap.py')]
    save('identity.json', dict(plan=schema.file_hash(ROOT / 'plan.json'),
        sources={str(p): schema.file_hash(p) for p in sources}, checkpoint=schema.file_hash(AP),
        prior_results=schema.file_hash(recheck.ROOT / 'results.json')))
    print(json.dumps(dict(prepared=True, contexts=len(bank), variants=len(rows), layers=26)), flush=True)


class Engine:
    letters = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tie_tolerance = 0.0

    def __init__(self, model, tokenizer, plan, budget):
        self.model, self.tokenizer, self.budget = model, tokenizer, budget
        self.contexts = [schema.Context(**c) for c in plan['contexts']]
        self.device = next(model.parameters()).device
        self.label_ids = old.read(old.DATA / 'protocol.json')['label_token_ids']
        self.q = torch.load(AP, map_location='cpu', weights_only=True)['basis'].float().to(self.device)
        self.encoded = [tokenizer(c.prompt, add_special_tokens=True, return_offsets_mapping=True) for c in self.contexts]
        self.width = max(len(e['input_ids']) for e in self.encoded)
        self.positions, self.color_positions = [], []
        for c, e in zip(self.contexts, self.encoded):
            found = []
            for offset in [*[m.start(1) for m in schema.CHOICE.finditer(c.prompt)], c.color_span[0]]:
                matching = [i for i, (a, b) in enumerate(e['offset_mapping']) if a <= offset < b]
                if len(matching) != 1:
                    raise ValueError('Token offset not unique')
                found.append(self.width-len(e['input_ids'])+matching[0])
            self.positions.append(found[:4])
            self.color_positions.append(found[4])

    def decode(self, token):
        return self.tokenizer.decode([token])

    def inputs(self, indices):
        padded = list(indices) + [indices[0]]*(BATCH-len(indices))
        ids = torch.full((BATCH, self.width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(padded):
            tokens = self.encoded[i]['input_ids']
            ids[row, -len(tokens):] = torch.tensor(tokens, device=self.device)
            mask[row, -len(tokens):] = 1
        return ids, mask, padded

    @torch.no_grad()
    def forward(self, indices, layer=None, donor=None, slots=None, subspace=False, capture=False):
        self.budget.tick()
        gpu.cooling(ROOT, self.budget)
        ids, mask, padded = self.inputs(indices)
        rows = torch.arange(BATCH, device=self.device)
        states, hooks = {}, []
        def hook_for(number):
            def hook(_module, _inputs, out):
                h = out[0] if isinstance(out, tuple) else out
                if number == layer:
                    pos = torch.full((BATCH,), self.width-1, device=self.device, dtype=torch.long) if subspace else torch.tensor(
                        [self.positions[i][slots[j]] for j, i in enumerate(padded)], device=self.device)
                    d = donor.to(self.device).float()
                    if len(d) < BATCH:
                        d = torch.cat((d, d[:1].expand(BATCH-len(d), -1)))
                    changed = h.clone()
                    changed[rows, pos] = (core.patch(h[rows, pos].float(), d, self.q) if subspace else d).to(h.dtype)
                    return (changed,) + out[1:] if isinstance(out, tuple) else changed
                if capture:
                    ps = torch.tensor([self.positions[i] for i in padded], device=self.device)
                    states[number] = h[rows[:, None], ps].detach().cpu()
                    if number == 0:
                        cp = torch.tensor([self.color_positions[i] for i in padded], device=self.device)
                        states['color'] = h[rows, cp].float().cpu()
                    if number == 17:
                        states['last'] = h[:, -1].float().cpu()
                return None
            return hook
        try:
            for number, block in enumerate(self.model.model.layers):
                if capture or number == layer:
                    hooks.append(block.register_forward_hook(hook_for(number)))
            out = self.model.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp(min=0), use_cache=False, return_dict=True)
            logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
            cap = self.model.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits/cap)*cap
        finally:
            for h in hooks:
                h.remove()
        self.budget.tick()
        result = dict(logits=logits[:len(indices)].cpu())
        if capture:
            result.update(states=torch.stack([states[i] for i in range(26)], dim=1)[:len(indices)],
                color=states['color'][:len(indices)], last=states['last'][:len(indices)])
        return result


@torch.no_grad()
def numerical(engine, natural):
    ref = audit.selected_functions(old.SHARED / 'mcqa_gradual_das.py', ['answer_logits', 'run_full_layer_intervention'])
    ref = audit.selected_functions(old.SHARED / 'mcqa_gradual_das_c.py', ['run_frozen_mediator'], ref)
    checks, indices = [], list(range(32))
    ids, mask, _ = engine.inputs(indices)
    bank = dict(base_input_ids=ids.cpu(), base_attention_mask=mask.cpu(), base_position_by_id={
        'label': torch.tensor([engine.positions[i][0] - (engine.width-len(engine.encoded[i]['input_ids'])) for i in indices]),
        'last_token': torch.tensor([len(engine.encoded[i]['input_ids'])-1 for i in indices])})
    clean = engine.forward(indices)['logits']
    outputs = engine.model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1)-1).clamp(min=0), use_cache=False, return_dict=True)
    old.compare('clean_reference', clean, ref['answer_logits'](engine.model, outputs, mask, None), checks)
    old.compare('repeat_clean', clean, engine.forward(indices)['logits'], checks)
    for layer in (0, 17, 25):
        donor = natural['states'][np.roll(indices, 1), layer, 0].float()
        actual = engine.forward(indices, layer=layer, donor=donor, slots=[0]*32)['logits']
        expected = ref['run_full_layer_intervention'](engine.model, bank, donor, layer, None, token_position='label', batch_size=32)
        old.compare('full_vector_L' + str(layer), actual, expected, checks)
    donor = natural['last'][np.roll(indices, 1)]
    actual = engine.forward(indices, layer=17, donor=donor, subspace=True)['logits']
    expected = ref['run_frozen_mediator'](engine.model, bank, donor, engine.q, 17, None, token_position='last_token', batch_size=32, require_grad=False)
    old.compare('learned_AP_reference', actual, expected, checks)
    if any(b._forward_hooks for b in engine.model.model.layers) or any(p.requires_grad for p in engine.model.parameters()):
        raise ValueError('Frozen model or hook cleanup failed')
    save('numerical.json', dict(passed=True, checks=checks, atol=2e-4, rtol=2e-4, batch=32))


def run():
    plan, identity = old.read(ROOT / 'plan.json'), old.read(ROOT / 'identity.json')
    for path, expected in identity['sources'].items():
        if schema.file_hash(path) != expected:
            raise ValueError('Pinned source changed: ' + path)
    if schema.file_hash(AP) != identity['checkpoint'] or schema.file_hash(ROOT / 'plan.json') != identity['plan']:
        raise ValueError('Inputs changed')
    model = engine = budget = None
    try:
        runtime.configure(ROOT)
        budget = gpu.Budget(ROOT, seconds=plan['GPU_limit_seconds'])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(old.MODEL, local_files_only=True)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(old.MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        for p in model.parameters():
            p.requires_grad_(False)
        engine = Engine(model, tokenizer, plan, budget)
        versions = [p._version for p in model.parameters()]
        save('execution.json', dict(interpreter=os.path.realpath(os.sys.executable), interpreter_sha256=schema.file_hash(os.sys.executable),
            torch=torch.__version__, width=engine.width, batch=32, power=runtime.power_status(),
            migration='Recorded signed Python 3.11.17; unchanged site packages; fresh matched numerical checks required'))
        pieces, preds = [], []
        for start in range(0, len(engine.contexts), BATCH):
            indices = list(range(start, min(start+BATCH, len(engine.contexts))))
            part = engine.forward(indices, capture=True)
            preds.append(core.score(part['logits'], [engine.contexts[i].answer for i in indices], engine))
            pieces.append(part)
        natural = {k: torch.cat([p[k] for p in pieces]) for k in ('states', 'color', 'last')}
        old.save_tensor(ROOT / 'natural_states.pt', natural)
        gpu.save_npz(ROOT / 'natural_predictions.npz', {k: np.concatenate([p[k] for p in preds]) for k in preds[0]})
        numerical(engine, natural)
        rows = plan['rows']
        bases, donors = [r['base'] for r in rows], [r['source'] for r in rows]
        old.compare('C_prefix_preservation', natural['color'][bases], natural['color'][donors], [])
        save('prefix_preservation.json', dict(max_error=(natural['color'][bases]-natural['color'][donors]).abs().max().item(), passed=True))
        def one(name, layer, subspace=False):
            path = ROOT / 'operations' / (name + '.npz')
            receipt = path.with_suffix('.json')
            if receipt.exists():
                if schema.file_hash(path) != old.read(receipt)['sha256']:
                    raise ValueError('Committed predictions changed')
                return
            parts = []
            for start in range(0, len(rows), 32):
                batch = rows[start:start+32]
                indices, slots, source = ([r[k] for r in batch] for k in ('base', 'start', 'source'))
                donor = natural['last'][source] if subspace else natural['states'][source, layer, slots].float()
                out = engine.forward(indices, layer=layer, donor=donor, slots=slots+[slots[0]]*(32-len(slots)), subspace=subspace)
                parts.append(core.score(out['logits'], [r['target'] for r in batch], engine))
            gpu.save_npz(path, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
            save(str(receipt.relative_to(ROOT)), dict(sha256=schema.file_hash(path), rows=len(rows), layer=layer, subspace=subspace))
            save('status.json', dict(stage=name, seconds=budget.elapsed(), pid=os.getpid()), False)
            print(json.dumps(dict(operation=name, seconds=budget.elapsed())), flush=True)
        start = time.perf_counter()
        one('full_L00', 0)
        measured = time.perf_counter()-start
        remaining = measured*26*1.25
        save('profile.json', dict(seconds_per_layer=measured, remaining_with_allowance=remaining,
            used=budget.elapsed(), launch_allowed=budget.elapsed()+remaining<budget.limit-15))
        if budget.elapsed()+remaining >= budget.limit-15:
            raise runtime.Paused('Measured diagnostic exceeds the registered 15-minute bound')
        for layer in range(1, 26):
            one('full_L' + str(layer).zfill(2), layer)
        one('learned_AP_L17', 17, True)
        if versions != [p._version for p in model.parameters()] or any(b._forward_hooks for b in model.model.layers):
            raise ValueError('Model mutation or hook leak')
        save('gpu_completion.json', dict(complete=True, no_training=True, GPU_seconds=budget.elapsed(), frozen_backbone=True, hook_cleanup=True, operations=27))
        save('status.json', dict(stage='complete', pid=os.getpid()), False)
    except BaseException as error:
        save('stop_' + str(time.time_ns()) + '.json', dict(reason=str(error), traceback=traceback.format_exc()))
        save('status.json', dict(stage='paused' if isinstance(error, runtime.Paused) else 'failed', reason=str(error), pid=os.getpid()), False)
        raise
    finally:
        engine = model = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if budget is not None:
            budget.close()
        runtime.close()


def report():
    plan = old.read(ROOT / 'plan.json')
    nat, rows = arrays(ROOT / 'natural_predictions.npz'), plan['rows']
    shared = {}
    for r in rows:
        shared.setdefault(r['pair_id'], True)
        shared[r['pair_id']] &= bool(nat['correct'][r['base']] and nat['correct'][r['source']])
    clean = np.array([shared[r['pair_id']] for r in rows])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(old.MODEL, local_files_only=True)
    summary = dict(total_pairs=96, shared_clean_pairs=sum(shared.values()), no_intervention_success_filter=True, operations={})
    for name in ['full_L' + str(i).zfill(2) for i in range(26)] + ['learned_AP_L17']:
        path = ROOT / 'operations' / (name + '.npz')
        if not path.with_suffix('.json').exists():
            continue
        if schema.file_hash(path) != old.read(path.with_suffix('.json'))['sha256']:
            raise ValueError('Prediction hash mismatch')
        pred = arrays(path)
        actual = np.array([tokenizer.decode([int(i)]).strip() for i in pred['token_id']])
        copy = np.array([r['copy_target'] for r in rows])
        cells = {}
        for population, pop in (('all_attempted', np.ones(len(rows), bool)), ('shared_clean', clean)):
            cells[population] = {}
            for variant, vmask in (('transposition', np.array([r['variant']=='transposition' for r in rows])),
                                  ('cycles', np.array([r['variant']!='transposition' for r in rows]))):
                mask = pop & vmask
                cells[population][variant] = dict(AP=core.rate(pred['correct'][mask]),
                    copy_label=core.rate(((actual == copy) & ~pred['tied'])[mask]), ties=int(pred['tied'][mask].sum()))
        summary['operations'][name] = cells
    save('AP_control_results.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['saved', 'prepare', 'run', 'report'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    globals()[args.command]()
