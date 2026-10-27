"""Joint four-token AP, frozen downstream replay, and matched native arithmetic."""
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import torch
from torch.nn import functional as F

from experiments.mcqa_ap_localization_sweep.engine import Engine as PreviousEngine
from experiments.mcqa_ap_localization_sweep.sites import Site, gather, scatter
from experiments.mcqa_ap_localization_sweep.banks import score
from experiments.mcqa_at_anchor_comparison.engine import Subspace, projection, checked
from experiments.mcqa_at_anchor_comparison.storage import read, write, save, load
from . import data

AP_SITE = Site(14, 'joint_color', 0)


def total_loss(terms, method):
    return terms['direct'] + (2 * sum(v for k, v in terms.items() if k != 'direct') if method == 'gradual' else 0)


class Engine(PreviousEngine):
    def __init__(self, root, session):
        super().__init__(root, session)
        if self.width != 37:
            raise ValueError('Registered width must remain 37')
        self.batch = 32

    def forward(self, contexts, donor=None, q=None, site=None, restore=False,
                c_donor=None, c_q=None, c_restore=False, at_donor=None, at_q=None,
                at_restore=False, reference=False, full=False):
        self.session.check()
        site = site or AP_SITE
        ids, mask, cp = self.encode(contexts)
        rr = torch.arange(self.batch, device=self.device)
        pos, available = self.positions(contexts, site)
        captures = {}
        before = [len(layer._forward_hooks) for layer in self.model.model.layers]
        src = None
        if donor is not None:
            src = self.pad(donor['values']).float()
            available = available & self.pad(donor['available']).bool()

        def swap(b, s, basis):
            return b + ((s - b) @ basis) @ basis.T if reference else projection(b, s, basis)

        def hook(layer):
            def apply(_module, _inputs, out):
                h = out[0] if isinstance(out, tuple) else out
                if layer == 0:
                    original = h[rr, cp].float()
                    captures['original_c'] = original
                    if c_donor is not None:
                        h = h.clone()
                        h[rr, cp] = swap(original, self.pad(c_donor).float(), c_q).to(h.dtype)
                        if c_restore:
                            h[rr, cp] = swap(h[rr, cp].float(), original, c_q).to(h.dtype)
                    captures['c'] = h[rr, cp].float()
                if layer == site.layer:
                    original = gather(h, pos).float()
                    captures['original_site'] = original
                    if src is not None:
                        changed = swap(original, src, q) if q is not None else src
                        h = scatter(h, pos, changed, available)
                        if restore:
                            h = scatter(h, pos, swap(gather(h, pos).float(), original, q), available)
                    captures['site'] = gather(h, pos).float()
                    captures['available'] = available
                if layer == 25:
                    original = h[rr, -1].float()
                    captures['at'] = original
                    if at_donor is not None:
                        h = h.clone()
                        h[rr, -1] = swap(original, self.pad(at_donor).float(), at_q).to(h.dtype)
                        if at_restore:
                            h[rr, -1] = swap(h[rr, -1].float(), original, at_q).to(h.dtype)
                    captures['at_after_override'] = h[rr, -1].float()
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return apply

        with ExitStack() as stack:
            for layer in sorted({0, 25, site.layer}):
                handle = self.model.model.layers[layer].register_forward_hook(hook(layer))
                stack.callback(handle.remove)
            args = dict(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1)-1).clamp(min=0),
                        use_cache=False, return_dict=True)
            if full:
                def head(_module, _inputs, output):
                    captures['raw'] = output[:, 0]
                handle = self.model.lm_head.register_forward_hook(head)
                stack.callback(handle.remove)
                native = self.model(**args, logits_to_keep=1).logits[:, -1]
                raw = captures['raw']
            else:
                hidden = self.model.model(**args).last_hidden_state
                raw = self.model.lm_head(hidden[:, -1:, :])[:, 0]
                cap = self.model.config.final_logit_softcapping
                native = torch.tanh(raw/cap)*cap if cap is not None else raw
        if before != [len(layer._forward_hooks) for layer in self.model.model.layers]:
            raise RuntimeError('Forward hooks leaked')
        captures['native'] = native
        return self.softcap(raw), captures

    def losses(self, variable, method, rows, states, sub, teachers=None, reference=False):
        n = len(rows)
        targets, classes = self.targets(rows)
        q = sub.basis()
        cs = [r['base'] for r in rows]
        if variable == 'AT':
            if reference:
                y, _ = self.forward(cs, at_donor=states['source_AT'], at_q=q, reference=True, full=True)
            else:
                b, s = self.pad(states['base_AT']).float(), self.pad(states['source_AT']).float()
                y = self.tail(projection(b, s, q).to(torch.bfloat16))
            return dict(direct=F.cross_entropy(y[:n], targets)), y[:n]
        args = dict(reference=reference, full=reference)
        if variable == 'AP':
            args.update(donor=dict(values=states['source_AP'], available=torch.ones(n, dtype=torch.bool)), q=q)
        else:
            args.update(c_donor=states['source_C'], c_q=q)
        y, trace = self.forward(cs, **args)
        terms = dict(direct=F.cross_entropy(y[:n], targets))
        if method == 'gradual':
            aq = teachers['AT_q']
            if reference:
                replay, _ = self.forward(cs, at_donor=trace['at'], at_q=aq, reference=True, full=True)
            else:
                replay = self.tail(projection(self.pad(states['base_AT']).float(), trace['at'], aq).to(torch.bfloat16))
            terms['AT_replay'] = F.cross_entropy(replay[:n], targets)
            head = teachers['AT_head']
            taught = (trace['at'][:n] @ aq) @ head['weight'].T + head['bias']
            terms['AT_readout'] = F.cross_entropy(taught, classes)
            if variable == 'C':
                pq, head = teachers['AP_q'], teachers['AP_head']
                ap_replay, _ = self.forward(cs, donor=dict(values=trace['site'], available=trace['available']),
                                             q=pq, reference=reference, full=reference)
                terms['AP_replay'] = F.cross_entropy(ap_replay[:n], targets)
                taught = (trace['site'][:n] @ pq) @ head['weight'].T + head['bias']
                positions = torch.tensor([r['metadata']['to_AP'] for r in rows], device=self.device)
                terms['AP_readout'] = F.cross_entropy(taught, positions)
        return terms, y[:n]

    def update(self, variable, method, rows, states, sub, opt, teachers=None):
        if len(rows) != 32:
            raise ValueError('Effective batch must equal 32')
        opt.zero_grad(set_to_none=True)
        totals = {}
        for start in range(0, 32, self.batch):
            small = rows[start:start+self.batch]
            local = {k: v[start:start+len(small)] for k, v in states.items()}
            terms, _ = self.losses(variable, method, small, local, sub, teachers)
            loss = total_loss(terms, method) * len(small)/32
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.) + float(v.detach()) * len(small)/32
        if sub.raw.grad is None or not torch.isfinite(sub.raw.grad).all():
            raise ValueError('Invalid projector gradient')
        if any(p.grad is not None or p.requires_grad for p in self.model.parameters()):
            raise ValueError('Backbone was not frozen')
        norm = float(sub.raw.grad.norm())
        opt.step()
        if not torch.isfinite(sub.raw).all():
            raise ValueError('Invalid projector state')
        return dict(losses=totals, gradient_norm=norm)


class NaturalCache:
    """Only the three registered states; hashes and clean scores persist."""

    def __init__(self, root, engine):
        self.root, self.engine = Path(root), engine
        self.folder = self.root / 'natural_cache'
        self.index = {}
        for p in sorted(self.folder.glob('*.ids.json')):
            value = read(p)
            tensor = p.with_name(p.name.replace('.ids.json', '.pt'))
            load(tensor)
            for i, ident in enumerate(value):
                if ident in self.index:
                    raise ValueError('Duplicate natural cache ID')
                self.index[ident] = (tensor, i)
        self.next = len(list(self.folder.glob('*.ids.json')))

    @lru_cache(maxsize=32)
    def chunk(self, path):
        return load(path)

    def ensure(self, contexts):
        todo = list({c['id']: c for c in contexts if c['id'] not in self.index}.values())
        for start in range(0, len(todo), self.engine.batch):
            cs = todo[start:start+self.engine.batch]
            n = len(cs)
            with torch.no_grad():
                y, h = self.engine.forward(cs)
                targets = [data.answer(c) for c in cs]
                pred = score(self.engine, y[:n], targets)
                native = score(self.engine, h['native'][:n], targets)
            result = dict(C=h['c'][:n].cpu().to(torch.bfloat16), AP=h['site'][:n].cpu().to(torch.bfloat16),
                          AT=h['at'][:n].cpu().to(torch.bfloat16), **pred,
                          native_token_id=native['token_id'], native_tied=native['tied'],
                          eligible=pred['correct'] & native['correct'])
            p = self.folder / f'{self.next:07d}.pt'
            save(p, result)
            write(p.with_suffix('.ids.json'), [c['id'] for c in cs])
            self.index.update({c['id']: (p, i) for i, c in enumerate(cs)})
            self.next += 1

    def field(self, contexts, key):
        self.ensure(contexts)
        return torch.stack([self.chunk(self.index[c['id']][0])[key][self.index[c['id']][1]] for c in contexts])

    def states(self, rows):
        base, source = [r['base'] for r in rows], [r['source'] for r in rows]
        return dict(base_AT=self.field(base, 'AT').float(), source_AT=self.field(source, 'AT').float(),
                    source_AP=self.field(source, 'AP').float(), source_C=self.field(source, 'C').float())

    def layout(self, rows, label):
        for side in ('base', 'source'):
            for start in range(0, len(rows), self.engine.batch):
                cs = [r[side] for r in rows[start:start+self.engine.batch]]
                with torch.no_grad():
                    y, h = self.engine.forward(cs)
                for scores, field in ((score(self.engine, y[:len(cs)], [data.answer(c) for c in cs]), 'token_id'),
                                      (score(self.engine, h['native'][:len(cs)], [data.answer(c) for c in cs]), 'native_token_id')):
                    if not scores['correct'].all() or not torch.equal(scores['token_id'], self.field(cs, field)):
                        raise ValueError('Matched natural layout changed: ' + label + '/' + side)
        write(self.root / 'engineering/layout' / (label + '.json'), dict(passed=True, count=len(rows)))
