"""Recompute all reported metrics on CPU from exact exported cases/predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from vendor import schema

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def rate(a):
    a=np.asarray(a,dtype=bool)
    return dict(correct=int(a.sum()),count=len(a),accuracy=float(a.mean()) if len(a) else None)


def interval(a,b,groups,mask):
    names,membership=np.unique(np.asarray(groups)[mask],return_inverse=True)
    counts=np.bincount(membership,minlength=len(names))
    sums=np.bincount(membership,weights=b[mask].astype(int)-a[mask].astype(int),minlength=len(names))
    rng=np.random.default_rng(20261020)
    draws=rng.integers(0,len(names),(2000,len(names)))
    multiplicities=np.array([np.bincount(x,minlength=len(names)) for x in draws])
    return np.quantile((multiplicities@sums)/(multiplicities@counts),[.025,.975]).tolist()


def verify_manifest():
    manifest=read(ROOT/'MANIFEST.json')
    for name,expected in manifest['files'].items():
        if schema.file_hash(ROOT/name)!=expected:
            raise ValueError('Manifest mismatch: '+name)
    return len(manifest['files'])


def check_score(a,targets,decodings,prefix=''):
    actual=np.array([decodings[str(int(i))]==t for i,t in zip(a[prefix+'token_id'],targets)]) & ~a[prefix+'tied']
    if not np.array_equal(actual,a[prefix+'correct']):
        raise ValueError('Strict full-vocabulary scoring mismatch: '+prefix)
    return actual


def analyze(predictions=None,strict=True):
    predroot=ROOT/'predictions' if predictions is None else Path(predictions)
    contexts,pairs=schema.import_pairs(read(ROOT/'data/c_at_pairs.json'))
    layout=read(ROOT/'data/c_at_schedule.json')
    decodings=read(predroot/'token_decodings.json' if (predroot/'token_decodings.json').exists() else ROOT/'data/token_decodings.json')
    if (ROOT/'data/contexts.json').exists():
        exported=[schema.Context(**c) for c in read(ROOT/'data/contexts.json')]
        if [c.id for c in contexts]!=[c.id for c in exported]:
            raise ValueError('Context IDs/order differ')
    b,s=np.asarray(layout['base']),np.asarray(layout['source'])
    indices=np.asarray(layout['original_pair_indices'])
    mask=np.zeros(len(pairs),bool)
    mask[indices]=True
    natural=arrays(predroot/'natural.npz')
    matched=arrays(predroot/'matched_base.npz')
    if strict:
        check_score(natural,[c.answer for c in contexts],decodings)
        check_score(matched,[contexts[i].answer for i in b],decodings)
    actual_mask=natural['correct'][b]&natural['correct'][s]&matched['correct']
    if strict and not np.array_equal(mask,actual_mask):
        raise ValueError('Recorded baseline-only mask differs')
    expected=[contexts[i].label_for(contexts[j].color) for i,j in zip(b,s)]
    groups=np.array([p.cluster for p in pairs])[indices]
    families=np.array([p.family for p in pairs])[indices]
    changing=np.array([contexts[i].color!=contexts[j].color for i,j in zip(b,s)])[indices]
    controls=arrays(predroot/'pair_controls.npz')
    donors=np.asarray(layout['pair_donors'])
    if any(contexts[d].answer!=expected[i] for i,ds in zip(indices,donors) for d in ds):
        raise ValueError('Clean AT donor letter differs from target')
    if strict:
        check_score(controls,np.repeat(np.asarray(expected)[indices],2),decodings)
    both=controls['correct'].reshape(len(indices),2).all(1)
    values={}
    for method in ('standard','gradual'):
        a=arrays(predroot/('c_at_'+method+'.npz'))
        if strict:
            for name in ('direct','replay'):
                check_score(a,expected,decodings,name+'_')
        d,r=a['direct_correct'][indices],a['replay_correct'][indices]
        values[method]=dict(direct=d,replay=r,joint=d&r)
    out=dict(original_pairs=len(pairs),recorded_clean_pairs=len(indices),changing_pairs=int(changing.sum()),
        recorded_clean_mask_reproduced=bool(np.array_equal(mask,actual_mask)),
        baseline_passes_on_recorded_subset=bool(actual_mask[indices].all()),primary={},conditional={},preservation={})
    for key,cell in (('primary',changing),('conditional',changing&both)):
        for metric in ('direct','replay','joint'):
            a,z=values['standard'][metric],values['gradual'][metric]
            out[key][metric]=dict(standard=rate(a[cell]),gradual=rate(z[cell]),percentile_95=interval(a,z,groups,cell))
    for method in values:
        out['preservation'][method]={f:{k:rate(v[families==f]) for k,v in values[method].items()} for f in ('AP_only','AT_only')}
    qualification=read(ROOT/'data/at_qualification.json')
    q=arrays(predroot/'clean_AT.npz')
    bases=np.asarray(qualification['bases'])
    targets=np.tile(np.repeat(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),2),len(bases))
    own=np.repeat([contexts[i].answer for i in bases],52)
    ds=np.asarray(qualification['donors']).reshape(-1)
    if [contexts[i].answer for i in ds]!=targets.tolist():
        raise ValueError('Qualification donor targets differ')
    if strict:
        check_score(q,targets,decodings)
    out['AT_clean_donors']={name:rate(q['correct'][cell]) for name,cell in
        (('all',np.ones(len(q['correct']),bool)),('preservation',own==targets),('changing',own!=targets))}
    if (predroot/'preservation_readout_features.npz').exists():
        f=arrays(predroot/'preservation_readout_features.npz')
        positions=f['retained_pair_positions']
        truth=np.array(['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(contexts[b[i]].answer) for i in indices[positions]])
        out['AT_readout_preservation']={name:rate(((f[name]@f['W'].T+f['b']).argmax(-1)==truth)) for name in ('clean','standard','gradual')}
    if not (predroot/'ap_natural.npz').exists():
        out['AP']=dict(evaluated=False)
        return out
    ap=read(ROOT/'data/ap_plan.json')
    nat=arrays(predroot/'ap_natural.npz')
    if strict:
        check_score(nat,[schema.Context(**c).answer for c in ap['contexts']],decodings)
    shared={}
    for row in ap['rows']:
        shared.setdefault(row['pair_id'],True)
        shared[row['pair_id']] &= bool(nat['correct'][row['base']] and nat['correct'][row['source']])
    clean=np.array([shared[r['pair_id']] for r in ap['rows']])
    out['AP']=dict(attempted_pairs=96,shared_clean_pairs=sum(shared.values()),layers={})
    for name in ['full_L'+str(i).zfill(2) for i in range(26)]+['learned_AP_L17']:
        path=predroot/'ap'/(name+'.npz')
        if not path.exists():
            continue
        a=arrays(path)
        if strict:
            check_score(a,[r['target'] for r in ap['rows']],decodings)
        copied=np.array([decodings.get(str(int(i)),'')==r['copy_target'] for i,r in zip(a['token_id'],ap['rows'])])&~a['tied']
        cells={}
        for label,population in (('all_attempted',np.ones(len(clean),bool)),('shared_clean',clean)):
            cells[label]={}
            for variant in ('transposition','cycles'):
                cell=population&np.array([(r['variant']=='transposition')==(variant=='transposition') for r in ap['rows']])
                cells[label][variant]=dict(AP=rate(a['correct'][cell]),copy_label=rate(copied[cell]))
        out['AP']['layers'][name]=cells
    return out


def verify():
    files=verify_manifest()
    out=analyze()
    expected=read(ROOT/'results/independent_verification.json')
    for metric in ('direct','replay','joint'):
        ref=expected['paired']['all_changing'][metric]
        actual=out['primary'][metric]
        if actual['standard']!=ref['standard'] or actual['gradual']!=ref['gradual'] or not np.allclose(actual['percentile_95'],ref['percentile_95'],atol=1e-12,rtol=0):
            raise ValueError('Primary counts or intervals differ')
    if out['original_pairs']!=1600 or out['recorded_clean_pairs']!=1476 or out['AP']['shared_clean_pairs']!=65:
        raise ValueError('Population count differs')
    ref=read(ROOT/'results/AP_control_results.json')
    for name,cells in out['AP']['layers'].items():
        for population in cells:
            for variant in cells[population]:
                for metric in ('AP','copy_label'):
                    if cells[population][variant][metric]!=ref['operations'][name][population][variant][metric]:
                        raise ValueError('AP scoring differs')
    print(json.dumps(dict(verified=True,files=files,primary=out['primary'],conditional=out['conditional'],
        AP_clean_pairs=out['AP']['shared_clean_pairs'],AT_clean_donors=out['AT_clean_donors']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['verify','summarize'])
    p.add_argument('--predictions-dir',type=Path)
    args=p.parse_args()
    if args.command=='verify':
        verify()
    else:
        print(json.dumps(analyze(args.predictions_dir,strict=args.predictions_dir is None),indent=2))
