"""CPU-only tables, figures and completion receipts from locked evidence."""
import csv
from pathlib import Path
import numpy as np
from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash
from . import verification,banks

def map_figures(root):
    root=Path(root)
    if not (root/'sites.json').exists():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .sites import Site
    catalog=read(root/'sites.json')['single'];width=max(s['index'] for s in catalog if s['kind']=='offset')+1
    for cell in [f'{b+1}->{s+1}' for b in range(4) for s in range(4)]:
        matrix=np.full((26,width),np.nan);struct=np.full((26,8),np.nan)
        for item in catalog:
            site=Site(**item);p=root/'maps'/f'{site.id}.json'
            if not p.exists():continue
            accuracy=read(p)['transition'][cell]['accuracy']
            if site.kind=='offset':matrix[site.layer,site.index]=accuracy
            else:struct[site.layer,site.index+(0 if site.kind=='label' else 4)]=accuracy
        if not np.isfinite(matrix).any():continue
        fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[3,1]})
        image=axes[0].imshow(matrix,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
        axes[0].set(title='End-relative token sites',xlabel='Offset from final token',ylabel='Decoder block')
        axes[1].imshow(struct,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
        axes[1].set(title='Structural option sites',xticks=range(8),xticklabels=['L1','L2','L3','L4','C1','C2','C3','C4'])
        fig.suptitle('Full activation transfer: AP '+cell);fig.colorbar(image,ax=list(axes),shrink=.75)
        fig.savefig(root/f"MAP_{cell.replace('->','_to_')}.png",dpi=140,bbox_inches='tight');plt.close(fig)

def partial(root):
    root=Path(root);budget=read(root/'engineering/gpu_budget.json') if (root/'engineering/gpu_budget.json').exists() else {}
    terminal=read(root/'terminal.json') if (root/'terminal.json').exists() else {}
    latest=[dict(configuration=p.parent.name,**read(p)) for p in sorted((root/'fits').glob('*/latest.json'))]
    receipt=dict(completed=False,reason=terminal.get('reason','Incomplete study'),budget=budget,
        completed_maps=len(list((root/'maps').glob('*.pt'))),completed_screens=len(list((root/'screens').glob('*/completion.json'))),
        completed_fits=len(list((root/'fits').glob('*/completion.json'))),latest_checkpoints=latest,
        final_opened=(root/'banks/final/completion.json').exists(),automatic_resume=False)
    write(root/'incomplete_study.json',receipt,immutable=False)
    (root/'STOP_REPORT.md').write_text('# Incomplete AP localization study\n\n'+receipt['reason']+'\n\n'+
        f"Completed maps: {receipt['completed_maps']}; screens: {receipt['completed_screens']}; confirmation fits: {receipt['completed_fits']}.\n"+
        f"GPU-session seconds: {budget.get('seconds',0):.2f}; active: {budget.get('active',False)}.\n\nNo scientific coverage was reduced. No automatic resume.\n",encoding='utf-8')
    map_figures(root);return receipt

def report(root):
    root=Path(root)
    if not (root/'results.json').exists():return partial(root)
    audit=verification.final(root);result=read(root/'results.json');locked=read(root/'locks/all_final.json')
    lines=['# Broad AP Localization and Composition Study','','Fixed endpoint standard DAS; all AP transitions retained.','',
        '| Seed | Configuration | Changing | Preservation |','|---|---|---:|---:|']
    table=[]
    for seed in (1,0,2):
        for slot,key in locked['aliases'].items():
            value=result['endpoints'][key+f'_S{seed}']['direct']
            lines.append(f"| {seed} | {slot} | {value['changing']['accuracy']:.2%} | {value['preservation']['accuracy']:.2%} |")
            for field in ('transition','parent_transition','permutation','target_letter','color','origin','destination','transfer'):
                for label,m in value[field].items():table.append(dict(seed=seed,configuration=slot,stratum=field,cell=label,**m))
    with (root/'TRANSITIONS.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    coverage={}
    for bank in ('final','c_diagnostic'):
        rows=banks.load_bank(root,bank);coverage[bank]=dict(count=len(rows),
            letters={x:sum(r['target']==x for r in rows) for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
            colors={c:sum(r['base']['color']==c for r in rows) for c in __import__(__package__+'.protocol',fromlist=['COLORS']).COLORS})
    write(root/'FULL_COVERAGE.json',coverage)
    map_figures(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,12),sharex=True,sharey=True)
    for row,seed in enumerate((1,0,2)):
        for slot,key in locked['aliases'].items():
            endpoint=locked['endpoints'][key+f'_S{seed}'];folder=Path(endpoint['checkpoint']).parent.parent/'evaluations'
            records=[(int(p.stem),read(p)) for p in sorted(folder.glob('[0-9][0-9][0-9][0-9][0-9][0-9].json'))]
            for col,label in enumerate(('changing','preservation')):
                axes[row,col].plot([x[0] for x in records],[100*x[1]['calibration'][label]['accuracy'] for x in records],label=slot)
                axes[row,col].set(title=f'Seed {seed}: calibration {label}',ylim=(0,100),xlabel='Update',ylabel='Accuracy (%)')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'LEARNING_CURVES.png',dpi=160);plt.close(fig)
    for seed in (1,0,2):
        fig,axes=plt.subplots(2,3,figsize=(13,8));image=None
        for ax,(slot,key) in zip(axes.flat,locked['aliases'].items()):
            cells=result['endpoints'][key+f'_S{seed}']['direct']['transition'];matrix=np.array([[cells[f'{b+1}->{s+1}']['accuracy'] for s in range(4)] for b in range(4)])
            image=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');ax.set(title=slot,xlabel='Destination AP',ylabel='Original AP',xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5))
            for b in range(4):
                for s in range(4):ax.text(s,b,f'{matrix[b,s]:.0%}',ha='center',va='center',color='white' if matrix[b,s]<.5 else 'black')
        fig.colorbar(image,ax=list(axes.flat),shrink=.75);fig.savefig(root/f'TRANSITIONS_S{seed}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    budget=read(root/'engineering/gpu_budget.json')
    lines += ['',f"Independently rescored predictions: {audit['predictions']}.",f"GPU-session hours: {budget['seconds']/3600:.3f} / 24.",
        '','Screening at 50 updates is not exhaustive full training. Final sites were selected on calibration only.',
        'The C diagnostic uses one uploaded standard/gradual pair, not three independent C trainings.',
        'Weak AT donor transfer and overlapping supports limit chain claims. No native switching or AP-absence claim.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(root/'completion.json',dict(completed=True,results_sha256=file_hash(root/'results.json'),
        independent_verification_sha256=file_hash(root/'independent_verification.json'),budget_closed=not budget['active'],GPU_seconds=budget['seconds']))
    return result
