"""CPU regressions for the full post-lock mediator/donor evaluation addendum."""
import copy
import unittest
from unittest.mock import patch


def tests():
    import torch
    from .full_transfer_plan import jobs,recipient_rows
    from .full_transfer import gate_without_gpu,require_closed,same_worker
    from experiments.mcqa_ap_localization_sweep import data

    class Tests(unittest.TestCase):
        def sample(self):
            base = dict(id='base',question='q_base',color='red',contents=['red','blue','green','black'],labels=['A','B','C','D'])
            source = dict(id='source',question='q_source',color='blue',contents=['red','blue','green','black'],labels=['A','B','C','D'])
            parent = dict(id='parent',base=base,source=source,target='B',metadata=dict(kind='changing',from_AP=0,to_AP=1,permutation='both'))
            same = dict(id='same',question='q_new_same',color='blue',contents=['red','blue','green','black'],labels=['A','B','C','D'])
            different = dict(id='different',question='q_new_diff',color='blue',contents=['red','blue','green','black'],labels=['A','C','B','D'])
            control = dict(parent=dict(AT_donor0=same,conflict0=different,
                recipient0=dict(same,labels=['E','F','G','H']),
                recipient2=dict(different,color='green',labels=['E','F','G','H'])))
            return parent,control

        def job(self,kind,role,condition):
            return dict(kind=kind,role=role,condition=condition)

        def test_inventory(self):
            values = jobs()
            self.assertEqual(len(values),144)
            self.assertEqual(sum(v['kind']=='AT_recipient' for v in values),96)
            self.assertEqual(sum(v['kind']=='AP_recipient_trace' for v in values),48)
            self.assertEqual({v['seed'] for v in values},{0,1,2})
            self.assertEqual({v['method'] for v in values},{'standard','gradual'})
            for seed in (0,1,2):
                for method in ('standard','gradual'):
                    local = [v for v in values if v['seed']==seed and v['method']==method]
                    self.assertEqual(len(local),24)
                    self.assertEqual(sum(v['condition']=='same' for v in local),12)
                    self.assertEqual(sum(v['condition']=='different' for v in local),12)

        def test_AT_same(self):
            p,c = self.sample()
            rows,ix = recipient_rows([p],c,self.job('AT_recipient','AT_donor0','same'))
            self.assertEqual(ix,[0])
            self.assertEqual(rows[0]['target'],'B')
            self.assertEqual(rows[0]['base']['question'],'q_new_same')
            self.assertEqual(rows[0]['metadata']['cluster_question'],'q_base')

        def test_AT_different_keeps_counterfactual_letter(self):
            p,c = self.sample()
            rows,_ = recipient_rows([p],c,self.job('AT_recipient','conflict0','different'))
            self.assertEqual(data.answer(rows[0]['base']),'C')
            self.assertEqual(rows[0]['target'],'B')

        def test_AP_uses_new_recipient_label_not_source_or_base_letter(self):
            p,c = self.sample()
            for role,condition in (('recipient0','same'),('recipient2','different')):
                rows,ix = recipient_rows([p],c,self.job('AP_recipient_trace',role,condition))
                self.assertEqual(rows[0]['target'],'F')
                self.assertNotEqual(rows[0]['target'],p['target'])
                self.assertEqual(ix,[0])

        def test_wrong_condition_rejected(self):
            p,c = self.sample()
            for kind,role in (('AT_recipient','AT_donor0'),('AP_recipient_trace','recipient0')):
                with self.assertRaises(ValueError):
                    recipient_rows([p],c,self.job(kind,role,'different'))

        def test_same_question_rejected(self):
            p,c = self.sample()
            c['parent']['AT_donor0']['question'] = p['base']['question']
            with self.assertRaises(ValueError):
                recipient_rows([p],c,self.job('AT_recipient','AT_donor0','same'))

        def test_missing_recipient_rejected_no_denominator_shrink(self):
            p,c = self.sample()
            del c['parent']['AT_donor0']
            with self.assertRaises(ValueError):
                recipient_rows([p],c,self.job('AT_recipient','AT_donor0','same'))

        def test_does_not_mutate_inputs(self):
            p,c = self.sample()
            old = copy.deepcopy((p,c))
            recipient_rows([p],c,self.job('AT_recipient','AT_donor0','same'))
            self.assertEqual((p,c),old)

        def test_counterfactual_condition_not_predicted_state(self):
            p,c = self.sample()
            p['predicted_AT']='Z'
            rr,_ = recipient_rows([p],c,self.job('AT_recipient','AT_donor0','same'))
            self.assertEqual(rr[0]['target'],'B')

        def test_budget_no_reset_or_missing_profile(self):
            ledger = dict(active=False,seconds=30000.,total_limit_seconds=43200,profile_seconds=853.1099996000121)
            self.assertTrue(gate_without_gpu(ledger,dict(total_seconds=1000))['allowed'])
            for change in (dict(active=True),dict(seconds=0.),dict(seconds=42900.),dict(total_limit_seconds=86400),dict(profile_seconds=0.)):
                with self.assertRaises(ValueError):
                    require_closed(dict(ledger,**change))
            with self.assertRaises(RuntimeError):
                gate_without_gpu(ledger,dict(total_seconds=14000.))

        def test_worker_command_and_creation_both_enforced(self):
            p = dict(pid=123,created=42.,cmdline=['python','-m','current','run'],executable='python.exe')
            with patch('psutil.Process') as factory:
                live = factory.return_value
                live.create_time.return_value=42.
                live.cmdline.return_value=p['cmdline']
                live.exe.return_value=p['executable']
                self.assertTrue(same_worker(p))
                live.cmdline.return_value=['python','unrelated.py']
                with self.assertRaises(ValueError):
                    same_worker(p)
                live.create_time.return_value=43.
                self.assertFalse(same_worker(p))

        def test_inventory_contains_both_AP_evaluators(self):
            for kind in ('AP_recipient_trace',):
                self.assertEqual({v['ap_evaluator'] for v in jobs() if v['kind']==kind},{'standard','gradual'})
            self.assertEqual({v['ap_evaluator'] for v in jobs() if v['route']=='via_AP'},{'standard','gradual'})

        def test_complete_report_and_independent_saved_score_rejection(self):
            import json
            import time
            from pathlib import Path
            from experiments.mcqa_at_anchor_comparison.storage import save,write,read,file_hash
            from experiments.mcqa_ap_gradual_chain_20261025 import evaluation,training
            from . import full_transfer_report as reporting
            import transformers
            root = Path('tmp')/('full_transfer_report_regression_'+str(time.time_ns()))
            output = root/'supplement'
            output.mkdir(parents=True)
            write(output/'plan.json',dict(synthetic=True))
            parent = {}
            controls = {}
            for bank in ('AT','AP','C'):
                parent[bank] = []
                controls[bank] = {}
                for i in (0,1):
                    base = dict(id=bank+'_base'+str(i),question='q'+str(i),color='red',
                                contents=['red','blue','green','black'],labels=['A','B','C','D'])
                    row = dict(id=bank+str(i),target='B' if i==0 else 'A',base=base,
                        source=dict(base,id=bank+'_source'+str(i),question='source'+str(i)),
                        metadata=dict(kind='changing' if i==0 else 'preservation',from_AP=0,to_AP=1-i,
                                      permutation='both' if i==0 else 'preservation'))
                    parent[bank].append(row)
                    accepted = {}
                    for role in ('recipient0','recipient1','recipient2','recipient3','AT_donor0','AT_donor1','conflict0','conflict1'):
                        position = row['metadata']['to_AP']
                        if role.startswith('recipient') and int(role[-1])>=2:
                            position = (position+int(role[-1])-1)%4
                        colors = ['red','blue','green','black']
                        labels = ['E','F','G','H']
                        if not role.startswith('recipient'):
                            position = 0
                            labels = [row['target'] if role.startswith('AT_') else 'Z','E','F','G']
                        accepted[role] = dict(id=bank+str(i)+role,question='new_'+role+str(i),
                            color=colors[position],contents=colors,labels=labels)
                    controls[bank][row['id']] = accepted
                write(root/'banks'/(bank+'_final')/'pairs.json',parent[bank])
                write(root/'banks'/(bank+'_final')/'completion.json',
                      dict(pairs_sha256=file_hash(root/'banks'/(bank+'_final')/'pairs.json')))
                if bank!='AT':
                    write(root/'controls'/bank/'completion.json',dict(complete=True,missing=[],primary_pairs_removed=0))
                    write(root/'controls'/bank/'page_000000.json',dict(accepted=controls[bank]))
            write(root/'identity.json',dict(model_root='synthetic-no-model'))
            for seed in (0,1,2):
                for label,classes in [(f'AT_S{seed}',26),*[(f'AP_{m}_S{seed}',4) for m in ('standard','gradual')]]:
                    save(root/'readouts'/(label+'.pt'),dict(weight=torch.eye(classes,128),bias=torch.zeros(classes)))

            def make(folder,label,bank,rr=None):
                rr = parent[bank] if rr is None else rr
                targets = [100+ord(r['target'])-ord('A') for r in rr]
                ids = torch.tensor(targets)
                if '_standard_' in label:
                    ids[0] = 100+(int(ids[0])-100+1)%26
                correct = ids==torch.tensor(targets)
                at_state,ap_state = torch.zeros(2,128),torch.zeros(2,128)
                for i,r in enumerate(rr):
                    at_state[i,ord(r['target'])-ord('A')] = 1.
                    ap_state[i,r['metadata']['to_AP']] = 1.
                arrays = dict(pair_ids=[r['id'] for r in rr],targets=[r['target'] for r in rr],
                    rows_digest=evaluation.digest(rr),label_ids=list(range(100,126)),target_token_ids=targets,
                    correct=correct,token_id=ids,tied=torch.zeros(2,dtype=torch.bool),
                    top_two_token_ids=torch.stack([ids,torch.full_like(ids,999)],1),
                    top_two_logits=torch.tensor([[1.,0.],[1.,0.]]),target_log_probability=torch.tensor([-.1,-.2]),
                    AT_state=at_state,AP_state=ap_state)
                save(folder/'evaluation'/(label+'.pt'),arrays)
                write(folder/'evaluation'/(label+'.json'),training.summarize(rr,arrays))
                write(folder/'evaluation_rows'/(label+'.json'),dict(rows=rr,parent_indices=None))

            for label,bank in evaluation.inventory().items():
                make(root,label,bank.removesuffix('_final'))
            for job in jobs():
                rr,_ = recipient_rows(parent[job['bank']],controls[job['bank']],job)
                make(output,job['label'],job['bank'],rr)

            class Tokenizer:
                def decode(self,ids):
                    return chr(ord('A')+int(ids[0])-100)

            with (patch.object(reporting,'ROOT',root),
                  patch.object(reporting,'verify_plan',return_value=dict(reference_pdf='synthetic',parents_per_bank=2)),
                  patch.object(training,'basis',return_value=torch.eye(128)),
                  patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=Tokenizer())):
                result = reporting.report(output)
                self.assertGreater(result['comparisons'],100)
                receipt = read(output/'independent_verification.json')
                self.assertTrue(receipt['passed'])
                self.assertFalse(receipt['cuda_initialized'])
                values = read(output/'results.json')['metrics']
                self.assertEqual({v['seed'] for v in read(output/'results.json')['paired_effects']},{0,1,2})
                self.assertTrue(any(v['path']=='C_to_AP_to_AT' and v['condition']=='different' for v in values))
                self.assertTrue(any(v['metric']=='AP_AT_readouts_and_Y' for v in values))
                self.assertTrue(all(v['count']==4 for v in values if v['setting']=='recipient' and v['group']=='all'))
                path = output/'evaluation'/(jobs()[0]['label']+'.pt')
                value = torch.load(path,map_location='cpu',weights_only=True)
                value['correct'][0] = ~value['correct'][0]
                torch.save(value,path)
                path.with_suffix('.receipt.json').write_text(json.dumps(dict(file=path.name,sha256=file_hash(path))))
                with self.assertRaisesRegex(ValueError,'Independent decoded-token score'):
                    reporting.report(output)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or torch.cuda.is_initialized():
        raise RuntimeError('CPU addendum regressions failed or initialized CUDA')
    return dict(passed=True,test_count=result.testsRun,new_training_updates=0,
        inventory=144,AT_recipient_outputs=96,AP_recipient_state_captures=48,
        all_seeds_and_methods_and_AP_evaluators=True,missing_controls_rejected=True,
        wrong_targets_conditions_workers_and_budgets_rejected=True,cuda_initialized=False)
