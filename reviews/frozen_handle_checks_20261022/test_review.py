"""Target, scoring and preservation checks independent of GPU execution."""
import unittest
import numpy as np
import review
from vendor import schema


class Tests(unittest.TestCase):
    def context(self, color, options):
        prompt='Question: The sky is '+color+'. What color is the sky?\n'+options+'\nAnswer:'
        return schema.parse_context(dict(base_prompt=prompt,question='sky'),'base')

    def test_AP_control_separates_predictions(self):
        b=self.context('blue','A. red\nB. blue\nC. green\nD. yellow')
        s=self.context('blue','A. red\nC. green\nB. blue\nD. yellow')
        z=self.context('blue','A. red\nD. yellow\nB. blue\nC. green')
        self.assertEqual(s.answer,z.answer)
        self.assertEqual(s.contents.index(s.color),z.contents.index(z.color))
        self.assertEqual(b.labels[z.contents.index(z.color)],'C')
        self.assertEqual(s.labels[b.contents.index(b.color)],'C')
        self.assertEqual(z.labels[b.contents.index(b.color)],'D')

    def test_C_target_uses_base_options(self):
        b=self.context('blue','A. red\nB. blue\nC. yellow\nD. green')
        s=self.context('green','A. red\nB. green\nC. yellow\nD. blue')
        self.assertEqual(b.answer,s.answer)
        self.assertEqual(b.label_for(s.color),'D')

    def test_ties_fail(self):
        a=dict(token_id=np.array([1]),tied=np.array([True]),correct=np.array([False]))
        self.assertFalse(review.check_score(a,['A'],{'1':'A'})[0])

    def test_corrupt_correctness_rejected(self):
        a=dict(token_id=np.array([1]),tied=np.array([False]),correct=np.array([True]))
        with self.assertRaises(ValueError):
            review.check_score(a,['B'],{'1':'A'})

    def test_original_failed_gate_preserved(self):
        self.assertFalse(review.read(review.ROOT/'results/original_baseline_gate.json')['passed'])


if __name__=='__main__':
    unittest.main()
