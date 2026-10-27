from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_round8 import (compute_headroom,refine_acquire_winners,
                            gate_stratum,summarize_primary_gate,clipped_infidelity)


GATE=dict(minimum_baseline_error=.001,minimum_relative_headroom=.2,
          minimum_leave_one_out_headroom=.1,minimum_eligible_strata=2,
          required_families=2,minimum_winner_seed_ids_per_type=2,
          minimum_absolute_action_advantage=.0001)


class TestHeadroomAnalysis(unittest.TestCase):
    def test_best_simple_and_hindsight_are_distinct(self):
        losses=dict(stop=[1.,1.,1.,1.,1.],refine_left=[0.,.8,0.,.8,0.],
                    refine_right=[.8,0.,.8,0.,.8],acquire_center=[.4]*5)
        result=compute_headroom(losses,seeds=[1,2,3,4,5])
        self.assertEqual(result['best_simple_policy'],'refine_left')
        self.assertAlmostEqual(result['baseline_mean'],.32)
        self.assertEqual(result['oracle_mean'],0.)
        self.assertEqual(result['relative_headroom'],1.)
        self.assertEqual(result['paired_improvements'],[0.,.8,0.,.8,0.])

    def test_leave_one_out_reselects_best_policy(self):
        losses=dict(stop=[2.,2.,2.,2.,2.],refine_left=[0.,1.,1.,1.,1.],
                    refine_right=[.9]*5,acquire_center=[1.5]*5)
        result=compute_headroom(losses,seeds=[10,20,30,40,50])
        self.assertEqual(result['best_simple_policy'],'refine_left')
        self.assertEqual(result['leave_one_out'][0]['best_simple_policy'],'refine_right')
        self.assertAlmostEqual(result['leave_one_out'][0]['baseline_mean'],.9)
        self.assertAlmostEqual(result['leave_one_out'][0]['oracle_mean'],.9)
        self.assertEqual(result['minimum_leave_one_out_headroom'],0.)

    def test_adaptive_simple_policy_can_remove_fixed_action_headroom(self):
        losses=dict(stop=[1.]*5,refine_left=[0.,.8,0.,.8,0.],
                    refine_right=[.8,0.,.8,0.,.8],acquire_center=[.4]*5)
        choices={'anchor_residual':['refine_left','refine_right','refine_left','refine_right','refine_left']}
        result=compute_headroom(losses,choices,list(losses)+['anchor_residual'])
        self.assertEqual(result['best_simple_policy'],'anchor_residual')
        self.assertEqual(result['baseline_mean'],0.)
        self.assertEqual(result['oracle_mean'],0.)
        self.assertEqual(result['relative_headroom'],0.)

    def test_left_right_only_cannot_pass_refine_acquire_gate(self):
        losses=dict(stop=[1.]*5,refine_left=[0.,.8,0.,.8,0.],
                    refine_right=[.8,0.,.8,0.,.8],acquire_center=[.4]*5)
        r=compute_headroom(losses,seeds=[1,2,3,4,5]);w=refine_acquire_winners(losses,[1,2,3,4,5],.0001)
        record=dict(**r,stop_mean=1.,refine_winner_seed_ids=w['refine_winner_seed_ids'],
                    acquire_winner_seed_ids=w['acquire_winner_seed_ids'])
        record.update(gate_stratum(record,GATE))
        cells=[dict(record,family=f,radius=.06,architecture='mlp') for f in ['chain','ladder']]
        summary=summarize_primary_gate(cells,GATE)
        self.assertTrue(summary['enough_eligible_strata'])
        self.assertTrue(summary['enough_families'])
        self.assertFalse(summary['enough_acquire_seed_ids'])
        self.assertFalse(summary['pass_gate'])

    def test_stop_solved_floor_rejects_large_relative_headroom(self):
        losses=dict(stop=[.0002]*5,refine_left=[0.,.0004,0.,.0004,0.],
                    refine_right=[.0004,0.,.0004,0.,.0004],acquire_center=[.0001]*5)
        r=compute_headroom(losses);record=dict(**r,stop_mean=.0002,acquire_winner_seed_ids=[])
        result=gate_stratum(record,GATE)
        self.assertGreater(r['relative_headroom'],.2)
        self.assertFalse(result['eligible'])
        self.assertIn('STOP already below baseline-error floor',result['failure_or_caveat_categories'])

    def test_type_winner_must_also_beat_stop_and_margin(self):
        losses=dict(stop=[.01,.01,.01],refine_left=[.02,.0002,.003],
                    refine_right=[.03,.0003,.004],acquire_center=[.04,.004,.0001])
        winners=refine_acquire_winners(losses,[1,2,3],.0001)
        self.assertEqual(winners['refine_winner_seed_ids'],[2])
        self.assertEqual(winners['acquire_winner_seed_ids'],[3])
        self.assertEqual(winners['rows'][0]['winner_type'],'none')
        self.assertEqual(clipped_infidelity(1.+1e-12),0.)
        self.assertAlmostEqual(clipped_infidelity(.3),.7)


if __name__=='__main__': unittest.main()
