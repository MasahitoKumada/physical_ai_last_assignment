"""Player feedback must preserve losses, unknown ice, and out-of-rink results."""
import unittest

from player_view import ice_hint, shot_outcome, shot_preview
from predictor import IceEstimate


def stone(index, score, position=(3., 0.), alive=True):
    return {"id": index, "score": score, "position": position, "alive": alive}


class PlayerFeedbackTests(unittest.TestCase):
    def test_prediction_reports_whole_board_change_and_keeps_losses(self):
        before = {"score": 90, "stones": [stone(0, 90)]}
        predicted = {"score": 45, "stones": [stone(0, 20), stone(1, 25)]}
        view = shot_preview(before, predicted)
        self.assertEqual(view['delta'], -45)
        self.assertIn('70点下がる', view['detail'])
        self.assertNotIn('押し込んで', view['detail'])

    def test_push_in_counts_only_existing_stones_and_out_warning_ignores_old_removed_stone(self):
        before = {"score": 20, "stones": [stone(0, 0, alive=False), stone(1, 20)]}
        predicted = {"score": 100, "stones": [stone(0, 0, alive=False), stone(1, 55), stone(2, 45)]}
        view = shot_preview(before, predicted)
        self.assertEqual(view['delta'], 80)
        self.assertIn('2番も押し込んで、さらに35点', view['detail'])
        self.assertNotIn('場外', view['detail'])

    def test_result_does_not_celebrate_net_loss_or_report_distance_for_removed_stone(self):
        before = {"score": 150, "stones": [stone(0, 80), stone(1, 70)]}
        after = {"score": 100, "stones": [stone(0, 0, alive=False), stone(1, 0, alive=False), stone(2, 100)]}
        outcome = shot_outcome(before, after, 2)
        self.assertEqual(outcome['delta'], -50)
        self.assertNotIn('ピタリ', outcome['title'])
        outcome = shot_outcome(before, after, 0)
        self.assertIsNone(outcome['distance_cm'])
        self.assertIn('場外', outcome['detail'])

    def test_ice_reading_stays_unknown_until_observed_and_labels_carryover(self):
        estimate = IceEstimate(.04, .03)
        unknown = ice_hint(estimate, .05, 0, 0, 'READY')
        self.assertNotIn('滑りやすそう', unknown[0])
        observed = ice_hint(estimate, .05, 4, 1, 'READY')
        self.assertIn('滑りやすそう', observed[0])
        self.assertIn('前の投球', observed[1])
        self.assertIsNone(shot_preview({'score': 0, 'stones': []}, None))


if __name__ == '__main__':
    unittest.main()
