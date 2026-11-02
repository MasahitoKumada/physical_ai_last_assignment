"""Checks for paired trials, durable ratings, and separation of automation."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from game import DEMO_COMMANDS, GameConfig, GameSession, verify_replay
from playtest import PlaytestStudy, QUESTIONS, plan_trials, summarize_study


class PlaytestTests(unittest.TestCase):
    def test_pairs_share_ice_with_opposite_orders_and_exclude_practice(self):
        plan = plan_trials(42)
        self.assertEqual(plan, plan_trials(42))
        self.assertEqual([row['practice'] for row in plan], [True, True, False, False, False, False])
        for group in (plan[2:4], plan[4:6]):
            self.assertEqual(group[0]['config'], group[1]['config'])
            self.assertEqual({row['forecast'] for row in group}, {False, True})
            worlds = [GameSession(GameConfig(**row['config'])) for row in group]
            self.assertEqual(worlds[0]._ice, worlds[1]._ice)
        self.assertEqual([row['forecast'] for row in plan[2:4]], list(reversed([row['forecast'] for row in plan[4:6]])))
        self.assertNotEqual(plan[2]['config'], plan[4]['config'])

    def test_completed_game_waits_for_explicit_ratings_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            study = PlaytestStudy(directory, automated=True)
            record = study.start_attempt()
            session = GameSession(GameConfig(**study.next_trial['config']))
            with self.assertRaises(ValueError):
                study.capture(session, 5., [1., 1., 1.], 0, [])
            for command in DEMO_COMMANDS:
                session.launch(command)
                session.run_until_rest()
            study.capture(session, 20., [1., 2., 3.], 0, [])
            with self.assertRaises(ValueError):
                study.start_attempt()
            with self.assertRaises(ValueError):
                study.rate({key: 0 for key in QUESTIONS}, '')
            resumed = PlaytestStudy(directory, automated=True)
            self.assertEqual(resumed.data['pending']['score'], session.world.result()['score'])
            self.assertEqual(resumed.next_trial['index'], 0)
            resumed.rate({key: 3 for key in QUESTIONS}, 'Test fixture')
            self.assertEqual(resumed.next_trial['index'], 1)
            self.assertTrue(verify_replay(record)['passed'])
            self.assertFalse(summarize_study(directory)['human_evidence'])
            with self.assertRaises(ValueError):
                PlaytestStudy(directory, automated=False)

    def test_interrupted_attempt_is_retained_and_cannot_overwrite_first_record(self):
        with tempfile.TemporaryDirectory() as directory:
            study = PlaytestStudy(directory, automated=True)
            first = study.start_attempt()
            first.write_text('partial record')
            resumed = PlaytestStudy(directory, automated=True)
            second = resumed.start_attempt()
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(), 'partial record')
            self.assertEqual(resumed.data['attempts'][0]['status'], 'interrupted')
            self.assertEqual(resumed.next_trial['index'], 0)
            self.assertEqual(summarize_study(directory)['interrupted_attempts'], 1)

    def test_summary_keeps_losses_and_practice_is_not_used_as_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            study = PlaytestStudy(directory, automated=True)
            for trial in study.data['trials']:
                study.data['completed'].append({
                    'trial_index': trial['index'], 'practice': trial['practice'], 'pair': trial['pair'],
                    'forecast': trial['forecast'], 'score': 300 if trial['practice'] else 90 if trial['forecast'] else 100,
                    'ratings': {key: 2 if trial['forecast'] else 4 for key in QUESTIONS},
                    'decision_seconds': [1., 2., 3.], 'comment': 'Synthetic unit-test fixture',
                    'forecast_completions': 3 if trial['forecast'] else 0, 'ui_errors': [],
                })
            study.save()
            result = summarize_study(directory)
            self.assertTrue(result['complete'])
            self.assertFalse(result['human_evidence'])
            self.assertIsNone(result['mean_with_minus_without'])
            self.assertEqual([pair['score_difference'] for pair in result['pairs']], [-10, -10])
            self.assertEqual(result['main_games_completed'], 4)

    def test_disabled_forecast_clears_overlays_and_schedules_no_worker(self):
        from ui import PocWindow
        window = object.__new__(PocWindow)
        window.prediction_enabled = False
        window.renderer = SimpleNamespace(prediction={'score': 123}, predicted_trace=[1, 2])
        window.prediction_status = Mock()
        window.executor = Mock()
        window._prediction_tick()
        self.assertIsNone(window.renderer.prediction)
        self.assertEqual(window.renderer.predicted_trace, [])
        window.executor.submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
