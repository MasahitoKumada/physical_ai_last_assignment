"""Behavioral checks for ice estimation and board forecasts; no test dependency."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from predictor import (FrictionRegressor, GlideObservation, IceEstimate, IceRegressor,
                       ice_features, observed_deceleration, predict_board, predict_free_glide, predict_straight_endpoint)


def glide(deceleration=.5):
    return tuple(GlideObservation(float(t), (float(1 + 1.4 * t - .5 * deceleration * t * t), 0.),
                                  (float(1.4 - deceleration * t), 0.)) for t in np.linspace(0, .4, 11))


class FrictionLearningTests(unittest.TestCase):
    def test_mapping_is_learned_from_labels_and_survives_save_load(self):
        # A deliberately different label law rules out a hardcoded a / g estimator.
        x = np.array([.3, .4, .6, .7])
        y = .02 + .06 * x
        model = FrictionRegressor.fit(x, y, regularization=0.)
        self.assertAlmostEqual(model.estimate(glide(.5)), .05)
        changed = FrictionRegressor.fit(x, y + .01, regularization=0.)
        self.assertAlmostEqual(changed.estimate(glide(.5)), .06)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(FrictionRegressor.load(path).estimate(glide()), model.estimate(glide()))

    def test_invalid_physics_windows_do_not_update_an_estimate(self):
        original = glide()
        cases = {
            "held": replace(original[5], held=True),
            "collision": replace(original[5], contact=True),
            "sweep": replace(original[5], sweeping=True),
            "spin": replace(original[5], angular_velocity=1.),
            "near_stop": replace(original[5], velocity=(.01, 0.)),
            "bad_clock": replace(original[5], time=original[4].time),
            "nonfinite": replace(original[5], velocity=(float("nan"), 0.)),
        }
        for label, changed in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                observed_deceleration(original[:5] + (changed,) + original[6:])
        with self.assertRaises(ValueError):
            observed_deceleration(original[:4])

    def test_time_and_position_origins_do_not_change_friction_inference(self):
        original = glide()
        translated = tuple(replace(o, time=o.time + 400., position=(o.position[0] + 1., .2)) for o in original)
        self.assertAlmostEqual(observed_deceleration(original), observed_deceleration(translated), places=10)

    def test_weak_drag_exits_but_strong_drag_stops_inside(self):
        state = GlideObservation(.5, (3., 0.), (1., 0.))
        weak = predict_straight_endpoint(state, .02)
        strong = predict_straight_endpoint(state, .09)
        self.assertFalse(weak["alive"])
        self.assertAlmostEqual(weak["position"][0], 3.93)
        self.assertGreater(weak["unbounded_stop"][0], 4.)
        self.assertTrue(strong["alive"])
        self.assertLess(strong["position"][0], 3.93)
        with self.assertRaises(ValueError):
            predict_straight_endpoint(replace(state, position=(4.1, 0.)), .05)


def robot_window(sweep=False):
    from physics import CurlingWorld, DT, ShotCommand
    world = CurlingWorld(.047, .036)
    world.start_shot(ShotCommand(3., 74., 1))
    observations, sweep_seconds = [], 0.
    while world.release is None or world.data.time - world.release < .7 - 1e-9:
        elapsed = -1. if world.release is None else float(world.data.time) - world.release
        world.step(sweep and (.18 <= elapsed < .30 or .46 <= elapsed < .58))
        if world.release is not None:
            sweep_seconds += world.sweeping * DT
            tick = round((world.data.time - world.release) / DT)
            if 50 <= tick <= 350 and (tick - 50) % 20 == 0:
                velocity = world.velocities()[0]
                observations.append(GlideObservation(float(world.data.time), tuple(world.positions()[0, :2]),
                                                     tuple(velocity[3:5]), float(velocity[2]),
                                                     sweeping=world.sweeping, sweep_seconds=sweep_seconds))
    return world, tuple(observations)


class IceEstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.window = robot_window()
        cls.model = IceRegressor((0., 1., 0., 1 / 9.81), (0., 1., 0., 1.), .05, .03, .00015, 0.)

    def test_unobservable_curl_preserves_previous_estimate(self):
        no_spin = tuple(replace(o, angular_velocity=0.) for o in self.window)
        estimate = self.model.estimate(no_spin, IceEstimate(.05, .044))
        self.assertFalse(estimate.curl_observable)
        self.assertEqual(estimate.curl, .044)

    def test_noisy_turns_reduce_update_weight_and_model_roundtrips(self):
        noisy = tuple(replace(o, velocity=(o.velocity[0], o.velocity[1] + (-1) ** i * .03))
                      for i, o in enumerate(self.window))
        clean_estimate = self.model.estimate(self.window)
        self.assertLess(self.model.estimate(noisy).curl_weight, clean_estimate.curl_weight)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ice.json"
            self.model.save(path)
            self.assertEqual(IceRegressor.load(path).estimate(self.window), clean_estimate)

    def test_executed_sweep_does_not_look_like_a_change_in_ice(self):
        _, swept = robot_window(sweep=True)
        plain_feature, swept_feature = ice_features(self.window), ice_features(swept)
        self.assertAlmostEqual(plain_feature.deceleration, swept_feature.deceleration, delta=.001)
        self.assertAlmostEqual(plain_feature.curvature, swept_feature.curvature, delta=.0003)
        missing_history = tuple(replace(o, sweep_seconds=0.) for o in swept)
        with self.assertRaises(ValueError):
            ice_features(missing_history)
        impossible_history = tuple(replace(o, sweep_seconds=2 * (o.time - swept[0].time)) for o in swept)
        with self.assertRaises(ValueError):
            ice_features(impossible_history)

    def test_observation_forecast_matches_continuation_without_mutating_source(self):
        world, window = robot_window(sweep=True)
        before = world.snapshot()["state"].copy()
        predicted = predict_free_glide(window[-1], IceEstimate(.047, .036))
        np.testing.assert_array_equal(world.snapshot()["state"], before)
        actual = world.run_until_rest()["stones"][0]
        self.assertEqual(predicted["alive"], actual["alive"])
        np.testing.assert_allclose(predicted["position"], actual["position"][:2], atol=1e-6)

    def test_features_do_not_depend_on_camera_or_rink_heading(self):
        theta = .65
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated = tuple(replace(o, position=tuple(rotation @ o.position), velocity=tuple(rotation @ o.velocity))
                        for o in self.window)
        a, b = ice_features(self.window), ice_features(rotated)
        self.assertAlmostEqual(a.deceleration, b.deceleration, places=10)
        self.assertAlmostEqual(a.curvature, b.curvature, places=10)
        with self.assertRaises(ValueError):
            ice_features((replace(self.window[0], contact=True),) + self.window[1:])


class BoardForecastTests(unittest.TestCase):
    def test_before_release_and_contact_forecasts_preserve_live_state_and_match(self):
        from collision_poc import fixture
        from physics import ShotCommand
        for case in ("head_on", "grazing", "chain"):
            world = fixture(case)
            command = ShotCommand(0., 79., 1)
            observations = [(world.observe_board(), command)]
            world.start_shot(command)
            world.step()
            observations.append((world.observe_board(), None))  # Held by robot.
            captured = False
            while world.phase not in ("READY", "RESULT", "ERROR"):
                world.step()
                if world.collisions and not captured:
                    observations.append((world.observe_board(), None))
                    captured = True
            self.assertTrue(captured)
            actual = {s["id"]: s for s in world.result()["stones"]}
            before = world.snapshot()["state"].copy()
            for observed, proposed in observations:
                with self.subTest(case=case, phase=observed.phase):
                    predicted, trace = predict_board(observed, IceEstimate(.05, .03), proposed, include_trace=True)
                    self.assertTrue(trace)
                    self.assertEqual({s["id"] for s in predicted["stones"]}, set(actual))
                    for stone in predicted["stones"]:
                        expected = actual[stone["id"]]
                        self.assertEqual(stone["alive"], expected["alive"])
                        np.testing.assert_allclose(stone["position"], expected["position"], atol=.005)
                    np.testing.assert_array_equal(before, world.snapshot()["state"])

    def test_hidden_ice_and_forces_do_not_enter_observations(self):
        from collision_poc import fixture
        from physics import ShotCommand
        world = fixture("chain")
        a = world.observe_board()
        world.friction, world.curl = .08, .08
        world.data.xfrc_applied[:] = 42.
        world.data.qfrc_applied[:] = 84.
        world.data.qacc_warmstart[:] = 100.
        b = world.observe_board()
        self.assertEqual(a, b)
        command = ShotCommand(0., 79., 1)
        first, _ = predict_board(a, IceEstimate(.05, .03), command)
        second, _ = predict_board(b, IceEstimate(.05, .03), command)
        self.assertEqual(first, second)

    def test_removed_stone_and_nonsequential_ids_survive_roundtrip(self):
        from physics import CurlingWorld, ShotCommand
        world = CurlingWorld(.035, .03)
        world.start_shot(ShotCommand(0., 100., 0))
        world.run_until_rest()
        self.assertFalse(world.alive[0])
        world.place_fixture(2, (2.22, 0.))
        observation = world.observe_board()
        reversed_order = replace(observation, stones=observation.stones[::-1])
        command = ShotCommand(0., 79., 1)
        predicted, _ = predict_board(reversed_order, IceEstimate(.035, .03), command)
        world.start_shot(command)
        self.assertEqual(world.current, 1)
        actual = world.run_until_rest()
        self.assertEqual([s["id"] for s in predicted["stones"]], [0, 1, 2])
        for a, b in zip(predicted["stones"], actual["stones"]):
            self.assertEqual(a["alive"], b["alive"])
            np.testing.assert_allclose(a["position"], b["position"], atol=1e-6)

    def test_rejects_duplicate_ids_and_changing_a_live_shot(self):
        from collision_poc import fixture
        from physics import ShotCommand
        world = fixture("chain")
        observed = world.observe_board()
        duplicated = replace(observed, stones=observed.stones + observed.stones[:1])
        with self.assertRaises(ValueError):
            predict_board(duplicated, IceEstimate(.05, .03), ShotCommand())
        with self.assertRaises(ValueError):
            predict_board(observed, IceEstimate(.05, .03))
        world.start_shot(ShotCommand(0., 79., 1))
        with self.assertRaises(ValueError):
            predict_board(world.observe_board(), IceEstimate(.05, .03), ShotCommand(1., 80., 0))


class GameSessionTests(unittest.TestCase):
    def test_retry_replays_three_shots_and_resets_only_online_estimates(self):
        from game import DEMO_COMMANDS, GameConfig, GameSession
        session = GameSession(GameConfig(123, .053, -.04, .034))
        prior = session.estimator.estimate
        model = session.model
        def play():
            results = []
            for command in DEMO_COMMANDS:
                session.launch(command)
                def sweep(s):
                    return s.world.release is not None and .2 <= s.world.data.time - s.world.release < .4
                results.append(session.run_until_rest(sweep))
            return results
        first = play()
        self.assertEqual(session.world.phase, "RESULT")
        self.assertGreater(session.estimator.updates, 0)
        with self.assertRaises(ValueError):
            session.launch(DEMO_COMMANDS[0])
        session.reset()
        self.assertEqual(session.estimator.estimate, prior)
        self.assertEqual(session.estimator.updates, 0)
        self.assertEqual(session.history, [])
        self.assertIs(session.model, model)
        self.assertEqual(play(), first)

    def test_correction_does_not_change_player_commands_or_actual_score(self):
        from game import DEMO_COMMANDS, GameConfig, GameSession
        outcomes = []
        for adaptive in (False, True):
            session = GameSession(GameConfig(42, .057, .04, .028), adaptive=adaptive)
            for command in DEMO_COMMANDS:
                session.launch(command)
                session.run_until_rest()
            outcomes.append(session.world.result())
            self.assertEqual(session.estimator.updates > 0, adaptive)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_last_estimate_is_carried_without_revealing_next_ice(self):
        from game import DEMO_COMMANDS, GameConfig, GameSession
        session = GameSession(GameConfig(41, .057, -.04, .028))
        session.launch(DEMO_COMMANDS[0])
        session.run_until_rest()
        last = session.estimator.estimate
        self.assertAlmostEqual(last.friction, .057, delta=.001)
        self.assertNotAlmostEqual(last.friction, session.world.friction, delta=.001)
        session.launch(DEMO_COMMANDS[1])
        self.assertEqual(session.estimator.estimate, last)
        self.assertEqual(session.estimator.window, [])


if __name__ == "__main__":
    unittest.main()
