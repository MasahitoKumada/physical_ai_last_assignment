"""Check the analytic baseline against independently constructed motion."""
from dataclasses import replace
import unittest

import numpy as np

from comparison import AnalyticIceModel
from predictor import GlideObservation, IceEstimate


def synthetic_window(swept=False, spin=1.):
    times = np.arange(16) * .04
    sweep = np.clip(times - .12, 0., .20) if swept else times * 0
    speed = 2. - .047 * 9.81 * (times - .2 * sweep)
    factor = np.tanh(spin) / (speed + .1)
    exposure = np.r_[0., np.cumsum((factor[:-1] + factor[1:]) / 2
                                   * (np.diff(times) - .3 * np.diff(sweep)))]
    heading = .036 * exposure
    return tuple(GlideObservation(float(t), (float(t), 0.),
                                  (float(v*np.cos(a)), float(v*np.sin(a))), spin,
                                  sweep_seconds=float(s))
                 for t,v,a,s in zip(times, speed, heading, sweep))


class AnalyticBaselineTests(unittest.TestCase):
    def setUp(self):
        self.model = AnalyticIceModel(.05, .03, .00015)

    def test_recovers_known_coefficients_with_and_without_executed_sweep(self):
        for swept in (False, True):
            with self.subTest(swept=swept):
                estimate = self.model.estimate(synthetic_window(swept))
                self.assertAlmostEqual(estimate.friction, .047, places=10)
                self.assertAlmostEqual(estimate.curl, .036, places=10)

    def test_zero_spin_cannot_identify_curl_and_retains_previous_estimate(self):
        estimate = self.model.estimate(synthetic_window(spin=0.), IceEstimate(.06, .042))
        self.assertAlmostEqual(estimate.friction, .047, places=10)
        self.assertFalse(estimate.curl_observable)
        self.assertEqual(estimate.curl, .042)

    def test_contacts_and_nonfinite_observations_are_rejected(self):
        original = synthetic_window()
        for replacement in (replace(original[7], contact=True),
                            replace(original[7], velocity=(float('nan'), 0.))):
            with self.assertRaises(ValueError):
                self.model.estimate(original[:7] + (replacement,) + original[8:])


if __name__ == '__main__':
    unittest.main()
