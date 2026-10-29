"""Level-1 scores use reference variability and flag monitored residuals only."""
from datetime import date
import unittest
import numpy as np
from subsystems.map.scripts.visualisation.plot_level1_anomalies import level1_scores


class Level1Tests(unittest.TestCase):
    def setUp(self):
        self.dates = [date(2020, 1, day) for day in range(1, 7)]
        self.payload = {'series': [{'phase': 'calibration' if i < 3 else 'monitoring'} for i in range(6)]}
        self.values = {'observed_mean_los_velocity': np.array([-1., 0., 1., -4., 3., np.nan]),
                       'predicted_baseline_velocity': np.zeros(6)}

    def test_sigma_comes_from_calibration_and_threshold_is_strict(self):
        z, flags, sigma, count = level1_scores(self.payload, self.dates, self.values)
        self.assertEqual(sigma, 1.)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(flags, [False, False, False, True, False, False])
        self.assertTrue(np.isnan(z[-1]))

    def test_explicit_sigma_handles_missing_calibration_predictions(self):
        self.values['predicted_baseline_velocity'][:3] = np.nan
        with self.assertRaisesRegex(ValueError, 'Supply --sigma'):
            level1_scores(self.payload, self.dates, self.values)
        _, flags, _, _ = level1_scores(self.payload, self.dates, self.values, sigma=0.5)
        self.assertTrue(flags[3] and flags[4])

    def test_invalid_sigma_and_threshold_are_rejected(self):
        for sigma in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                level1_scores(self.payload, self.dates, self.values, sigma=sigma)
        with self.assertRaises(ValueError):
            level1_scores(self.payload, self.dates, self.values, threshold=0)

    def test_overlapping_calendar_windows_ignore_phase_labels(self):
        self.payload['temporal_windows'] = {
            'calibration': {'configured_start_date': '2020-01-01', 'configured_end_date': '2020-01-04'},
            'monitoring': {'configured_start_date': '2020-01-01', 'configured_end_date': '2020-01-06'},
        }
        _, flags, sigma, _ = level1_scores(self.payload, self.dates, self.values, threshold=0.5)
        self.assertEqual(sigma, 1.)
        self.assertTrue(flags[0] and flags[2])
