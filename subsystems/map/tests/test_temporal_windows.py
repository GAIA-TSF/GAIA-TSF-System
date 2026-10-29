"""Independent MAP windows support full-period retrospective monitoring."""

import unittest

from subsystems.map.utils.temporal_windows import resolve_temporal_window


class TemporalWindowTests(unittest.TestCase):
    dates = ('2020-01-01', '2020-01-13', '2020-01-25')

    def test_full_period_includes_last_calibration_acquisition(self):
        config = {'temporal_windows': {
            'calibration': {'full_period': True},
            'monitoring': {'full_period': True},
        }}
        calibration = resolve_temporal_window(
            self.dates, config, 'calibration', end_inclusive=False,
        )
        monitoring = resolve_temporal_window(self.dates, config, 'monitoring')
        self.assertEqual(calibration, monitoring)
        self.assertEqual((calibration.start_index, calibration.end_index), (0, 3))
        self.assertEqual(calibration.end_date, self.dates[-1])

    def test_explicit_windows_can_overlap_and_preserve_boundaries(self):
        config = {'temporal_windows': {
            'calibration': {'start_date': self.dates[0], 'end_date': self.dates[-1]},
            'monitoring': {'start_date': self.dates[0], 'end_date': self.dates[-1]},
        }}
        calibration = resolve_temporal_window(
            self.dates, config, 'calibration', end_inclusive=False,
        )
        monitoring = resolve_temporal_window(self.dates, config, 'monitoring')
        self.assertEqual(calibration.end_index, 2)
        self.assertEqual(monitoring.end_index, 3)
        self.assertEqual(calibration.start_index, monitoring.start_index)

    def test_full_period_rejects_empty_acquisitions(self):
        with self.assertRaisesRegex(ValueError, 'no acquisition dates'):
            resolve_temporal_window(
                (), {'temporal_windows': {'calibration': {'full_period': True}}},
                'calibration',
            )

    def test_full_period_takes_precedence_over_explicit_bounds(self):
        window = resolve_temporal_window(self.dates, {'temporal_windows': {
            'calibration': {'full_period': True, 'start_date': '2030-01-01'},
        }}, 'calibration', end_inclusive=False)
        self.assertEqual((window.start_index, window.end_index), (0, 3))
