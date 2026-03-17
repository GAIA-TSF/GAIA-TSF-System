import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3])) 

from subsystems.map.learning.validate_lstm import expanding_window_splits


class TestTimeSeriesValidation:
    """
    Tests for time-series cross validation.
    """

    def test_expanding_window_split_count(self):
        """
        Ensure expanding window produces correct number of folds.
        """

        splits = expanding_window_splits(
            n=200,
            look_back=5,
            horizon=5,
            folds=3,
        )

        assert len(splits) == 3

    def test_expanding_window_indices(self):
        """
        Ensure train/test indices are non-overlapping.
        """

        splits = expanding_window_splits(200, 5, 5, folds=2)

        train_idx, test_idx = splits[0]

        assert max(train_idx) < min(test_idx)

