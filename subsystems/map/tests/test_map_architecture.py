from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_builder import DatasetBuilder
from core.pipeline_executor import PipelineExecutor
from plugins.models.rf import RandomForestModel
from plugins.selection.stable_pixel_selector import StablePixelSelector


def test_dataset_builder_and_stable_pixel_selector(tmp_path: Path) -> None:
    velocity = np.arange(20, dtype=float).reshape(5, 2, 2)
    acceleration = np.arange(20, 40, dtype=float).reshape(5, 2, 2)

    np.save(tmp_path / "velocity.npy", velocity)
    np.save(tmp_path / "acceleration.npy", acceleration)

    config = SimpleNamespace(
        features=SimpleNamespace(
            selected=["velocity", "acceleration"],
            source_dir=str(tmp_path),
            look_back=2,
            target_feature="velocity",
        ),
        dataset=SimpleNamespace(
            split=SimpleNamespace(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        ),
        experiment=SimpleNamespace(seed=42),
    )

    dataset = DatasetBuilder(config).build()

    assert dataset.X_train.shape[0] > 0
    assert dataset.y_train.shape[0] == dataset.X_train.shape[0]
    assert dataset.X_val.shape[0] > 0
    assert dataset.X_test.shape[0] > 0

    selector = StablePixelSelector(SimpleNamespace(stable_pixel_std_threshold=0.008))
    mask = selector.select(np.stack([velocity.mean(axis=0), acceleration.mean(axis=0)], axis=0))

    assert mask.shape == (2, 2)
    assert mask.dtype == bool


def test_random_forest_model_persistence(tmp_path: Path) -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    model = RandomForestModel(SimpleNamespace(n_estimators=10, random_state=42))
    model.train(X, y)

    predictions = model.predict(X)
    assert predictions.shape[0] == X.shape[0]

    model_path = tmp_path / "model.joblib"
    model.save(model_path)
    assert model_path.exists()

    loaded = RandomForestModel.load(model_path)
    assert loaded is not None


def test_pipeline_executor_runs_configured_dag() -> None:
    config = SimpleNamespace(
        pipelines=SimpleNamespace(
            learning=SimpleNamespace(
                dag=SimpleNamespace(
                    nodes=SimpleNamespace(
                        first=SimpleNamespace(op="append", inputs=["input"], value="first"),
                        second=SimpleNamespace(op="append", inputs=["first"], value="second"),
                    ),
                    output="second",
                )
            )
        )
    )

    def append_op(config, inputs, context, node_config):  # noqa: ANN001
        values = list(inputs[0] or [])
        values.append(node_config.value)
        return values

    result = PipelineExecutor(config, {"append": append_op}).run(
        "learning",
        initial_context={"input": []},
    )

    assert result == ["first", "second"]
