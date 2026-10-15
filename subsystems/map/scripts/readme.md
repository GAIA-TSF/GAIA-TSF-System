# Simulated MAP Monitoring Animation

`simulate_monitoring_animation.py` runs the existing MAP calibration and inference
pipeline as an acquisition-by-acquisition operational replay. It creates a scientific
animation of baseline behaviour, deformation dynamics, regime-change probability,
and risk level.

The simulation uses only information available at each acquisition date. Future
monitoring observations are not included in the current frame.

## Requirements

Run the script from the GAIA-TSF repository root using the same Python environment as
the MAP subsystem. The environment must contain the MAP dependencies and Matplotlib.

Additional output-specific requirements are:

- PNG: Matplotlib only.
- GIF: Pillow and Matplotlib.
- MP4: Matplotlib and `ffmpeg`.

If `ffmpeg` is unavailable when MP4 is requested, the script automatically writes a
GIF with the same base filename.

The configured feature rasters, TSF mask, and other MAP input data must already exist.

## Configuration

Pass the existing MAP YAML configuration using `--config`. The standard configuration
is located at:

```text
subsystems/map/config.yaml
```

### Dataset and model

The script uses the selected MAP dataset and registered model:

```yaml
model: trf
dataset: slope_dataset

datasets:
  slope_dataset:
    mask_path: /path/to/tsf_mask.tif
    features:
      - velocity_lag2
      - velocity_lag3
      - acceleration_lag1
      - acceleration_lag2
    target_feature: velocity_lag1
```

Any model supported through the existing MAP model registry can be used. Model-specific
parameters remain in the normal `models` section of the configuration.

### Input and output paths

Configure the feature directories and MAP output root as usual:

```yaml
data:
  features_directory: /path/to/results/features
  temporal_features_directory: /path/to/results/temporal_features
  meteo_features_directory: /path/to/results/meteo_features
  temporal_alignment_method: exact

outputs:
  root: /path/to/results
```

Paths may be absolute or relative to the configuration file.

### Temporal windows

Define calibration and monitoring windows under the selected dataset:

```yaml
datasets:
  slope_dataset:
    temporal_windows:
      calibration:
        start_date: '2018-01-01'
        end_date: '2020-01-01'
      monitoring:
        start_date: '2020-01-01'
        end_date: '2020-08-30'
```

Window boundaries are interpreted as follows:

```text
Calibration: start_date <= acquisition date < end_date
Monitoring:  start_date <= acquisition date <= end_date
```

Frames are created only for actual acquisition dates. No synthetic daily dates are
inserted, and observations after `monitoring.end_date` are not processed.

### Monitoring and risk settings

The animation reuses the standard temporal monitoring settings:

```yaml
monitoring:
  dashboard:
    enabled: true
    anomaly_magnitude_threshold: 0.02
    cusum:
      instability_direction: negative
      signal: observed_velocity
      reference_value: 0.5
      decision_threshold: 5.0
      smoothing_span: 5
      persistence_window: 20
      persistence_threshold: 0.25
    regime:
      smoothing_span: 5
      medium_risk_threshold: 0.3
      high_risk_threshold: 0.7
```

`instability_direction` must match the deformation sign convention of the input data.
Use `negative` when increasingly negative deformation represents instability, or
`positive` for the opposite convention.

The displayed probability is the existing MAP regime-change probability. It is not a
probability of failure. Risk levels are classified from the configured thresholds:

```text
probability < medium threshold                 -> NORMAL
medium threshold <= probability < high        -> MEDIUM
probability >= high threshold                  -> HIGH
```

Acceleration and deceleration states come from the configured MAP CUSUM and persistence
logic rather than from animation-specific derivative thresholds.

### Display units

Axis labels and value scaling use the normal plotting configuration:

```yaml
plotting:
  deformation_unit: mm/day
  value_scale: 1000.0
```

## Running the simulation

From the repository root, train the configured baseline model and create an MP4:

```bash
python3 subsystems/map/scripts/simulate_monitoring_animation.py \
  --config subsystems/map/config.yaml \
  --output animation/map_monitoring.mp4 \
  --fps 4 \
  --dpi 120
```

Create a GIF:

```bash
python3 subsystems/map/scripts/simulate_monitoring_animation.py \
  --config subsystems/map/config.yaml \
  --output animation/map_monitoring.gif
```

Create only the final PNG frame:

```bash
python3 subsystems/map/scripts/simulate_monitoring_animation.py \
  --config subsystems/map/config.yaml \
  --output animation/map_monitoring.png
```

### Reusing an existing model

By default, the script runs MAP learning before replaying monitoring. To load the model
artifact already stored under `outputs.root/models/<experiment.name>/model.pkl`, use:

```bash
python3 subsystems/map/scripts/simulate_monitoring_animation.py \
  --config subsystems/map/config.yaml \
  --output animation/map_monitoring.mp4 \
  --reuse-model
```

The artifact must have been trained with the same model, dataset, features, and
experiment configuration.

### Showing the figure interactively

Use `--show` to display the figure after writing it:

```bash
python3 subsystems/map/scripts/simulate_monitoring_animation.py \
  --config subsystems/map/config.yaml \
  --output outputs/map_monitoring.gif \
  --show
```

## Command-line options

```text
--config PATH    Required MAP YAML configuration file.
--output PATH    Required .png, .gif, or .mp4 output path.
--fps INTEGER    Animation frames per second; default: 4.
--dpi INTEGER    Output resolution; default: 120.
--reuse-model    Skip calibration and load the existing model artifact.
--show           Display the completed figure interactively.
```

Both `--fps` and `--dpi` must be positive integers.

## Outputs

The requested visualization is written to `--output`. A CSV trajectory is always
created beside it with the same base filename, for example:

```text
outputs/map_monitoring.mp4
outputs/map_monitoring.csv
```

The CSV contains:

```text
date
observed_los
predicted_los
prediction_std
residual
velocity
acceleration
regime_change_probability
dynamics
risk_level
```

When MP4 falls back to GIF, the CSV is named after the resulting GIF base path.

## Causal monitoring behaviour

For each monitoring acquisition at date `t`, model prediction and monitoring analysis
are limited to data available through that acquisition:

```text
observation date <= t
```

The baseline is calibrated only from the calibration window. During monitoring, the
fitted baseline is retained while residual, acceleration/deceleration evidence,
regime-change probability, and risk level are updated sequentially. Plot lines are
revealed only through the current animation cursor.

## Troubleshooting

- **Model artifact not found with `--reuse-model`:** Run once without `--reuse-model`,
  or verify `outputs.root` and `experiment.name`.
- **Temporal window contains no acquisition dates:** Check the configured dates against
  the acquisition dates in the target feature stack.
- **No finite samples remain:** Verify the mask, feature coverage, temporal alignment,
  and target feature.
- **MP4 becomes GIF:** Install `ffmpeg` and ensure it is available on `PATH`.
- **Pillow writer error:** Install Pillow in the MAP Python environment.
