"""Run Sentinel-1 acquisition and processing for a configured project.

From docker/, process the configured Cadia epochs independently:
    docker compose exec -T -u "$(id -u):$(id -g)" gaiatesting bash -c \
      'ulimit -Sn 65536 && exec python3 -u -m subsystems.dpr.run_s1_eou_dpr \
       --project slope_monitoring_cadia --epoch all --skip-download --insar-only'

Use --dry-run to inspect dates and paths. --epoch pre_failure/post_failure runs
one period; --epoch full explicitly requests the unsplit monitoring period.
With epochs, --result-dir specifies a base containing one subdirectory per epoch.
"""

import argparse
from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from lib.config import ProjectConfigReader
from tests.utils import TestUtils


def load_reference_area(path, aoi):
    """Load all stable-reference polygons and validate before expensive processing."""
    import geopandas as gpd

    reference = gpd.read_file(path)
    if reference.crs is None or reference.empty:
        raise ValueError('Reference file must contain polygons and a declared CRS.')
    if (reference.geometry.isna().any() or reference.geometry.is_empty.any()
            or not reference.geometry.is_valid.all()
            or not reference.geom_type.isin(['Polygon', 'MultiPolygon']).all()):
        raise ValueError('Reference file must contain only valid nonempty polygons.')
    geometry = reference.to_crs(4326).geometry.union_all()
    if not aoi.covers(geometry):
        raise ValueError('All stable reference polygons must be inside the processing AOI.')
    return geometry.wkt


def processing_runs(project_config, epoch=None, result_dir=None):
    """Resolve non-overlapping epochs and isolated paths before any processing."""
    import pandas as pd

    site_dir = Path(project_config['project']['data_dir'])
    period = project_config['project']['monitoring_period']
    epochs = project_config.get('sentinel1', {}).get('epochs', {})
    epoch = epoch or ('all' if epochs else 'full')
    if epoch == 'full':
        selections = {'full': period}
    else:
        if not epochs or any(name not in ('pre_failure', 'post_failure') for name in epochs):
            raise ValueError('Configure sentinel1.epochs with pre_failure and/or post_failure bounds.')
        ordered = sorted(epochs.items(), key=lambda item: pd.Timestamp(item[1]['start']))
        previous_end = None
        for name, bounds in ordered:
            start, end = pd.Timestamp(bounds['start']), pd.Timestamp(bounds['end'])
            if start > end or start < pd.Timestamp(period['start']) or end > pd.Timestamp(period['end']):
                raise ValueError(f'Invalid or out-of-monitoring-period bounds for {name}')
            if previous_end is not None and start <= previous_end:
                raise ValueError('Processing epochs must not overlap.')
            previous_end = end
        if epoch != 'all' and epoch not in epochs:
            raise ValueError(f'No configured epoch: {epoch}')
        selections = dict(ordered) if epoch == 'all' else {epoch: epochs[epoch]}
    runs = []
    for name, bounds in selections.items():
        base = Path(result_dir) if result_dir else site_dir / 'sentinel1' / ('results' if name == 'full' else 'results_epochs')
        runs.append({'name': name, 'start': str(bounds['start']), 'end': str(bounds['end']),
                     'workdir': site_dir / 'processing' / 'sentinel1' / name,
                     'result_dir': base if name == 'full' else base / name})
    return runs


def main():
    parser = argparse.ArgumentParser(description='Process independent configured Sentinel-1 epochs.')
    parser.add_argument('--project', default='slope_monitoring_jagersfontein')
    parser.add_argument('--skip-download', action='store_true', help='Reuse local SLCs; date filtering still applies')
    parser.add_argument('--result-dir', type=Path, help='Output base; independent epochs get named subdirectories')
    parser.add_argument('--reference-area', type=Path, help='Stable-ground polygon file with a declared CRS')
    parser.add_argument('--epoch', choices=['all', 'pre_failure', 'post_failure', 'full'],
                        help='Default: all configured epochs, or full when no epochs are configured')
    parser.add_argument('--insar-only', action='store_true', help='Skip environmental and risk processing')
    parser.add_argument('--dry-run', action='store_true', help='Print periods and paths without processing or downloads')
    args = parser.parse_args()
    config_path = Path(TestUtils.get_project_config_path(args.project))
    project_config = ProjectConfigReader(config_path)
    sentinel1 = project_config.get('sentinel1', {})
    runs = processing_runs(project_config, args.epoch, args.result_dir)
    for run in runs:
        print(f"{run['name']}: {run['start']} through {run['end']} UTC; "
              f"workdir={run['workdir']}; results={run['result_dir']}", flush=True)
    if args.dry_run:
        return
    reference_wkt = sentinel1.get('reference_area_wkt', '')
    reference_path = args.reference_area
    if reference_path is None and sentinel1.get('reference_area'):
        reference_path = Path(sentinel1['reference_area'])
        if not reference_path.is_absolute():
            reference_path = config_path.parent / reference_path
    if reference_path is not None:
        reference_wkt = load_reference_area(reference_path, project_config.aoi())
    data_dir = Path(project_config['project']['data_dir'], 'sentinel1')
    for run in runs:
        if not args.skip_download:
            dag_module = DataAcquisitionGateway(backend='asf')
            results = dag_module.backend.search(
                geom=project_config.aoi(), start=run['start'], end=run['end'],
                direction=sentinel1.get('direction', 'A'), path_number=sentinel1.get('path_number'))
            dag_module.backend.download_all(results, target_dir=data_dir)
        # A new instance and work directory give each epoch its own alignment,
        # reference acquisition, network, unwrapping and inversion.
        pipeline = PreprocessingPipelines().pipelines['sentinel1']
        pipeline.configure(
            datadir=data_dir, aoi=project_config.aoi(),
            dem_path=data_dir / 'dem.nc', landmask_path=data_dir / 'landmask.nc',
            workdir=run['workdir'], result_dir=run['result_dir'],
            processing_start=run['start'], processing_end=run['end'],
            source_safe_only=True, insar_only=args.insar_only,
            excluded_dates=sentinel1.get('excluded_dates') or [],
            reference_area_wkt=reference_wkt, diagnostics=sentinel1.get('diagnostics', True),
            retain_pair_checkpoints=sentinel1.get('retain_pair_checkpoints', True),
            diagnostics_root=str(Path(project_config['project']['data_dir'])))
        pipeline.run()


if __name__ == '__main__':
    main()
