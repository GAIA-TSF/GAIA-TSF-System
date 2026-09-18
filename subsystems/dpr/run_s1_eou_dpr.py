"""
Task: run EOU-DPR subsystems to download Sentinel-1 images for slope stability monitoring
and prepo-cessing the images for given test site.

Prep:
1. set start & end dates
2. create project /home/lukas/GAIA-TSF/src/GAIA-TSF-System/tests/projects/slope_monitoring_cadia
3. update project config.yaml
4. set config.yaml data_dir:  /home/lukas/GAIA-TSF/tsf_experiments/slope_monitoring_cadia

Cadia site:
Direction	Orbit path	Burst products
Ascending   (A)	  82	3—all in 2016
Descending  (D)	  45	207

Usage:
cd docker/
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 run_s1_eou_dpr.py
new way:
docker compose exec -T -u "$(id -u):$(id -g)" gaiatesting bash -c 'ulimit -Sn 65536 && exec python3 -u -m subsystems.dpr.run_s1_eou_dpr' 2>&1 | tee s1_run.log
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a configured Sentinel-1 project.')
    parser.add_argument('--project', default='slope_monitoring_jagersfontein')
    parser.add_argument('--skip-download', action='store_true', help='Reuse local SLC inputs')
    parser.add_argument('--result-dir', type=Path, help='Separate output directory for comparison runs')
    parser.add_argument('--reference-area', type=Path, help='Stable-ground polygon file with a declared CRS')
    args = parser.parse_args()
    config_path = Path(TestUtils.get_project_config_path(args.project))
    project_config = ProjectConfigReader(config_path)
    sentinel1 = project_config.get('sentinel1', {})
    period = project_config['project']['monitoring_period']
    reference_wkt = sentinel1.get('reference_area_wkt', '')
    reference_path = args.reference_area
    if reference_path is None and sentinel1.get('reference_area'):
        reference_path = Path(sentinel1['reference_area'])
        if not reference_path.is_absolute():
            reference_path = config_path.parent / reference_path
    if reference_path is not None:
        reference_wkt = load_reference_area(reference_path, project_config.aoi())

    # download input data
    data_dir = Path(project_config['project']['data_dir'], 'sentinel1')
    print(data_dir)

    if not args.skip_download:
        dag_module = DataAcquisitionGateway(backend='asf')
        results = dag_module.backend.search(
            geom=project_config.aoi(),
            start=period['start'],
            end=period['end'],
            direction=sentinel1.get('direction', 'A'),
            path_number=sentinel1.get('path_number'),
        )
        dag_module.backend.download_all(results, target_dir=data_dir)

    # configure & run the pipeline
    pipeline = PreprocessingPipelines().pipelines['sentinel1']

    pipeline.configure(
        datadir=data_dir,
        aoi=project_config.aoi(),
        dem_path= data_dir / 'dem.nc',
        landmask_path= data_dir / 'landmask.nc',
        workdir= data_dir / 'workdir',
        result_dir=args.result_dir or data_dir / 'results',
        excluded_dates=sentinel1.get('excluded_dates') or [],
        reference_area_wkt=reference_wkt,
        diagnostics=sentinel1.get('diagnostics', True),
        diagnostics_root=str(Path(project_config['project']['data_dir'])),
    )

    pipeline.run()
