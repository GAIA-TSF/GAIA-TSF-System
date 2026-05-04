from .base import PreprocessingBasePipeline

from pathlib import Path
from pygmtsar import S1, Tiles, Stack
import dask
from dask.distributed import Client
from collections import defaultdict
import numpy as np


class Sentinel1Pipeline(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
        'params': {},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.s1 = None
        self.stack = None
        self.dem = None
        self.landmask = None
        self.sbas = None
        self.dem_masked = None
        self.baseline_pairs = None
        self.corr = None
        self.intf = None
        self.unwrap = None
        self.detrend = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _download_orbits(self, datadir):
        """Download precise orbit files for Sentinel-1 BURST data.

        :param str datadir: Path to directory with downloaded BURST data
        :return: None.
        """
        self.s1 = S1.scan_slc(datadir)
        S1.download_orbits(datadir, self.s1)

    def _download_dem(self, aoi, output_dem):
        """Download DEM.
        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_dem: Path to output DEM file.
        :return: None
        """
        if not output_dem.exists():
            Tiles().download_dem(aoi, filename=output_dem, skip_exist=True)
        self.dem = output_dem

    def _download_landmask(self, aoi, output_landmask):
        """Download landmask.
        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :param Path output_landmask: Path to output landmask file.
        :return: None
        """
        if not output_landmask.exists():
            Tiles().download_landmask(
                aoi, filename=output_landmask, skip_exist=True
            ).fillna(0)
        self.landmask = output_landmask

    def _run_dask_cluster(self, **kwargs):
        """Run Dask Cluster Client for computation.

        :return: None
        """
        if self.client is None:
            self.client = Client(**kwargs)

    def _stack_scenes(self, datadir, workdir):
        """Stack Sentinel-1 scenes together.

        :param str datadir: Directory path with downloaded Sentinel-1 BURST data.
        :param str workdir: Directory path where future computed data will be stored (cannot be the same as datadir).
        :return: None
        """
        data_path = Path(datadir).resolve()
        work_path = Path(workdir).resolve()

        if data_path == work_path:
            raise ValueError(
                f'Safety Triggered: datadir and workdir are the same location ({data_path}). '
                'Aborting to prevent accidental data deletion.'
            )
        if work_path in data_path.parents:
            raise ValueError(
                f'Safety Triggered: workdir ({work_path}) is a parent of datadir ({data_path}). '
                'Aborting to prevent accidental data deletion.'
            )

        self.s1 = S1.scan_slc(datadir)
        self.sbas = Stack(workdir, drop_if_exists=True).set_scenes(self.s1)

    def _reframe_scenes(self, aoi):
        """Reframe stacked Sentinel-1 data to smaller area of interest and stitch them together.

        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :return: None
        """
        self.sbas.compute_reframe(aoi)

    def _load_dem_and_landmask(self, aoi):
        """Load DEM and landmask to stacked and reframed Sentinel-1 data.

        :param str | Polygon aoi: WKT string or Shapely Polygon representing the area of interest.
        :return: None
        """
        self.sbas.load_dem(str(self.dem), aoi)
        self.sbas.load_landmask(str(self.landmask))
        self.dem_masked = self.sbas.get_dem().where(
            self.sbas.get_landmask()
        )  # maybe not needed

    def _align_images(self):
        """Align Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_align()

    def _geocoding_transform(self):
        """Geocode Sentinel-1 images.

        :return: None
        """
        self.sbas.compute_geocode(coarsen=10.0)

    def _find_optimal_network(self):
        """Analyzes all possible scene combinations to find the optimal fully connected SBAS network.

        :return: None
        """
        basedays = [36, 48, 60, 80]
        basemeters = [80, 100, 120, 150]

        stack_df = self.sbas.to_dataframe()
        all_dates = list(stack_df.index.astype(str))
        results = []

        for days in basedays:
            for meters in basemeters:
                # Build candidate pairs
                pairs_df = (
                    self.sbas.sbas_pairs(days=days, meters=meters)
                    if hasattr(self.sbas, 'sbas_pairs')
                    else self.sbas.baseline_pairs(days=days, meters=meters)
                )

                # Connectivity check (DFS)
                adj = defaultdict(set)
                for a, b in pairs_df.iloc[:, :2].astype(str).values:
                    adj[a].add(b)
                    adj[b].add(a)

                seen, components = set(), 0
                for node in all_dates:
                    if node not in seen:
                        components += 1
                        stack = [node]
                        while stack:
                            curr = stack.pop()
                            if curr not in seen:
                                seen.add(curr)
                                stack.extend(adj[curr] - seen)

                results.append(
                    {
                        'days': days,
                        'meters': meters,
                        'n_pairs': len(pairs_df),
                        'n_comp': components,
                        'df': pairs_df,
                    }
                )

        valid = [r for r in results if r['n_comp'] == 1]
        if not valid:
            raise RuntimeError('No connected network found. Try increasing thresholds.')

        best_config = min(valid, key=lambda x: (x['n_pairs'], x['days'], x['meters']))
        self.baseline_pairs = best_config['df']

    def _compute_interferograms(self):
        """
        Compute interferograms from baseline pairs.

        :return: None
        """
        intensity_wavelength = 20  # metres
        phase_wavelength = 30  # metres
        coarsen = (1, 4)
        goldstein_patch = 8  # pixels

        # 1. Load Data
        topo = self.sbas.get_topo()
        data = self.sbas.open_data()

        # 2. Process Intensity (for correlation weights)
        intensity = self.sbas.multilooking(
            np.square(np.abs(data)), wavelength=intensity_wavelength, coarsen=coarsen
        )

        # 3. Compute Phase Difference
        phase = self.sbas.phasediff(self.baseline_pairs, data, topo)
        phase = self.sbas.multilooking(
            phase, wavelength=phase_wavelength, coarsen=coarsen
        )

        # 4. Filter & Synthesis
        self.corr = self.sbas.correlation(phase, intensity)
        phase_goldstein = self.sbas.goldstein(phase, self.corr, goldstein_patch)

        self.intf = self.sbas.interferogram(phase_goldstein)

    def _unwrap_interferograms(self, corr_limit=0.20, unwrap_m=10.0):
        """
        Unwrap interferograms using SNAPHU.

        :param float corr_limit: Minimum correlation threshold for masking.
        :param float unwrap_m: Target spatial resolution for unwrapping in meters.
        :return: None
        """
        if self.intf is None or self.corr is None:
            raise RuntimeError(
                'Interferograms and correlation must be computed before unwrapping.'
            )

        # 1. Decimate to target unwrap spacing
        dec_u = self.sbas.decimator(unwrap_m)
        corr_u, intf_u = dask.persist(dec_u(self.corr), dec_u(self.intf))

        # 2. Build Correlation Mask
        corr_mask = corr_u.where(corr_u >= corr_limit)

        # Verify we have valid pixels to unwrap
        n_valid = int(np.isfinite(corr_mask).sum().compute())
        if n_valid == 0:
            raise RuntimeError(
                f'No pixels found above corr_limit={corr_limit}. Unwrapping aborted.'
            )

        # 3. Run SNAPHU Unwrapping
        self.unwrap = self.sbas.unwrap_snaphu(
            intf_u.where(corr_mask), corr_mask
        ).persist()

        # Trigger computation to catch SNAPHU execution errors immediately
        _ = float(self.unwrap.phase.isel(pair=0).mean().compute())

    def _detrend_phase(self, ramp_factor=3.0):
        """
        Removes long-wavelength ramps from the unwrapped phase in radar geometry.

        :param float ramp_factor: Multiplier for the smoothing wavelength. Higher values remove only very broad trends.
        :return: None
        """
        if self.unwrap is None:
            raise RuntimeError('Phase must be unwrapped before detrending.')

        # 1. Determine spatial scales
        base_wavelength = 30
        ramp_wavelength = ramp_factor * base_wavelength

        # 2. Ensure Dask chunking
        chunksize = 256
        phase_data = self.unwrap.phase
        chunk_spec = {d: chunksize for d in phase_data.dims if d in ('y', 'x')}
        if chunk_spec:
            phase_data = phase_data.chunk(chunk_spec)

        # 3. Build and subtract the ramp
        ramp = self.sbas.gaussian(phase_data, wavelength=ramp_wavelength).persist()

        # Subtracting the ramp leaves only the localized displacement signal
        self.detrend = (phase_data - ramp).persist()

        # 4. Trigger computation
        _ = float(self.detrend.isel(pair=0).mean().compute())

    def _run(self):
        self._build_sbas_stack()
        self._reframe_sbas
        # ...
