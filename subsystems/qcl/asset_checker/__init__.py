from datetime import datetime
from typing import Dict, List, Optional

from shapely.geometry import box
from shapely.ops import unary_union

from lib.base import GaiaBase, SubsystemId


class AssetChecker(GaiaBase):
    """
    Pre-flight quality check for STAC assets before running downstream computation.

    Scores a list of assets on a 100-point scale. A task should only proceed
    if the score meets the configured conformance threshold (default: 80).
    """

    def __init__(self):
        super().__init__(SubsystemId.QCL)
        cfg = self.settings.get('qcl', {}).get('asset_checker', {})
        self._task_required_types: Dict[str, List[str]] = cfg.get(
            'task_required_types', {}
        )
        self._s2_platforms = set(cfg.get('s2_platforms', []))
        self._insitu_mime_types = set(cfg.get('insitu_mime_types', []))
        self._coverage_threshold: float = cfg.get('coverage_threshold', 0.95)

    def _get_asset_type(self, asset: Dict) -> str:
        platform = (asset.get('platform') or '').lower()
        mime_type = (asset.get('type') or '').lower()

        if platform in self._s2_platforms:
            return 's2'
        if mime_type in self._insitu_mime_types:
            return 'insitu'
        return 'unknown'

    def _compute_coverage(self, assets: List[Dict], aoi) -> float:
        """
        Return the fraction of the AOI covered by the union of S2 asset bboxes.

        :param assets: Full asset list; only assets that carry a bbox field contribute
        :param aoi: Shapely geometry of the AOI in WGS84 (EPSG:4326)
        :return: Coverage ratio in [0, 1]; 1.0 if no asset has a bbox
        """
        spatial_geoms = [
            box(*a['bbox'])
            for a in assets
            if a.get('bbox')
        ]
        if not spatial_geoms:
            return 1.0  # no bbox data — cannot penalise
        covered = unary_union(spatial_geoms)
        return covered.intersection(aoi).area / aoi.area

    def analyze_assets(
        self,
        assets: List[Dict],
        key_variable: str = 'amd',
        aoi=None,
        verbose: bool = True,
    ) -> Dict:
        """
        Score STAC assets on a 100-point scale.

        Rules:
          - Must-Have Data (50 pts): all required types present; returns score=0 immediately if any missing
          - Cloud Cover   (20 pts): avg S2 cloud cover <10%=20, 10-30%=10, >30%=0; 10 if data unavailable
          - Time Spread   (20 pts): >=3 distinct acquisition dates = 20, else 0
          - Map Coverage  (10 pts): union of S2 bboxes must cover >=coverage_threshold of AOI; 10 if no AOI provided

        :param assets: List of asset dicts returned by SdiReader.search_assets()
        :param key_variable: Task type key used to look up required data types
        :param aoi: Shapely geometry of the AOI in WGS84 (EPSG:4326); if None coverage check is skipped (10 pts)
        :param verbose: Print a conformance report to stdout
        :return: {'conformance': int, 'result': dict}
        :raises ValueError: If key_variable is not a recognised task type
        """
        if key_variable not in self._task_required_types:
            raise ValueError(
                f"Unknown key_variable '{key_variable}'. "
                f'Supported: {list(self._task_required_types)}'
            )

        if not assets:
            if verbose:
                print('No assets to analyze')
            return {'conformance': 0, 'result': {}}

        required_types = self._task_required_types[key_variable]
        present_types = {self._get_asset_type(a) for a in assets}

        # Rule 1: Must-Have Data (50 pts, blocker)
        missing_types = [t for t in required_types if t not in present_types]
        if missing_types:
            if verbose:
                print(f'Conformance: 0 — missing required data types: {missing_types}')
            return {'conformance': 0, 'result': {'missing_types': missing_types}}
        score_must_have = 50

        # Rule 2: Cloud Cover (20 pts)
        needs_s2 = 's2' in required_types
        avg_cloud: Optional[float] = None
        if not needs_s2:
            score_cloud = 20
        else:
            cloud_values = [
                a['cloud_cover']
                for a in assets
                if self._get_asset_type(a) == 's2' and a.get('cloud_cover') is not None
            ]
            if not cloud_values:
                score_cloud = 10  # cloud cover unavailable — uncertain, not perfect
            else:
                avg_cloud = sum(cloud_values) / len(cloud_values)
                if avg_cloud < 10:
                    score_cloud = 20
                elif avg_cloud <= 30:
                    score_cloud = 10
                else:
                    score_cloud = 0

        # Rule 3: Time Spread (20 pts)
        distinct_dates = set()
        for asset in assets:
            ts_str = asset.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    distinct_dates.add(ts.date())
                except (ValueError, AttributeError):
                    continue
        score_time_spread = 20 if len(distinct_dates) >= 3 else 0

        # Rule 4: Map Coverage (10 pts)
        coverage_ratio: Optional[float] = None
        if aoi is None:
            score_coverage = 10  # no AOI provided — cannot check
        else:
            coverage_ratio = self._compute_coverage(assets, aoi)
            score_coverage = 10 if coverage_ratio >= self._coverage_threshold else 0

        total = score_must_have + score_cloud + score_time_spread + score_coverage

        result = {
            'scores': {
                'must_have_data': score_must_have,
                'cloud_cover': score_cloud,
                'time_spread': score_time_spread,
                'map_coverage': score_coverage,
            },
            'present_types': list(present_types),
            'distinct_dates': len(distinct_dates),
            'avg_cloud_cover': avg_cloud,
            'coverage_ratio': coverage_ratio,
        }

        if verbose:
            cloud_str = f'{avg_cloud:.1f}%' if avg_cloud is not None else 'n/a'
            coverage_str = (
                f'{coverage_ratio:.1%}' if coverage_ratio is not None else 'n/a'
            )
            print('=' * 60)
            print('ASSET CONFORMANCE CHECK')
            print('=' * 60)
            print(
                f'  Must-Have Data : {score_must_have:>3}/50  types present: {sorted(present_types)}'
            )
            print(f'  Cloud Cover    : {score_cloud:>3}/20  avg: {cloud_str}')
            print(
                f'  Time Spread    : {score_time_spread:>3}/20  distinct dates: {len(distinct_dates)}'
            )
            print(
                f'  Map Coverage   : {score_coverage:>3}/10  coverage: {coverage_str}'
            )
            print('-' * 60)
            print(f'  TOTAL          : {total:>3}/100')
            print('=' * 60)

        return {'conformance': total, 'result': result}
