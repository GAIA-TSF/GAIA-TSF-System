import zipfile
import json
import os
import tempfile
import shutil
import psycopg
from psycopg import sql
import requests
import boto3
from abc import ABC, abstractmethod
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from lib.base import GaiaBase, SubsystemId


class SdiLoader(ABC, GaiaBase):
    """
    Base class defining the workflow for loading
    a ZIP package into SDI.
    """

    def __init__(self, zip_path, pg_config=None, stac_api_url=None):
        ABC.__init__(self)
        GaiaBase.__init__(self, SubsystemId.SDI)

        self.zip_path = zip_path
        # TBD: raise GaiaSettingsError
        self.pg_config = pg_config or self.settings['sdi']['db']
        self.stac_api_url = stac_api_url or self.settings['sdi']['stac']['url']

        self.temp_dir = None
        self.json_file = None
        self.table_name = None
        self.stac_json = None

    def import_zip(self, append_data: bool = False):
        """
        Template method defining the full import workflow.

        :param bool append_data: True to append data otherwise overwrite
        """
        self._extract_zip()
        self._load_stac_json()
        self._import_data(append_data)
        if not append_data:
            self._update_stac_json()
            self._post_to_stac()
        # TODO maybe return back cleanup

    def _extract_zip(self):
        """
        Extract ZIP file and locate data and JSON files.
        """
        self.temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(self.zip_path, 'r') as z:
            z.extractall(self.temp_dir)

        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.lower().endswith('.json'):
                    self.json_file = os.path.join(root, file)

        if not self.json_file:
            raise FileNotFoundError('ZIP must contain one JSON file.')

    def _load_stac_json(self):
        """
        Load STAC JSON file.
        """
        with open(self.json_file) as f:
            self.stac_json = json.load(f)

    def _cleanup(self):
        """
        Remove temporary directory after processing.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _post_to_stac(self):
        """
        Post updated STAC feature to STAC API.
        Uses bbox and temporal extent from JSON.
        """
        headers = {'Content-Type': 'application/json'}

        if (
            'properties' not in self.stac_json
            or 'datetime' not in self.stac_json['properties']
            or 'bbox' not in self.stac_json
            or 'collection' not in self.stac_json
        ):
            raise ValueError(
                'Missing required fields: Not all items are in the metadata. Requiring datetime, bbox and collection'
            )

        # Temporal interval from JSON
        start_dt = self.stac_json['properties'].get(
            'start_datetime', self.stac_json['properties'].get('datetime')
        )
        end_dt = self.stac_json['properties'].get(
            'end_datetime', self.stac_json['properties'].get('datetime')
        )
        interval = [[start_dt, end_dt]] if start_dt and end_dt else [[None, None]]

        # Spatial bbox from JSON
        bbox = self.stac_json.get('bbox', [0, 0, 1, 1])

        # Create a new collection
        collection_id = self.stac_json['collection']

        collection_url = f'{self.stac_api_url}/collections/{collection_id}'

        # Check if collection exists
        check_response = requests.get(collection_url)

        if check_response.status_code == 200:
            # Collection exists, do not do anything
            self.logger.debug(f'Collection "{collection_id}" already exists')

        elif check_response.status_code == 404:
            # Collection does not exist

            collection_payload = {
                'id': collection_id,
                'title': 'Test Collection',
                'description': 'Collection created dynamically from STAC JSON',
                'extent': {
                    'spatial': {'bbox': [bbox]},
                    'temporal': {'interval': interval},
                },
                'license': 'proprietary',
            }

            response = requests.post(
                f'{self.stac_api_url}/collections', json=collection_payload
            )
            if response.status_code not in (200, 201):
                raise requests.exceptions.HTTPError(f'STAC API error: {response.text}')
        else:
            # Other problem
            raise requests.exceptions.HTTPError(
                f'STAC API Error checking collection: {check_response.status_code} {check_response.text}'
            )

        # We are deleting the item first. Not sure if this is the right approach.
        item_id = self.stac_json['id']
        requests.delete(
            f'{self.stac_api_url}/collections/{collection_id}/items/{item_id}'
        )

        # Post the STAC feature as item
        response = requests.post(
            f'{self.stac_api_url}/collections/{collection_id}/items',
            headers=headers,
            json=self.stac_json,
        )
        if response.status_code not in (200, 201):
            raise requests.exceptions.HTTPError(f'STAC API error: {response.text}')

        self.logger.debug('STAC item successfully posted.')

    @abstractmethod
    def _import_data(self, append_data: bool = False):
        """
        Import content into SDI.

        :param bool append_data: True to append data otherwise overwrite
        """
        pass


class InSituDataLoader(SdiLoader):
    """
    Concrete implementation for CSV datasets described in STAC JSON.
    """

    def _import_data(self, append_data: bool = False):
        """
        Import all assets defined in STAC JSON into PostGIS.
        Dynamically creates tables based on 'table:columns' for each asset.
        'lat' and 'lon' columns are always used to build geometry.

        :param bool append_data: True to append data otherwise overwrite
        """
        assets = self.stac_json.get('assets', {})
        if not assets:
            raise Exception('STAC JSON contains no assets')

        for asset_key, asset in assets.items():
            href = asset.get('href')
            columns = asset.get('table:columns', [])

            if not href or not columns:
                self.logger.debug(
                    f'Skipping asset {asset_key}, missing href or columns'
                )
                continue

            # Determine table name from STAC id + asset key
            self.schema_name = f'{self.stac_json["collection"]}'
            self.table_name = f'{self.stac_json["id"]}_{asset_key}'

            # Build SQL column definitions dynamically
            sql_columns = []
            for col in columns:
                col_name = col['name']
                col_type = col['type']
                if col_type == 'number':
                    sql_type = 'DOUBLE PRECISION'
                elif col_type == 'datetime':
                    sql_type = 'TIMESTAMP'
                else:
                    sql_type = 'TEXT'
                sql_columns.append(f'{col_name} {sql_type}')

            # Always add geom column
            sql_columns.append('geom geometry(Point, 4326)')

            try:
                with psycopg.connect(**self.pg_config) as conn:
                    with conn.cursor() as cur:
                        # Create schema if it does not exist
                        cur.execute(
                            sql.SQL(f'CREATE SCHEMA IF NOT EXISTS {self.schema_name};')
                        )
                        conn.commit()

                        table_ident = sql.Identifier(self.schema_name, self.table_name)

                        # Table handling
                        if not append_data:
                            cur.execute(
                                sql.SQL('DROP TABLE IF EXISTS {}').format(table_ident)
                            )

                            cur.execute(
                                sql.SQL('CREATE TABLE {} ({})').format(
                                    table_ident,
                                    sql.SQL(', ').join(sql.SQL(c) for c in sql_columns),
                                )
                            )

                        else:
                            cur.execute(
                                'SELECT to_regclass(%s);',
                                (self.schema_name + '.' + self.table_name,),
                            )
                            table_exists = cur.fetchone()[0] is not None

                            if not table_exists:
                                cur.execute(
                                    sql.SQL('CREATE TABLE {} ({})').format(
                                        table_ident,
                                        sql.SQL(', ').join(
                                            sql.SQL(c) for c in sql_columns
                                        ),
                                    )
                                )

                        # Bulk load CSV
                        csv_path = os.path.join(self.temp_dir, href)

                        copy_query = sql.SQL(
                            'COPY {} ({}) FROM STDIN WITH CSV HEADER'
                        ).format(
                            table_ident,
                            sql.SQL(', ').join(
                                sql.Identifier(c['name']) for c in columns
                            ),
                        )

                        with open(csv_path, 'rb') as f:
                            with cur.copy(copy_query) as copy:
                                while data := f.read(8192):
                                    copy.write(data)

                        # Update geom column

                        sql.SQL(
                            """
                            UPDATE {}
                            SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326)
                            """
                        ).format(table_ident)

                        # Create spatial index
                        cur.execute(
                            sql.SQL('CREATE INDEX ON {} USING GIST (geom)').format(
                                table_ident
                            )
                        )

                        self.logger.debug(
                            f'Table "{self.schema_name}.{self.table_name}" successfully imported.'
                        )

            except psycopg.Error as e:
                raise RuntimeError(
                    f"""
                    PostgreSQL error while importing {asset_key}
                    Table: {self.table_name}
                    SQLSTATE: {e.sqlstate}
                    Message: {e}
                    """
                ) from e

    def _update_stac_json(self):
        """
        Update STAC JSON asset hrefs to PostGIS connection URLs.
        Keeps original PG URL format without changing credentials.
        """
        assets = self.stac_json.get('assets', {})
        for asset_key, asset in assets.items():
            table_name = f'{self.stac_json["id"]}_{asset_key}'
            pg_url = (
                f'postgresql://user:password'
                f'@{self.pg_config["host"]}:{self.pg_config["port"]}'
                f'/{self.pg_config["dbname"]}#{table_name}'
            )
            asset['href'] = pg_url


class EarthObservationDataLoader(SdiLoader):
    """
    Concrete implementation for raster datasets.
    Handles:
    - GeoTIFF/SAFE upload to S3
    - COG conversion
    - STAC item publishing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cog_temp_dir = None

    def _import_data(self, append_data: bool = False):
        """
        Locate raster assets in STAC JSON, convert to COG,
        upload to S3, and update asset href dynamically.

        :param bool append_data: currently ignored
        """
        assets = self.stac_json.get('assets', {})
        if not assets:
            raise Exception('STAC JSON contains no assets')

        self.raster_files = []

        # Create temp directory for COG files
        self.cog_temp_dir = os.path.join(self.temp_dir, 'cog_outputs')
        os.makedirs(self.cog_temp_dir, exist_ok=True)

        # Initialize S3 client
        s3 = boto3.client(
            's3',
            endpoint_url='http://localstack:4566',
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1',
        )
        bucket_name = 'gaia-tsf-private'

        # Create bucket if missing
        try:
            s3.create_bucket(Bucket=bucket_name)
        except (
            s3.exceptions.BucketAlreadyOwnedByYou,
            s3.exceptions.BucketAlreadyExists,
        ):
            pass

        # Process each asset
        for asset_key, asset in list(assets.items()):
            href = asset.get('href')
            if not href:
                self.logger.debug(f'Skipping asset {asset_key}, missing href')
                continue

            # Resolve raster file path from ZIP
            raster_path = os.path.join(self.temp_dir, href)
            if not os.path.exists(raster_path):
                self.logger.debug(f'Raster file {href} not found, skipping')
                continue

            self.raster_files.append(raster_path)

            # Check if SAFE format or GeoTIFF
            if self._is_safe_format(raster_path):
                self._process_safe_to_cog(
                    raster_path, asset_key, s3, bucket_name, asset
                )
            else:
                self._process_geotiff_to_cog(
                    raster_path, asset_key, s3, bucket_name, asset
                )

    def _is_safe_format(self, path: str) -> bool:
        """Check if path is Sentinel-2 SAFE format"""
        return path.endswith('.SAFE') or 'SAFE' in path

    def _process_geotiff_to_cog(
        self, raster_path: str, asset_key: str, s3, bucket_name: str, asset: Dict
    ):
        """
        Upload original GeoTIFF AND create COG version

        :param raster_path: Path to original GeoTIFF
        :param asset_key: STAC asset key
        :param s3: boto3 S3 client
        :param bucket_name: S3 bucket name
        :param asset: STAC asset dictionary to update
        """
        filename = os.path.basename(raster_path)

        # UPLOAD ORIGINAL GeoTIFF
        self.logger.info(f'📤 Uploading original GeoTIFF: {filename}')
        s3_key_original = f'rasters/original/{asset_key}_{filename}'

        s3.upload_file(Filename=raster_path, Bucket=bucket_name, Key=s3_key_original)

        # Update original asset href
        asset['href'] = f'http://localstack:4566/{bucket_name}/{s3_key_original}'
        asset['type'] = 'image/tiff; application=geotiff'
        asset['roles'] = ['data']

        self.logger.info(f'Original uploaded to s3://{bucket_name}/{s3_key_original}')

        # CREATE AND UPLOAD COG VERSION
        self.logger.info(f'Converting to COG: {raster_path}')

        cog_filename = f'{os.path.splitext(filename)[0]}_cog.tif'
        cog_path = os.path.join(self.cog_temp_dir, cog_filename)

        # Convert to COG
        success = self._convert_to_cog(raster_path, cog_path, target_epsg=3857)

        if success:
            # Upload COG
            s3_key_cog = f'rasters/cog/{asset_key}_{cog_filename}'
            s3.upload_file(Filename=cog_path, Bucket=bucket_name, Key=s3_key_cog)

            # Add COG as NEW asset (not replacing original)
            cog_asset_key = f'{asset_key}_cog'
            self.stac_json['assets'][cog_asset_key] = {
                'href': f'http://localstack:4566/{bucket_name}/{s3_key_cog}',
                'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                'title': f'{asset.get("title", asset_key)} (COG)',
                'roles': ['data', 'visual', 'cloud-optimized'],
                'derived_from': asset_key,  # Reference to original
            }

            self.logger.info(f'COG uploaded to s3://{bucket_name}/{s3_key_cog}')
        else:
            self.logger.warning(f'COG conversion failed for {filename}')

    def _process_safe_to_cog(
        self, safe_path: str, asset_key: str, s3, bucket_name: str, asset: Dict
    ):
        """
        Upload original SAFE AND create COG files for each band

        :param safe_path: Path to .SAFE directory
        :param asset_key: STAC asset key
        :param s3: boto3 S3 client
        :param bucket_name: S3 bucket name
        :param asset: STAC asset dictionary
        """
        self.logger.info(f'Processing SAFE format: {safe_path}')
        safe_name = os.path.basename(safe_path).replace('.SAFE', '')

        # UPLOAD ORIGINAL SAFE (if it's a ZIP or directory reference)
        # Keep original asset reference intact
        # (Assuming original asset already has correct href to SAFE)

        # EXTRACT AND PROCESS BANDS
        bands = self._extract_safe_bands(safe_path)

        if not bands:
            self.logger.warning(f'No bands found in SAFE: {safe_path}')
            return

        # Process each band
        for band_info in bands:
            band_path = band_info['path']
            band_name = band_info['name']

            self.logger.info(f'Processing band {band_name}')

            # A) Upload original JP2 band
            original_filename = os.path.basename(band_path)
            s3_key_original = (
                f'rasters/safe_original/{safe_name}/{band_name}/{original_filename}'
            )

            s3.upload_file(Filename=band_path, Bucket=bucket_name, Key=s3_key_original)

            # Add original band asset
            original_asset_key = f'{asset_key}_{band_name}_original'
            self.stac_json['assets'][original_asset_key] = {
                'href': f'http://localstack:4566/{bucket_name}/{s3_key_original}',
                'type': 'image/jp2',
                'title': f'{band_name} Band (Original JP2)',
                'roles': ['data'],
                'eo:bands': [
                    {
                        'name': band_name,
                        'common_name': self._get_common_band_name(band_name),
                    }
                ],
            }

            self.logger.info(f'Original {band_name} uploaded')

            # B) Convert to COG and upload
            self.logger.info(f'Converting {band_name} to COG')

            cog_filename = f'{safe_name}_{band_name}_cog.tif'
            cog_path = os.path.join(self.cog_temp_dir, cog_filename)

            success = self._convert_to_cog(
                band_path,
                cog_path,
                compression='JPEG',
                overview_levels=[2, 4, 8, 16, 32],
                target_epsg=3857,
            )

            if success:
                # Upload COG
                s3_key_cog = f'rasters/safe_cog/{safe_name}/{band_name}_cog.tif'
                s3.upload_file(Filename=cog_path, Bucket=bucket_name, Key=s3_key_cog)

                # Add COG asset
                cog_asset_key = f'{asset_key}_{band_name}_cog'
                self.stac_json['assets'][cog_asset_key] = {
                    'href': f'http://localstack:4566/{bucket_name}/{s3_key_cog}',
                    'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                    'title': f'{band_name} Band (COG)',
                    'roles': ['data', 'visual', 'cloud-optimized'],
                    'eo:bands': [
                        {
                            'name': band_name,
                            'common_name': self._get_common_band_name(band_name),
                        }
                    ],
                    'derived_from': original_asset_key,  # Link to original
                }

                self.logger.info(f'COG {band_name} uploaded')
            else:
                self.logger.warning(f'COG conversion failed for {band_name}')

    def _extract_safe_bands(self, safe_path: str) -> List[Dict]:
        """
        Extract band files from SAFE directory

        :param safe_path: Path to .SAFE directory
        :return: List of band info dictionaries
        """
        bands = []
        safe_path_obj = Path(safe_path)

        # Find GRANULE/*/IMG_DATA directory
        img_data_dirs = list(safe_path_obj.glob('GRANULE/*/IMG_DATA'))

        if not img_data_dirs:
            return bands

        img_data_dir = img_data_dirs[0]

        # Check for L2A structure (R10m, R20m, R60m)
        resolutions = ['R10m', 'R20m', 'R60m']
        search_dirs = []

        for res in resolutions:
            res_dir = img_data_dir / res
            if res_dir.exists():
                search_dirs.append(res_dir)

        # If no resolution dirs, search in IMG_DATA directly (L1C)
        if not search_dirs:
            search_dirs = [img_data_dir]

        # Find all band files (.jp2)
        for search_dir in search_dirs:
            for band_file in search_dir.glob('*_B*.jp2'):
                # Extract band name (e.g., B02, B03, B04)
                filename = band_file.name
                band_name = None

                for part in filename.split('_'):
                    if part.startswith('B') and len(part) <= 4:
                        band_name = part
                        break

                if band_name:
                    bands.append({'name': band_name, 'path': str(band_file)})

        return bands

    def _get_common_band_name(self, band_name: str) -> Optional[str]:
        """Map Sentinel-2 band names to common names"""
        band_mapping = {
            'B01': 'coastal',
            'B02': 'blue',
            'B03': 'green',
            'B04': 'red',
            'B05': 'rededge',
            'B06': 'rededge',
            'B07': 'rededge',
            'B08': 'nir',
            'B8A': 'nir08',
            'B09': 'nir09',
            'B10': 'cirrus',
            'B11': 'swir16',
            'B12': 'swir22',
        }
        return band_mapping.get(band_name)

    def _convert_to_cog(
        self,
        input_file: str,
        output_file: str,
        compression: str = 'DEFLATE',
        overview_levels: Optional[List[int]] = None,
        blocksize: int = 512,
        target_epsg: Optional[int] = None,
        resampling: str = 'BILINEAR',
    ) -> bool:
        """
        Convert GeoTIFF to Cloud Optimized GeoTIFF (COG)

        :param input_file: Input GeoTIFF path
        :param output_file: Output COG path
        :param compression: Compression (DEFLATE, JPEG, ZSTD)
        :param overview_levels: Overview pyramid levels
        :param blocksize: Internal tile size
        :param target_epsg: Target EPSG code (e.g., 3857 for Web Mercator), None = keep original
        :param resampling: Resampling method for reprojection (BILINEAR, CUBIC, LANCZOS, etc.)
        :return: True if successful
        """
        if overview_levels is None:
            overview_levels = [2, 4, 8, 16]

        try:
            if target_epsg is not None:
                temp_reprojected = input_file.replace(
                    '.tif', f'_epsg{target_epsg}_temp.tif'
                )

                self.logger.info(f'Reprojectování do EPSG:{target_epsg}...')

                reproject_cmd = [
                    'gdalwarp',
                    '-t_srs',
                    f'EPSG:{target_epsg}',
                    '-r',
                    resampling,
                    '-co',
                    'TILED=YES',
                    '-co',
                    'COMPRESS=LZW',
                    '-multi',
                    '-wo',
                    'NUM_THREADS=ALL_CPUS',
                    input_file,
                    temp_reprojected,
                ]

                result = subprocess.run(
                    reproject_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 min timeout pro reprojekci
                )

                source_file = temp_reprojected
            else:
                source_file = input_file

            cmd = [
                'gdal_translate',
                source_file,
                output_file,
                '-of',
                'COG',
                '-co',
                f'COMPRESS={compression}',
                '-co',
                f'BLOCKSIZE={blocksize}',
                '-co',
                'TILED=YES',
                '-co',
                'NUM_THREADS=ALL_CPUS',
                '-co',
                f'OVERVIEWS={",".join(map(str, overview_levels))}',
                '-co',
                'BIGTIFF=IF_SAFER',
            ]

            # Add JPEG quality for JPEG compression
            if compression == 'JPEG':
                cmd.extend(['-co', 'QUALITY=85'])

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
            )

            if target_epsg is not None and os.path.exists(temp_reprojected):
                os.remove(temp_reprojected)
                self.logger.debug(f'Odstraněn dočasný soubor: {temp_reprojected}')

            # 📊 Log file sizes
            input_size = os.path.getsize(input_file) / (1024 * 1024)
            output_size = os.path.getsize(output_file) / (1024 * 1024)

            epsg_info = f' (EPSG:{target_epsg})' if target_epsg else ''
            self.logger.debug(
                f'COG conversion{epsg_info}: {input_size:.2f} MB → {output_size:.2f} MB'
            )

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f'COG conversion failed: {e.stderr}')

            if target_epsg is not None and os.path.exists(temp_reprojected):
                os.remove(temp_reprojected)
            return False

        except subprocess.TimeoutExpired:
            self.logger.error(f'COG conversion timeout for {input_file}')
            if target_epsg is not None and os.path.exists(temp_reprojected):
                os.remove(temp_reprojected)
            return False

        except Exception as e:
            self.logger.error(f'COG conversion error: {str(e)}')
            if (
                target_epsg is not None
                and 'temp_reprojected' in locals()
                and os.path.exists(temp_reprojected)
            ):
                os.remove(temp_reprojected)
            return False

    def _update_stac_json(self):
        """
        Update STAC JSON with COG-specific metadata
        """
        # Add COG extension to STAC
        if 'stac_extensions' not in self.stac_json:
            self.stac_json['stac_extensions'] = []

        # Add projection extension if not present
        proj_ext = 'https://stac-extensions.github.io/projection/v1.0.0/schema.json'
        if proj_ext not in self.stac_json['stac_extensions']:
            self.stac_json['stac_extensions'].append(proj_ext)

        # Update processing metadata
        if 'properties' not in self.stac_json:
            self.stac_json['properties'] = {}

        self.stac_json['properties']['processing:level'] = 'L2A'
        self.stac_json['properties']['processing:software'] = {'gdal': 'COG conversion'}
        self.stac_json['properties']['updated'] = datetime.utcnow().isoformat() + 'Z'
