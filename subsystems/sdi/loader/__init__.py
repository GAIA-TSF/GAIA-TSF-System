import zipfile
import json
import os
import tempfile
import shutil
import psycopg2
from psycopg2 import sql
import requests
import uuid
import boto3
from abc import ABC, abstractmethod

from qcl.logger import Logger

ALLOWED_RASTER_EXTENSIONS = {'.tif', '.zip'}

DB_CONFIG = {
    'host': 'postgis',
    'port': 5432,
    'dbname': 'geodata',
    'user': 'postgres',
    'password': 'fevcfQBu3b3CfxFU',
}

STAC_URL = 'http://stacapi:8000'


class SdiLoader(ABC):
    """
    Base class defining the workflow for loading
    a ZIP package into SDI.
    """

    def __init__(self, zip_path, pg_config=None, stac_api_url=None):
        self.zip_path = zip_path
        self.pg_config = pg_config or DB_CONFIG
        self.stac_api_url = stac_api_url or STAC_URL

        self.temp_dir = None
        self.json_file = None
        self.table_name = None
        self.stac_json = None
        self.id = 'SDI'

        self.logger = Logger(subsystem=self.id)

    def import_zip(self, append_data=False):
        """
        Template method defining the full import workflow.
        """
        self._extract_zip()
        self._load_stac_json()
        self._import_data(append_data)
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
        collection_id = f'testcollection{uuid.uuid4().hex[:8]}'
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
    def _import_data(self):
        """
        Import content into SDI.
        """
        pass


class InSituDataLoader(SdiLoader):
    """
    Concrete implementation for CSV datasets described in STAC JSON.
    """

    def _import_data(self, append_data=False):
        """
        Import all assets defined in STAC JSON into PostGIS.
        Dynamically creates tables based on 'table:columns' for each asset.
        'lat' and 'lon' columns are always used to build geometry.
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
                with psycopg2.connect(**self.pg_config) as conn:
                    with conn.cursor() as cur:
                        # Check if table exists
                        table_exists = False
                        if append_data:
                            cur.execute(
                                "SELECT to_regclass(%s);",
                                (self.table_name,)
                            )
                            table_exists = cur.fetchone()[0] is not None

                        # Table handling
                        if not append_data:
                            # původní režim
                            cur.execute(sql.SQL(f'DROP TABLE IF EXISTS {self.table_name};'))
                            cur.execute(
                                sql.SQL(
                                    f'CREATE TABLE {self.table_name} ({", ".join(sql_columns)});'
                                )
                            )

                        else:
                            # append režim
                            if not table_exists:
                                cur.execute(
                                    sql.SQL(
                                        f'CREATE TABLE {self.table_name} ({", ".join(sql_columns)});'
                                    )
                                )

                        # Bulk load CSV
                        csv_path = os.path.join(self.temp_dir, href)

                        with open(csv_path, 'r') as f:
                            cur.copy_expert(
                                sql.SQL(
                                    f'COPY {self.table_name}({", ".join([c["name"] for c in columns])}) '
                                    'FROM STDIN WITH CSV HEADER'
                                ).as_string(conn),
                                f,
                            )

                        # Update geom column from lat/lon
                        cur.execute(
                            sql.SQL(f"""
                                UPDATE {self.table_name}
                                SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);
                                CREATE INDEX ON {self.table_name} USING GIST (geom);
                            """)
                        )

                self.logger.debug(f'Table "{self.table_name}" successfully imported.')

            except psycopg2.Error as e:
                raise RuntimeError(
                    f"""
                    PostgreSQL error while importing {asset_key}
                    Table: {self.table_name}
                    SQLSTATE: {e.pgcode}
                    Message: {e.pgerror}
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
    Handles uploading raster files to S3 and publishing STAC items.
    """

    def _import_data(self, append_data=False):
        """
        Locate raster assets in STAC JSON, upload each to S3,
        and update asset href dynamically.
        """
        assets = self.stac_json.get('assets', {})
        if not assets:
            raise Exception('STAC JSON contains no assets')

        self.raster_files = []  # store all raster paths processed

        s3 = boto3.client(
            's3',
            endpoint_url='http://localstack:4566',
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1',
        )
        bucket_name = 'gaia-tsf-private'
        # create bucket if missing
        try:
            s3.create_bucket(Bucket=bucket_name)
        except s3.exceptions.BucketAlreadyOwnedByYou:
            pass
        except s3.exceptions.BucketAlreadyExists:
            pass

        for asset_key, asset in assets.items():
            href = asset.get('href')
            if not href:
                self.logger.debug(f'Skipping asset {asset_key}, missing href')
                continue

            # Resolve raster file path from ZIP
            raster_path = os.path.join(self.temp_dir, href)
            if not os.path.exists(raster_path):
                self.logger.debug(
                    f'Raster file {href} not found in extracted ZIP, skipping'
                )
                continue

            # Save path to raster_files list
            self.raster_files.append(raster_path)

            # S3 key can include asset_key to avoid collisions
            filename = os.path.basename(href)
            s3_key = f'rasters/{asset_key}_{filename}'

            # Upload to S3
            s3.upload_file(Filename=raster_path, Bucket=bucket_name, Key=s3_key)
            asset['href'] = f'http://localstack:4566/{bucket_name}/{s3_key}'
            self.logger.debug(f'Uploaded {raster_path} to s3://{bucket_name}/{s3_key}')

    def _update_stac_json(self):
        """
        All asset hrefs already updated during S3 upload in _import_data.
        No additional changes needed.
        """
        pass  # hrefs already updated
