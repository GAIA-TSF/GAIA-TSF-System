import zipfile
import json
import os
import tempfile
import psycopg2
from psycopg2 import sql
import requests
import uuid

from qcl.logger import Logger

DB_CONFIG = {
    'host': 'postgis',
    'port': 5432,
    'dbname': 'geodata',
    'user': 'postgres',
    'password': 'fevcfQBu3b3CfxFU',
}

STAC_URL = 'http://stacapi:8000'

class IsuDataZipImporter:
    """
    Handles importing CSV data from ZIP into PostGIS
    and publishing updated STAC Item to a STAC API.
    """

    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.pg_config = DB_CONFIG
        self.stac_api_url = STAC_URL

        self.temp_dir = None
        self.csv_file = None
        self.json_file = None
        self.table_name = None
        self.stac_json = None
        self.id = 'SDI'
        self.logger = Logger(subsystem=self.id)

    def import_zip(self):
        """
        Main execution workflow.
        """
        self._extract_zip()
        self._load_stac_json()
        self._derive_table_name()
        self._import_csv_to_postgis()
        self._update_stac_json()
        self._post_to_stac()

    def _extract_zip(self):
        """
        Extract ZIP file and locate CSV and JSON files.
        """
        self.temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(self.zip_path, 'r') as z:
            z.extractall(self.temp_dir)

        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    self.csv_file = os.path.join(root, file)
                elif file.lower().endswith(".json"):
                    self.json_file = os.path.join(root, file)

        if not self.csv_file or not self.json_file:
            message = 'ZIP must contain one CSV file and one JSON file.'
            self.logger.debug(message)
            raise Exception(message)

    def _load_stac_json(self):
        """
        Load STAC JSON file.
        """
        with open(self.json_file) as f:
            self.stac_json = json.load(f)

    def _derive_table_name(self):
        """
        Extract table name from STAC asset href.
        """
        try:
            original_url = self.stac_json["assets"]["data"]["href"]
            self.table_name = original_url.split("/")[-1]
        except KeyError:
            message = 'STAC JSON does not contain assets.data.href'
            self.logger.debug(message)
            raise Exception(message)

    def _import_csv_to_postgis(self):
        """
        Create table, import CSV data including date column,
        build geometry column and create indexes.
        """

        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cur:

                    table_identifier = sql.Identifier(self.table_name)

                    # Drop and recreate table
                    cur.execute(sql.SQL("""
                                        DROP TABLE IF EXISTS {table};
    
                                        CREATE TABLE {table} (
                                                                 id SERIAL PRIMARY KEY,
                                                                 lon DOUBLE PRECISION,
                                                                 lat DOUBLE PRECISION,
                                                                 value DOUBLE PRECISION,
                                                                 measurement_date DATE
                                        );
                                        """).format(table=table_identifier))

                    # Bulk load CSV
                    with open(self.csv_file, "r") as f:
                        cur.copy_expert(
                            sql.SQL("""
                                COPY {table}(lon, lat, value, measurement_date)
                                FROM STDIN WITH CSV HEADER
                            """).format(table=table_identifier).as_string(conn),
                            f
                        )

                    # Add geometry column
                    cur.execute(sql.SQL("""
                                        ALTER TABLE {table}
                                            ADD COLUMN geom geometry(Point, 4326);
    
                                        UPDATE {table}
                                        SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);
                                        """).format(table=table_identifier))

                    # Create indexes
                    cur.execute(sql.SQL("""
                                        CREATE INDEX ON {table} USING GIST (geom);
                                        CREATE INDEX ON {table} (measurement_date);
                                        """).format(table=table_identifier))

            self.logger.debug(f'Table "{self.table_name}" successfully imported.')

        except Exception as e:
            # Connection context manager automatically rolls back
            message = f'Failed to import CSV into PostGIS: {e}'
            self.logger.debug(message)
            raise RuntimeError(message) from e



    def _update_stac_json(self):
        """
        Update STAC asset href to reflect actual PostgreSQL connection.
        """
        pg_url = (
            f"postgresql://user:password"
            f"@{self.pg_config['host']}:{self.pg_config['port']}"
            f"/{self.pg_config['dbname']}#{self.table_name}"
        )

        self.stac_json["assets"]["data"]["href"] = pg_url

    def _post_to_stac(self):
        """
        Send updated STAC Item to STAC API.
        """
        headers = {"Content-Type": "application/json"}

        print(self.stac_json)

        # create a new collection
        collection_id = f'testcollection{uuid.uuid4().hex[:8]}'
        collection_payload = {
            'id': collection_id,
            'title': 'Test Collection',
            'description': 'Collection created for pytest test',
            'extent': {
                'spatial': {'bbox': [[0, 0, 1, 1]]},
                'temporal': {
                    'interval': [['2025-10-01T00:00:00Z', '2025-10-31T23:59:59Z']]
                },
            },
            'license': 'proprietary',
        }

        response = requests.post(f'{STAC_URL}/collections', json=collection_payload)

        if response.status_code not in (200, 201):
            message = f'STAC API error: {response.text}'
            self.logger.debug(message)
            raise Exception(message)

        response = requests.post(
            f'{STAC_URL}/collections/{collection_id}/items',
            headers=headers,
            json=self.stac_json
        )

        if response.status_code not in (200, 201):
            message = f'STAC API error: {response.text}'
            self.logger.debug(message)
            raise Exception(message)

        self.logger.debug('STAC item successfully posted.')

