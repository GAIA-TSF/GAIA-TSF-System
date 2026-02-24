import zipfile
import json
import os
import tempfile
import requests
import uuid
import boto3

from qcl.logger import Logger

STAC_URL = 'http://stacapi:8000'
ALLOWED_RASTER_EXTENSIONS = {'.tif', '.zip'}

# TODO Make parent class

class EouDataZipImporter:
    """
    Handles importing raster data from ZIP into S3
    and publishing updated STAC Item to a STAC API.
    """

    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.stac_api_url = STAC_URL

        self.temp_dir = None
        self.raster_file = None
        self.json_file = None
        self.stac_json = None
        self.id = 'SDI'
        self.logger = Logger(subsystem=self.id)

    def import_zip(self):
        """
        Main execution workflow.
        """
        self._extract_zip()
        self._load_stac_json()
        self._copy_file_to_s3()
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
                ext = os.path.splitext(file)[1].lower()
                if ext in ALLOWED_RASTER_EXTENSIONS:
                    self.raster_file = os.path.join(root, file)
                elif file.lower().endswith('.json'):
                    self.json_file = os.path.join(root, file)

        if not self.raster_file or not self.json_file:
            message = 'ZIP must contain one raster file (TIF or SAFE in ZIP) and one JSON file.'
            self.logger.debug(message)
            raise Exception(message)

    def _load_stac_json(self):
        """
        Load STAC JSON file.
        """
        with open(self.json_file) as f:
            self.stac_json = json.load(f)

    def _copy_file_to_s3(self):
        """
        Upload raster file to S3 bucket.
        """

        if not self.raster_file:
            raise Exception('Raster file not found. Cannot upload to S3.')

        s3 = boto3.client(
            's3',
            endpoint_url='http://localstack:4566',
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1',
        )

        bucket_name = 'gaia-tsf-private'

        # Create bucket if it does not exist
        try:
            s3.create_bucket(Bucket=bucket_name)
        except s3.exceptions.BucketAlreadyOwnedByYou:
            pass
        except s3.exceptions.BucketAlreadyExists:
            pass

        # Use filename as S3 key (you can adjust folder structure if needed)
        filename = os.path.basename(self.raster_file)
        s3_key = f'rasters/{filename}'

        # Upload file (better for large files than put_object)
        s3.upload_file(Filename=self.raster_file, Bucket=bucket_name, Key=s3_key)

        self.logger.debug(f'Uploaded {self.raster_file} to s3://{bucket_name}/{s3_key}')

        self.s3_bucket = bucket_name
        self.s3_key = s3_key

    def _update_stac_json(self):
        """
        Update STAC asset href to reflect actual S3 location.
        """

        if not hasattr(self, 's3_bucket') or not hasattr(self, 's3_key'):
            raise Exception('S3 location not available. Did upload run?')

        s3_url = f'http://localstack:4566/{self.s3_bucket}/{self.s3_key}'

        # pokud je asset pojmenovaný B01
        self.stac_json['assets']['B01']['href'] = s3_url

        self.logger.debug(f'STAC asset href updated to {s3_url}')

    def _post_to_stac(self):
        """
        Send updated STAC Item to STAC API.
        """
        headers = {'Content-Type': 'application/json'}

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
            json=self.stac_json,
        )

        if response.status_code not in (200, 201):
            message = f'STAC API error: {response.text}'
            self.logger.debug(message)
            raise Exception(message)

        self.logger.debug('STAC item successfully posted.')
