import os
import json
import uuid
import zipfile
import io
import pandas as pd
from typing import Tuple, List, Dict


class SDIExporter:
    """
    Exporter for STAC Table Extension (v1.2.0).
    Automatically senses DataFrame columns to generate Table-Extension metadata.
    Enforces Wide-Format and ISO 8601 high-precision timestamps.
    """

    def __init__(self, logger):
        self.logger = logger
        self.logger.debug('SDIExporter (STAC Table Extension) initialized.')

    def create_sdi_package(
        self, df: pd.DataFrame, filename: str
    ) -> Tuple[io.BytesIO, str]:
        """
        Main workflow to package data and metadata into a ZIP buffer.
        """
        table_name = self._generate_table_name(filename)
        self.logger.info(f'Generating STAC Table package for: {table_name}')

        # 1. Standardize Timestamps to ISO 8601 (STAC Requirement)
        df = self._standardize_dataframe(df)

        # 2. Dynamic Column Sensing (Build table:columns metadata)
        column_definitions = self._build_column_definitions(df)

        # 3. Build the STAC Item JSON
        stac_json = self._generate_stac_table_json(table_name, column_definitions)

        # 4. Create ZIP in memory
        zip_buffer = self._create_zip(df, stac_json)

        self.logger.debug(
            f"Packaged {len(df.columns)} columns for table '{table_name}'."
        )
        return zip_buffer, table_name

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures wide-format consistency and ISO 8601 timestamp precision.
        """
        # use 'iso_timestamp' as the standardized column from ParsingEngine
        if 'iso_timestamp' in df.columns:
            df['iso_timestamp'] = pd.to_datetime(df['iso_timestamp']).dt.strftime(
                '%Y-%m-%dT%H:%M:%SZ'
            )

        return df

    def _build_column_definitions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Pure dynamic sensing logic.
        Maps Pandas dtypes to STAC Table types without hardcoded sensor names.
        """
        cols = []
        for col_name in df.columns:
            dtype = str(df[col_name].dtype)

            # Type Mapping Logic
            if 'int' in dtype or 'float' in dtype:
                stac_type = 'number'
            elif 'datetime' in dtype or 'timestamp' in col_name.lower():
                stac_type = 'datetime'
            else:
                stac_type = 'string'

            # Construct the column metadata object
            col_def = {'name': col_name, 'type': stac_type}
            cols.append(col_def)

        return cols

    def _generate_stac_table_json(
        self, table_name: str, column_definitions: List[Dict]
    ) -> dict:
        """
        Constructs the STAC Item JSON using the Table Extension schema.
        """
        return {
            'type': 'Feature',
            'stac_version': '1.0.0',
            'stac_extensions': [
                'https://stac-extensions.github.io/table/v1.2.0/schema.json'
            ],
            'id': f'isu_{uuid.uuid4().hex[:8]}',
            'properties': {
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(),
                'table:row_count': None,
            },
            'geometry': None,
            'assets': {
                'data': {
                    # The table name is routed via the href fragment (#)
                    'href': f'postgresql://postgis:5432/geodata#{table_name}',
                    'type': 'text/csv',
                    'roles': ['data'],
                    'table:columns': column_definitions,
                }
            },
        }

    def _create_zip(self, df: pd.DataFrame, stac_json: dict) -> io.BytesIO:
        """
        Compresses the CSV and JSON into a ZIP file in memory.
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Wide-Format CSV
            zf.writestr('data.csv', df.to_csv(index=False))

            # 2. Metadata JSON
            zf.writestr('metadata.json', json.dumps(stac_json, indent=4))

        zip_buffer.seek(0)
        return zip_buffer

    def _generate_table_name(self, filename: str) -> str:
        """
        Sanitizes the filename for safe use as a PostgreSQL table name.
        """
        base_name = os.path.splitext(filename)[0].lower()
        sanitized = base_name.replace('-', '_').replace('.', '_')
        return f'isu_{sanitized}'
