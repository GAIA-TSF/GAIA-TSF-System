from typing import Dict, Any
import pandas as pd
import io
import os
from .base import BaseParser, _read_csv_bytes


class SlopeStabilityParser(BaseParser):
    """
    Parser for Slope Stability data (GNSS, InSAR, Inclinometers).
    """

    def get_parser_name(self) -> str:
        return 'Slope_Stability_Parser_v1'

    def detect(self, signature: Dict[str, Any]) -> float:
        """
        Check if file contains slope stability keywords (displacement, velocity, etc.).
        """
        filename = signature.get('filename', '')
        ext = signature.get('extension', '')
        content = signature.get('content', b'')

        score = 0.0

        # 1. Filename Indicators
        filename_indicators = [
            'slope',
            'gnss',
            'insar',
            'piezo',
            'ground_motion',
            'egms',
            'displacement',
        ]
        if any(x in filename.lower() for x in filename_indicators):
            score += 0.2

        # 2. Header Signature (Critical)
        # Read the first few lines to avoid repeatedly writing the reading logic.
        df = self._read_file_sample(content, ext)

        if df is not None:
            headers = [str(c).lower().strip() for c in df.columns]

            strong_indicators = {
                'displacement',
                'velocity',
                'def_x',
                'def_y',
                'def_z',
                'pressure',
                'pore_water',
                'kpa',
                'kilopascal',
                'pascal',
                'piezo',
                'inclinometer',
                'tilt',
                'angle',
                'depth',
                'dataset',   # dam/structure monitoring: DataSetI, DataSetII, …
                'celsius',   # temperature paired with pressure sensors
            }

            # Calculate the number of matched keywords
            matches = [h for h in headers if any(ind in h for ind in strong_indicators)]
            if matches:
                # The more matches you get, the higher your score.
                score += 0.4 + (0.15 * len(matches))

            # mm-unit columns (e.g. X(mm), Y(mm)) indicate structural monitoring
            if any('(mm)' in h for h in headers):
                score += 0.15

            # 3. Negative Indicators (Excluding water quality data)
            negative_indicators = {'ph', 'conductivity', 'turbidity', 'sulfate'}
            if any(neg in h for h in headers for neg in negative_indicators):
                score -= 0.6

        return min(max(score, 0.0), 1.0)

    def parse(self, content: bytes, filename: str) -> pd.DataFrame:
        try:
            # 1. Load Data
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.csv', '.txt']:
                # Peek at headers: if a QC column exists, preserve it as string so
                # that multi-bit flag values (e.g. '000000000000000000') are not
                # silently coerced to integer 0 by pandas type inference.
                header_df, _ = _read_csv_bytes(content, nrows=0)
                qc_original = next(
                    (c for c in header_df.columns if str(c).strip().upper() == 'QC'),
                    None,
                )
                converters = {qc_original: str} if qc_original else {}
                df, _ = _read_csv_bytes(content, converters=converters)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise ValueError(f'Unsupported format: {ext}')

            # Clean headers
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Standardize Timestamp
            df = self.standardize_timestamp(
                df, ['timestamp', 'date', 'time', 'reading_time', 'epoch']
            )
            df = self.ensure_qc_column(df)

            # Return cleaned DataFrame
            return df

        except (pd.errors.ParserError, ValueError) as e:
            raise ValueError(f'Slope parser failed to process {filename}: {str(e)}')
