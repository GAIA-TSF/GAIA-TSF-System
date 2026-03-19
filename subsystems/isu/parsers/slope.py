from typing import Dict, Any
import pandas as pd
import io
import os

from .base import BaseParser
from lib.exceptions import GaiaUnsupportedDataError, GaiaReadDataError

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
                'piezo',
                'inclinometer',
                'tilt',
                'angle',
                'depth',
            }

            # Calculate the number of matched keywords
            matches = [h for h in headers if any(ind in h for ind in strong_indicators)]
            if matches:
                # The more matches you get, the higher your score.
                score += 0.4 + (0.15 * len(matches))

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
                df = pd.read_csv(io.BytesIO(content))
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise GaiaUnsupportedDataError(f'Unsupported format: {ext}')

            # Clean headers
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Standardize Timestamp
            df = self.standardize_timestamp(
                df, ['timestamp', 'date', 'time', 'reading_time', 'epoch']
            )

            # Return cleaned DataFrame
            return df

        except (pd.errors.ParserError, ValueError) as e:
            raise GaiaReadDataError(f'Slope parser failed to process {filename}: {str(e)}')
