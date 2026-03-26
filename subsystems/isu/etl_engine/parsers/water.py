from typing import Dict, Any
import pandas as pd
import io
import os

from .base import BaseParser
from lib.exceptions import GaiaUnsupportedDataError, GaiaReadDataError


class WaterQualityParser(BaseParser):
    """
    Parser for Water Quality data (Sondes, Lab samples).
    """

    def get_parser_name(self) -> str:
        return 'Water_Quality_Parser_v1'

    def detect(self, signature: Dict[str, Any]) -> float:
        """
        Check if file contains water quality keywords (ph, turbidity, etc.).
        """
        filename = signature.get('filename', '')
        ext = signature.get('extension', '')
        content = signature.get('content', b'')
        score = 0.0

        # 1. Filename Indicators
        filename_indicators = [
            'water',
            'quality',
            'sonde',
            'hydro',
            'chem',
            'lab',
            'sample',
        ]
        if any(x in filename.lower() for x in filename_indicators):
            score += 0.2

        # 2. Header Indicators
        df = self._read_file_sample(content, ext)

        if df is not None:
            headers = [str(c).lower().strip() for c in df.columns]

            strong_indicators = {
                'ph',
                'conductivity',
                'ec',
                'turbidity',
                'do',
                'orp',
                'sulfate',
                'iron',
                'fe',
                'tds',
                'nitrate',
                'mg/l',
            }

            matches = [h for h in headers if any(ind in h for ind in strong_indicators)]
            if matches:
                score += 0.4 + (0.15 * len(matches))

            # 3. Negative Indicators
            negative_indicators = {'displacement', 'velocity', 'inclinometer', 'gnss'}
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

            df.columns = [str(c).strip().lower() for c in df.columns]

            # 2. Standardize Timestamp
            #
            df = self.standardize_timestamp(df)

            return df

        except (pd.errors.ParserError, ValueError) as e:
            raise GaiaReadDataError(f'Water Quality parser failed: {str(e)}')
