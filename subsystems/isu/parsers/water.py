from typing import Dict, Any
import pandas as pd
import io
from .base import BaseParser, FileSignature


class WaterQualityParser(BaseParser):
    """
    Parser for Water Quality data (Sondes, Lab samples).
    """

    def get_parser_name(self) -> str:
        return 'Water_Quality_Parser_v1'

    def detect(self, signature: FileSignature) -> float:
        score = 0.0

        # 1. Strong Indicators
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

        matches = [
            h for h in signature.headers if any(ind in h for ind in strong_indicators)
        ]

        if matches:
            score += 0.4 + (0.15 * (len(matches) - 1))

        # 2. Filename Indicators
        filename_indicators = [
            'water',
            'quality',
            'sonde',
            'hydro',
            'chem',
            'lab',
            'sample',
        ]
        if any(x in signature.filename.lower() for x in filename_indicators):
            score += 0.2

        # 3. Negative Indicators
        negative_indicators = {'displacement', 'velocity', 'inclinometer', 'gnss'}
        if any(neg in h for h in signature.headers for neg in negative_indicators):
            score -= 0.6

        return min(max(score, 0.0), 1.0)

    def parse(self, content: bytes, signature: FileSignature) -> Dict[str, Any]:
        try:
            if signature.extension in ['.csv', '.txt']:
                df = pd.read_csv(io.BytesIO(content))
            elif signature.extension == '.xlsx':
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise ValueError('Unsupported format during parsing')

            df.columns = [str(c).strip().lower() for c in df.columns]

            # Validation / QC Logic
            validation_notes = []

            # Check pH range (0-14)
            ph_cols = [c for c in df.columns if 'ph' in c]
            if ph_cols:
                ph_col = ph_cols[0]
                if not df[ph_col].between(0, 14).all():
                    validation_notes.append(
                        'Warning: pH values out of range (0-14) detected.'
                    )

            # Check negative values
            non_negative_cols = [
                c
                for c in df.columns
                if any(x in c for x in ['conductivity', 'ec', 'sulfate', 'turbidity'])
            ]
            for col in non_negative_cols:
                if (df[col] < 0).any():
                    validation_notes.append(
                        f'Warning: Negative values detected in {col}.'
                    )

            return {
                'type': 'Water Quality Data',
                'row_count': len(df),
                'columns': list(df.columns),
                'validation_notes': validation_notes,
                'preview': df.head(5).to_dict(orient='records'),
            }

        except Exception as e:
            raise ValueError(f'Water Quality Parsing Error: {str(e)}')
