from typing import Dict, Any
import pandas as pd
import io
import re
from .base import BaseParser, FileSignature


class SlopeStabilityParser(BaseParser):
    """
    Parser for Slope Stability data (GNSS, InSAR, Inclinometers).
    """

    def get_parser_name(self) -> str:
        return 'Slope_Stability_Parser_v1'

    def detect(self, signature: FileSignature) -> float:
        score = 0.0

        # 1. Header Signature
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

        matches = [
            h for h in signature.headers if any(ind in h for ind in strong_indicators)
        ]

        if matches:
            score += 0.4 + (0.15 * (len(matches) - 1))

        # 2. Filename Indicators
        filename_indicators = [
            'slope',
            'gnss',
            'insar',
            'piezo',
            'ground_motion',
            'egms',
            'displacement',
        ]
        if any(x in signature.filename.lower() for x in filename_indicators):
            score += 0.2

        # 3. Negative Indicators
        negative_indicators = {'ph', 'conductivity', 'turbidity', 'sulfate', 'do_mg'}
        if any(neg in h for h in signature.headers for neg in negative_indicators):
            score -= 0.6

        return min(max(score, 0.0), 1.0)

    def extract_metadata(self, signature: FileSignature) -> dict:
        metadata = {}

        # Extract Site ID
        site_match = re.search(
            r'(Site[A-Z0-9]+|TUD[A-Z0-9]*)', signature.filename, re.IGNORECASE
        )
        metadata['site_id'] = site_match.group(1) if site_match else 'Unknown_Site'

        # Infer Sensor Type
        headers_str = ' '.join(signature.headers)
        if 'pressure' in headers_str or 'piezo' in headers_str:
            metadata['sensor_type'] = 'Piezometer'
        elif 'inclinometer' in headers_str or 'tilt' in headers_str:
            metadata['sensor_type'] = 'Inclinometer'
        elif 'gnss' in signature.filename.lower() or 'lat' in headers_str:
            metadata['sensor_type'] = 'GNSS'
        else:
            metadata['sensor_type'] = 'Generic_Displacement'

        return metadata

    def parse(self, content: bytes, signature: FileSignature) -> Dict[str, Any]:
        try:
            if signature.extension in ['.csv', '.txt']:
                df = pd.read_csv(io.BytesIO(content))
            elif signature.extension == '.xlsx':
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise ValueError('Unsupported format during parsing')

            df.columns = [str(c).strip().lower() for c in df.columns]

            # Standardize Timestamp
            df = self.standardize_timestamp(
                df, ['timestamp', 'date', 'time', 'reading_time', 'epoch']
            )

            # Basic Validation
            validation_notes = []

            if 'depth' in df.columns and not df['depth'].is_monotonic_increasing:
                validation_notes.append(
                    'Info: Depth column is not monotonic (check sensor ordering).'
                )

            disp_cols = [c for c in df.columns if 'disp' in c or 'def' in c]
            for col in disp_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].abs().max() > 500:
                        validation_notes.append(
                            f'Warning: Large displacement detected in {col} (>500). Check sensor health.'
                        )

            extracted_meta = self.extract_metadata(signature)

            return {
                'type': 'Slope Stability Data',
                'subtype': extracted_meta.get('sensor_type'),
                'site_id': extracted_meta.get('site_id'),
                'row_count': len(df),
                'columns': list(df.columns),
                'validation_notes': validation_notes,
                'preview': df.head(5).to_dict(orient='records'),
            }

        except Exception as e:
            raise ValueError(f'Slope Stability Parsing Error: {str(e)}')
