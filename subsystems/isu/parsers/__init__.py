import os
import io
import pandas as pd
from typing import Dict, List
from .base import BaseParser, FileSignature
from .slope import SlopeStabilityParser
from .water import WaterQualityParser


class ParsingEngine:
    """
    Core logic for identifying file content and selecting the correct parser.
    Replaces the legacy ImportCatalogue.
    """

    def __init__(self):
        # Register available parsers
        self.registered_parsers: List[BaseParser] = [
            SlopeStabilityParser(),
            WaterQualityParser(),
        ]
        self.confidence_threshold = 0.6

    def _create_signature(self, file_content: bytes, filename: str) -> FileSignature:
        """Create a lightweight file signature from the first 10 rows."""
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in ['.csv', '.txt']:
                df_head = pd.read_csv(io.BytesIO(file_content), nrows=10)
            elif ext in ['.xlsx']:
                df_head = pd.read_excel(io.BytesIO(file_content), nrows=10)
            else:
                raise ValueError('Unsupported container format')

            return FileSignature(
                filename=filename,
                headers=list(df_head.columns),
                sample_df=df_head,
                file_ext=ext,
            )
        except Exception as e:
            raise ValueError(f'Failed to create file signature: {str(e)}')

    def route_and_parse(self, file_content: bytes, filename: str) -> Dict:
        """
        Main Entry: Identify -> Score -> Gate -> Parse.
        """
        # Step 1: Structural Summary
        try:
            signature = self._create_signature(file_content, filename)
        except ValueError as e:
            return {'status': 'failed', 'error': str(e)}

        # Step 2: Scoring
        candidates = []
        for parser in self.registered_parsers:
            score = parser.detect(signature)
            if score > 0:
                candidates.append(
                    {
                        'parser': parser,
                        'score': score,
                        'name': parser.get_parser_name(),
                    }
                )

        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Step 3: Gating & Routing
        if not candidates:
            return self._fallback_procedure(filename, 'No parser matched')

        best_match = candidates[0]

        # Threshold Check
        if best_match['score'] < self.confidence_threshold:
            return self._fallback_procedure(
                filename,
                f'Top match ({best_match["name"]}) confidence {best_match["score"]:.2f} is too low.',
            )

        # Ambiguity Check
        if len(candidates) > 1:
            gap = candidates[0]['score'] - candidates[1]['score']
            if gap < 0.1:
                return self._fallback_procedure(
                    filename,
                    f'Ambiguous file. Close match between {candidates[0]["name"]} and {candidates[1]["name"]}',
                )

        # Step 4: Execute Parse
        try:
            parsed_data = best_match['parser'].parse(file_content, signature)
            return {
                'status': 'success',
                'parser_applied': best_match['name'],
                'confidence': best_match['score'],
                'data': parsed_data,
            }
        except Exception as e:
            return {'status': 'failed', 'error': f'Parser execution failed: {str(e)}'}

    def _fallback_procedure(self, filename: str, reason: str) -> Dict:
        """Return quarantine status."""
        return {
            'status': 'quarantine',
            'message': 'Automatic parsing failed, file moved to quarantine.',
            'reason': reason,
            'filename': filename,
            'action_required': 'Manual Inspection',
        }
