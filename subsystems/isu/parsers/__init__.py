import os
from typing import Dict, List, Any
from .base import BaseParser
from .slope import SlopeStabilityParser
from .water import WaterQualityParser


class ParsingEngine:
    """
    Core logic for identifying file content and selecting the correct parser.
    Replaces the legacy ImportCatalogue.
    """

    def __init__(self, logger):
        self.logger = logger
        # Register available parsers
        self.registered_parsers: List[BaseParser] = [
            SlopeStabilityParser(self.logger),
            WaterQualityParser(self.logger),
        ]
        self.confidence_threshold = 0.6

    def route_and_parse(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Main Entry: Identify -> Score -> Gate -> Parse.
        """
        ext = os.path.splitext(filename)[1].lower()

        # 1. Create simple signature (dict) instead of object
        signature = {'filename': filename, 'extension': ext, 'content': file_content}

        # 2. Scoring: Ask parsers if they recognize the file
        candidates = []
        for parser in self.registered_parsers:
            try:
                score = parser.detect(signature)
                if score > 0:
                    candidates.append(
                        {
                            'parser': parser,
                            'score': score,
                            'name': parser.get_parser_name(),
                        }
                    )

            except (ValueError, TypeError, AttributeError) as e:
                self.logger.warning(
                    f'Error matching parser {parser.get_parser_name()}: {e}'
                )
                continue

        # Check if any parser matched BEFORE sorting
        if not candidates:
            return self._fallback_procedure(filename, 'No parser matched')

        # Sort by score (highest confidence first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_match = candidates[0]

        # 3. Gating: Check confidence threshold
        if best_match['score'] < self.confidence_threshold:
            return self._fallback_procedure(
                filename, f'Confidence {best_match["score"]:.2f} is too low.'
            )

        # Check for ambiguity (two parsers with similar scores)
        if len(candidates) > 1:
            gap = candidates[0]['score'] - candidates[1]['score']
            if gap < 0.1:
                return self._fallback_procedure(
                    filename,
                    f'Ambiguous match between {candidates[0]["name"]} and {candidates[1]["name"]}',
                )

        # 4. Execution: Parse the file
        try:
            self.logger.info(f'Selected {best_match["name"]} for {filename}')

            # Call the parser (now returns a DataFrame)
            parsed_df = best_match['parser'].parse(file_content, filename)

            # Return success response with data preview
            return {
                'status': 'success',
                'parser_applied': best_match['name'],
                'confidence': best_match['score'],
                'row_count': len(parsed_df),
                'columns': list(parsed_df.columns),
                'data_preview': parsed_df.head(5).to_dict(orient='records'),
            }

        except (ValueError, IOError) as e:
            self.logger.error(f'Parsing execution failed: {str(e)}')
            return {'status': 'failed', 'error': f'Parser execution failed: {str(e)}'}

    def _fallback_procedure(self, filename: str, reason: str) -> Dict[str, Any]:
        """Handle files that cannot be parsed (Quarantine)."""
        self.logger.info(f'Quarantine: {filename} - {reason}')
        return {
            'status': 'quarantine',
            'message': 'Automatic parsing failed.',
            'reason': reason,
            'filename': filename,
            'action_required': 'Manual Inspection',
        }
