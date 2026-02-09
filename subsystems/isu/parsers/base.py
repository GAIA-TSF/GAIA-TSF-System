from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import logging

logger = logging.getLogger('gaia.isu.parser')


class BaseParser(ABC):
    """
    Abstract Base Class for all parser plugins.
    """

    @abstractmethod
    def get_parser_name(self) -> str:
        """Return unique identifier for the parser."""
        pass

    def _read_file_sample(self, content: bytes, ext: str, nrows: int = 5) -> Optional[pd.DataFrame]:
        """
        Helper method to read a small sample of the file for detection/preview.
        Reduces code duplication across parsers.
        """
        try:
            if ext == '.csv':
                return pd.read_csv(io.BytesIO(content), nrows=nrows)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(io.BytesIO(content), nrows=nrows)
            return None
        except (ValueError, pd.errors.ParserError):
            # Return None if parsing fails (not a valid CSV/Excel)
            return None
        except Exception as e:
            logger.debug(f"Sample read failed: {str(e)}")
            return None
    @abstractmethod
    def detect(self, signature: Dict[str, Any]) -> float:
        """Calculate confidence score (0.0 - 1.0) based on file signature."""
        pass

    @abstractmethod
    def parse(self, content: bytes, filename: str) -> pd.DataFrame:
        """Parse content into structured dictionary."""
        pass

    def standardize_timestamp(
        self,
        df: pd.DataFrame,
        time_col_candidates: List[str] = None,
    ) -> pd.DataFrame:
        """
        Standardize timestamp columns to ISO-8601 UTC.
        """
        if time_col_candidates is None:
            time_col_candidates = ['timestamp', 'time', 'date', 'datetime', 'epoch']

        target_col = None
        # Exact match first
        for col in df.columns:
            if col.lower().strip() in time_col_candidates:
                target_col = col
                break

        # Fuzzy match
        if not target_col:
            for col in df.columns:
                if any(candidate in col.lower() for candidate in time_col_candidates):
                    target_col = col
                    break

        if not target_col:
            logger.warning(f'[{self.get_parser_name()}] No timestamp column found.')
            return df

        try:
            # Convert to datetime (coerce errors, dayfirst for EU format)
            df['iso_timestamp'] = pd.to_datetime(
                df[target_col],
                errors='coerce',
                dayfirst=True,
            )

            # QC: Drop invalid timestamps
            invalid_count = df['iso_timestamp'].isna().sum()
            if invalid_count > 0:
                logger.warning(
                    f'[{self.get_parser_name()}] Dropping {invalid_count} rows with invalid timestamps.'
                )
                df = df.dropna(subset=['iso_timestamp'])

            # Format as ISO string (SDI standard)
            df['iso_timestamp'] = df['iso_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

            return df

        except (KeyError, ValueError) as e:
            logger.error(f'Timestamp standardization failed: {str(e)}')
            raise ValueError(
                f"Failed to standardize timestamp column '{target_col}': {str(e)}"
            )
