from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger('gaia.isu.parser')


class FileSignature:
    """
    File signature object to pass metadata between parsers.
    """

    def __init__(
        self, filename: str, headers: List[str], sample_df: pd.DataFrame, file_ext: str
    ):
        self.filename = filename
        # Lowercase and strip headers
        self.headers = [str(h).lower().strip() for h in headers]
        self.sample_df = sample_df
        self.extension = file_ext.lower()


class BaseParser(ABC):
    """
    Abstract Base Class for all parser plugins.
    """

    @abstractmethod
    def get_parser_name(self) -> str:
        """Return unique identifier for the parser."""
        pass

    @abstractmethod
    def detect(self, signature: FileSignature) -> float:
        """Calculate confidence score (0.0 - 1.0) based on file signature."""
        pass

    @abstractmethod
    def parse(self, content: bytes, signature: FileSignature) -> Dict[str, Any]:
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

        except Exception as e:
            logger.error(f'Timestamp standardization failed: {str(e)}')
            raise ValueError(
                f"Failed to standardize timestamp column '{target_col}': {str(e)}"
            )
