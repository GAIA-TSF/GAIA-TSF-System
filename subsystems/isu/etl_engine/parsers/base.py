from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
import io


class BaseParser(ABC):
    """
    Abstract Base Class for all parser plugins.
    """

    _ENCODINGS_TO_TRY = ['utf-8', 'gbk', 'gb18030', 'latin-1']

    def __init__(self, logger, encodings=None, keyword_overrides=None):
        self.logger = logger
        if encodings:
            self._ENCODINGS_TO_TRY = encodings
        # Config-driven detect() keyword lists (config.yaml: isu.parsers.<name>).
        # Falls back to each parser's hardcoded defaults when absent, so
        # parsers keep working when instantiated without settings (e.g. tests).
        self._keyword_overrides = keyword_overrides or {}

    def _keywords(self, key: str, default: List[str]) -> List[str]:
        return self._keyword_overrides.get(key) or default

    def _read_csv_bytes(self, content: bytes, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Try common encodings in order; return (DataFrame, encoding_used)."""
        last_err = None
        for enc in self._ENCODINGS_TO_TRY:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, **kwargs)
                return df, enc
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_err = e
        raise ValueError(
            f'Could not decode CSV with any supported encoding: {last_err}'
        )

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for SDI compatibility.

        Maps latitude/longitude variants to lat/lon, then replaces
        non-alphanumeric characters with underscores.
        """
        rename = {}
        for col in df.columns:
            if col.startswith('lat') and col != 'lat':
                rename[col] = 'lat'
            elif col.startswith('lon') and col != 'lon':
                rename[col] = 'lon'
        if rename:
            df = df.rename(columns=rename)
        df.columns = [re.sub(r'[^\w]+', '_', c).strip('_') or c for c in df.columns]
        return df

    @abstractmethod
    def get_parser_name(self) -> str:
        """Return unique identifier for the parser."""
        pass

    def _read_file_sample(
        self, content: bytes, ext: str, nrows: int = 5
    ) -> Optional[pd.DataFrame]:
        """
        Helper method to read a small sample of the file for detection/preview.
        Reduces code duplication across parsers.
        """
        try:
            if ext in ['.csv', '.txt']:
                return self._read_csv_bytes(content, nrows=nrows)[0]
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(io.BytesIO(content), nrows=nrows)
            return None
        except (ValueError, pd.errors.ParserError, TypeError, OSError) as e:
            self.logger.debug(f'Sample read failed: {str(e)}')
            return None

    @abstractmethod
    def detect(self, signature: Dict[str, Any]) -> float:
        """Calculate confidence score (0.0 - 1.0) based on file signature."""
        pass

    @abstractmethod
    def parse(self, content: bytes, filename: str) -> pd.DataFrame:
        """Parse content into structured dictionary."""
        pass

    def ensure_qc_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a valid integer QC column exists. QC=1 means valid sensor data.
        If the column is absent, add it. If it contains non-integer values such as
        multi-bit flag strings (e.g. '000000000000000000'), normalise to 1 (pass)."""
        qc_col = next((c for c in df.columns if c.upper() == 'QC'), None)
        if qc_col is None:
            df = df.copy()
            df['QC'] = 1
            self.logger.debug(f'[{self.get_parser_name()}] Added default QC=1 column.')
            return df

        # Detect multi-bit flag strings (e.g. '000000000000000000'):
        # pd.to_numeric converts them to 0, which QCL reads as sensor-fault.
        # Check length first: any value longer than 1 char is a bit-string.
        if df[qc_col].astype(str).str.len().max() > 1:
            df = df.copy()
            df[qc_col] = (
                df[qc_col]
                .astype(str)
                .apply(lambda x: 1 if set(x.strip()) <= {'0'} else 0)
                .astype(int)
            )
            self.logger.debug(
                f'[{self.get_parser_name()}] Normalised multi-bit QC flag strings to 0/1.'
            )
            return df

        coerced = pd.to_numeric(df[qc_col], errors='coerce')
        if coerced.isna().any():
            df = df.copy()
            df[qc_col] = coerced.fillna(1).astype(int)
            self.logger.debug(
                f'[{self.get_parser_name()}] Normalised non-integer QC values to 1.'
            )
        return df

    def standardize_timestamp(
        self,
        df: pd.DataFrame,
        time_col_candidates: List[str] = None,
    ) -> pd.DataFrame:
        """
        Standardize timestamp columns to ISO-8601 UTC.
        """
        if time_col_candidates is None:
            time_col_candidates = [
                'timestamp',
                'time',
                'date',
                'datetime',
                'epoch',
                'cjtime',
            ]

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
            self.logger.warning(
                f'[{self.get_parser_name()}] No timestamp column found.'
            )
            return df

        try:
            # Year-first strings (ISO-8601 "YYYY-MM-DD..." / "YYYY/MM/DD...")
            # are unambiguous. Forcing dayfirst=True on them anyway makes
            # pandas' batch format inference mis-swap month/day, which turns
            # any row whose day-of-month is > 12 into an invalid date (e.g.
            # "2018-01-13" gets read as month=13) and silently drops it.
            # Only apply the dayfirst heuristic to genuinely ambiguous
            # (non year-first, e.g. EU DD/MM/YYYY) formats.
            sample = df[target_col].dropna().astype(str)
            year_first = (
                sample.str.match(r'^\d{4}[-/]').all() if not sample.empty else False
            )

            df['iso_timestamp'] = pd.to_datetime(
                df[target_col],
                errors='coerce',
                dayfirst=not year_first,
            )

            # QC: Drop invalid timestamps
            invalid_count = df['iso_timestamp'].isna().sum()
            if invalid_count > 0:
                self.logger.warning(
                    f'[{self.get_parser_name()}] Dropping {invalid_count} rows with invalid timestamps.'
                )
                df = df.dropna(subset=['iso_timestamp'])

            # Format as ISO string (SDI standard)
            df['iso_timestamp'] = df['iso_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

            return df

        except (KeyError, ValueError) as e:
            self.logger.error(f'Timestamp standardization failed: {str(e)}')
            raise ValueError(
                f"Failed to standardize timestamp column '{target_col}': {str(e)}"
            )
