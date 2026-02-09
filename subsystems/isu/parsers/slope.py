from typing import Dict, Any
import pandas as pd
import io
import os
import logging
import re
from .base import BaseParser

logger = logging.getLogger('gaia.isu.parsers.slope')
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
        filename_indicators = ['slope', 'gnss', 'insar', 'piezo', 'ground_motion', 'egms', 'displacement']
        if any(x in filename.lower() for x in filename_indicators):
            score += 0.2

        # 2. Header Signature (Critical)
        # 使用父类的 helper 读取前几行，避免重复写读取逻辑
        df = self._read_file_sample(content, ext)

        if df is not None:
            # 转换为小写并去除空格
            headers = [str(c).lower().strip() for c in df.columns]

            strong_indicators = {
                'displacement', 'velocity', 'def_x', 'def_y', 'def_z',
                'pressure', 'pore_water', 'kpa', 'piezo', 'inclinometer',
                'tilt', 'angle', 'depth'
            }

            # 计算匹配到的关键词数量
            matches = [h for h in headers if any(ind in h for ind in strong_indicators)]
            if matches:
                # 匹配得越多，分数越高
                score += 0.4 + (0.15 * len(matches))

            # 3. Negative Indicators (排除水质数据)
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
                raise ValueError(f"Unsupported format: {ext}")

            # Clean headers
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Standardize Timestamp
            df = self.standardize_timestamp(
                df, ['timestamp', 'date', 'time', 'reading_time', 'epoch']
            )

            # Quality Control
            if 'depth' in df.columns:
                if not df['depth'].is_monotonic_increasing and not df['depth'].is_monotonic_decreasing:
                    logger.warning(f"[{filename}] Depth column is not monotonic. Check sensor ordering.")
            disp_cols = [c for c in df.columns if 'disp' in c or 'def' in c]
            for col in disp_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    max_disp = df[col].abs().max()
                    if max_disp > 500:
                        logger.warning(
                            f"[{filename}] Large displacement detected in {col}: {max_disp:.2f}. Check sensor health.")
            # Return cleaned DataFrame
            return df

        except (pd.errors.ParserError, ValueError) as e:
            #
            raise ValueError(f"Slope parser failed to process {filename}: {str(e)}")
