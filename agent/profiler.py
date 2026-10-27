import os
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DatasetProfiler:
    """Universal dataset profiler for arbitrary CSV, Excel, and in-memory tabular data."""

    @staticmethod
    def load_dataset(
        source: Union[str, Path, io.BytesIO, bytes],
        file_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Safely ingests CSV or Excel datasets with automatic encoding and format detection.
        
        Returns:
            Tuple of (loaded DataFrame, file metadata dict)
        """
        name = file_name or (str(source) if isinstance(source, (str, Path)) else "dataset.csv")
        name_lower = name.lower()
        metadata: Dict[str, Any] = {
            "file_name": name,
            "format": "csv" if not name_lower.endswith((".xlsx", ".xls")) else "excel",
            "file_size_bytes": 0,
            "encoding": "utf-8"
        }

        # Calculate byte size if possible
        if isinstance(source, (str, Path)):
            if os.path.exists(str(source)):
                metadata["file_size_bytes"] = os.path.getsize(str(source))
        elif isinstance(source, bytes):
            metadata["file_size_bytes"] = len(source)
            source = io.BytesIO(source)
        elif isinstance(source, io.BytesIO):
            metadata["file_size_bytes"] = source.getbuffer().nbytes
            source.seek(0)

        # Ingestion logic
        df: Optional[pd.DataFrame] = None
        if name_lower.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(source, engine="openpyxl")
            except Exception as e:
                try:
                    df = pd.read_excel(source)
                except Exception as inner_e:
                    raise ValueError(f"Failed to parse Excel file '{name}': {str(inner_e)}")
        else:
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            last_err = None
            for enc in encodings:
                try:
                    if hasattr(source, "seek"):
                        source.seek(0)
                    df = pd.read_csv(source, encoding=enc, on_bad_lines="skip")
                    metadata["encoding"] = enc
                    break
                except UnicodeDecodeError as ude:
                    last_err = ude
                    continue
                except Exception as e:
                    last_err = e
                    break
            
            if df is None:
                raise ValueError(f"Failed to parse CSV file '{name}': {str(last_err)}")

        if df is None or df.empty:
            if df is not None and len(df.columns) == 0:
                raise ValueError(f"File '{name}' contains no readable columns.")
            raise ValueError(f"File '{name}' is empty or contains no records.")

        metadata["rows"] = len(df)
        metadata["columns"] = len(df.columns)
        return df, metadata

    @staticmethod
    def infer_candidate_type(series: pd.Series) -> str:
        """Infers the most appropriate semantic type for a column."""
        s_clean = series.dropna()
        if s_clean.empty:
            return "empty"

        # Check boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if s_clean.dtype == object:
            unique_vals = set(s_clean.astype(str).str.strip().str.lower().unique())
            if unique_vals.issubset({"true", "false", "yes", "no", "t", "f", "1", "0"}):
                return "boolean"

        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            return "numeric"
        
        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
            
        # Check string that can be coerced to numeric (>=75% match)
        if series.dtype == object:
            converted_num = pd.to_numeric(s_clean, errors="coerce")
            num_ratio = converted_num.notna().sum() / len(s_clean)
            if num_ratio >= 0.75:
                return "numeric"

            # Check if parseable as dates (>=75% match)
            if s_clean.astype(str).str.len().mean() > 4:
                parsed_dates = pd.to_datetime(s_clean.head(100), errors="coerce")
                date_ratio = parsed_dates.notna().sum() / len(parsed_dates)
                if date_ratio >= 0.75:
                    return "datetime"

            # Check categorical
            distinct_ratio = series.nunique(dropna=True) / len(series)
            if distinct_ratio < 0.10 or series.nunique(dropna=True) <= 20:
                return "categorical"

        return "text"

    @classmethod
    def profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates comprehensive, non-hardcoded summary profiling on any tabular dataset."""
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        total_nulls = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        col_profiles = {}
        candidate_types = {}

        for col in df.columns:
            s = df[col]
            null_count = int(s.isna().sum())
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
            distinct_count = int(s.nunique(dropna=True))
            inferred_type = cls.infer_candidate_type(s)
            candidate_types[col] = inferred_type

            is_numeric = (inferred_type in ["numeric", "integer"]) or pd.api.types.is_numeric_dtype(s)
            is_datetime = (inferred_type == "datetime") or pd.api.types.is_datetime64_any_dtype(s)

            col_info = {
                "name": str(col),
                "dtype": str(s.dtype),
                "inferred_type": inferred_type,
                "is_numeric": bool(is_numeric),
                "is_datetime": bool(is_datetime),
                "null_count": null_count,
                "null_pct": null_pct,
                "distinct_count": distinct_count,
                "unique_ratio": round(distinct_count / total_rows, 4) if total_rows > 0 else 0.0,
                "sample_values": s.dropna().head(5).astype(str).tolist()
            }

            clean_num = pd.to_numeric(s.dropna(), errors="coerce").dropna() if is_numeric else pd.Series(dtype=float)
            if not clean_num.empty:
                col_info.update({
                    "min": float(clean_num.min()),
                    "max": float(clean_num.max()),
                    "mean": round(float(clean_num.mean()), 3),
                    "median": round(float(clean_num.median()), 3),
                    "std": round(float(clean_num.std()), 3) if len(clean_num) > 1 else 0.0,
                    "q25": round(float(clean_num.quantile(0.25)), 3),
                    "q75": round(float(clean_num.quantile(0.75)), 3),
                })
            elif distinct_count <= 25 and total_rows > 0:
                top_vals = s.value_counts(dropna=False).head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in top_vals.items()}

            col_profiles[col] = col_info

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "total_null_cells": total_nulls,
            "overall_missingness_pct": round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0.0,
            "duplicate_rows_count": duplicate_rows,
            "duplicate_rows_pct": round(duplicate_rows / total_rows * 100, 2) if total_rows > 0 else 0.0,
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "memory_usage_mb": round(int(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2),
            "columns": list(df.columns),
            "candidate_types": candidate_types,
            "column_profiles": col_profiles
        }

    @staticmethod
    def apply_schema_overrides(df: pd.DataFrame, overrides: Dict[str, str]) -> pd.DataFrame:
        """Applies explicit user-specified schema type corrections to a dataframe copy."""
        work_df = df.copy()
        for col, target_type in overrides.items():
            if col not in work_df.columns:
                continue
            target = target_type.lower().strip()
            if target in ["numeric", "float"]:
                work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
            elif target == "integer":
                num_s = pd.to_numeric(work_df[col], errors="coerce")
                work_df[col] = num_s.round().astype("Int64")
            elif target == "datetime":
                work_df[col] = pd.to_datetime(work_df[col], errors="coerce")
            elif target == "categorical":
                work_df[col] = work_df[col].astype("category")
            elif target in ["string", "text"]:
                work_df[col] = work_df[col].astype(str)
            elif target == "boolean":
                bool_map = {"true": True, "yes": True, "1": True, "t": True,
                            "false": False, "no": False, "0": False, "f": False}
                work_df[col] = work_df[col].astype(str).str.strip().str.lower().map(bool_map)
        return work_df
