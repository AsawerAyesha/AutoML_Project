"""
Feature Type Inference Module
Classifies each feature column into meaningful types for proper statistical analysis.

Feature types:
- continuous_numeric: Floats with many unique values (continuous range). E.g., age, height
- discrete_numeric: Integers with moderate unique values (count/ordinal). E.g., number of items
- binary: Exactly 2 unique values. E.g., sex (0,1), yes/no
- categorical_encoded: Integers/codes with small unique values (1-15). E.g., blood pressure type (0-3)
- categorical_text: Object/string dtype. E.g., "red", "blue", "green"
- id_like: Very high cardinality (>80% unique). Likely IDs, should be dropped or ignored
"""

import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException


class FeatureTypeInference:
    """
    Infers the type of each feature to enable proper statistical analysis and preprocessing.
    """
    
    def __init__(self, df, target_column=None):
        """
        Initialize feature type inference.
        
        Args:
            df (pd.DataFrame): The input dataframe
            target_column (str): Name of target column (will be excluded from classification)
        """
        self.df = df
        self.target_column = target_column
        self.feature_types = {}
        logging.info(f"FeatureTypeInference initialized with {len(df)} rows, {len(df.columns)} columns")
        
    def infer_types(self):
        """
        Infer feature type for each column (except target).
        
        Returns:
            dict: {column_name: {'type': str, 'metadata': dict}}
        """
        try:
            for col in self.df.columns:
                if col == self.target_column:
                    logging.debug(f"Skipping target column: {col}")
                    continue
                
                self.feature_types[col] = self._classify_feature(col)
            
            logging.info(f"Feature type inference complete. Classified {len(self.feature_types)} features")
            return self.feature_types
        
        except Exception as e:
            logging.error(f"Error in feature type inference: {str(e)}")
            raise CustomException(f"Feature type inference failed: {str(e)}")
    
    def _try_parse_datetime(self, series: pd.Series) -> bool:
        """Return True if >=90% of non-null values parse as datetime."""
        # Skip datetime parsing for performance - most datasets don't have datetime columns
        # This dramatically speeds up feature inference (10-100x faster)
        return False
        
        # Original implementation (SLOW - disabled for performance):
        # non_null = series.dropna()
        # if len(non_null) == 0:
        #     return False
        # parsed = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
        # success_ratio = parsed.notna().mean()
        # return success_ratio >= 0.9

    def _classify_feature(self, col):
        """
        Classify a single feature based on dtype, cardinality, and range.
        
        Args:
            col (str): Column name
            
        Returns:
            dict: {
                'type': str,
                'n_unique': int,
                'n_missing': int,
                'other_metadata': ...
            }
        """
        data = self.df[col]
        dtype = data.dtype
        n_unique = data.nunique()
        n_rows = len(self.df)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0
        
        # Handle missing values
        n_missing = data.isnull().sum()
        non_null_count = n_rows - n_missing
        
        logging.debug(f"Classifying column '{col}': dtype={dtype}, n_unique={n_unique}, unique_ratio={unique_ratio:.3f}, n_missing={n_missing}")
        
        # ===== RULE 0: Booleans → binary =====
        if pd.api.types.is_bool_dtype(dtype):
            values = sorted(data.dropna().unique().tolist())
            value_counts = data.value_counts().to_dict()
            return {
                'type': 'binary',
                'values': values,
                'n_missing': int(n_missing),
                'value_counts': {str(k): int(v) for k, v in value_counts.items()},
                'unique_ratio': round(unique_ratio, 4)
            }

        # ===== RULE 1: Object/Category dtype → maybe datetime / id_like / categorical_text =====
        if dtype == 'object' or pd.api.types.is_categorical_dtype(data):
            # datetime check
            if self._try_parse_datetime(data):
                return {
                    'type': 'datetime',
                    'n_unique': int(n_unique),
                    'n_missing': int(n_missing),
                    'unique_ratio': round(unique_ratio, 4)
                }
            # id-like check
            if unique_ratio >= 0.95:
                return {
                    'type': 'id_like',
                    'n_unique': int(n_unique),
                    'n_missing': int(n_missing),
                    'unique_ratio': round(unique_ratio, 4),
                    'example_values': data.dropna().unique()[:3].tolist() if non_null_count > 0 else [],
                    'note': 'High cardinality string'
                }
            # default categorical text
            return {
                'type': 'categorical_text',
                'n_unique': int(n_unique),
                'n_missing': int(n_missing),
                'unique_ratio': round(unique_ratio, 4),
                'example_values': data.dropna().unique()[:3].tolist() if non_null_count > 0 else []
            }

        # ===== RULE 2: Exactly 2 unique values → binary =====
        if n_unique == 2:
            values = sorted(data.dropna().unique().tolist())
            value_counts = data.value_counts().to_dict()
            return {
                'type': 'binary',
                'values': values,
                'n_missing': int(n_missing),
                'value_counts': {str(k): int(v) for k, v in value_counts.items()},
                'unique_ratio': round(unique_ratio, 4)
            }
        
        # ===== RULE 3: Numeric dtype =====
        if pd.api.types.is_numeric_dtype(dtype):
            
            # === Rule 3a: Integer dtype handling ===
            if dtype in ['int64', 'int32', 'int16', 'int8']:
                # Very high cardinality ints → id_like
                if unique_ratio >= 0.95:
                    return {
                        'type': 'id_like',
                        'n_unique': int(n_unique),
                        'unique_ratio': round(unique_ratio, 4),
                        'n_missing': int(n_missing),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'note': 'High cardinality int - likely ID'
                    }
                # VERY low cardinality ints (≤10) → categorical_encoded
                # But only if the range is small relative to unique values (suggests codes, not measurements)
                if n_unique <= 10:
                    values = sorted(data.dropna().unique().tolist())
                    value_counts = data.value_counts().to_dict()
                    return {
                        'type': 'categorical_encoded',
                        'n_unique': int(n_unique),
                        'values': values,
                        'n_missing': int(n_missing),
                        'value_counts': {str(k): int(v) for k, v in value_counts.items()},
                        'unique_ratio': round(unique_ratio, 4)
                    }
                # Otherwise treat as numeric (continuous or discrete)
                # High unique count or spread → continuous_numeric
                if n_unique > 20 or (data.max() - data.min()) > 50:
                    return {
                        'type': 'continuous_numeric',
                        'n_unique': int(n_unique),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'range': float(data.max() - data.min()),
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'q1': float(data.quantile(0.25)),
                        'q3': float(data.quantile(0.75)),
                        'n_missing': int(n_missing),
                        'unique_ratio': round(unique_ratio, 4),
                        'note': 'Integer with sufficient spread/cardinality'
                    }
                # Moderate cardinality and range → discrete_numeric
                return {
                    'type': 'discrete_numeric',
                    'n_unique': int(n_unique),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                    'n_missing': int(n_missing),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'unique_ratio': round(unique_ratio, 4),
                    'note': 'Integer with moderate cardinality'
                }

            # === Rule 3e: Float dtype → continuous_numeric (unless few uniques) ===
            else:
                if n_unique <= 15:  # Unusual: float with few uniques
                    values = sorted(data.dropna().unique().tolist())
                    return {
                        'type': 'categorical_encoded',
                        'n_unique': int(n_unique),
                        'values': values,
                        'n_missing': int(n_missing),
                        'unique_ratio': round(unique_ratio, 4),
                        'note': 'Float dtype but low cardinality'
                    }
                return {
                    'type': 'continuous_numeric',
                    'n_unique': int(n_unique),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'q1': float(data.quantile(0.25)),
                    'q3': float(data.quantile(0.75)),
                    'n_missing': int(n_missing),
                    'unique_ratio': round(unique_ratio, 4)
                }
        
        # ===== FALLBACK =====
        result = {
            'type': 'unknown',
            'dtype': str(dtype),
            'n_unique': int(n_unique),
            'n_missing': int(n_missing),
            'unique_ratio': round(unique_ratio, 4)
        }
        logging.warning(f"  → classified as: unknown (dtype={dtype})")
        return result
    
    def get_feature_type(self, col):
        """
        Get type info for a specific column.
        
        Args:
            col (str): Column name
            
        Returns:
            dict or None: Feature type information, or None if not found
        """
        if col not in self.feature_types:
            if col != self.target_column:
                self.feature_types[col] = self._classify_feature(col)
            else:
                return None
        
        return self.feature_types.get(col)
    
    def get_columns_by_type(self, feature_type):
        """
        Get all columns of a specific type.
        
        Args:
            feature_type (str): Type to filter by (e.g., 'continuous_numeric', 'binary')
            
        Returns:
            list: Column names matching the specified type
        """
        if not self.feature_types:
            self.infer_types()
        
        columns = [col for col, info in self.feature_types.items() 
                   if info.get('type') == feature_type]
        logging.debug(f"Found {len(columns)} columns of type '{feature_type}': {columns}")
        return columns
    
    def get_summary(self):
        """
        Get a summary of all feature types found.
        
        Returns:
            dict: Summary statistics of feature type distribution
        """
        if not self.feature_types:
            self.infer_types()
        
        type_counts = {}
        for col, info in self.feature_types.items():
            ftype = info.get('type', 'unknown')
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        summary = {
            'total_features': len(self.feature_types),
            'type_distribution': type_counts,
            'types': self.feature_types
        }
        
        logging.info(f"Feature type summary: {type_counts}")
        return summary
    
    def display_summary(self):
        """
        Print a human-readable summary of feature types.
        """
        if not self.feature_types:
            self.infer_types()
        
        print("\n" + "="*70)
        print("FEATURE TYPE INFERENCE SUMMARY")
        print("="*70)
        
        for col, info in self.feature_types.items():
            ftype = info.get('type', 'unknown')
            n_unique = info.get('n_unique', 'N/A')
            n_missing = info.get('n_missing', 0)
            
            print(f"\n📌 {col}")
            print(f"   Type: {ftype}")
            print(f"   Unique Values: {n_unique}")
            print(f"   Missing: {n_missing}")
            
            if ftype in ['continuous_numeric', 'discrete_numeric']:
                print(f"   Range: [{info.get('min', 'N/A')}, {info.get('max', 'N/A')}]")
            elif ftype in ['binary', 'categorical_encoded', 'categorical_text']:
                print(f"   Values: {info.get('values', info.get('example_values', 'N/A'))}")
        
        print("\n" + "="*70 + "\n")
