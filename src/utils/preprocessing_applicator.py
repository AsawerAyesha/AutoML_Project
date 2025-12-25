"""
Preprocessing Applicator Module

Applies user-selected preprocessing decisions to the dataset:
1. Outlier handling (cap/remove)
2. Constant feature removal
3. Class imbalance remediation (class_weights/SMOTE/undersampling)
4. Missing value imputation (median/mean/mode/constant)

Tracks all preprocessing decisions in a structured log for reporting.
"""

import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.feature_type_inference import FeatureTypeInference


class PreprocessingApplicator:
    """
    Applies preprocessing actions to the dataset and tracks all decisions.
    
    Returns a modified dataset and a detailed log of all transformations applied.
    """
    
    def __init__(self, df, feature_types_dict, target_column=None):
        """
        Initialize preprocessing applicator.
        
        Args:
            df (pd.DataFrame): The input dataframe
            feature_types_dict (dict): Output from FeatureTypeInference.infer_types()
            target_column (str): Name of target column
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.feature_types = feature_types_dict
        self.target_column = target_column
        self.preprocessing_log = []  # Track all decisions
        
        logging.info(f"PreprocessingApplicator initialized with {len(df)} rows, {len(df.columns)} columns")
    
    def apply_missing_value_imputation(self, imputation_config):
        """
        Apply missing value imputation strategies.
        
        Args:
            imputation_config (dict): {column_name: strategy}
                Strategies: 'median', 'mean', 'mode', 'constant:<value>', 'drop_rows'
        
        Returns:
            dict: Summary of imputation applied
        """
        logging.info("Applying missing value imputation...")
        
        summary = {
            'action': 'missing_value_imputation',
            'columns_affected': [],
            'details': {}
        }
        
        for col, strategy in imputation_config.items():
            if col not in self.df.columns:
                logging.warning(f"Column '{col}' not found in dataframe")
                continue
            
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                logging.debug(f"Column '{col}' has no missing values")
                continue
            
            feature_type = self.feature_types.get(col, {}).get('type', 'unknown')
            
            # SAFETY: prevent numeric strategies on non-numeric columns
            if strategy in ("median", "mean") and feature_type not in ("continuous_numeric", "discrete_numeric"):
                logging.warning(
                    f"Invalid imputation strategy '{strategy}' for non-numeric column '{col}' "
                    f"(type={feature_type}). Falling back to mode."
                )
                strategy = "mode"
            
            if strategy == 'drop_rows':
                # Drop rows with missing values
                self.df = self.df.dropna(subset=[col])
                rows_removed = missing_count
                
                log_entry = {
                    'step': 'missing_value_imputation',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'drop_rows',
                    'rows_removed': int(rows_removed),
                    'rows_remaining': len(self.df)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Dropped {rows_removed} rows with missing values in '{col}'")
            
            elif strategy == 'median':
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                
                log_entry = {
                    'step': 'missing_value_imputation',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'median',
                    'fill_value': float(median_value),
                    'values_imputed': int(missing_count)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Imputed {missing_count} missing values in '{col}' with median={median_value:.4f}")
            
            elif strategy == 'mean':
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)
                
                log_entry = {
                    'step': 'missing_value_imputation',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'mean',
                    'fill_value': float(mean_value),
                    'values_imputed': int(missing_count)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Imputed {missing_count} missing values in '{col}' with mean={mean_value:.4f}")
            
            elif strategy == 'mode':
                m = self.df[col].mode(dropna=True)
                mode_value = m.iloc[0] if len(m) > 0 else "Unknown"
                self.df[col] = self.df[col].fillna(mode_value)
              
                log_entry = {
                    'step': 'missing_value_imputation',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'mode',
                    'fill_value': str(mode_value),
                    'values_imputed': int(missing_count)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Imputed {missing_count} missing values in '{col}' with mode={mode_value}")
            
            elif strategy.startswith('constant:'):
                const_value = strategy.split(':')[1]
                try:
                    const_value = float(const_value) if col in self.df.select_dtypes(include=[np.number]).columns else const_value
                except:
                    pass
                
                self.df[col].fillna(const_value, inplace=True)
                
                log_entry = {
                    'step': 'missing_value_imputation',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'constant',
                    'fill_value': const_value,
                    'values_imputed': int(missing_count)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Imputed {missing_count} missing values in '{col}' with constant={const_value}")
        
        self.preprocessing_log.append(summary)
        return summary
    
    def apply_outlier_action(self, outlier_config):
        """
        Apply outlier handling strategies.
        
        Args:
            outlier_config (dict): {column_name: action, ...}
                Actions: 'remove', 'cap (IQR)', 'no action'
                Must also include bounds info: {column_name: {'action': str, 'bounds': {...}}}
        
        Returns:
            dict: Summary of outliers handled
        """
        logging.info("Applying outlier handling...")
        
        summary = {
            'action': 'outlier_handling',
            'columns_affected': [],
            'details': {}
        }
        
        for col, config in outlier_config.items():
            if col not in self.df.columns:
                logging.warning(f"Column '{col}' not found in dataframe")
                continue
            
            # Skip categorical columns - outlier handling only for numeric
            if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                logging.warning(f"Skipping outlier handling for categorical column '{col}'")
                continue
            
            action = config.get('action') if isinstance(config, dict) else config
            bounds = config.get('bounds', {}) if isinstance(config, dict) else {}
            
            if action == 'no action':
                logging.debug(f"No outlier action for '{col}'")
                continue
            
            feature_type = self.feature_types.get(col, {}).get('type', 'unknown')
            data = self.df[col].dropna()
            
            # Calculate bounds if not provided
            if not bounds:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                bounds = {
                    'lower': float(Q1 - 1.5 * IQR),
                    'upper': float(Q3 + 1.5 * IQR)
                }
            
            if action == 'remove':
                # Remove rows where value is outside bounds
                rows_before = len(self.df)
                self.df = self.df[(self.df[col] >= bounds['lower']) & (self.df[col] <= bounds['upper'])]
                rows_removed = rows_before - len(self.df)
                
                log_entry = {
                    'step': 'outlier_handling',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'remove',
                    'bounds': bounds,
                    'rows_removed': int(rows_removed),
                    'rows_remaining': len(self.df)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Removed {rows_removed} rows with outliers in '{col}'")
            
            elif action == 'cap (IQR)':
                # Cap values to bounds
                outlier_count_below = (self.df[col] < bounds['lower']).sum()
                outlier_count_above = (self.df[col] > bounds['upper']).sum()
                
                self.df[col] = self.df[col].clip(lower=bounds['lower'], upper=bounds['upper'])
                
                log_entry = {
                    'step': 'outlier_handling',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'cap (IQR)',
                    'bounds': bounds,
                    'values_capped_below': int(outlier_count_below),
                    'values_capped_above': int(outlier_count_above),
                    'total_capped': int(outlier_count_below + outlier_count_above)
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
                logging.info(f"Capped {outlier_count_below + outlier_count_above} outliers in '{col}'")
        
        self.preprocessing_log.append(summary)
        return summary
    
    def apply_constant_feature_removal(self, constant_config):
        """
        Remove constant/near-constant features.
        
        Args:
            constant_config (dict): {column_name: action, ...}
                Actions: 'drop feature', 'keep feature'
        
        Returns:
            dict: Summary of features removed
        """
        logging.info("Applying constant feature removal...")
        
        summary = {
            'action': 'constant_feature_removal',
            'columns_affected': [],
            'details': {}
        }
        
        columns_to_drop = []
        
        for col, action in constant_config.items():
            if col not in self.df.columns:
                logging.warning(f"Column '{col}' not found in dataframe")
                continue
            
            if action == 'drop feature':
                columns_to_drop.append(col)
                feature_type = self.feature_types.get(col, {}).get('type', 'unknown')
                
                log_entry = {
                    'step': 'constant_feature_removal',
                    'column': col,
                    'feature_type': feature_type,
                    'action': 'drop',
                    'reason': 'constant or near-constant feature'
                }
                
                summary['columns_affected'].append(col)
                summary['details'][col] = log_entry
            
            elif action == 'keep feature':
                logging.debug(f"Keeping constant feature '{col}'")
        
        # Drop all marked columns at once
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            logging.info(f"Dropped {len(columns_to_drop)} constant features: {columns_to_drop}")
        
        self.preprocessing_log.append(summary)
        return summary
    
    def apply_class_imbalance_handling(self, imbalance_action, target_col=None):
        """
        Apply class imbalance remediation.
        
        Note: SMOTE and undersampling modify the dataset size. 
        Class weights are applied during model training instead.
        
        Args:
            imbalance_action (str): 'class_weights', 'undersampling', 'oversampling', 'no action'
            target_col (str): Target column name (uses self.target_column if not provided)
        
        Returns:
            dict: Summary of imbalance handling
        """
        if target_col is None:
            target_col = self.target_column
        
        if target_col is None or target_col not in self.df.columns:
            logging.warning(f"Target column not found, skipping imbalance handling")
            return None
        
        logging.info(f"Applying class imbalance handling: {imbalance_action}...")
        
        summary = {
            'action': 'class_imbalance_handling',
            'columns_affected': [target_col] if target_col else [],
            'target_column': target_col,
            'method': imbalance_action,
            'details': {}
        }
        
        if imbalance_action == 'no action':
            logging.debug("No class imbalance action selected")
            summary['details']['note'] = 'No action taken'
        
        elif imbalance_action == 'class_weights':
            # Class weights are applied during model training, not here
            value_counts = self.df[target_col].value_counts()
            logging.info(f"Class weights will be applied during model training: {value_counts.to_dict()}")
            summary['details'] = {
                'method': 'class_weights',
                'note': 'Applied during model training',
                'class_distribution': value_counts.to_dict()
            }
        
        elif imbalance_action == 'undersampling':
            # Random undersampling: reduce majority class to match minority class
            value_counts = self.df[target_col].value_counts()
            min_class = value_counts.idxmin()
            max_class = value_counts.idxmax()
            min_count = value_counts.min()
            
            rows_before = len(self.df)
            
            # Separate majority and minority
            minority_df = self.df[self.df[target_col] == min_class]
            majority_df = self.df[self.df[target_col] == max_class]
            
            # Undersample majority class
            majority_df_sampled = majority_df.sample(n=min_count, random_state=42)
            
            # Recombine
            self.df = pd.concat([minority_df, majority_df_sampled], ignore_index=True)
            
            rows_after = len(self.df)
            rows_removed = rows_before - rows_after
            
            summary['details'] = {
                'method': 'undersampling',
                'minority_class': min_class,
                'majority_class': max_class,
                'target_count': min_count,
                'rows_removed': int(rows_removed),
                'rows_before': int(rows_before),
                'rows_after': int(rows_after)
            }
            
            logging.info(f"Undersampled majority class. Removed {rows_removed} rows. "
                        f"New distribution: {self.df[target_col].value_counts().to_dict()}")
        
        elif imbalance_action == 'oversampling':
            # For SMOTE-like oversampling, we'll use simple random oversampling
            value_counts = self.df[target_col].value_counts()
            min_class = value_counts.idxmin()
            max_class = value_counts.idxmax()
            max_count = value_counts.max()
            min_count = value_counts.min()
            
            rows_before = len(self.df)
            
            # Separate classes
            minority_df = self.df[self.df[target_col] == min_class]
            majority_df = self.df[self.df[target_col] == max_class]
            
            # Oversample minority class
            samples_needed = max_count - min_count
            minority_df_sampled = minority_df.sample(n=samples_needed, replace=True, random_state=42)
            
            # Recombine
            self.df = pd.concat([majority_df, minority_df, minority_df_sampled], ignore_index=True)
            
            rows_after = len(self.df)
            rows_added = rows_after - rows_before
            
            summary['details'] = {
                'method': 'oversampling',
                'minority_class': min_class,
                'majority_class': max_class,
                'rows_added': int(rows_added),
                'rows_before': int(rows_before),
                'rows_after': int(rows_after),
                'new_distribution': self.df[target_col].value_counts().to_dict()
            }
            
            logging.info(f"Oversampled minority class. Added {rows_added} rows. "
                        f"New distribution: {summary['details']['new_distribution']}")
        
        self.preprocessing_log.append(summary)
        return summary
    
    def get_processed_dataframe(self):
        """
        Get the processed dataframe.
        
        Returns:
            pd.DataFrame: Modified dataframe after all preprocessing
        """
        return self.df
    
    def get_preprocessing_log(self):
        """
        Get the complete preprocessing log for reporting.
        
        Returns:
            list: List of preprocessing steps applied with details
        """
        return self.preprocessing_log
    
    def get_summary(self):
        """
        Get a summary of all preprocessing applied.
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'rows_original': len(self.original_df),
            'rows_final': len(self.df),
            'rows_removed': len(self.original_df) - len(self.df),
            'columns_original': len(self.original_df.columns),
            'columns_final': len(self.df.columns),
            'columns_removed': len(self.original_df.columns) - len(self.df.columns),
            'preprocessing_steps': len(self.preprocessing_log),
            'steps': self.preprocessing_log
        }
        
        return summary
    
    def display_summary(self):
        """Print a human-readable summary of all preprocessing applied"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        
        print(f"\nDataset Changes:")
        print(f"  Rows: {summary['rows_original']} → {summary['rows_final']} "
              f"({summary['rows_removed']} removed)")
        print(f"  Columns: {summary['columns_original']} → {summary['columns_final']} "
              f"({summary['columns_removed']} removed)")
        
        print(f"\nPreprocessing Steps Applied ({summary['preprocessing_steps']}):")
        for step in self.preprocessing_log:
            action = step.get('action', 'unknown')
            cols_affected = step.get('columns_affected', [])
            print(f"\n  {action.upper()}")
            
            if cols_affected:
                print(f"    Columns affected: {', '.join(cols_affected)}")
                for col, details in step.get('details', {}).items():
                    if isinstance(details, dict):
                        for key, value in details.items():
                            if key not in ['step', 'column', 'feature_type']:
                                print(f"      {col}: {key}={value}")
            else:
                print(f"    {step.get('details', {})}")
        
        print("\n" + "="*80 + "\n")
