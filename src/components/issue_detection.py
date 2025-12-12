import numpy as np
import pandas as pd
from scipy import stats
from src.logger import logging
from src.exception import CustomException

class IssueDetector:
    """Detects and flags data quality issues"""
    
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.issues = []
        self.suggestions = []
        
    def detect_all_issues(self):
        """Run all issue detection methods"""
        self.detect_missing_values()
        self.detect_outliers()
        self.detect_imbalanced_classes()
        self.detect_high_cardinality()
        self.detect_constant_features()
        return self.issues, self.suggestions
    
    def detect_missing_values(self):
        """Check 1: Missing Values (per feature + global %)"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        for col in missing_data[missing_data > 0].index:
            if missing_percent[col] > 0:
                issue = {
                    'type': 'Missing Values',
                    'column': col,
                    'percent': round(missing_percent[col], 2),
                    'count': int(missing_data[col]),
                    'severity': 'high' if missing_percent[col] > 50 else 'medium'
                }
                self.issues.append(issue)
                
                suggestion = {
                    'type': 'Missing Values',
                    'column': col,
                    'options': ['mean (for numeric)', 'median (for numeric)', 
                               'mode (for categorical)', 'constant value', 'drop rows'],
                    'recommended': self._recommend_imputation_strategy(col)
                }
                self.suggestions.append(suggestion)
    
    def detect_outliers(self):
        """Check 2: Outliers using IQR and Z-score"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == self.target_column:
                continue
                
            # IQR Method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            # Z-score Method
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers_zscore = (z_scores > 3).sum()
            
            if len(outliers_iqr) > 0 or outliers_zscore > 0:
                issue = {
                    'type': 'Outliers',
                    'column': col,
                    'iqr_count': len(outliers_iqr),
                    'zscore_count': int(outliers_zscore),
                    'severity': 'high' if len(outliers_iqr) > len(self.df) * 0.05 else 'medium'
                }
                self.issues.append(issue)
                
                suggestion = {
                    'type': 'Outliers',
                    'column': col,
                    'options': ['Remove outliers', 'Cap outliers (IQR method)', 'No action'],
                    'recommended': 'Cap outliers (IQR method)'
                }
                self.suggestions.append(suggestion)
    
    def detect_imbalanced_classes(self):
        """Check 3: Highly Imbalanced Classes"""
        class_dist = self.df[self.target_column].value_counts()
        class_ratio = class_dist.min() / class_dist.max()
        
        if class_ratio < 0.2:  # Less than 20% representation
            issue = {
                'type': 'Class Imbalance',
                'class_distribution': class_dist.to_dict(),
                'imbalance_ratio': round(class_ratio, 3),
                'severity': 'high'
            }
            self.issues.append(issue)
            
            suggestion = {
                'type': 'Class Imbalance',
                'options': ['Use class_weight in models', 'SMOTE oversampling', 
                           'Random undersampling', 'Stratified split'],
                'recommended': 'Use class_weight="balanced" in models'
            }
            self.suggestions.append(suggestion)
    
    def detect_high_cardinality(self):
        """Check 4: High Cardinality Categorical Features"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count > 20:
                issue = {
                    'type': 'High Cardinality',
                    'column': col,
                    'unique_values': unique_count,
                    'severity': 'medium'
                }
                self.issues.append(issue)
                
                suggestion = {
                    'type': 'High Cardinality',
                    'column': col,
                    'options': ['Keep top categories', 'Group rare categories', 'Ordinal encoding'],
                    'recommended': 'Group rare categories (frequency < 1%)'
                }
                self.suggestions.append(suggestion)
    
    def detect_constant_features(self):
        """Check 5: Constant or Near-Constant Features"""
        for col in self.df.columns:
            if col == self.target_column:
                continue
            
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < 0.05:  # Less than 5% unique values
                issue = {
                    'type': 'Constant/Near-Constant Feature',
                    'column': col,
                    'unique_ratio': round(unique_ratio, 3),
                    'severity': 'high'
                }
                self.issues.append(issue)
                
                suggestion = {
                    'type': 'Constant/Near-Constant Feature',
                    'column': col,
                    'options': ['Drop feature', 'Keep'],
                    'recommended': 'Drop feature'
                }
                self.suggestions.append(suggestion)
    
    def _recommend_imputation_strategy(self, col):
        """Helper to recommend imputation based on column type"""
        if self.df[col].dtype in ['int64', 'float64']:
            return 'median (for numeric columns)'
        else:
            return 'mode (for categorical columns)'