"""
STEP 10 TEST: Report Generator with Preprocessing Log, Feature Types, and Full Metrics
Tests the complete HTML report generation including all new enhancements.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.report_generator import ReportGenerator

def test_report_generator_with_preprocessing_log():
    """Test report generator includes preprocessing log"""
    print("\n" + "="*70)
    print("TEST 1: Report Generator with Preprocessing Log")
    print("="*70)
    
    # Sample dataset
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'income': [50000, 60000, 70000, 80000, 90000, 100000],
        'has_loan': [0, 1, 0, 1, 0, 1],
        'target': [0, 1, 0, 1, 0, 1]
    })
    
    # Sample preprocessing log
    preprocessing_log = [
        {'action': 'missing_value_handling', 'description': 'Imputed 5 missing values in age column'},
        {'action': 'outlier_handling', 'description': 'Removed 2 outliers from income column'},
        {'action': 'class_imbalance_handling', 'description': 'Applied class weights to handle imbalance'}
    ]
    
    # Sample model results
    model_results = {
        'Random Forest': {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.96,
            'F1-Score': 0.95,
            'ROC-AUC': 0.98,
            'training_time': 0.15,
            'confusion_matrix': np.array([[45, 5], [3, 47]]),
            'model': RandomForestClassifier()
        }
    }
    
    best_model = pd.Series({
        'Model': 'Random Forest',
        'Accuracy': 0.95,
        'Precision': 0.94,
        'Recall': 0.96,
        'F1-Score': 0.95,
        'ROC-AUC': 0.98,
        'Training Time (s)': 0.15
    })
    
    # Generate report
    report_gen = ReportGenerator(
        dataset=data,
        target_column='target',
        issues=[],
        user_decisions={},
        preprocessing_config={'strategy': 'median'},
        model_results=model_results,
        best_model=best_model,
        preprocessing_log=preprocessing_log
    )
    
    html = report_gen.generate_html_report()
    
    # Assertions
    assert 'Preprocessing Decisions Log' in html, "Preprocessing log section missing"
    assert 'missing_value_handling' in html, "Missing value action not in report"
    assert 'outlier_handling' in html, "Outlier handling action not in report"
    assert 'class_imbalance_handling' in html, "Class imbalance action not in report"
    assert 'Imputed 5 missing values' in html, "Description not in report"
    
    print(" Preprocessing log properly integrated into report")


def test_report_generator_with_feature_types():
    """Test report generator includes feature type analysis"""
    print("\n" + "="*70)
    print("TEST 2: Report Generator with Feature Types")
    print("="*70)
    
    # Sample dataset
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'income': [50000, 60000, 70000, 80000, 90000, 100000],
        'has_loan': [0, 1, 0, 1, 0, 1],
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'target': [0, 1, 0, 1, 0, 1]
    })
    
    # Feature types
    feature_types = {
        'age': 'continuous_numeric',
        'income': 'continuous_numeric',
        'has_loan': 'binary',
        'category': 'categorical_encoded'
    }
    
    # Sample model results
    model_results = {
        'Decision Tree': {
            'accuracy': 0.88,
            'precision': 0.87,
            'recall': 0.89,
            'F1-Score': 0.88,
            'ROC-AUC': 0.91,
            'training_time': 0.05,
            'confusion_matrix': np.array([[42, 8], [6, 44]]),
            'model': DecisionTreeClassifier()
        }
    }
    
    best_model = pd.Series({
        'Model': 'Decision Tree',
        'Accuracy': 0.88,
        'F1-Score': 0.88
    })
    
    # Generate report
    report_gen = ReportGenerator(
        dataset=data,
        target_column='target',
        issues=[],
        user_decisions={},
        preprocessing_config={'strategy': 'mean'},
        model_results=model_results,
        best_model=best_model,
        feature_types=feature_types
    )
    
    html = report_gen.generate_html_report()
    
    # Assertions
    assert 'Feature Type Analysis' in html, "Feature type section missing"
    assert 'continuous_numeric' in html, "Continuous numeric type not in report"
    assert 'binary' in html, "Binary type not in report"
    assert 'categorical_encoded' in html, "Categorical type not in report"
    assert 'age' in html and 'income' in html, "Feature names missing"
    
    print(" Feature types properly displayed in report")


def test_report_generator_with_all_metrics():
    """Test report generator displays all metrics correctly"""
    print("\n" + "="*70)
    print("TEST 3: Report Generator with All Metrics")
    print("="*70)
    
    # Sample dataset
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 2, 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Comprehensive model results
    model_results = {
        'Random Forest': {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.96,
            'F1-Score': 0.95,
            'ROC-AUC': 0.98,
            'training_time': 0.15,
            'confusion_matrix': np.array([[45, 5], [3, 47]]),
            'model': RandomForestClassifier()
        },
        'Decision Tree': {
            'accuracy': 0.88,
            'precision': 0.87,
            'recall': 0.89,
            'F1-Score': 0.88,
            'ROC-AUC': 0.91,
            'training_time': 0.05,
            'confusion_matrix': np.array([[42, 8], [6, 44]]),
            'model': DecisionTreeClassifier()
        }
    }
    
    best_model = pd.Series({
        'Model': 'Random Forest',
        'Accuracy': 0.95,
        'Precision': 0.94,
        'Recall': 0.96,
        'F1-Score': 0.95,
        'ROC-AUC': 0.98,
        'Training Time (s)': 0.15
    })
    
    # Generate report
    report_gen = ReportGenerator(
        dataset=data,
        target_column='target',
        issues=[],
        user_decisions={},
        preprocessing_config={},
        model_results=model_results,
        best_model=best_model
    )
    
    html = report_gen.generate_html_report()
    
    # Assertions for metrics display
    assert 'Accuracy' in html, "Accuracy metric missing"
    assert 'Precision' in html, "Precision metric missing"
    assert 'Recall' in html, "Recall metric missing"
    assert 'F1-Score' in html, "F1-Score metric missing"
    assert 'ROC-AUC' in html, "ROC-AUC metric missing"
    assert 'Training Time' in html, "Training time missing"
    
    # Check both models are in report
    assert 'Random Forest' in html, "Random Forest not in report"
    assert 'Decision Tree' in html, "Decision Tree not in report"
    
    # Check confusion matrices
    assert 'Confusion Matrices by Model' in html, "Confusion matrix section missing"
    assert '45' in html and '47' in html, "Confusion matrix values not displayed"
    
    print("All metrics properly displayed in report")


def test_report_generator_section_numbering():
    """Test that report sections are properly numbered"""
    print("\n" + "="*70)
    print("TEST 4: Report Section Numbering")
    print("="*70)
    
    data = pd.DataFrame({
        'x': [1, 2, 3],
        'target': [0, 1, 0]
    })
    
    model_results = {
        'Random Forest': {
            'accuracy': 0.90,
            'F1-Score': 0.89,
            'confusion_matrix': np.array([[1, 0], [0, 2]]),
            'model': RandomForestClassifier()
        }
    }
    
    best_model = pd.Series({'Model': 'Random Forest', 'F1-Score': 0.89})
    
    preprocessing_log = [{'action': 'test', 'description': 'Test action'}]
    feature_types = {'x': 'continuous_numeric'}
    
    report_gen = ReportGenerator(
        dataset=data,
        target_column='target',
        issues=[],
        user_decisions={},
        preprocessing_config={},
        model_results=model_results,
        best_model=best_model,
        preprocessing_log=preprocessing_log,
        feature_types=feature_types
    )
    
    html = report_gen.generate_html_report()
    
    # Check section numbering
    assert '<h2>2. Exploratory Data Analysis' in html, "Section 2 wrong"
    assert '<h2>3. Feature Type Analysis' in html, "Section 3 wrong"
    assert '<h2>4. Data Quality Issues' in html, "Section 4 wrong"
    assert '<h2>5. Preprocessing Configuration' in html, "Section 5 wrong"
    assert '<h2>6. Preprocessing Decisions Log' in html, "Section 6 wrong"
    assert '<h2>7. Model Configurations' in html, "Section 7 wrong"
    assert '<h2>8. Model Performance Comparison' in html, "Section 8 wrong"
    assert '<h2>9. Confusion Matrices by Model' in html, "Section 9 wrong"
    assert '<h2>10.  Best Model Selected' in html, "Section 10 wrong"
    
    print(" All sections properly numbered from 1-10")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" STEP 10: REPORT GENERATOR COMPREHENSIVE TEST")
    print("="*70)
    
    test_report_generator_with_preprocessing_log()
    test_report_generator_with_feature_types()
    test_report_generator_with_all_metrics()
    test_report_generator_section_numbering()
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED - STEP 10 COMPLETE")
    print("="*70)
    print("\n Report Generator successfully enhanced with:")
    print("  - Preprocessing decisions log")
    print("  - Feature type analysis section")
    print("  - Complete metrics display (Accuracy, Precision, Recall, F1, ROC-AUC, Time)")
    print("  - Confusion matrices for all models")
    print("  - Proper section numbering (1-10)")
    print("\n 10-STEP AUTOML PIPELINE COMPLETE!")
