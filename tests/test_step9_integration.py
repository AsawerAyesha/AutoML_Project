"""
STEP 9 Integration Test - Validate app.py enhancements
Tests: test_size propagation, feature type display, preprocessing log, metrics dict handling
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_eda_with_test_size():
    """Test EDA generator with dynamic test_size parameter"""
    print("\n✓ TEST 1: EDA with Dynamic test_size")
    from src.components.eda_generator import EDAGenerator
    
    df = pd.read_csv('artifacts/data.csv')
    
    # Test different test_sizes
    for test_size in [0.15, 0.2, 0.3]:
        eda_gen = EDAGenerator(df, target_column='target', test_size=test_size)
        summary, fig = eda_gen.generate_train_test_split_summary()
        
        # Verify test ratio matches
        actual_ratio = summary['Test Samples'] / len(df)
        print(f"  - test_size={test_size}: Train={summary['Train Samples']}, Test={summary['Test Samples']}, Ratio={actual_ratio:.2%}")
        assert abs(actual_ratio - test_size) < 0.01, f"Test size mismatch: expected {test_size}, got {actual_ratio}"
    
    print("  PASS: test_size correctly propagates to EDA")

def test_feature_type_inference():
    """Test feature type inference for UI display"""
    print("\n✓ TEST 2: Feature Type Inference")
    from src.utils.feature_type_inference import FeatureTypeInference
    
    df = pd.read_csv('artifacts/data.csv')
    
    ft_infer = FeatureTypeInference(df, target_column='target')
    feature_types = ft_infer.infer_types()
    
    print(f"  - Inferred {len(feature_types)} feature types")
    for col, info in list(feature_types.items())[:3]:
        print(f"    • {col}: {info['type']}")
    
    assert len(feature_types) > 0, "No feature types inferred"
    assert all('type' in info for info in feature_types.values()), "Missing 'type' key in feature info"
    print("   PASS: Feature types inferred successfully")

def test_data_transformation_with_user_decisions():
    """Test data transformation with user decisions and test_size"""
    print("\n✓ TEST 3: Data Transformation with User Decisions & test_size")
    from src.components.data_transformation import DataTransformation
    from src.components.issue_detection import IssueDetector
    
    df = pd.read_csv('artifacts/data.csv')
    target_col = 'target'
    
    # Simulate issue detection
    detector = IssueDetector(df, target_col)
    issues, suggestions = detector.detect_all_issues()
    
    # Create user decisions
    user_decisions = {}
    for i, issue in enumerate(issues[:3]):
        key = f"{issue['type']}_{i}_{issue.get('column', 'global')}"
        user_decisions[key] = "remove"  # Example decision
    
    print(f"  - {len(issues)} issues detected, {len(user_decisions)} user decisions created")
    
    # Test with different test_sizes
    for test_size in [0.2, 0.25]:
        transformer = DataTransformation()
        result = transformer.initiate_data_transformation(
            df=df,
            target_column_name=target_col,
            issues=issues,
            user_decisions=user_decisions,
            test_size=test_size,
            random_state=42
        )
        
        train_arr, test_arr, preproc_path, preprocessing_log, class_weights = result
        
        # Verify array shapes
        total_samples = len(train_arr) + len(test_arr)
        actual_test_ratio = len(test_arr) / total_samples
        
        print(f"  - test_size={test_size}: Train={len(train_arr)}, Test={len(test_arr)}, Ratio={actual_test_ratio:.2%}")
        print(f"    Preprocessing steps: {len(preprocessing_log)}")
        
        assert len(preprocessing_log) > 0, "No preprocessing log generated"
        assert abs(actual_test_ratio - test_size) < 0.02, f"Test ratio mismatch: expected {test_size}, got {actual_test_ratio}"
    
    print("   PASS: Data transformation with user decisions & test_size working")

def test_metrics_dict_structure():
    """Test that metrics dict is properly structured for dashboard"""
    print("\n✓ TEST 4: Metrics Dict Structure")
    from src.components.model_trainer import ModelTrainer
    from src.components.data_transformation import DataTransformation
    
    df = pd.read_csv('artifacts/data.csv')
    target_col = 'target'
    test_size = 0.2
    
    # Transform data
    transformer = DataTransformation()
    result = transformer.initiate_data_transformation(
        df=df,
        target_column_name=target_col,
        issues=[],
        user_decisions={},
        test_size=test_size,
        random_state=42
    )
    
    train_arr, test_arr, _, _, class_weights = result
    
    # Train models
    trainer = ModelTrainer()
    report_dict, best_model_name, model_path = trainer.initiate_model_trainer(
        train_arr,
        test_arr,
        'classification',
        search_type='random',
        class_weights=class_weights
    )
    
    print(f"  - {len(report_dict)} models trained")
    print(f"  - Best model: {best_model_name}")
    
    # Validate metrics dict structure
    for model_name, metrics in list(report_dict.items())[:2]:
        print(f"  - {model_name}:")
        assert isinstance(metrics, dict), f"Metrics for {model_name} is not a dict"
        
        # Check required keys
        required_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
        for key in required_keys:
            assert key in metrics, f"Missing key '{key}' in metrics for {model_name}"
            print(f"    • {key}: {type(metrics[key]).__name__}")
    
    print("  PASS: Metrics dict structure valid for dashboard")

def test_preprocessing_log_structure():
    """Test preprocessing log contains actionable information"""
    print("\n✓ TEST 5: Preprocessing Log Structure")
    from src.components.data_transformation import DataTransformation
    
    df = pd.read_csv('artifacts/data.csv')
    target_col = 'target'
    
    transformer = DataTransformation(
        imputation_strategy='mean',
        scaling_strategy='StandardScaler',
        encoding_strategy='OneHotEncoder'
    )
    
    result = transformer.initiate_data_transformation(
        df=df,
        target_column_name=target_col,
        issues=[],
        user_decisions={},
        test_size=0.2,
        random_state=42
    )
    
    train_arr, test_arr, preproc_path, preprocessing_log, class_weights = result
    
    print(f"  - {len(preprocessing_log)} preprocessing steps logged")
    for i, step in enumerate(preprocessing_log[:3], 1):
        print(f"    Step {i}: {step.get('action', 'unknown')}")
        assert 'action' in step, "Missing 'action' in preprocessing log entry"
        assert isinstance(step, dict), "Preprocessing log entry is not a dict"
    
    print("  PASS: Preprocessing log properly structured for UI display")

def main():
    print("\n" + "="*70)
    print("STEP 9 INTEGRATION TEST SUITE")
    print("="*70)
    
    try:
        test_eda_with_test_size()
        test_feature_type_inference()
        test_data_transformation_with_user_decisions()
        test_metrics_dict_structure()
        test_preprocessing_log_structure()
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED - STEP 9 INTEGRATION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
