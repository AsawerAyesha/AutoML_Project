"""Test STEP 3: Preprocessing Applicator"""
import pandas as pd
from src.utils.feature_type_inference import FeatureTypeInference
from src.utils.preprocessing_applicator import PreprocessingApplicator

# Load heart.csv
df = pd.read_csv('datasets/heart.csv')
print(f"Loaded dataset: {df.shape}")

# Initialize feature type inference
inference = FeatureTypeInference(df, target_column='target')
feature_types = inference.infer_types()

# Initialize preprocessing applicator
applicator = PreprocessingApplicator(df, feature_types, target_column='target')

print(f"\n✅ STEP 3 Test: Preprocessing Applicator")
print("="*80)

# Test 1: Outlier capping (non-destructive)
print("\nTest 1: OUTLIER CAPPING")
print("-"*80)

outlier_config = {
    'chol': {
        'action': 'cap (IQR)',
        'bounds': {'lower': 130.0, 'upper': 310.0}
    },
    'trestbps': {
        'action': 'cap (IQR)',
        'bounds': {'lower': 94.0, 'upper': 200.0}
    },
    'thalach': {
        'action': 'no action'
    }
}

result = applicator.apply_outlier_action(outlier_config)
print(f"Applied outlier capping to {len(result['columns_affected'])} columns")
for col in result['columns_affected']:
    print(f"  - {col}: {result['details'][col]}")

print(f"Rows affected: {len(applicator.get_processed_dataframe())} (no rows removed by capping)")

# Test 2: Constant feature removal
print("\n\nTest 2: CONSTANT FEATURE REMOVAL")
print("-"*80)

# For this test, create a mock constant feature
test_df = applicator.get_processed_dataframe().copy()
test_df['mock_constant'] = 1  # All same value

applicator2 = PreprocessingApplicator(test_df, feature_types, target_column='target')

constant_config = {
    'mock_constant': 'drop feature'
}

result = applicator2.apply_constant_feature_removal(constant_config)
print(f"Removed {len(result['columns_affected'])} constant features: {result['columns_affected']}")
print(f"Columns before: {len(test_df.columns)} → after: {len(applicator2.get_processed_dataframe().columns)}")

# Test 3: Missing value imputation
print("\n\nTest 3: MISSING VALUE IMPUTATION")
print("-"*80)

# Create a new dataframe with some missing values
test_df = applicator.get_processed_dataframe().copy()
test_df.loc[0:5, 'age'] = None  # Add missing values
test_df.loc[0:3, 'chol'] = None

applicator3 = PreprocessingApplicator(test_df, feature_types, target_column='target')

imputation_config = {
    'age': 'median',
    'chol': 'mean'
}

result = applicator3.apply_missing_value_imputation(imputation_config)
print(f"Applied imputation to {len(result['columns_affected'])} columns")
for col in result['columns_affected']:
    print(f"  - {col}: {result['details'][col]['action']} "
          f"({result['details'][col]['values_imputed']} values imputed)")

# Test 4: Class imbalance handling - class_weights
print("\n\nTest 4: CLASS IMBALANCE HANDLING (class_weights)")
print("-"*80)

result = applicator.apply_class_imbalance_handling('class_weights', target_col='target')
print(f"Method: {result['details']['method']}")
print(f"Note: {result['details']['note']}")
print(f"Current class distribution: {result['details']['class_distribution']}")

# Test 5: Summary
print("\n\nTest 5: PREPROCESSING SUMMARY")
print("-"*80)

summary = applicator.get_summary()
print(f"Rows: {summary['rows_original']} → {summary['rows_final']} ({summary['rows_removed']} removed)")
print(f"Columns: {summary['columns_original']} → {summary['columns_final']} ({summary['columns_removed']} removed)")
print(f"Preprocessing steps applied: {summary['preprocessing_steps']}")

print("\n" + "="*80)
print("STEP 3 TEST COMPLETE")
print("="*80)
