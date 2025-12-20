# test_feature_inference.py (in project root)
import pandas as pd
from src.utils.feature_type_inference import FeatureTypeInference

# Load heart.csv
df = pd.read_csv('datasets/heart.csv')

# Initialize and run inference
inference = FeatureTypeInference(df, target_column='target')
feature_types = inference.infer_types()

# Display summary
inference.display_summary()

# Validate heart.csv specific columns
print("\nVALIDATION CHECKS FOR heart.csv:")
print("=" * 70)

# Expected results
expected = {
    'sex': 'binary',           # Should be binary (0 or 1)
    'exang': 'binary',         # Should be binary (0 or 1)
    'fbs': 'binary',           # Should be binary (0 or 1)
    'cp': 'categorical_encoded',    # Should be categorical (0-3)
    'slope': 'categorical_encoded', # Should be categorical (0-2)
    'ca': 'categorical_encoded',    # Should be categorical (0-4)
    'thal': 'categorical_encoded',  # Should be categorical (1-3)
    'age': 'continuous_numeric',    # Should be continuous
    'chol': 'continuous_numeric',   # Should be continuous
    'trestbps': 'continuous_numeric',  # Should be continuous
    'thalach': 'continuous_numeric',   # Should be continuous
    'oldpeak': 'continuous_numeric',   # Should be continuous
}

all_passed = True
for col, expected_type in expected.items():
    actual_type = feature_types[col]['type']
    status = "PASS" if actual_type == expected_type else "FAIL"
    print(f"{status} | {col:15s} | Expected: {expected_type:20s} | Got: {actual_type}")
    if actual_type != expected_type:
        all_passed = False

print("=" * 70)
if all_passed:
    print("ALL TESTS PASSED! Feature inference is working correctly.")
else:
    print("SOME TESTS FAILED! Check the implementation.")