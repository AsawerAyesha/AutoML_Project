# src/utils/constants.py

# Model names - Must match requirement E exactly
CLASSIFICATION_MODELS = {
    'Logistic Regression': 'LogisticRegression',
    'K-Nearest Neighbors': 'KNeighborsClassifier',
    'Decision Tree': 'DecisionTreeClassifier',
    'Naive Bayes': 'GaussianNB',
    'Random Forest': 'RandomForestClassifier',
    'Support Vector Machine': 'SVC',
    'Rule-based Classifier': 'DecisionTreeClassifier'  # Can be replaced with custom rule engine
}

# Evaluation metrics for classification
CLASSIFICATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
]

# Preprocessing strategies
IMPUTATION_STRATEGIES = ['mean', 'median', 'mode', 'constant']
OUTLIER_STRATEGIES = ['removal', 'capping', 'no_action']
SCALING_STRATEGIES = ['StandardScaler', 'MinMaxScaler']
ENCODING_STRATEGIES = ['OneHotEncoder', 'OrdinalEncoder']

# Issue thresholds
MISSING_VALUE_THRESHOLD = 0.5  # 50%
CARDINALITY_THRESHOLD = 20  # High cardinality if > 20 unique values
IMBALANCE_RATIO_THRESHOLD = 0.2  # Flag if minority class < 20% of majority
CONSTANT_FEATURE_THRESHOLD = 0.95  # Near-constant if same value > 95%