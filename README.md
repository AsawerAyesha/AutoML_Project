# AutoML Classification System

A comprehensive automated machine learning system that makes building classification models accessible to everyone. Upload your data, and let the system handle exploratory analysis, preprocessing, model training, and evaluation for you.

## What This Does

This application takes the complexity out of machine learning workflows. Instead of writing hundreds of lines of code, you can upload a CSV file and get a complete analysis along with trained models in minutes. The system automatically detects data quality issues, suggests fixes, trains multiple algorithms, and tells you which one works best for your data.

Think of it as having a data scientist assistant that walks you through the entire process, explains what it finds, and asks for your input when decisions need to be made.

## Key Features

**Smart Data Analysis**
- Automatically analyzes your dataset and shows you what's inside
- Detects missing values, outliers, and imbalanced classes
- Generates correlation matrices and distribution plots
- Identifies problematic features that might hurt model performance

**Interactive Preprocessing**
- The system flags data quality issues and suggests fixes
- You decide which preprocessing steps to apply
- Handles missing values with multiple imputation strategies
- Deals with outliers through removal or capping
- Applies proper scaling and encoding for machine learning

**Multiple Algorithm Training**
- Trains seven different classification algorithms simultaneously
- Performs hyperparameter optimization automatically
- Compares models using accuracy, precision, recall, and F1-score
- Shows confusion matrices and ROC curves for evaluation
- Recommends the best model based on performance metrics

**Comprehensive Reporting**
- Downloads a complete HTML report with all findings
- Includes preprocessing decisions and model configurations
- Shows detailed performance comparisons
- Exports metrics as CSV for further analysis

## Supported Algorithms

The system trains and compares these classifiers:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Naive Bayes
5. Random Forest
6. Support Vector Machine
7. Rule-based Classifier (OneR)

Each model goes through hyperparameter tuning using Grid Search or Randomized Search to find the best configuration for your data.

## How to Run Locally

**Prerequisites**
- Python 3.11.1 (pinned in `runtime.txt`; 3.10+ also works)
- pip package manager

**Installation Steps**

1. Clone this repository or download the files:
```bash
git clone <your-repo-url>
cd AutoML_Project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Mac/Linux:
```bash
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run app.py
```

6. Open your browser and go to `http://localhost:8501`

**Quick Start with Sample Data**

To test the application immediately, use one of the sample datasets provided in the `datasets/` folder:
- `heart.csv` - Heart disease classification (1,025 patients, 13 features)
- `Churn_Modelling.csv` - Bank customer churn prediction (10,000 customers, 13 features)

Simply upload one of these files in Step 1 of the application.

## How to Use

**Step 1: Upload Your Data**
Click the upload button and select a CSV file with your classification dataset. The system will display basic information about your data including the number of rows, columns, and class distribution.

**Step 2: Explore Your Data**
Review the automated exploratory data analysis. Check the tabs for missing values, outliers, correlations, distributions, and more. This helps you understand what you're working with.

**Step 3: Select Target Variable**
Choose which column contains the values you want to predict. The system will automatically infer the type of each feature in your dataset.

**Step 4: Review Data Issues**
The system identifies potential problems in your data. Read each issue and decide whether to apply the suggested fix. You have full control over what preprocessing happens.

**Step 5: Configure Preprocessing**
Set your preferences for handling missing values, outliers, scaling, and encoding. Choose your train-test split ratio (default is 80/20).

**Step 6: Train Models**
Click the button to start training. The system will preprocess your data, train all seven algorithms, optimize their hyperparameters, and evaluate their performance. This might take a minute or two depending on your dataset size.

**Step 7: Compare Results**
Review the model comparison dashboard. See which algorithm performed best, examine confusion matrices, and check ROC curves. Download the metrics as CSV if you want to analyze them further.

**Step 8: Download Report**
Generate and download a comprehensive HTML report that documents everything: your data, the issues found, preprocessing decisions, model configurations, and performance results.

## Example Datasets

Sample datasets are included in the `datasets/` folder for testing:

**1. heart.csv**
- 1,025 patient records
- 13 features (age, blood pressure, cholesterol, resting ECG, etc.)
- Binary classification: heart disease present or not
- Good for testing binary classification workflows

**2. Churn_Modelling.csv**
- 10,000 customer records
- 13 features (credit score, geography, age, balance, etc.)
- Binary classification: customer churn prediction
- Good for testing class imbalance handling

Upload either dataset in the app to see the AutoML pipeline in action!

## Project Structure

```
AutoML_Project/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── datasets/                   # Sample datasets for testing
│   ├── heart.csv              # Heart disease dataset
│   └── Churn_Modelling.csv    # Bank churn dataset
├── artifacts/                  # Generated files (models, reports)
├── src/
│   ├── components/            # Core pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── eda_generator.py
│   │   ├── issue_detection.py
│   │   └── model_trainer.py
│   ├── pipeline/              # Pipeline orchestration
│   │   └── report_generator.py
│   └── utils/                 # Helper functions
│       ├── feature_type_inference.py
│       ├── preprocessing_applicator.py
│       └── metrics_utils.py
└── tests/                     # Test suite
```

## Technical Details

**Machine Learning Pipeline**
The system follows a structured workflow: feature type inference, issue detection, preprocessing, train-test split, model training with hyperparameter tuning, evaluation, and reporting. Each step is modular and tested independently.

**Preprocessing Strategy**
All preprocessing happens before splitting the data to avoid leakage. The system logs every transformation applied so you can see exactly what happened to your data.

**Model Selection Criteria**
The best model is selected based on F1-score, which balances precision and recall. This is particularly important for imbalanced datasets where accuracy alone can be misleading.

**Hyperparameter Optimization**
Adaptive hyperparameter search based on dataset size. For large datasets (>20,000 rows), the system uses default parameters to ensure fast training. For smaller datasets, it performs hyperparameter tuning using Grid Search or Randomized Search for optimal performance.

## Live Application

🚀 **Hosted App:** [Link will be added after deployment]

The application is deployed on Streamlit Cloud for easy access without local installation.

## Requirements

All dependencies are listed in `requirements.txt`:
- pandas - data manipulation
- numpy - numerical operations
- scikit-learn - machine learning algorithms
- streamlit - web application framework
- plotly - interactive visualizations
- ydata-profiling - automated EDA reports
- imbalanced-learn - handling class imbalance
- mlxtend - additional ML utilities

## Testing

The project includes a comprehensive test suite covering all components:

```bash
# Run integration tests
python test_step9_integration.py
python test_report_generator.py

# Run individual component tests
python test_feature_inference.py
python test_issue_detection.py
python test_data_transformation.py
python test_model_trainer.py
```

All tests validate that the system works correctly across different datasets and scenarios.

## Limitations

- Currently supports only classification tasks (not regression)
- Works best with tabular data in CSV format
- Requires all data to fit in memory
- Some algorithms may be slow on very large datasets (>100,000 rows)
- Hyperparameter search space is predefined and not customizable through the UI

## Future Improvements

- Add support for regression problems
- Include more advanced algorithms (XGBoost, LightGBM, CatBoost)
- Implement feature selection and engineering
- Add model interpretability tools (SHAP values, feature importance)
- Support for time series data
- Export models in production-ready formats

## License

This project is for educational purposes as part of the CS-245 Machine Learning course.

## Contact

For questions or issues, please contact
Alisha Siddiqui - 464647 
Asawer Ayesha - 470860
