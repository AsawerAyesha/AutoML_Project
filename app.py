import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
from pathlib import Path

# Import custom modules
from src.components.data_ingestion import DataIngestion
from src.components.issue_detection import IssueDetector
from src.components.eda_generator import EDAGenerator
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.report_generator import ReportGenerator
from src.logger import logging
from src.exception import CustomException

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="AutoML Pro | Enterprise Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Change Progress Bar Color to match Primary Theme */
    .stProgress > div > div > div > div {
        background-color: #2C3E50;
    }
    .metric-card {
        border-top: 4px solid #2C3E50;
        border-right: 1px solid #e0e0e0;
        border-bottom: 1px solid #e0e0e0;
        border-left: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    default_states = {
        'current_page': 'page_upload',
        'df_uploaded': False,
        'df': None,
        'train_path': None,
        'test_path': None,
        'eda_report_path': None,
        'issues': [],
        'suggestions': [],
        'user_decisions': {},
        'preprocessing_config': {},
        'train_array': None,
        'test_array': None,
        'model_results': None,
        'best_model': None,
        'report_generated': False
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_progress_bar(active_step_index):
    """
    Renders a visual progress bar at the top of the page.
    0=Upload, 1=EDA, 2=Issues, 3=Config, 4=Train, 5=Compare, 6=Report
    """
    steps = ["Upload", "EDA", "Quality", "Config", "Train", "Compare", "Report"]
    
    # Calculate progress (0.0 to 1.0)
    progress_val = (active_step_index + 1) / len(steps)
    st.progress(progress_val)
    
    # Breadcrumbs
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i == active_step_index:
                st.markdown(f"** {step}**")  # Active
            elif i < active_step_index:
                st.markdown(f" {step}")      # Completed
            else:
                st.markdown(f"<span style='color:grey'> {step}</span>", unsafe_allow_html=True) # Future
    st.divider()

# ============================================================================
# PAGE 1: DATASET UPLOAD & BASIC INFO
# ============================================================================
def page_upload_and_info():
    render_progress_bar(0)
    st.title(" Data Ingestion Hub")
    st.markdown("Upload your dataset to begin the automated classification pipeline.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.container(border=True):
            st.subheader("Upload File")
            uploaded_file = st.file_uploader("Drop CSV file here", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.df_uploaded = True
                    
                    # Save uploaded file
                    os.makedirs('artifacts', exist_ok=True)
                    df.to_csv('artifacts/data.csv', index=False)
                    
                    st.success(f" Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f" Error: {str(e)}")
                    return
    
    with col2:
        if st.session_state.df_uploaded and st.session_state.df is not None:
            df = st.session_state.df
            
            st.subheader("Dataset Snapshot")
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", len(df))
                m2.metric("Columns", len(df.columns))
                m3.metric("Numeric", len(df.select_dtypes(include=[np.number]).columns))
                m4.metric("Categorical", len(df.select_dtypes(include=['object']).columns))

            st.markdown("###### Data Preview")
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Target Selection
            st.subheader("Target Configuration")
            target_column = st.selectbox(
                "Select Target Variable (Label)",
                df.columns,
                index=len(df.columns)-1, # Default to last column often
                help="The column you want the model to predict."
            )
            
            if target_column:
                st.session_state.target_column = target_column
                
                # Mini distribution chart
                chart_data = df[target_column].value_counts().reset_index()
                chart_data.columns = ['Class', 'Count']
                st.bar_chart(chart_data, x='Class', y='Count', color='#2C3E50', height=200)

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("Proceed to Analysis ", type="primary", use_container_width=True):
                    st.session_state.current_page = "page_eda"
                    st.rerun()
        else:
            # Placeholder content
            st.info(" Please upload a CSV file to see dataset details.")
            st.markdown("""
            **Supported Format:** CSV
            \n**Requirements:**
            - Must have headers
            - Clean tabular data
            """)

# ============================================================================
# PAGE 2: AUTOMATED EXPLORATORY DATA ANALYSIS
# ============================================================================
def page_eda():
    render_progress_bar(1)
    st.title(" Exploratory Data Analysis")
    
    if not st.session_state.df_uploaded:
        st.error(" No data found. Please upload a dataset first.")
        return
    
    df = st.session_state.df
    target_col = st.session_state.get('target_column')
    
    # Get test_size from preprocessing config or default
    test_size = st.session_state.preprocessing_config.get('test_size', 0.2)
    
    # Initialize EDA generator with dynamic test_size
    eda_gen = EDAGenerator(df, target_column=target_col, test_size=test_size)
    
    # Modern Tab interface
    tab_names = [" Overview", " Missing Data", " Outliers", " Correlations", " Distributions", " Categorical", " Split Info"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]: # Overview
        stats = eda_gen.generate_basic_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame(stats['Column Types'].items(), columns=['Feature', 'Data Type']), use_container_width=True)
        with col2:
            st.write("### Statistical Summary")
            st.dataframe(eda_gen.generate_summary_statistics(), use_container_width=True)
    
    with tabs[1]: # Missing
        fig, missing_analysis = eda_gen.generate_missing_value_analysis()
        # Display global missing percent prominently
        st.metric("Global Missing Percentage", f"{missing_analysis['global_missing_percent']}%")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View Missing Data Details"):
                st.dataframe(pd.DataFrame(missing_analysis['per_column']), use_container_width=True)
        else:
            st.success(" This dataset is clean! No missing values detected.")
            
    with tabs[2]: # Outliers
        outlier_data, outlier_figs = eda_gen.generate_outlier_analysis()
        st.dataframe(pd.DataFrame(outlier_data).T, use_container_width=True)
        if outlier_figs:
            st.plotly_chart(outlier_figs[0], use_container_width=True) # Show first one as example
            if len(outlier_figs) > 1:
                with st.expander(f"View remaining {len(outlier_figs)-1} box plots"):
                    for i in range(1, len(outlier_figs)):
                        st.plotly_chart(outlier_figs[i], use_container_width=True)
                        
    with tabs[3]: # Correlations
        fig, corr_matrix = eda_gen.generate_correlation_matrix()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient numeric data for correlation analysis.")
            
    with tabs[4]: # Distributions
        dist_figs = eda_gen.generate_distribution_plots()
        if dist_figs:
            col1, col2 = st.columns(2)
            for i, fig in enumerate(dist_figs):
                with (col1 if i % 2 == 0 else col2):
                    st.plotly_chart(fig, use_container_width=True)
                    
    with tabs[5]: # Categorical
        cat_figs = eda_gen.generate_categorical_plots()
        if cat_figs:
            col1, col2 = st.columns(2)
            for i, fig in enumerate(cat_figs):
                with (col1 if i % 2 == 0 else col2):
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical features found.")
            
    with tabs[6]: # Split Info
        summary, fig = eda_gen.generate_train_test_split_summary()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Train Samples", summary['Train Samples'])
            st.metric("Test Samples", summary['Test Samples'])
            st.metric("Test Size", f"{summary['Test Ratio %']}%")
        with c2:
            st.plotly_chart(fig, use_container_width=True)

    # Navigation Footer
    st.divider()
    c1, _, c2 = st.columns([1, 4, 1])
    if c1.button(" Back"):
        st.session_state.current_page = "page_upload"
        st.rerun()
    if c2.button("Next: Quality Check ", type="primary"):
        st.session_state.current_page = "page_issues"
        st.rerun()

# ============================================================================
# PAGE 3: ISSUE DETECTION
# ============================================================================
def page_issue_detection():
    render_progress_bar(2)
    st.title(" Data Quality & Integrity Check")
    
    if not st.session_state.df_uploaded:
        st.error(" No data loaded.")
        return
        
    df = st.session_state.df
    target_col = st.session_state.get('target_column')
    
    # Infer feature types for UI display
    from src.utils.feature_type_inference import FeatureTypeInference
    ft_infer = FeatureTypeInference(df, target_column=target_col)
    feature_types = ft_infer.infer_types()
    st.session_state.feature_types = feature_types
    
    # Run Detection
    detector = IssueDetector(df, target_col)
    issues, suggestions = detector.detect_all_issues()
    st.session_state.issues = issues
    st.session_state.suggestions = suggestions
    
    if not issues:
        st.canvas_balloons()
        st.success(" Excellent! No significant data quality issues detected.")
    else:
        st.warning(f"Found {len(issues)} potential issues requiring attention.")
        
        # Group issues for cleaner UI
        issue_groups = {}
        for issue in issues:
            grp = issue['type']
            if grp not in issue_groups: issue_groups[grp] = []
            issue_groups[grp].append(issue)
            
        for grp_name, grp_issues in issue_groups.items():
            with st.expander(f" {grp_name} ({len(grp_issues)})", expanded=True):
                for i, issue in enumerate(grp_issues):
                    col_issue, col_ftype, col_fix = st.columns([2, 1, 1])
                    with col_issue:
                        st.markdown(f"**Column:** `{issue.get('column', 'Global')}`")
                        st.text(f"Details: {issue}")
                    
                    with col_ftype:
                        # Display feature type from inference
                        col_name = issue.get('column')
                        if col_name and col_name in feature_types:
                            ftype = feature_types[col_name].get('type', 'unknown')
                            st.caption(f"Type: **{ftype}**")
                    
                    with col_fix:
                        # Find suggestion
                        issue_id = issue.get("issue_id")

                        sug = next((s for s in suggestions if s.get("issue_id") == issue_id), None)
                        if sug:
                            options = sug["options"]
                            recommended = sug.get("recommended", options[0])

                            # ✅ decision_key MUST equal issue_id so DataTransformation can find it
                            decision_key = issue_id

                            prev_decision = st.session_state.user_decisions.get(decision_key, recommended)

                            choice = st.selectbox(
                                "Action",
                                options,
                                key=f"sel_{decision_key}",
                                index=options.index(prev_decision) if prev_decision in options else 0
                            )
                            st.session_state.user_decisions[decision_key] = choice
                    st.divider()

    # Navigation Footer
    c1, _, c2 = st.columns([1, 4, 1])
    if c1.button(" Back"):
        st.session_state.current_page = "page_eda"
        st.rerun()
    if c2.button("Next: Configure ", type="primary"):
        st.session_state.current_page = "page_preprocessing"
        st.rerun()

# ============================================================================
# PAGE 4: PREPROCESSING CONFIG
# ============================================================================
def page_preprocessing_config():
    render_progress_bar(3)
    st.title(" Pipeline Configuration")
    st.markdown("Customize how the AutoML pipeline handles data transformation.")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Cleaning Strategy")
            st.session_state.preprocessing_config['missing_strategy'] = st.selectbox(
                "Missing Values Imputation",
                ["median", "mean", "mode", "constant"],
                help="Median is robust to outliers."
            )
            # Ask for constant value only if 'constant' is selected
            if st.session_state.preprocessing_config['missing_strategy'] == "constant":
                const_val = st.text_input(
                    "Constant value to fill missing values",
                    value="Unknown",
                    help="Used to replace missing values (especially useful for categorical/text columns)."
                )
                st.session_state.preprocessing_config['missing_constant_value'] = const_val

            st.session_state.preprocessing_config['outlier_strategy'] = st.selectbox(
                "Outlier Handling",
                ["capping", "removal", "no_action"],
                help="Capping limits extreme values to the IQR bounds."
            )
            
        with col2:
            st.subheader(" Feature Engineering")
            st.session_state.preprocessing_config['scaling_strategy'] = st.selectbox(
                "Feature Scaling",
                ["StandardScaler", "MinMaxScaler"],
                help="StandardScaler normalizes to mean=0, std=1."
            )
            st.session_state.preprocessing_config['encoding_strategy'] = st.selectbox(
                "Categorical Encoding",
                ["OneHotEncoder", "OrdinalEncoder"],
                help="OneHot creates binary columns. Ordinal uses integer mapping."
            )

        st.divider()
        st.subheader(" Validation Split")
        split_val = st.slider("Test Set Percentage", 10, 40, 20, 5)
        st.session_state.preprocessing_config['test_size'] = split_val / 100
        st.caption(f"Training: {100-split_val}% | Testing: {split_val}%")

    # Navigation Footer
    c1, _, c2 = st.columns([1, 4, 1])
    if c1.button(" Back"):
        st.session_state.current_page = "page_issues"
        st.rerun()
    if c2.button("Start Pipeline ", type="primary"):
        with st.status(" Initializing Pipeline...", expanded=True) as status:
            try:
                st.write(" Ingesting Data...")
                data_ingestor = DataIngestion()
                # Pass test_size from config
                test_size = st.session_state.preprocessing_config.get('test_size', 0.2)
                st.write("DEBUG user_decisions keys:", list(st.session_state.user_decisions.keys()))
                train_path, test_path, eda_path = data_ingestor.initiate_data_ingestion(
                    st.session_state.df, test_size=test_size
                )
                
                st.session_state.train_path = train_path
                st.session_state.test_path = test_path
                st.session_state.eda_report_path = eda_path
                
                st.write(" Applying Transformations...")
                data_transformer = DataTransformation(
                    imputation_strategy=st.session_state.preprocessing_config.get('missing_strategy', 'median'),
                    scaling_strategy=st.session_state.preprocessing_config.get('scaling_strategy', 'StandardScaler'),
                    encoding_strategy=st.session_state.preprocessing_config.get('encoding_strategy', 'OneHotEncoder')
                )
                
                # Pass issues, user_decisions, and test_size to transformation
                train_arr, test_arr, preproc_path, preprocessing_log, class_weights = data_transformer.initiate_data_transformation(
                    df=st.session_state.df,
                    target_column_name=st.session_state.get('target_column'),
                    issues=st.session_state.issues,
                    user_decisions=st.session_state.user_decisions,
                    test_size=test_size,
                    random_state=42
                )
                
                # Store preprocessing decisions and log
                st.session_state.preprocessing_log = preprocessing_log
                st.session_state.class_weights = class_weights
                st.session_state.train_array = train_arr
                st.session_state.test_array = test_arr
                
                st.write(f"Preprocessing applied: {len(preprocessing_log)} steps")
                with st.expander("View Preprocessing Log"):
                    for step in preprocessing_log:
                        st.json(step)
                
                status.update(label=" Preprocessing Complete!", state="complete", expanded=False)
                time.sleep(1)
                st.session_state.current_page = "page_model_training"
                st.rerun()
                
            except Exception as e:
                status.update(label=" Pipeline Failed", state="error")
                st.error(f"Error: {str(e)}")
                logging.error(f"Pipeline Error: {str(e)}")

# ============================================================================
# PAGE 5: MODEL TRAINING
# ============================================================================
def page_model_training():
    render_progress_bar(4)
    st.title(" Model Training Lab")
    
    if st.session_state.train_array is None:
        st.warning(" Pipeline not initialized. Go back to Configuration.")
        return

    st.markdown("""
    The system will now train **7 distinct classification algorithms**, optimize their hyperparameters, 
    and validate them against the test set.
    """)
    
    with st.container(border=True):
        st.write("### Algorithms Queued")
        cols = st.columns(4)
        models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes", "Rule Based Classifier"]
        for i, m in enumerate(models):
            cols[i%4].markdown(f"- {m}")
            
    if st.button(" Begin Training Sequence", type="primary", use_container_width=True):
        with st.status(" AutoML Engine Running...", expanded=True) as status:
            try:
                model_trainer = ModelTrainer()
                
                st.write(" Training Base Models...")
                # Simulate steps for UX
                time.sleep(0.5)
                
                st.write(" Optimizing Hyperparameters...")
                
                # Train model and capture results
                train_results = model_trainer.initiate_model_trainer(
                    st.session_state.train_array,
                    st.session_state.test_array,
                    'classification',
                    search_type='random',  # Random search is faster than grid search
                    class_weights=st.session_state.class_weights
                )

                # Extract model report dict and best model name
                st.session_state.model_results = train_results[0]  # Full metrics dict
                st.session_state.best_model_name = train_results[1]  # Best model name
                
                # 3. Extract X_test and y_test for ROC curve visualization
                test_array = st.session_state.test_array
                st.session_state.X_test = test_array[:, :-1]
                st.session_state.y_test = test_array[:, -1]
                
                st.write(" Calculating Metrics...")
                status.update(label=" Training Successfully Completed!", state="complete", expanded=False)
                time.sleep(1)
                st.session_state.current_page = "page_comparison"
                st.rerun()
                
            except Exception as e:
                st.error(f"Training Error: {str(e)}")
                logging.error(f"Training Error: {str(e)}")

# ============================================================================
# PAGE 6: COMPARISON DASHBOARD
# ============================================================================
def page_model_comparison():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    render_progress_bar(5)
    st.title(" Leaderboard & Analysis")
    
    if not st.session_state.model_results:
        st.error("No results found.")
        return
        
    results = st.session_state.model_results
    
    # Process results into DataFrame - metrics are now full dicts from evaluate_models
    data = []
    for model, metrics in results.items():
        row = {'Model': model}
        
        # Metrics should be dictionaries (from step 7 enhancements)
        if isinstance(metrics, dict):
            # Add scalar metrics directly
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time']:
                if key in metrics:
                    row[key] = metrics[key]
            # Store model and predictions for later use
            if 'model' in metrics:
                row['model'] = metrics['model']
            if 'confusion_matrix' in metrics:
                row['confusion_matrix'] = metrics['confusion_matrix']
            if 'y_test_pred' in metrics:
                row['y_test_pred'] = metrics['y_test_pred']
        else:
            # Fallback for non-dict metrics
            row['Score'] = metrics
            
        data.append(row)

    df_results = pd.DataFrame(data)
    st.session_state.comparison_df = df_results
    
    # 1. Best Model Highlight
    if 'f1_score' in df_results.columns:
        best_idx = df_results['f1_score'].idxmax()
        best_model = df_results.loc[best_idx]
        st.session_state.best_model = best_model
        
        st.markdown("###  Champion Model")
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Algorithm", best_model['Model'])
            col2.metric("F1 Score", f"{best_model['f1_score']:.4f}", delta="Best")
            col3.metric("Accuracy", f"{best_model['accuracy']:.4f}")
            col4.metric("Recall", f"{best_model['recall']:.4f}")

    # 2. Detailed Table with Styling
    st.markdown("###  Comparative Metrics")
    
    # Get numeric columns only (exclude non-numeric/object columns and non-serializable)
    numeric_cols = df_results.select_dtypes(include=['number']).columns.tolist()
    exclude_cols = ['confusion_matrix', 'model', 'y_test_pred']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Create display dataframe with only numeric columns for styling
    display_df = df_results[['Model'] + numeric_cols].copy()
    
    # Apply gradient styling only to numeric columns
    styled_df = display_df.style.highlight_max(
        subset=numeric_cols, 
        axis=0, 
        color='#D6EAF8' 
    ).format("{:.4f}", subset=numeric_cols)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # CSV Download Button
    st.markdown("###  Export Results")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Prepare CSV data (exclude non-serializable columns)
    csv_df = df_results.copy()
    csv_df = csv_df.drop(columns=['model', 'y_test_pred', 'confusion_matrix'], errors='ignore')
    
    csv_data = csv_df.to_csv(index=False)
    
    col1.download_button(
        label=" Download Metrics as CSV",
        data=csv_data,
        file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # 3. Visual Comparison
    st.markdown("###  Metric Analysis")
    available_metrics = [m for m in ['f1_score', 'accuracy', 'precision', 'recall', 'training_time'] if m in df_results.columns]
    metric = st.selectbox("Select Metric to Visualize", available_metrics) if available_metrics else None
    
    if metric and metric in df_results.columns:
        chart_data = df_results[['Model', metric]].sort_values(metric, ascending=False)
        st.bar_chart(chart_data.set_index('Model'), color="#2C3E50")

    # 4. ROC Curves (for binary classification)
    st.markdown("###  ROC Curves (Binary Classification)")
    
    # Check if we have binary classification and roc_auc values
    if 'roc_auc' in df_results.columns:
        has_roc = df_results['roc_auc'].notna().any()
        
        if has_roc:
            # Recreate ROC curves for all models with predictions
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Get the test set from session state (need to store it during training)
            if hasattr(st.session_state, 'y_test') and hasattr(st.session_state, 'X_test'):
                y_test = st.session_state.y_test
                
                for idx, row in df_results.iterrows():
                    model_name = row['Model']
                    if pd.notna(row.get('roc_auc')) and 'model' in row:
                        try:
                            model = row['model']
                            X_test = st.session_state.X_test
                            
                            # Get probability predictions
                            y_proba = model.predict_proba(X_test)[:, 1]
                            
                            # Calculate ROC curve
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            roc_auc_val = auc(fpr, tpr)
                            
                            # Plot
                            ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_val:.3f})', linewidth=2)
                        except Exception as e:
                            logging.warning(f"Could not generate ROC curve for {model_name}: {str(e)}")
                
                # Plot diagonal line
                ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            else:
                st.info(" Test data not available for ROC curve visualization. ROC-AUC values are shown in the metrics table.")
        else:
            st.info(" ROC-AUC only available for binary classification problems.")
    else:
        st.info("ROC-AUC values not computed (multiclass classification detected).")

    # 5. Precision-Recall Curves (for binary classification)
    st.markdown("###  Precision-Recall Curves (Binary Classification)")
    
    if hasattr(st.session_state, 'y_test') and hasattr(st.session_state, 'X_test'):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        y_test = st.session_state.y_test
        if len(np.unique(y_test)) == 2:
            try:
                fig_pr, ax_pr = plt.subplots(figsize=(10, 7))
                
                for idx, row in df_results.iterrows():
                    model_name = row['Model']
                    if 'model' in row:
                        try:
                            model = row['model']
                            X_test = st.session_state.X_test
                            
                            # Get probability predictions
                            y_proba = model.predict_proba(X_test)[:, 1]
                            
                            # Calculate P-R curve
                            precision, recall, _ = precision_recall_curve(y_test, y_proba)
                            ap = average_precision_score(y_test, y_proba)
                            
                            # Plot
                            ax_pr.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})', linewidth=2)
                        except Exception as e:
                            logging.warning(f"Could not generate P-R curve for {model_name}: {str(e)}")
                
                ax_pr.set_xlabel('Recall', fontsize=12)
                ax_pr.set_ylabel('Precision', fontsize=12)
                ax_pr.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
                ax_pr.legend(loc='best', fontsize=10)
                ax_pr.grid(True, alpha=0.3)
                ax_pr.set_xlim([0, 1])
                ax_pr.set_ylim([0, 1])
                
                st.pyplot(fig_pr)
            except Exception as e:
                st.warning(f"Could not generate Precision-Recall curves: {str(e)}")
        else:
            st.info(" Precision-Recall curves only available for binary classification problems.")
    else:
        st.info(" Test data not available for P-R curve visualization.")

    # 6. Confusion Matrices
    st.markdown("###  Confusion Matrices")
    
    if hasattr(st.session_state, 'y_test'):
        y_test = st.session_state.y_test
        
        # Select which models' confusion matrices to display
        models_to_show = st.multiselect(
            "Select models to view confusion matrices:",
            list(results.keys()),
            default=list(results.keys())[:3] if len(results) > 3 else list(results.keys())
        )
        
        if models_to_show:
            cols = st.columns(min(2, len(models_to_show)))
            
            for idx, model_name in enumerate(models_to_show):
                row = df_results[df_results['Model'] == model_name]
                if not row.empty and 'confusion_matrix' in row.columns:
                    cm = row.iloc[0]['confusion_matrix']
                    
                    with cols[idx % 2]:
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        
                        # Plot heatmap
                        import seaborn as sns
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
                        ax_cm.set_title(f'{model_name}', fontsize=12, fontweight='bold')
                        ax_cm.set_ylabel('Actual', fontsize=11)
                        ax_cm.set_xlabel('Predicted', fontsize=11)
                        
                        st.pyplot(fig_cm)

    # 7. Feature Importance (for tree-based and ensemble models)
    st.markdown("###  Feature Importance Analysis")
    
    tree_based_models = ['Random Forest', 'Decision Tree']
    tree_models_available = [m for m in tree_based_models if m in results.keys()]
    
    if tree_models_available and hasattr(st.session_state, 'X_test'):
        selected_model = st.selectbox(
            "Select tree-based model for feature importance:",
            tree_models_available
        )
        
        row = df_results[df_results['Model'] == selected_model]
        if not row.empty and 'model' in row.columns:
            model = row.iloc[0]['model']
            
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create feature importance dataframe
                feature_names = [f"Feature {i}" for i in range(len(importances))]
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(10)
                
                # Plot feature importance
                fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                ax_fi.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db')
                ax_fi.set_xlabel('Importance Score', fontsize=12)
                ax_fi.set_title(f'Top 10 Feature Importances - {selected_model}', fontsize=14, fontweight='bold')
                ax_fi.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig_fi)
            else:
                st.info(f" {selected_model} does not have feature importance scores.")
    else:
        st.info(" Feature importance visualization only available for tree-based models (Random Forest, Decision Tree).")

    # Navigation Footer
    c1, _, c2 = st.columns([1, 4, 1])
    if c1.button(" Retrain"):
        st.session_state.current_page = "page_model_training"
        st.rerun()
    if c2.button("Generate Report ", type="primary"):
        st.session_state.current_page = "page_report"
        st.rerun()

# ============================================================================
# PAGE 7: REPORT
# ============================================================================
def page_report_generation():
    render_progress_bar(6)
    st.title("Final Execution Report")
    
    with st.container(border=True):
        st.success(" The comprehensive pipeline report has been compiled.")
        st.markdown("This report includes:")
        st.markdown("""
        - Data Health Summary
        - Preprocessing Decisions
        - Full Model Leaderboard
        - Champion Model Specifications
        """)
        
        try:
            report_gen = ReportGenerator(
                dataset=st.session_state.df,
                target_column=st.session_state.get('target_column'),
                issues=st.session_state.issues,
                user_decisions=st.session_state.user_decisions,
                preprocessing_config=st.session_state.preprocessing_config,
                model_results=st.session_state.model_results,
                best_model=st.session_state.best_model,
                preprocessing_log=st.session_state.get('preprocessing_log', []),
                feature_types=st.session_state.get('feature_types', {})
            )
            html_report = report_gen.generate_html_report()
            
            st.download_button(
                label=" Download HTML Report",
                data=html_report,
                file_name=f"AutoML_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                type="primary",
                use_container_width=True
            )
            st.session_state.report_generated = True
            
        except Exception as e:
            st.error(f"Generation Error: {str(e)}")

    if st.button(" Reset Application", type="secondary"):
        for key in st.session_state.keys():
            del st.session_state[key]
        initialize_session_state()
        st.rerun()

# ============================================================================
# MAIN APP ROUTING
# ============================================================================
def main():
    initialize_session_state()
    
    # --- Professional Sidebar ---
    with st.sidebar:
        st.title(" AutoML Pro")
        st.caption("v2.0 | Enterprise Edition")
        st.divider()
        
        # Status Badge
        if st.session_state.df_uploaded:
            st.info(f" Data: Loaded\n({len(st.session_state.df)} rows)")
        else:
            st.warning(" Data: Not Loaded")
            
        st.divider()
        
        # Navigation Map
        pages = {
            "page_upload": " Upload & Info",
            "page_eda": " Analysis (EDA)",
            "page_issues": " Quality Check",
            "page_preprocessing": " Configuration",
            "page_model_training": " Training Lab",
            "page_comparison": " Leaderboard",
            "page_report": " Final Report"
        }
        
        # Auto-select based on session state
        curr = st.session_state.current_page
        idx = list(pages.keys()).index(curr) if curr in pages else 0
        
        selection = st.radio(
            "Pipeline Stages:",
            list(pages.values()),
            index=idx,
            label_visibility="collapsed"
        )
        
        # Sync selection back to session state
        selected_key = list(pages.keys())[list(pages.values()).index(selection)]
        if selected_key != st.session_state.current_page:
            st.session_state.current_page = selected_key
            st.rerun()
            
        st.divider()
        st.markdown("Made with ❤️ using Streamlit")

    # --- Page Routing ---
    page_map = {
        "page_upload": page_upload_and_info,
        "page_eda": page_eda,
        "page_issues": page_issue_detection,
        "page_preprocessing": page_preprocessing_config,
        "page_model_training": page_model_training,
        "page_comparison": page_model_comparison,
        "page_report": page_report_generation
    }
    
    # Execute current page function
    if st.session_state.current_page in page_map:
        page_map[st.session_state.current_page]()

if __name__ == "__main__":
    main()