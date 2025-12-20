import pandas as pd
import datetime
import json

class ReportGenerator:
    """
    Generates an HTML report summarizing the entire AutoML process:
    - Dataset Statistics
    - EDA Findings
    - Issues Detected & User Decisions
    - Preprocessing Configuration
    - Model Comparison Table
    - Best Model Metrics with Justification
    - Model Configurations & Hyperparameters
    """
    def __init__(self, dataset, target_column, issues, user_decisions, preprocessing_config, model_results, best_model, preprocessing_log=None, feature_types=None):
        self.dataset = dataset
        self.target_column = target_column
        self.issues = issues
        self.user_decisions = user_decisions
        self.preprocessing_config = preprocessing_config
        self.model_results = model_results
        self.best_model = best_model
        self.preprocessing_log = preprocessing_log or []
        self.feature_types = feature_types or {}

    def generate_html_report(self):
        # Current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Basic HTML Template using f-strings
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoML Project Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                h3 {{ color: #34495e; margin-top: 20px; }}
                .metric-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .best-model-box {{ background: #d5f4e6; padding: 20px; border-left: 5px solid #27ae60; border-radius: 5px; margin: 15px 0; }}
                .config-box {{ background: #fef5e7; padding: 15px; border-left: 5px solid #f39c12; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .highlight {{ font-weight: bold; color: #27ae60; }}
                .warning {{ color: #e74c3c; font-weight: bold; }}
                ul {{ line-height: 1.8; }}
                .justification {{ background-color: #e8f8f5; padding: 15px; border-radius: 5px; margin-top: 10px; }}
                .config-details {{ background-color: white; padding: 15px; border: 1px solid #bdc3c7; border-radius: 5px; margin-top: 10px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 AutoML Execution Report</h1>
                <p><strong>Generated on:</strong> {timestamp}</p>

                <h2>1. Dataset Overview</h2>
                <div class="metric-box">
                    <p><strong>Target Column:</strong> {self.target_column}</p>
                    <p><strong>Total Rows:</strong> {len(self.dataset)}</p>
                    <p><strong>Total Columns:</strong> {len(self.dataset.columns)}</p>
                    <p><strong>Features Used:</strong> {len(self.dataset.columns) - 1}</p>
                </div>

                <h2>2. Exploratory Data Analysis (EDA) Findings</h2>
                {self._generate_eda_findings()}

                <h2>3. Feature Type Analysis</h2>
                {self._generate_feature_types_section()}

                <h2>4. Data Quality Issues & User Decisions</h2>
                {self._generate_issues_table()}

                <h2>5. Preprocessing Configuration</h2>
                <ul>
                    <li><strong>Missing Value Strategy:</strong> {self.preprocessing_config.get('missing_strategy', 'N/A')}</li>
                    <li><strong>Outlier Strategy:</strong> {self.preprocessing_config.get('outlier_strategy', 'N/A')}</li>
                    <li><strong>Scaling Strategy:</strong> {self.preprocessing_config.get('scaling_strategy', 'N/A')}</li>
                    <li><strong>Encoding Strategy:</strong> {self.preprocessing_config.get('encoding_strategy', 'N/A')}</li>
                    <li><strong>Test Split Size:</strong> {float(self.preprocessing_config.get('test_size', 0.2))*100:.0f}%</li>
                </ul>

                <h2>6. Preprocessing Decisions Log</h2>
                {self._generate_preprocessing_log()}

                <h2>7. Model Configurations & Hyperparameters</h2>
                {self._generate_model_configs()}

                <h2>8. Model Performance Comparison</h2>
                {self._generate_model_table()}

                <h2>9. Confusion Matrices by Model</h2>
                {self._generate_confusion_matrices()}

                <h2>10. 🏆 Best Model Selected & Justification</h2>
                {self._generate_best_model_section()}

                <hr>
                <p style="text-align: center; font-size: 0.9em; color: #7f8c8d;">Generated by Custom AutoML Pipeline</p>
            </div>
        </body>
        </html>
        """
        return html_content

    def _generate_eda_findings(self):
        """Generate EDA findings from dataset analysis"""
        findings_html = """
        <div class="metric-box">
            <h3>Data Distribution & Quality Metrics</h3>
            <ul>
        """
        
        # Missing values analysis
        missing_count = self.dataset.isnull().sum().sum()
        missing_pct = (missing_count / (len(self.dataset) * len(self.dataset.columns))) * 100
        findings_html += f"<li><strong>Missing Values:</strong> {missing_count} ({missing_pct:.2f}% of total data)</li>"
        
        # Numerical columns analysis
        numeric_cols = self.dataset.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            findings_html += f"<li><strong>Numerical Features:</strong> {len(numeric_cols)} columns</li>"
            findings_html += f"<li><strong>Numerical Statistics:</strong>"
            findings_html += "<ul>"
            for col in numeric_cols:
                if col != self.target_column:
                    mean_val = self.dataset[col].mean()
                    std_val = self.dataset[col].std()
                    findings_html += f"<li>{col}: μ={mean_val:.2f}, σ={std_val:.2f}</li>"
            findings_html += "</ul></li>"
        
        # Categorical columns analysis
        categorical_cols = self.dataset.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            findings_html += f"<li><strong>Categorical Features:</strong> {len(categorical_cols)} columns</li>"
        
        # Target variable distribution
        if self.target_column in self.dataset.columns:
            if self.dataset[self.target_column].dtype == 'object':
                class_dist = self.dataset[self.target_column].value_counts()
                findings_html += f"<li><strong>Target Distribution:</strong>"
                findings_html += "<ul>"
                for cls, count in class_dist.items():
                    pct = (count / len(self.dataset)) * 100
                    findings_html += f"<li>{cls}: {count} samples ({pct:.1f}%)</li>"
                findings_html += "</ul></li>"
            else:
                findings_html += f"<li><strong>Target Variable Range:</strong> [{self.dataset[self.target_column].min():.2f}, {self.dataset[self.target_column].max():.2f}]</li>"
        
        # Duplicates check
        duplicates = self.dataset.duplicated().sum()
        findings_html += f"<li><strong>Duplicate Rows:</strong> {duplicates} ({(duplicates/len(self.dataset))*100:.2f}%)</li>"
        
        findings_html += """
            </ul>
        </div>
        """
        return findings_html

    def _generate_feature_types_section(self):
        """Generate feature type analysis section"""
        if not self.feature_types:
            return "<p> No feature type information available.</p>"
        
        # Normalize feature types: handle both str and dict formats
        # Expected formats:
        # - {'col': 'continuous_numeric'}
        # - {'col': {'type': 'continuous_numeric', ...metadata}}
        type_counts = {}
        normalized_items = []
        for col, ftype in self.feature_types.items():
            if isinstance(ftype, dict):
                ftype_name = ftype.get('type', 'unknown')
            else:
                ftype_name = str(ftype)
            type_counts.setdefault(ftype_name, []).append(col)
            normalized_items.append((col, ftype_name))
        
        html = """
        <div class="metric-box">
            <h3> Feature Type Distribution</h3>
            <ul>
        """
        
        for ftype, cols in sorted(type_counts.items()):
            html += f"<li><strong>{ftype}:</strong> {len(cols)} features</li>"
        
        html += "</ul><h3> Feature Type Details</h3><table><thead><tr><th>Feature</th><th>Type</th></tr></thead><tbody>"
        
        for col, ftype_name in sorted(normalized_items):
            html += f"<tr><td>{col}</td><td>{ftype_name}</td></tr>"
        
        html += "</tbody></table></div>"
        return html

    def _generate_preprocessing_log(self):
        """Generate preprocessing decisions log"""
        if not self.preprocessing_log:
            return "<p> No preprocessing steps were logged.</p>"
        
        html = """
        <div class="metric-box">
            <h3>Applied Preprocessing Steps</h3>
            <table>
                <thead>
                    <tr>
                        <th>Action</th>
                        <th>Columns Affected</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for log_entry in self.preprocessing_log:
            action = log_entry.get('action', 'Unknown')
            columns = log_entry.get('columns_affected', [])
            details_dict = log_entry.get('details', {})
            
            # Format column list
            if isinstance(columns, list):
                columns_str = ', '.join(str(c) for c in columns) if columns else 'N/A'
            else:
                columns_str = str(columns)
            
            # Format details from nested dict
            details_list = []
            if isinstance(details_dict, dict):
                for col, detail in details_dict.items():
                    if isinstance(detail, dict):
                        action_taken = detail.get('action', 'N/A')
                        if action == 'outlier_handling':
                            total = detail.get('total_capped', detail.get('rows_removed', 0))
                            details_list.append(f"{col}: {action_taken} ({total} values)")
                        elif action == 'missing_value_handling':
                            imputed = detail.get('values_imputed', 0)
                            strategy = detail.get('action', 'N/A')
                            details_list.append(f"{col}: {strategy} ({imputed} values)")
                        else:
                            details_list.append(f"{col}: {action_taken}")
            
            details_str = '<br>'.join(details_list) if details_list else log_entry.get('description', 'Applied')
            
            html += f"<tr><td>{action.replace('_', ' ').title()}</td><td>{columns_str}</td><td>{details_str}</td></tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        """
        return html

    def _generate_issues_table(self):
        """Helper to create HTML table for issues"""
        if not self.issues:
            return "<p> No data quality issues were detected.</p>"
        
        rows = ""
        for i, issue in enumerate(self.issues):
            # Try multiple key formats to find user decision
            issue_type = issue['type']
            column = issue.get('column', 'global')
            
            # Try format used in app: {type}_{i}_{column}
            decision_key1 = f"{issue_type}_{i}_{column}"
            # Try alternative format: {type}_{i}
            decision_key2 = f"{issue_type}_{i}"
            
            decision = self.user_decisions.get(decision_key1, 
                                              self.user_decisions.get(decision_key2, "Default/No Action"))
            
            rows += f"""
            <tr>
                <td>{issue_type}</td>
                <td>{column}</td>
                <td>{issue.get('severity', 'Medium')}</td>
                <td>{decision}</td>
            </tr>
            """
            
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Issue Type</th>
                    <th>Column</th>
                    <th>Severity</th>
                    <th>User Decision</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """

    def _generate_model_configs(self):
        """Generate model configurations and hyperparameters section"""
        if not self.model_results:
            return "<p> No model results available.</p>"
        
        configs_html = ""
        
        # Predefined hyperparameters for each model
        model_params = {
            "Logistic Regression": "No hyperparameters tuned (default configuration)",
            "Decision Tree": "Criterion: ['gini', 'entropy']",
            "Random Forest": "n_estimators: [8, 16, 32, 64, 128, 256]",
            "AdaBoost Classifier": "learning_rate: [0.1, 0.01, 0.5, 0.001], n_estimators: [8, 16, 32, 64, 128, 256]",
            "K-Nearest Neighbors": "n_neighbors: [3, 5, 7, 9, 11, 13], metric: ['euclidean', 'manhattan']",
            "Naive Bayes": "No hyperparameters tuned (distribution-based)",
            "Support Vector Machine": "C: [0.1, 1, 10, 100], kernel: ['linear', 'rbf']",
            "OneR Classifier": "No hyperparameters tuned (rule-based)"
        }
        
        for model_name in self.model_results.keys():
            params = model_params.get(model_name, "Parameters not defined")
            configs_html += f"""
            <div class="config-box">
                <h3>{model_name}</h3>
                <p><strong>Hyperparameter Search Space:</strong></p>
                <div class="config-details">
                    {params}
                </div>
            </div>
            """
        
        return configs_html

    def _generate_best_model_section(self):
        """Generate best model summary with detailed justification"""
        if self.best_model is None or (hasattr(self.best_model, 'empty') and self.best_model.empty):
            return "<p> No best model available.</p>"
        
        model_name = self.best_model.get('Model', 'Unknown')
        # Handle both key formats
        f1_score = self.best_model.get('f1_score', self.best_model.get('F1-Score', 0))
        accuracy = self.best_model.get('accuracy', self.best_model.get('Accuracy', 0))
        precision = self.best_model.get('precision', self.best_model.get('Precision', 0))
        recall = self.best_model.get('recall', self.best_model.get('Recall', 0))
        roc_auc = self.best_model.get('roc_auc', self.best_model.get('ROC-AUC', None))
        training_time = self.best_model.get('training_time', self.best_model.get('Training Time (s)', 0))
        
        # Calculate performance ranking
        if self.model_results:
            all_f1_scores = []
            for metrics in self.model_results.values():
                if isinstance(metrics, dict):
                    f1 = metrics.get('f1_score', metrics.get('F1-Score', 0))
                    if f1:
                        all_f1_scores.append(f1)
            
            if all_f1_scores:
                rank = sorted(all_f1_scores, reverse=True).index(f1_score) + 1
                total_models = len(all_f1_scores)
            else:
                rank = 1
                total_models = 1
        else:
            rank = 1
            total_models = 1
        
        # Generate justification
        justification = f"""
        <div class="justification">
            <h4> Performance Justification:</h4>
            <ul>
                <li><strong>Ranking:</strong> #{rank} out of {total_models} models</li>
                <li><strong>Primary Metric (F1-Score):</strong> {f1_score:.4f} - Balances precision and recall effectively</li>
                <li><strong>Accuracy:</strong> {accuracy:.4f} - Overall correctness of predictions</li>
                <li><strong>Precision:</strong> {precision:.4f} - Minimizes false positives</li>
                <li><strong>Recall:</strong> {recall:.4f} - Minimizes false negatives</li>
                {f'<li><strong>ROC-AUC:</strong> {roc_auc:.4f} - Binary classification discriminative ability</li>' if roc_auc else ''}
                <li><strong>Training Time:</strong> {training_time:.4f}s - Computational efficiency</li>
            </ul>
            
            <h4> Why This Model?</h4>
            <p>This model achieved the <strong>highest F1-score</strong>, which represents the best balance between 
            precision and recall. In classification tasks, F1-score is crucial because it avoids the pitfall of 
            high accuracy with imbalanced classes. The model demonstrates robust generalization and reliable 
            predictions for production deployment.</p>
        </div>
        """
        
        return f"""
        <div class="best-model-box">
            <h3> {model_name}</h3>
            <table style="width: 100%; margin-top: 15px;">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td><strong>F1-Score</strong></td>
                    <td><span class="highlight">{f1_score:.4f}</span></td>
                </tr>
                <tr>
                    <td><strong>Accuracy</strong></td>
                    <td>{accuracy:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Precision</strong></td>
                    <td>{precision:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Recall</strong></td>
                    <td>{recall:.4f}</td>
                </tr>
                {f'<tr><td><strong>ROC-AUC</strong></td><td>{roc_auc:.4f}</td></tr>' if roc_auc else ''}
                <tr>
                    <td><strong>Training Time</strong></td>
                    <td>{training_time:.4f}s</td>
                </tr>
            </table>
            {justification}
        </div>
        """

    def _generate_confusion_matrices(self):
        """Generate confusion matrices for all models as HTML tables"""
        if not self.model_results:
            return "<p> No confusion matrices available.</p>"
        
        cm_html = ""
        
        for model_name, metrics in self.model_results.items():
            if isinstance(metrics, dict) and 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                
                # Create HTML table from confusion matrix
                cm_html += f"""
                <div class="metric-box">
                    <h3>{model_name}</h3>
                    <table style="width: auto; margin: 10px 0;">
                        <tr>
                            <th>Predicted \\ Actual</th>
                            <th style="background-color: #3498db; color: white;">Negative</th>
                            <th style="background-color: #3498db; color: white;">Positive</th>
                        </tr>
                        <tr>
                            <th style="background-color: #3498db; color: white;">Negative</th>
                            <td style="background-color: #d5f4e6; font-weight: bold;">{cm[0,0]}</td>
                            <td style="background-color: #fef5e7;">{cm[0,1] if cm.shape[1] > 1 else 'N/A'}</td>
                        </tr>
                        <tr>
                            <th style="background-color: #3498db; color: white;">Positive</th>
                            <td style="background-color: #fef5e7;">{cm[1,0] if cm.shape[0] > 1 else 'N/A'}</td>
                            <td style="background-color: #d5f4e6; font-weight: bold;">{cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 'N/A'}</td>
                        </tr>
                    </table>
                </div>
                """
        
        if cm_html:
            return f"<p><strong>Confusion matrices for each model:</strong></p>{cm_html}"
        else:
            return "<p> No confusion matrices available.</p>"

    def _generate_model_table(self):
        """Helper to create HTML table for model results"""
        if not self.model_results:
            return "<p> No model results available.</p>"
        
        # Convert dict to DataFrame for easier handling
        rows_list = []
        for model_name, metrics in self.model_results.items():
            row = {}
            if isinstance(metrics, dict):
                # Extract only the metrics we want to display
                # Handle both underscore and hyphen key formats
                row['Model'] = model_name
                row['Accuracy'] = metrics.get('accuracy', 0)
                row['Precision'] = metrics.get('precision', 0)
                row['Recall'] = metrics.get('recall', 0)
                row['F1-Score'] = metrics.get('f1_score', metrics.get('F1-Score', 0))
                row['ROC-AUC'] = metrics.get('roc_auc', metrics.get('ROC-AUC', 'N/A'))
                row['Training Time (s)'] = metrics.get('training_time', 0)
            rows_list.append(row)
            
        df = pd.DataFrame(rows_list)
        
        # Sort by F1 Score descending
        if 'F1-Score' in df.columns:
            df = df.sort_values('F1-Score', ascending=False)
            
        return df.to_html(classes='table', index=False, float_format="%.4f")