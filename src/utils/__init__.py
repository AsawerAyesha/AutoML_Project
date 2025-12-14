import os
import sys
import pickle
import numpy as np
import logging
import time
from scipy.sparse import issparse
from sklearn.metrics import (
    make_scorer, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """Save object to file using pickle"""
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load object from file using pickle"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param, metric, search_type='grid'):
    """
    Evaluate classification models with hyperparameter optimization and compute comprehensive metrics.
    
    Args:
        search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        
    Returns:
        model_report: Dictionary with detailed metrics for each model
    """
    try:
        # Ensure target variables are integers for classification and properly shaped
        y_train = np.ravel(y_train).astype(int)
        y_test = np.ravel(y_test).astype(int)
        
        # Determine appropriate cv splits based on minimum class size
        unique, counts = np.unique(y_train, return_counts=True)
        min_class_size = counts.min()
        cv_splits = min(5, min_class_size)  # Use minimum of 5 or smallest class size
        
        if cv_splits < 2:
            logging.warning(f"Dataset too small or imbalanced for cross-validation. Min class size: {min_class_size}")
            cv_splits = 2  # Use at least 2 splits
        
        model_report = {}
        for model_name, model in models.items():
            # Track training time
            start_time = time.time()
            
            if param[model_name]:  # If there are hyperparameters to tune
                if search_type == 'random':
                    # Use RandomizedSearchCV
                    search = RandomizedSearchCV(
                        model, 
                        param[model_name], 
                        cv=cv_splits, 
                        scoring=make_scorer(metric), 
                        n_jobs=-1,
                        n_iter=20,  # Number of combinations to try
                        random_state=42
                    )
                else:
                    # Use GridSearchCV (default)
                    search = GridSearchCV(
                        model, 
                        param[model_name], 
                        cv=cv_splits, 
                        scoring=make_scorer(metric), 
                        n_jobs=-1
                    )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                search_method = 'RandomizedSearchCV' if search_type == 'random' else 'GridSearchCV'
                logging.info(f"{search_method} - Best model params for {model_name}: {search.best_params_}")
            else:  # If no hyperparameters to tune
                best_model = model
                best_model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Classification metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            
            # Handle both binary and multiclass
            average_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
            precision = precision_score(y_test, y_test_pred, average=average_method, zero_division=0)
            recall = recall_score(y_test, y_test_pred, average=average_method, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average=average_method, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            
            # ROC-AUC for binary classification
            roc_auc = None
            if len(np.unique(y_test)) == 2:
                try:
                    # For binary classification
                    y_test_proba = best_model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_test_proba)
                except (AttributeError, IndexError):
                    logging.warning(f"{model_name} does not support probability predictions for ROC-AUC")
                    roc_auc = None
            
            model_report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'roc_auc': roc_auc,
                'training_time': training_time,
                'model': best_model,
                'y_test_pred': y_test_pred
            }
            logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        return model_report
    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)
