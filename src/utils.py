import logging
import os
import pickle
import sys
import time

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param,
    metric,
    search_type: str = "grid",
    problem_type: str = "classification",
):
    """Evaluate models with hyperparameter search and return rich metrics."""
    try:
        if problem_type == "classification":
            y_train = np.ravel(y_train).astype(int)
            y_test = np.ravel(y_test).astype(int)

            unique, counts = np.unique(y_train, return_counts=True)
            min_class_size = counts.min()
            cv_splits = min(5, min_class_size) if min_class_size >= 2 else 2

            model_report = {}
            for model_name, model in models.items():
                start_time = time.time()

                if param.get(model_name):
                    if search_type == "random":
                        search = RandomizedSearchCV(
                            model,
                            param[model_name],
                            cv=cv_splits,
                            scoring=make_scorer(metric),
                            n_jobs=-1,
                            n_iter=min(20, len(ParameterGrid(param[model_name]))),
                            random_state=42,
                        )
                    else:
                        search = GridSearchCV(
                            model,
                            param[model_name],
                            cv=cv_splits,
                            scoring=make_scorer(metric),
                            n_jobs=-1,
                        )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    logging.info(f"Search params for {model_name}: {search.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)

                training_time = time.time() - start_time

                y_test_pred = best_model.predict(X_test)

                average_method = "binary" if len(np.unique(y_test)) == 2 else "weighted"
                accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred, average=average_method, zero_division=0)
                recall = recall_score(y_test, y_test_pred, average=average_method, zero_division=0)
                f1 = f1_score(y_test, y_test_pred, average=average_method, zero_division=0)
                cm = confusion_matrix(y_test, y_test_pred)

                roc_auc = None
                if len(np.unique(y_test)) == 2:
                    try:
                        y_test_proba = best_model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_test_proba)
                    except (AttributeError, IndexError):
                        logging.warning(f"{model_name} lacks predict_proba; ROC-AUC skipped")

                model_report[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "confusion_matrix": cm,
                    "roc_auc": roc_auc,
                    "training_time": training_time,
                    "model": best_model,
                    "y_test_pred": y_test_pred,
                }

                logging.info(
                    f"{model_name} - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, "
                    f"F1: {f1:.4f}, Time: {training_time:.2f}s"
                )

            return model_report

        elif problem_type == "clustering":
            model_report = {}
            for model_name, model in models.items():
                best_score = -np.inf
                best_params = None
                for param_combination in ParameterGrid(param.get(model_name, {})):
                    model_instance = model.set_params(**param_combination)
                    y_pred = model_instance.fit_predict(X_train)

                    unique_labels = np.unique(y_pred)
                    n_clusters = len(unique_labels[unique_labels != -1])
                    if n_clusters <= 1:
                        continue

                    X_dense = X_train.toarray() if issparse(X_train) else X_train
                    score = silhouette_score(X_dense, y_pred)
                    if score > best_score:
                        best_score = score
                        best_params = param_combination

                if best_params is None:
                    logging.warning(f"No valid clustering found for {model_name}; skipping")
                    continue

                best_model = model.set_params(**best_params)
                best_model.fit(X_train)
                model_report[model_name] = {
                    "silhouette_score": best_score,
                    "model": best_model,
                }

            return model_report

        else:
            raise CustomException("Unsupported problem type")

    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)