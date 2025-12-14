import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from mlxtend.classifier import OneRClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model_info.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, problem_type='classification', search_type='grid'):
        """
        Train and optimize classification models.
        
        Args:
            search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV (default: 'grid')
        """
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            if problem_type == 'classification':
                # Convert target variables to integers for classification
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
                
                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Logistic Regression": LogisticRegression(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "Support Vector Machine": SVC(probability=True),
                    "OneR Classifier": OneRClassifier(),
                }
                params = {
                    "Decision Tree": {
                        'criterion': ['gini', 'entropy'],
                    },
                    "Random Forest": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Logistic Regression": {},
                    "AdaBoost Classifier": {
                        'learning_rate': [.1, .01, 0.5, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "K-Nearest Neighbors": {
                        'n_neighbors': [3, 5, 7, 9, 11, 13],
                        'metric': ['euclidean', 'manhattan']
                    },
                    "Naive Bayes": {},
                    "Support Vector Machine": {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['linear', 'rbf']
                    },
                    "OneR Classifier": {}
                }
                metric = accuracy_score

            else:
                raise CustomException("Only classification problem type is supported")

            # Evaluate Models
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, param=params, metric=metric, search_type=search_type)

            # --- Logic to find Best Model ---
            best_model_name = ""
            
            # Find best model based on F1-score
            best_model_name = max(model_report.items(), key=lambda x: x[1].get('f1_score', -np.inf))[0]

            logging.info(f"Best found model: {best_model_name}")

            # RETURN 2 ITEMS: detailed report and best model name
            return model_report, best_model_name

        except Exception as e:
            logging.error(f"Exception occurred in model training: {e}")
            raise CustomException(e, sys)