import os
import sys
import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self, imputation_strategy='median', outlier_strategy='capping',
                 scaling_strategy='StandardScaler', encoding_strategy='OneHotEncoder',
                 remove_outliers=False, handle_missing=True):
        
        self.data_transformation_config = DataTransformationConfig()
        
        # User preferences for preprocessing
        self.imputation_strategy = imputation_strategy
        self.outlier_strategy = outlier_strategy
        self.scaling_strategy = scaling_strategy
        self.encoding_strategy = encoding_strategy
        self.remove_outliers = remove_outliers
        self.handle_missing = handle_missing

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        Build preprocessing pipeline based on user preferences
        '''
        try:
            # 1. Configure Imputer
            if self.handle_missing:
                imputer_strategy = self.imputation_strategy if self.imputation_strategy in \
                                 ['mean', 'median', 'most_frequent'] else 'median'
                num_imputer = SimpleImputer(strategy=imputer_strategy)
            else:
                num_imputer = SimpleImputer(strategy='median')
            
            # 2. Configure Scaler
            if self.scaling_strategy == 'MinMaxScaler':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            # 3. Create Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", num_imputer),
                    ("scaler", scaler)
                ]
            )

            # 4. Configure Encoder
            if self.encoding_strategy == 'OrdinalEncoder':
                # Use OrdinalEncoder for ordinal strategies
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            else:
                # Default to OneHotEncoder
                encoder = OneHotEncoder(handle_unknown='ignore')

            # 5. Create Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
                    ("scaler", StandardScaler(with_mean=False) if self.encoding_strategy != 'OrdinalEncoder' else 'passthrough')
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # 6. Combine into ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path, problem_type, target_column_name):
        try:
            train_df = pd.read_csv(train_path, 
                                   on_bad_lines='warn',    # Warn about skipped lines
                                   quoting=csv.QUOTE_MINIMAL,  # Only quote fields which contain special characters
                                   escapechar='\\')        # Use backslash as escape character
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Train data columns: {train_df.columns.tolist()}")

            test_df = pd.read_csv(test_path, 
                                  on_bad_lines='warn',    # Warn about skipped lines
                                  quoting=csv.QUOTE_MINIMAL,  # Only quote fields which contain special characters
                                  escapechar='\\')        # Use backslash as escape character
            logging.info(f"Test data shape: {test_df.shape}")
            logging.info(f"Test data columns: {test_df.columns.tolist()}")

            logging.info("Read train and test data completed")

            # Automatically identify numerical and categorical columns
            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

            if target_column_name:
                if target_column_name in numerical_columns:
                    numerical_columns.remove(target_column_name)
                if target_column_name in categorical_columns:
                    categorical_columns.remove(target_column_name)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            if problem_type in ['regression', 'classification']:
                if target_column_name is None:
                    raise CustomException("Target column name must be provided for regression and classification problems.")

                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info("Applying preprocessing object on training and testing dataframes.")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

            elif problem_type == 'clustering':
                input_feature_train_df = train_df
                input_feature_test_df = test_df

                logging.info("Applying preprocessing object on training and testing dataframes for clustering.")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = input_feature_train_arr
                test_arr = input_feature_test_arr

            else:
                raise CustomException("Unsupported problem type")

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)