
# basic modules
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# specialized modules
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
# custom modules
from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
       self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            '''
            This function is responsible for data transformation
            '''
            cat_features = ['kpi_cause_code','light_code', 'weather_code',
                        'traffic_code','road_type_code','road_code',
                        'collision_code','damage_code','location_code']

            cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('onehotencoder', OneHotEncoder())                                                                   
                                       ])
            #------------------------------------------------------------------------------------------
            logging.info('Categorical columns encoding completed')
            #------------------------------------------------------------------------------------------
            preprocessor = ColumnTransformer([('cat_pipeline', cat_pipeline, cat_features)])

            #------------------------------------------------------------------------------------------
            logging.info("Pipeline Colum Transformation completed for..")
            logging.info(f"Categorical columns: {cat_features}")
            #------------------------------------------------------------------------------------------
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            #------------------------------------------------------------------------------------------
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            #------------------------------------------------------------------------------------------
            target_col_name = 'emp_liability_code'
                        
            X_train = train_df.drop(columns=[target_col_name], axis=1)
            y_train = train_df[target_col_name]

            X_test = test_df.drop(columns=[target_col_name], axis=1)
            y_test = test_df[target_col_name]
            
            preprocessing = self.get_data_transformer()
            #------------------------------------------------------------------------------------------
            logging.info("Applying preprocessing object on train and test data")
            #------------------------------------------------------------------------------------------
            input_train_arr = preprocessing.fit_transform(X_train)
            input_test_arr = preprocessing.transform(X_test)

            #-----------------------------------------------------------------------------------------------------------
            # Reshaping the Sparse matrices so that they can be passed to the trainer 
            #-----------------------------------------------------------------------------------------------------------
            
            # reshaping the y variables into 2d array and coverting them to float
            y_train_2d = y_train.values.reshape(-1, 1)
            y_train_2df = y_train_2d.astype(float)
            
            y_test_2d = y_test.values.reshape(-1, 1)
            y_test_2df = y_test_2d.astype(float)
            
            # Convert the dense array to a column-wise sparse matrix
            dense_sparse_matrix_train = np.column_stack([y_train_2df])
            dense_sparse_matrix_test = np.column_stack([y_test_2df])

            # Combine the sparse matrix and the conerted y sparse matrix horizontally
            train_combined_matrix = hstack([input_train_arr, dense_sparse_matrix_train]) 
            test_combined_matrix = hstack([input_test_arr, dense_sparse_matrix_test])   
            
            # Converting back the sparse matrix to a dense matrix (inot a numpy array)
            train_combined_matrix = train_combined_matrix.toarray()
            test_combined_matrix = test_combined_matrix.toarray()

            save_object(file_path = self.data_transformation_config.preprocessor_file_path, obj = preprocessing)

            #------------------------------------------------------------------------------------------
            logging.info(f"Saved preprocessing object")
            #------------------------------------------------------------------------------------------

            return (train_combined_matrix, test_combined_matrix, self.data_transformation_config.preprocessor_file_path)

        except Exception as e:
            raise CustomException(e, sys)
