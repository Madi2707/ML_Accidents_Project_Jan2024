# basic modules
import os 
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# custom modules
from logger import logging
from exception import CustomException
from data_transformation import DataTransformation
from model_trainer import ModelTrainerConfig, ModelTrainer

# specialized modules
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    #all the outputs would be in the path below
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        #------------------------------------------------------------------------------------------
        logging.info("Entered the data ingestion method or component")
        #------------------------------------------------------------------------------------------
        try:
            data = pd.read_csv('C:/Users/MAnsari/ml_accidents/notebook/Accidents_dataset01.csv')
            #----------------------------------------------------------------------------------------------
            # Cleaning the raw data
            #----------------------------------------------------------------------------------------------
            
            # 1: Drop duplicates
            df = data.drop_duplicates() 

            # 2: Handling empty values
            mask = (df['location_code'].isna()) & (df['emp_liability_code'] == 'UNREP')
            df.loc[mask, 'location_code'] = 'DEPOT'

            # 3: Categorical to numerical conversion simple using pandas cat.codes
            df['emp_liability_code'] = df['emp_liability_code'].astype('category').cat.codes

            #----------------------------------------------------------------------------------------------
            logging.info("Reading  the dataset as dataframe and basic cleaning completed")
            #----------------------------------------------------------------------------------------------
            
            os.makedirs(os.path.dirname(self.ingestion_config. train_data_path), exist_ok = True)

            # send cleaned version of the dataset to the artifacts directory
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            #----------------------------------------------------------------------------------------------
            logging.info('Train Test Split initiated')
            #----------------------------------------------------------------------------------------------
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            #----------------------------------------------------------------------------------------------
            # Check for unseen categories in test data before feeding into the pipeline
            #----------------------------------------------------------------------------------------------

            # 1. Extract categorical feature names
            cat_features = train_set.select_dtypes(include=['object']).columns.tolist()

            # 2. Identify unseen categories
            unseen_categories = {}
            for feature in cat_features:
                unseen_categories[feature] = set(test_set[feature]) - set(train_set[feature])

            # 3. Remove rows with unseen categories from the test data
            for feature in cat_features:
                test_set = test_set[~test_set[feature].isin(unseen_categories[feature])]

            # 4. Save the data split into .csv files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            #------------------------------------------------------------------------------------------
            logging.info('Ingestion of the data completed')
            #------------------------------------------------------------------------------------------
            
            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
                  )

        except Exception as e:
            raise CustomException(e, sys)          


if __name__=='__main__':
       obj = DataIngestion()
       train_data, test_data = obj.initiate_data_ingestion()

       data_transform = DataTransformation()
       train_arr, test_arr,_ = data_transform.initiate_data_transformation(train_data, test_data)

       modeltrainer = ModelTrainer()
       modeltrainer.initiate_model_training(train_arr, test_arr)