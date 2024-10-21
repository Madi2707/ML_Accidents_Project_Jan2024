# basic modules
import os
import sys
from dataclasses import dataclass
# specialized modules
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score #, classification_report, confusion_matrix
#from sklearn import metrics
# custom modules
from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Split training and test input data')

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1])

            # dictionary of models to train
            models = {'Gradient Boosting': GradientBoostingClassifier(),'Support Vector Classifier': SVC()}

            # hyperparameter tuning
            params = { 

                "Gradient Boosting": { 'n_estimators': [50, 100],
                                       'learning_rate': [0.05, 0.1, 0.2],
                                       'max_depth': [5, 6],
                                       'min_samples_split': [2, 5, 10],
                                       'min_samples_leaf': [2, 4],
                                       'subsample': [0.8, 1.0]},
                
                "Support Vector Classifier": {'C': [0.1, 1, 10], 
                                              'kernel': ['linear', 'rbf', 'poly'], 
                                              'gamma': ['scale', 'auto'], 
                                              'probability': [True]}

                }
                        
            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test=X_test, y_test = y_test, models=models, param = params)

            # get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name from the dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                print(f'No best model found: {best_model_score}')
            else:
                print(f'Best model found to be: {best_model_name} with {best_model_score} accuracy')

            #-------------------------------------------------------------------------
            logging.info('Best found model on both training and testing dataset')
            #-------------------------------------------------------------------------

            save_object(file_path=self.model_trainer_config.trained_model_filepath, obj= best_model)

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)