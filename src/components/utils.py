import sys
import os
import dill

from src.components.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
         
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv = 3, n_jobs=-1)

            # train the model using parameters with GridSearchCV
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # make predictions 
            y_train_pred = model.predict(X_train) 
            y_test_pred = model.predict(X_test)

            # get the accuracy score from predictions and actuals train and test data
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            # create a report for findings
            report[list(models.keys())[i]] = test_model_score

        # send report back to model trainer.py
        return report
    
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj: # open the file in read byte mode 'rb'
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException