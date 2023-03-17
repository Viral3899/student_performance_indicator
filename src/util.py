import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


from src.logger import logging
from src.exception import CustomeException


def save_object(file_path,obj):

    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)


    except Exception as e:
        logging.info(f"error Occuredd in save_object {CustomeException(e,sys)}")
        raise CustomeException(e,sys)
    




def evaluate_models(X_train, y_train,X_test,y_test,models) -> dict:
    try:
        report = {}

        for i in range(len(list(models))):
            # print(len(model))

            estimator = list(models.values())[i]['model']
            
            logging.info(f"Current Estimator is {list(models.keys())[i]}")
            
            model= GridSearchCV(estimator, list(models.values())[i]['parameters'],scoring='neg_mean_squared_error',cv=5)
            
            model.fit(X_train, y_train)  # Train model
            
            logging.info(f"{estimstor} with best parameters {modes.besr_params_} and best score of {model.best_score_}")
            
            estimator.set_params(**model.best_params_)
            
            estimator.fit(X_train, y_train)
            
            
            y_train_pred = estimator.predict(X_train)

            y_test_pred = estimator.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
         logging.info(f'Error Occured {CustomeException(e,sys)}')
         raise CustomeException(e,sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        logging.info(f"Error Occured {CustomeException(e,sys)}")
        raise CustomeException(e, sys)
