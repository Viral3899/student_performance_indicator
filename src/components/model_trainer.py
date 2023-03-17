import os
import sys
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass


from src.logger import logging
from src.exception import CustomeException
from src.util import save_object,evaluate_models

import pandas as pd
import numpy as np
import sklearn

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error






@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and Test Input Data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Linear Regression":{'model':LinearRegression(),"parameters":{"fit_intercept":[True,False]}},

                # "Ridge Regression":{'model':Ridge(),"parameters":{"alpha":[0.0001, 0.001,0.01, 0.1, 1, 10,15,20],"solver":['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']}},

                # "Lasso Regression":{'model':Ridge(),"parameters":{"alpha":[0.0001, 0.001,0.01, 0.1, 1, 10,15,20],'selection':['cyclic', 'random']}},

                "Random Forest Regressor":{'model':RandomForestRegressor(n_jobs=-1),"parameters":{'n_estimators':[i for i in range (50,251,50)],'max_features' : ["sqrt", "log2", None],'max_depth':[i for i in range(2,10)],'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],}} ,

                "Decision Tree Regressor":{'model':DecisionTreeRegressor(),"parameters":{'max_depth':[i for i in range(2,10)],'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],'max_features' : ["auto", "sqrt", "log2"]}} ,

                "Gradient Boosting Regressor":{'model':GradientBoostingRegressor(),"parameters":{
                    # 'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'n_estimators':[i for i in range (50,251,50)],
                'learning_rate':[.005,.001,.05,.01,.1,.5,1],
                # 'criterion' : ['friedman_mse', 'squared_error'],'max_features' : ['auto', 'sqrt', 'log2'],
                # 'max_depth':[i for i in range(2,10)]
                }} ,

                "K-Neighbors Regressor":{'model':KNeighborsRegressor(n_jobs=-1),"parameters":{'weights' :['uniform', 'distance'],
                'n_neighbors':[i for i in range(1,15)],'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
                }} ,

                "XGBRegressor": {'model':XGBRegressor(),"parameters":{'n_estimators':[i for i in range (50,251,50)],}},

                "CatBoosting Regressor": {'model':CatBoostRegressor(verbose=False),"parameters":{'max_depth':[i for i in range(2,10)]}},

                "AdaBoost Regressor":{'model':AdaBoostRegressor(),"parameters":{'loss' : ['linear', 'square', 'exponential'],'n_estimators':[i for i in range (50,251,50)]}},


            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]['model']

            if best_model_score<0.6:
                raise CustomeException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"best model found is {best_model_name} : with accuracy of {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square


          
        
        except Exception as e:
            logging.info(f'Error Occured {CustomeException(e,sys)}')
            raise CustomeException(e,sys)
