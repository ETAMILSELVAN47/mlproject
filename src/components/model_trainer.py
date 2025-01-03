import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr:np.ndarray,test_arr:np.ndarray):
        logging.info('Model Trainer has been started')
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'Linear Regression':LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'AdaBoost Regressor':AdaBoostRegressor(),
                'XGBoost Regressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=False),
                'Gradient Boosting Regressor':GradientBoostingRegressor()
            }

            params_list={
                'Linear Regression':{},
                'Decision Tree Regressor': {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                'Random Forest Regressor': {'n_estimators': [8,16,32,64,128,256]},
                'AdaBoost Regressor':{'learning_rate':[.1,.01,0.5,.001],'n_estimators': [8,16,32,64,128,256]},
                'XGBoost Regressor':{'learning_rate':[.1,.01,.05,.001],'n_estimators': [8,16,32,64,128,256]},
                'CatBoostRegressor':{'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]},
                'Gradient Boosting Regressor':{ 'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]}
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params_list=params_list)

            logging.info(f'model_report:{model_report}')
        
            # To get best model score
            best_model_score=max(sorted(model_report.values()))

            # To get best model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # To get best model object
            best_model=models.get(best_model_name)

            if best_model_score<0.6:
                raise CustomException('No best model found')
            
            logging.info('Best model found')
            logging.info(f'best_model_name:{best_model_name}')
            logging.info(f'best_model:{best_model}')
            logging.info(f'best_model_score:{best_model_score}')

            save_object(file_path=self.model_trainer_config.model_file_path,obj=best_model)

            y_test_pred=best_model.predict(X_test)
            
            logging.info('Model Trainer has been completed')
            
            return np.round(r2_score(y_test,y_test_pred)*100,2)

        except Exception as e:
            raise CustomException(e,sys)    
