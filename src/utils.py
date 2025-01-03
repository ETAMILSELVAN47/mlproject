import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        pickle.dump(obj=obj,file=open(file=file_path,mode='wb'))

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params_list)->dict:
    try:
        report=dict()

        for i in range(len(models)):
            model=list(models.values())[i]

            params=params_list[list(models.keys())[i]]

            gs=GridSearchCV(estimator=model,param_grid=params,cv=5)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            ### Predict train and test
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            ### R2 Score Train and Test
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report    

    except Exception as e:
        raise CustomException(e,sys)    
    
def load_object(file_path):
    try:
        return pickle.load(file=open(file=file_path,mode='rb'))
    except Exception as e:
        raise CustomException(e,sys)