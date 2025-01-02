import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle



def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        pickle.dump(obj=obj,file=open(file=file_path,mode='wb'))

    except Exception as e:
        raise CustomException(e,sys)