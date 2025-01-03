import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has been started')
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            
            logging.info('Exported the Raw data file')
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            
            # Train Test Split
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            logging.info('Exported the Train and Test data file')
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Data Ingestion has been completed')
            return self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e,sys)        
        

if __name__=='__main__':

    logging.info('=========================================================')
    data_ingestion=DataIngestion()   
    train_file_path,test_file_path=data_ingestion.initiate_data_ingestion() 

    logging.info('=========================================================')
    data_transformation=DataTransformation()
    train_arr,test_arr,preprocessor_obj_file_path=data_transformation.initiate_data_transformation(
                                                     train_file_path=train_file_path,
                                                     test_file_path=test_file_path)   
    logging.info('=========================================================') 
    model_trainer=ModelTrainer()
    r2_score=model_trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
    print(f'R2 score:{r2_score}')
    

    

