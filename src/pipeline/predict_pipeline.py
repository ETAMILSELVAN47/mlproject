from src.exception import CustomException
import pandas as pd
import sys
from src.utils import load_object
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformationConfig
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_file_path:str=ModelTrainerConfig().model_file_path
        self.preprocessor_file_path:str=DataTransformationConfig().preprocessor_obj_file_path

    def predict(self,data:pd.DataFrame):
        try:
            model=load_object(file_path=self.model_file_path)
            preprocessor=load_object(file_path=self.preprocessor_file_path)
            print(f'data:{data}')
            data_scaled=preprocessor.transform(data)
            predicted=model.predict(data_scaled)
            return predicted
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:float,
                 writing_score:float):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_dataframe(self)-> pd.DataFrame:
        try:
            custom_dict={'gender':[self.gender],
                         'race_ethnicity':[self.race_ethnicity],
                         'parental_level_of_education':[self.parental_level_of_education],
                         'lunch':[self.lunch],
                         'test_preparation_course':[self.test_preparation_course],
                         'reading_score':[self.reading_score],
                         'writing_score':[self.writing_score]}
            
            return pd.DataFrame(data=custom_dict)
        except Exception as e:
            raise CustomException(e,sys) 

