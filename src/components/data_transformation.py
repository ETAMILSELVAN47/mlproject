import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(
                          steps=[
                              ('imputer',SimpleImputer(strategy='median')),
                              ('scaler',StandardScaler())
                              ]
                              )
            cat_pipeline=Pipeline(
                         steps=[
                             ('imputer',SimpleImputer(strategy='most_frequent')),
                             ('one_hot_encoder',OneHotEncoder()),
                             ('scaler',StandardScaler(with_mean=False))
                         ]
            )

            preprocessor=ColumnTransformer(transformers=
                                           [
                                            ('num_pipeline',num_pipeline,numerical_columns),
                                            ('cat_pipeline',cat_pipeline,categorical_columns)
                                           ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    
    def initiate_data_transformation(self,train_file_path,test_file_path):
        logging.info('Data Transformation has been started')
        try:
            logging.info('Read the Train and Test Data file')
            train_df=pd.read_csv(train_file_path)
            test_df=pd.read_csv(test_file_path)
            
            logging.info('Obtain the Processor obj file')
            preprocessing_obj=self.get_data_transformer_object()

            target_feature=['math_score']

            input_feature_train_df=train_df.drop(columns=target_feature)
            target_feature_train_df=train_df[target_feature]

            input_feature_test_df=test_df.drop(columns=target_feature)
            target_feature_test_df=test_df[target_feature]
           
            logging.info('Transform the Train and Test data')
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info('Save the Processor obj file')
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            logging.info('Data Transformation has been completed')

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)