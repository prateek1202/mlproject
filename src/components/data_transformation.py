import sys
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

sys.path.append('src/')
from exception import CustomException
from logger import logging
from utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifact","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_trasnformation_config = DataTransformationConfig()
    
    def get_data_transformer(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # numerial_columns = ["writing_score","reading_score"]
            # categorical_columns = [
            #     "gender",
            #     "race_ethinicity",
            #     "parental_level_of_education",
            #     "lunch",
            #     "test_preparation_course"
            # ]
            # num_pipeline = Pipeline(
            #     steps=[
            #         ("imputer", SimpleImputer(strategy="median")),
            #         ("scaler", StandardScaler())
            #     ]
            # )
            # cat_pipeline = Pipeline(
            #     steps = [
            #         ("imputer",SimpleImputer(strategy="most-frequent")),
            #         ("one_hot_encoder",OneHotEncoder()),
            #         ("scaler",StandardScaler())
            #     ]
            # )
            
            # logging.info("Categorical columns encoding completed.")
            # logging.info("Numerical columns encoding completed.")
            
            # preprocessor = ColumnTransformer([
            #     ("num_pipeline", num_pipeline, numerial_columns),
            #     ("cat_pipeline", cat_pipeline, categorical_columns),
            # ])
            # return preprocessor
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        except Exception as e:
            CustomException(e, sys)
        
    def init_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("train test data reading completed.")
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer()
            
            target_column = "math_score"
            numerial_columns = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns = [target_column],axis = 1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns = [target_column],axis = 1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying preprocessing object on traning dataframe and testing dataframe.")
            logging.info("preprocessing step begins")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("preprocessing step done")
            logging.info("np.c_ setp begins")
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            logging.info("np.c_ setp done")
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("np.c_ setp done")
            
            logging.info("Saved preprocessing object")
            
            save_obj(
                file_path = self.data_trasnformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj)
            return (
                train_arr,
                test_arr,
                self.data_trasnformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            CustomException(e, sys)