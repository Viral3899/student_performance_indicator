import sys
import os
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomeException
from src.logger import logging
from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transormation_config=DataTransformationConfig()

    def get_data_transformaer_obj(self):
        """
        This Function is Responsible to Transformation of the data

        """
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                    'gender',
                    'race_ethnicity',
                    'parental_level_of_education',
                    'lunch',
                    'test_preparation_course'
            ]

            num_pippeline=Pipeline(
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

            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                    [
                    ('num_pipeline',num_pippeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                    ]
            )

            logging.info('Numerical Columns Imputation and Scaling Completed')

            logging.info('Categorical Columns Imputation and Encoding Completed')

            return preprocessor

        except Exception as e:
            logging.info(f"error Occured in get_data_transformaer_obj {CustomeException(e,sys)}")
            raise CustomeException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
         try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reding Train/Tets Data Completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformaer_obj()

            target_column_name='math_score'
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Start applying preprocessing object on Training and Testing Dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            
            logging.info('Saving preprocessed Objects')


            save_object(
                file_path=self.data_transormation_config.preprocessor_obj_file_path,

                obj=preprocessing_obj


            )

            return(
                train_arr,
                test_arr,
                self.data_transormation_config.preprocessor_obj_file_path,
            )


         
         except Exception as e:
            logging.info(f"error Occuredd in initiate_data_transformation {CustomeException(e,sys)}")
            raise CustomeException(e,sys)
