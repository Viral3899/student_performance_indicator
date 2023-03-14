import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomeException
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv') 
    test_data_path:str=os.path.join('artifacts','test.csv') 
    raw_data_path:str=os.path.join('artifacts','data.csv')




class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    


    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method pr Component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read Dataset in Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split Initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=1)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is Completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
             )


        except Exception as e:
            logging.info(f"error Occured in initiate_data_ingestion {CustomeException(e,sys)}")
            raise CustomeException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transormation=DataTransformation()
    train_arr,test_arr,_=data_transormation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))