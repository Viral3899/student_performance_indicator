import os
import sys

import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomeException


def save_object(file_path,obj):

    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)


    except Exception as e:
        logging.info(f"error Occuredd in save_object {CustomeException(e,sys)}")
        raise CustomeException(e,sys)
