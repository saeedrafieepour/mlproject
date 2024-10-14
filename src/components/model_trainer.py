import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import(

    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:

            logging.info('Splitting train and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-neighbors" : KNeighborsRegressor(),
                "Xgboost" : XGBRegressor(),
                "Catboosting" : CatBoostRegressor(),
                "AdaBoost" : AdaBoostRegressor()
            
            }
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               models=models)
            
            best_model_score = max(sorted(model_report.values()))

            #best_model_name = list(filter(lambda x: model_report[x]==best_model_score, model_report))
            best_model_name = list(model_report.keys())[
                                        list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found!")
            
            logging.info("Best model on both trainig and test datasets!")

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info("Best model object was saved!")
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)