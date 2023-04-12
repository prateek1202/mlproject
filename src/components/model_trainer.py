import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

sys.path.append('src/')
from exception import CustomException
from logger import logging
from utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('slicing over. models inintialized')
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            logging.info("evaluate models begins")
            # model_report: dict = evaluate_models(x_train = x_train, y_train = y_train , x_test = x_test, y_test = y_test, models = models)
            model_report:dict = evaluate_models(x_train = X_train, x_test = X_test, y_train = y_train, y_test = y_test, models= models)
            
            logging.info("To get the best model score from dict")
            best_model_score = max(sorted(model_report.values()))
            
            logging.info("To get best model name from dict")
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score< 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing datasets")
            
            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                )
            predicted = best_model.predict(X_test)
            
            r2_score_value = r2_score(y_test,predicted)
            return r2_score_value

        except Exception as e:
            CustomException(e, sys)
            