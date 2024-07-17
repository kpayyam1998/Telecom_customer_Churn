import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.utills import save_object,evaluate_model
from src.logger import logging

@dataclass
class  ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_train_file_path=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1] ,
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = [
                ("Logistic Regression", LogisticRegression()),
                ("Support Vector Machines (SVM)", SVC()),
                ("Random Forest", RandomForestClassifier()),
                ("Gradient Boosting Machines (XGBoost)", XGBClassifier()),
                ("K-Nearest Neighbors (KNN)", KNeighborsClassifier()),
                ("Decision Trees", DecisionTreeClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Extra Trees", ExtraTreesClassifier()),
                ("Ensemble Methods (AdaBoost)", AdaBoostClassifier()),
                (
                    "Ensemble Methods (Voting Classifier)",
                    VotingClassifier(
                        estimators=[
                            ("lr", LogisticRegression()),
                            ("svm", SVC()),
                            ("rf", RandomForestClassifier()),
                        ]
                    ),
                ),
                (
                    "Ensemble Methods (Bagging)",
                    BaggingClassifier(base_estimator=DecisionTreeClassifier()),
                ),
                ("Ensemble Methods (Gradient Boosting)", GradientBoostingClassifier()),
            ]
            
            #Hyperparameter Tuning

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_accuracy,final_model=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models) # param=params
           
            
            
            # Saveing best_model as pkl file
            save_object(file_path=self.model_train_file_path.trained_model_path,obj=final_model)

            logging.info("Best model founded")

            logging.info(f"Model accuracy {model_accuracy}")

            return model_accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
        