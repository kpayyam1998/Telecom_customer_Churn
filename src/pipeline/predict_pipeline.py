import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utills import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            script_dif=os.path.join(os.path.dirname(__file__))
            preprocessor_path=os.path.join(script_dif,"artifacts","preprocessor.pkl")
            model_path=os.path.join(script_dif,"artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise CustomException(e,sys)
    
#----------------------------------------------------------------
#  cat_columns=['gender', 'Partner', 'PhoneService', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group', 'SeniorCitizen']
#  numerical_columns=['MonthlyCharges', 'TotalCharges']
#----------------------------------------------------------------

class CustomData:
    def __init__(self,
                 gender:str,
                 Partner:str,
                 PhoneService:str,
                 OnlineBackup:str,
                 DeviceProtection:str,
                 TechSupport:str,
                 StreamingTV:str,
                 StreamingMovies:str,
                 Contract:str,
                 PaperlessBilling:str,
                 PaymentMethod:str,
                 tenure_group:str,
                 SeniorCitizen:str,
                 MonthlyCharges:float,
                 TotalCharges:float):
        self.gender = gender
        self.Partner = Partner
        self.PhoneService = PhoneService
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.tenure_group = tenure_group
        self.SeniorCitizen = SeniorCitizen
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
        
        
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    "gender": [self.gender],
                    "Partner": [self.Partner],
                    "PhoneService": [self.PhoneService],
                    "OnlineBackup": [self.OnlineBackup],
                    "DeviceProtection": [self.DeviceProtection],
                    "TechSupport": [self.TechSupport],
                    "StreamingTV": [self.StreamingTV],
                    "StreamingMovies": [self.StreamingMovies],
                    "Contract": [self.Contract],
                    "PaperlessBilling": [self.PaperlessBilling],
                    "PaymentMethod": [self.PaymentMethod],
                    "tenure_group": [self.tenure_group],
                    "SeniorCitizen": [self.SeniorCitizen],
                    "MonthlyCharges": [self.MonthlyCharges],
                    "TotalCharges": [self.TotalCharges],
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e,sys)
            
