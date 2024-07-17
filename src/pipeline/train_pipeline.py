from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Tranformation
from src.components.model_trainer import Model_trainer

# DataLoad
obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()

# Preprocessing
data_transformation=Data_Tranformation()
train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

# Training
model_trainer=Model_trainer()
acc_score=model_trainer.initiate_model_trainer(train_arr,test_arr)
