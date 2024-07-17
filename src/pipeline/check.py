import os 
from src.utills import load_object
def predict(features):
            script_dif=os.path.join(os.path.dirname(__file__))
            preprocessor_path=os.path.join(script_dif,"artifacts","preprocessor.pkl")
            model_path=os.path.join(script_dif,"artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            # model=load_object(model_path)
            
            # scaled_data=preprocessor.transform(features)
            
            # pred=model.predict(scaled_data)
            
            return preprocessor_path


if __name__=="__main__":
    df=predict("kpop")
    print(df)