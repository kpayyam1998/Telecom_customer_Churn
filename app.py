from flask import Flask,jsonify,request
app = Flask(__name__)



# Get the data from frontend convert to dataframe & perform preprocessing then predict

# Hello world
@app.route('/api/hello', methods=['GET'])
def hello_world():
    return jsonify(message='Hello, World!')


# We need to write POST other api with exception and logs 

# @app.route('/api/add',methods=['POST'])
# def add():
#     data=request.get_json()
#     num1=data['num1']
#     num2=data['num2']
    
#     if num1 is None or num2 is None:
#         return jsonify(error="Missing required parameters"),400
    
#     # if not isinstance(num1,int) or not isinstance(num2,int):
#     #     return jsonify(error="Invalid parameters"),400
#     try:
#         result=int(num1)+int(num2)
#         return jsonify(result=result)
#     except ValueError :#Exception as e:
#         return jsonify(error="Number is right format"),400
    

#     #return jsonify(result=result)

# Prediction
@app.route('/api/predictions',methods=['POST'])
def prediction():
    from src.pipeline.predict_pipeline import CustomData,PredictPipeline
    # TODO: Implement prediction logic

    try:
        # Get Json data from input
        data=request.get_json()

        # Validate the input data
        required_fields = [
            'gender', 'partner', 'phone_service', 'online_backup', 'device_protection', 
            'tech_support', 'streaming_tv', 'streaming_movies', 'contract', 
            'paperless_billing', 'payment_method', 'tenure_group', 'senior_citizen', 
            'monthly_charges', 'total_charges'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify(error=f"Missing required field: {field}"), 400

        df=CustomData(
            gender=data['gender'],
            Partner=data['partner'],
            PhoneService=data['phone_service'],
            OnlineBackup=data['online_backup'],
            DeviceProtection=data['device_protection'],
            TechSupport=data['tech_support'],
            StreamingTV=data['streaming_tv'],                    
            StreamingMovies=data['streaming_movies'],
            Contract=data['contract'],
            PaperlessBilling=data['paperless_billing'],
            PaymentMethod=data['payment_method'],
            tenure_group=data['tenure_group'],
            SeniorCitizen=data['senior_citizen'],
            MonthlyCharges=data['monthly_charges'],
            TotalCharges=data['total_charges']
        )
        # Convert data to dataframe
        final_df=df.get_data_as_dataframe()

        # Predcition
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_df)

        result=round(pred[0],2)

        return jsonify(prediction=result),200
    except KeyError as e:
        return jsonify(error=f"Invalid data format: {str(e)}"), 400
    except ValueError as e:
        return jsonify(error=f"Value Error: {str(e)}"), 400
    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}"), 500


if __name__ == '__main__':
    app.run(debug=True)


    