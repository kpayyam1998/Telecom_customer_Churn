import streamlit as st
import requests 

# if st.button("Click Hello World"):
#     response=requests.get("http://127.0.0.1:5000/api/hello")
#     if response.status_code==200:
#         st.write(response.json()['message'])
#     else:
#         st.write("Failed to get response from server")

# if st.button("Add two"):
#     response=requests.post("http://127.0.0.1:5000/api/add",json={'num1':a,'num2':b})
#     if response.status_code==200:
#         st.write(f"Addition of 2 number is : {response.json()['result']}")
#     else:
#         st.write(response.json().get('error'))

st.title('Telecom Customer Churn Prediction ')

# Define input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
partner = st.selectbox('Partner', ['Yes', 'No'])
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
tenure_group = st.selectbox('Tenure Group', ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
# Senior Citizen selection with conditional assignment
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
senior_citizen = 1 if senior_citizen == 'Yes' else 0
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, format='%f')
total_charges = st.number_input('Total Charges', min_value=0.0, format='%f')





# Submit button
if st.button('Submit'):
    # Create a dictionary with the input data
    # validate all input if empty input
    try:


    # Make API request and display prediction
        employee_data = {
            "gender": gender,
            "partner": partner,
            "phone_service": phone_service,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "tenure_group": tenure_group,
            "senior_citizen": senior_citizen,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges
        }
        
        if not all(employee_data.values()):
            st.error('Please fill in all required fields.')

        else:
            import pandas as pd

            df=pd.DataFrame([employee_data])
            st.write(df)
            # Flask API endpoint
            api_endpoint = "http://127.0.0.1:5000/api/predictions"

            # Send POST request with JSON data
            response = requests.post(api_endpoint, json=employee_data)

            # Display confirmation or handle response as needed
            if response.status_code == 200:
                st.success(f'Prediction result as :{response.json()['prediction']}')
            else:
                st.error(f'Error: {response.status_code}')

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the API request: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")





