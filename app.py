import streamlit
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

model = load_model('model.h5')

gender = streamlit.radio("Gender",options=['Male','Female'])
isactive = streamlit.radio("IsActive", options=['Yes','No'])
hascreditcard = streamlit.radio("HasCreditCard", options=['Yees','No'])
numberOfProducts = streamlit.slider("Number of products", min_value=1, max_value=10)
tenure = streamlit.slider("Tenure", min_value=1, max_value=10)
geography = streamlit.selectbox("Geography", ['France','Germany','Spain'])
age = streamlit.slider("Age",min_value=18,max_value=80)
salary = streamlit.number_input("Salary",min_value=0,max_value=10000000)
balance = streamlit.number_input("Balance")
credit_score = streamlit.number_input('CreditScore')

input = { 'CreditScore': credit_score,'Gender': gender, 'Age': age,
    'Tenure':tenure,'Balance': balance, 'NumOfProducts': numberOfProducts,\
        'HasCrCard':0,'IsActiveMember':0, 'EstimatedSalary':salary,     'Geography': geography}

if isactive == 'Yes':
    input['IsActiveMember']=1    
else:
    input['IsActiveMember']=0

if hascreditcard=='Yes':
    input['HasCrCard'] = 1
else:
    input['HasCrCard'] = 0

with open('label_encoder_gender.pkl','rb') as file:
    gender_encoder=pickle.load(file=file)

with open('onehot_encoded_geographies','rb') as file:
    geo_encoder=pickle.load(file=file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file=file)

input_df=pd.DataFrame([input])
streamlit.write(input_df.head())
input_df['Gender']=gender_encoder.transform(input_df['Gender'])

geotransformed = geo_encoder.transform([[input['Geography']]])
transformed_df=pd.DataFrame(data=geotransformed,columns=geo_encoder.get_feature_names_out())

final_df=pd.concat([input_df.drop(['Geography'],axis=1),transformed_df],axis=1)

streamlit.write(final_df.head())

final_df = scaler.transform(final_df)

probabilities=model.predict(final_df)

probability=probabilities[0][0]

streamlit.write(f'Churn Probability: {probability:.2f}')

if probability > 0.5:
    streamlit.write('The customer is likely to churn.')
else:
    streamlit.write('The customer is not likely to churn.')