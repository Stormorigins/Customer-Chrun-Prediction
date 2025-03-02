import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import tensorflow as tf 

#model loading
model=tf.keras.models.load_model("Model.h5") 

#Scaler and encoder loading
with open("Onehot_encoder_geo.pkl","rb") as file:
    Onehot_encoder_geo=pickle.load(file)
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender=pickle.load(file)
with open("Scalar.pkl","rb") as file:
    Scalar=pickle.load(file)

#streamlit app

st.title("Customer Chrun Prediction")

#codes in predic data file will implement to app.py

#user input
geography=st.selectbox("Geography",options=["France","Germany","Spain"])
gender=st.selectbox("Gender",options=["Male","Female"])
col1,col2=st.columns(2)
with col1:
    age=st.slider("Age",18,92)
with col2:
    tenure=st.slider("Tenure",0,10)
Balance=st.number_input("Balance")
Credit=st.number_input("Credit Score")
est=st.number_input("Estimated salary")
noofpro=st.slider("Number of Product",0,4)
HasCC=st.selectbox("Has Credit Card",options=[0,1])
Active=st.selectbox("Active member",options=[0,1])

#covert all the inputo dataframe
input_data = pd.DataFrame({
    'CreditScore': [Credit],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [Balance],
    'NumOfProducts': [noofpro],
    'HasCrCard': [HasCC],
    'IsActiveMember': [Active],
    'EstimatedSalary': [est]
})

#implement OHE,label and scaler

ohe=Onehot_encoder_geo.transform([[geography]]).toarray()
df_ohe=pd.DataFrame(ohe,columns=Onehot_encoder_geo.get_feature_names_out(["Geography"]))

df=pd.concat([input_data,df_ohe],axis=1)

df["Gender"]=label_encoder_gender.transform(df["Gender"])
df.drop("Geography",axis=1,inplace=True)

df_scaler=Scalar.transform(df)

# Predict churn
predic=model.predict(df_scaler)

st.write(f'Churn Probability: {predic}')
if predic[0][0] > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
