import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            overflow: hidden;
        }
        section.main > div {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#Loading the trained model
model=tf.keras.models.load_model('model.h5')

#loading the encoders and scalers
with open('Onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#Streamlit app

st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    **Customer Churn Prediction App**

    This application predicts whether a customer is likely to leave the bank  
    based on basic demographic and financial details.

    **Tech Stack**
    - Streamlit
    - TensorFlow
    - Scikit-learn
    """
)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to leave based on basic profile details")
st.divider()

col1, col2 = st.columns(2)

#User input

with col1:
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92)
    tenure = st.slider("📆 Tenure", 0, 10)
    num_of_products = st.slider("📦 Number of Products", 1, 4)

with col2:
    credit_score = st.number_input("💳 Credit Score")
    balance = st.number_input("💰 Balance")
    estimated_salary = st.number_input("💼 Estimated Salary")
    has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
    is_active_member = st.selectbox("⚡ Is Active Member", [0, 1])

st.divider()    

#Preparing the input data
input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine one hot encoded columns with input data 
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the imput data
input_data_scaled=scaler.transform(input_data)


#Prediction Churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.subheader("Prediction Result")
st.metric("Churn Probability", f"{prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("⚠️ High Risk: Customer is likely to churn.")
else:
    st.success("✅ Low Risk: Customer is likely to stay.")

#footer
st.markdown("---")
st.caption("📌 Built with Streamlit • TensorFlow • Scikit-learn")