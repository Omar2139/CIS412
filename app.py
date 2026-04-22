import streamlit as st
import pandas as pd
import joblib

# Load the trained model and features
model = joblib.load('gradient_boosting_model.joblib')
model_features = joblib.load('model_features.joblib')

st.title('Customer Response Prediction App')
st.write('Enter customer details to predict their response to a campaign.')

# Create input widgets for each feature
# You will need to carefully consider the range and type of each input
# For simplicity, let's create numerical inputs for now.
# In a real app, you'd want to handle categorical features with select boxes/radio buttons
# and ensure data types match your training data.

st.header('Customer Information')

income = st.number_input('Income', min_value=0.0, value=50000.0, step=100.0)
kidhome = st.slider('Number of Kids at Home', 0, 2, 0)
teenhome = st.slider('Number of Teens at Home', 0, 2, 0)
recency = st.number_input('Days since last purchase (Recency)', min_value=0, max_value=99, value=50)
mnt_wines = st.number_input('Amount spent on Wines', min_value=0, value=100)
mnt_fruits = st.number_input('Amount spent on Fruits', min_value=0, value=10)
mnt_meat_products = st.number_input('Amount spent on Meat Products', min_value=0, value=50)
mnt_fish_products = st.number_input('Amount spent on Fish Products', min_value=0, value=10)
mnt_sweet_products = st.number_input('Amount spent on Sweet Products', min_value=0, value=10)
mnt_gold_prods = st.number_input('Amount spent on Gold Products', min_value=0, value=20)
num_deals_purchases = st.number_input('Number of purchases with a deal', min_value=0, value=2)
num_web_purchases = st.number_input('Number of web purchases', min_value=0, value=3)
num_catalog_purchases = st.number_input('Number of catalog purchases', min_value=0, value=1)
num_store_purchases = st.number_input('Number of store purchases', min_value=0, value=4)
num_web_visits_month = st.number_input('Number of web visits per month', min_value=0, value=5)
complain = st.selectbox('Has the customer complained?', [0, 1], index=0)
age = st.number_input('Age', min_value=18, max_value=90, value=40)
customer_tenure = st.number_input('Customer Tenure (days)', min_value=0, value=1000)
total_spending = mnt_wines + mnt_fruits + mnt_meat_products + mnt_fish_products + mnt_sweet_products + mnt_gold_prods
total_purchases = num_web_purchases + num_catalog_purchases + num_store_purchases

# Education and Marital Status (example for handling dummy variables)
st.header('Demographics')
education = st.selectbox('Education', ['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'])
marital_status = st.selectbox('Marital Status', ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'])

# Prepare input data for prediction
input_data = pd.DataFrame(columns=model_features)

# Initialize with zeros for all dummy variables
for col in model_features:
    input_data.loc[0, col] = 0

input_data.loc[0, 'Income'] = income
input_data.loc[0, 'Kidhome'] = kidhome
input_data.loc[0, 'Teenhome'] = teenhome
input_data.loc[0, 'Recency'] = recency
input_data.loc[0, 'MntWines'] = mnt_wines
input_data.loc[0, 'MntFruits'] = mnt_fruits
input_data.loc[0, 'MntMeatProducts'] = mnt_meat_products
input_data.loc[0, 'MntFishProducts'] = mnt_fish_products
input_data.loc[0, 'MntSweetProducts'] = mnt_sweet_products
input_data.loc[0, 'MntGoldProds'] = mnt_gold_prods
input_data.loc[0, 'NumDealsPurchases'] = num_deals_purchases
input_data.loc[0, 'NumWebPurchases'] = num_web_purchases
input_data.loc[0, 'NumCatalogPurchases'] = num_catalog_purchases
input_data.loc[0, 'NumStorePurchases'] = num_store_purchases
input_data.loc[0, 'NumWebVisitsMonth'] = num_web_visits_month
input_data.loc[0, 'Complain'] = complain
input_data.loc[0, 'Age'] = age
input_data.loc[0, 'Customer_Tenure'] = customer_tenure
input_data.loc[0, 'Total_Spending'] = total_spending
input_data.loc[0, 'Total_Purchases'] = total_purchases

# Set dummy variables based on selection
if education != 'Graduation': # Graduation is the dropped first column
    edu_col = f'Education_{education}'
    if edu_col in input_data.columns:
        input_data.loc[0, edu_col] = 1

if marital_status != 'Alone': # Assuming 'Alone' or similar was dropped, adjust if different
    marital_col = f'Marital_Status_{marital_status}'
    if marital_col in input_data.columns:
        input_data.loc[0, marital_col] = 1

# Ensure all columns are in the correct order and type
input_data = input_data[model_features].astype(float)

# Make prediction
if st.button('Predict Response'):
    prediction_proba = model.predict_proba(input_data)[:, 1][0] # Probability of class 1
    prediction = model.predict(input_data)[0]

    st.subheader('Prediction Results')
    if prediction == 1:
        st.success(f'Customer is likely to respond! (Probability: {prediction_proba:.2f})')
    else:
        st.info(f'Customer is unlikely to respond. (Probability: {prediction_proba:.2f})')

    st.write('---')
    st.write('### Feature Importances (Top 10)')
    # This part requires having feature importances available, 
    # which we can get from the trained model if it supports it (like GradientBoostingClassifier)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': model_features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)
        st.bar_chart(importance_df.set_index('Feature'))
