# import libraries
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Set page configuration
st.set_page_config(page_title="Iris Classification" , layout="wide")

# Importing models
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.header("IRIS CLASSIFICATION MODEL")

# load dataset
df = pd.read_csv("dataset/iris.csv")
cols = list(df.drop(columns=['species']).columns)

# Output(y)
species = list(df['species'].unique())

st.text("This is the sample dataset!")
st.dataframe(df.head(3))

st.subheader("Select Sepal And Petal Lenght and Width")
sl = st.slider(cols[0],0.0,10.0)
sw = st.slider(cols[1],0.0,10.0)
pl = st.slider(cols[2],0.0,10.0)
pw = st.slider(cols[3],0.0,10.0)

data = [[sl,sw,pl,pw]]
scaled_data = scaler.transform(data)

y_pred = model.predict(scaled_data)
st.success(species[y_pred[0]])