import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
 

from matplotlib import style
style.use("seaborn")
from IPython.display import HTML
import plotly.express as px


st.write("## House Price Prediction ")
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSiGkaHRFu6n1RIqDTi8cmUNWQ4oNbHlRN36dwu2fbr6g&usqp=CAU&ec=48665701")
st.write("Well, house prices are an important reflection of the economy. The price of a property is important in real estate transactions as it provides information to stakeholders, including real estate agents, investors, and developers, to enable them to make informed decisions.Governments also use such information to formulate appropriate regulatory policies. Overall, it helps all parties involved to determine the selling price of a house. With such information, they will then decide when to buy or sell a house. We will use machine learning with Python to try to predict the price of a house. ")

option = st.selectbox('Contents',('Home' , 'Prediction' , 'Model'))

def user_input_features():
    global longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity
    longitude = st.sidebar.number_input("Enter the longitude values in negative number : ")
    latitude = st.sidebar.number_input("Enter the latitude values in positive number : ")
    housing_median_age = st.sidebar.number_input("Enter the median age of the building : ")
    total_rooms = st.sidebar.number_input("Enter the number of rooms in the house : ")
    total_bedrooms = st.sidebar.number_input("Enter the number of bedrooms in the house : ")
    population = st.sidebar.number_input("Population of the people within the block : ")
    households = st.sidebar.number_input("Population of a household : ")
    median_income = st.sidebar.number_input("Median income of a household in Dollars : ")
    ocean_proximity = st.sidebar.selectbox('How close to the sea is the house?', ('<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'))
    data = {
    "longitude" : longitude,
    "latitude" : latitude,
    "housing_median_age" : housing_median_age,
    "total_rooms" : total_rooms,
    "total_bedrooms" : total_bedrooms,
    "population" : population,
    "households" : households,
    "median_income" : median_income,
    "ocean_proximity" : ocean_proximity
    }

    data_model = {
    "longitude" : longitude,
    "latitude" : latitude,
    "housing_median_age" : housing_median_age,
    "total_rooms" : total_rooms,
    "total_bedrooms" : total_bedrooms,
    "population" : population,
    "households" : households,
    "median_income" : median_income,
    "ocean_proximity" : ocean_proximity
    }
    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data, index=[0])
    return features , data

df , data = user_input_features()
st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

data1 = pd.read_csv("housing.csv")
data1 = data1[:1000]
label_encoder = LabelEncoder()
obj = (data1.dtypes == 'object')
for col in list(obj[obj].index):
    data1[col] = label_encoder.fit_transform(data1[col])

for col in data1.columns:
    data1[col] = data1[col].fillna(data1[col].mean())

x = data1.drop(['median_house_value'], axis=1)
y = data1.median_house_value
x = x.values
y = y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

std = StandardScaler()
scaler = std.fit(x_train)
rescaledx = scaler.transform(x_train)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

ocean_proximity = 0 if ocean_proximity == '<1H OCEAN' else 1 if ocean_proximity == 'INLAND' else 2 if ocean_proximity == 'ISLAND' else 3 if ocean_proximity == 'NEAR BAY' else 4 
med_income = median_income / 5
lists = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, med_income, ocean_proximity]
df = pd.DataFrame(lists).transpose()
scaler.transform(df)
prediction = model.predict(df)

def predict(df):
    st.header("Prediction : ")
    latest_iteration = st.empty()
    bar=st.progress(0)
    for i in range(100):
        bar.progress(i+1)
        time.sleep(0.01)
    df = pd.DataFrame(lists).transpose()
    result = int(prediction)
    st.write("The value of the house is : $ ",result)

def model(dataframe):
    st.title("Linear Regression")
    st.write("Linear regression is one of the easiest and most popular Machine Learning algorithms. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables. Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) variables, hence called as linear regression. Since linear regression shows the linear relationship, which means it finds how the value of the dependent variable is changing according to the value of the independent variable.")
    st.write("LinearRegression Mean Absolute Error(MAE) :" ,round(metrics.mean_absolute_error(y_test,y_pred),2))
    st.write("LinearRegression Mean Squared Error(MSE) : ",round(metrics.mean_squared_error(y_test,y_pred),2))

import warnings
warnings.filterwarnings('ignore')

if  option == 'Prediction':
  predict(df)
elif option =='Model':
  model(df)