import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
np.random.seed(42)

def fetch_details(df):
    return list(sorted(df.name.unique())),list(df.fuel.unique()),list(df.seller_type.unique()),list(df.transmission.unique()),list(df.owner.unique())

df=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

df=df[~df.duplicated()]
df=df.reset_index(drop=True)
df=df[df.km_driven<600000]

#Creating the data to be used during prediction
name,fuel,seller_type,transmission,owner=fetch_details(df)
year=list(range(1995,datetime.now().year+1))

    # creation of features
df['age']=datetime.now().year-df.year
df['brand']=df.name.str.split(' ').str[0]

    # label encoding 
le={}
for col in ['fuel','seller_type','transmission','owner']:
    le[col]=LabelEncoder()
    df[col]=le[col].fit_transform(df[col])

    # target_encoding on brand and name 
idx1=df.groupby(['name'])['selling_price'].mean().sort_values().index
dict1={key:index for index,key in enumerate(idx1,0)}
idx2=df.groupby(['brand'])['selling_price'].mean().sort_values().index
dict2={key:index for index,key in enumerate(idx2,0)}
df['name']=df['name'].map(dict1)
df['brand']=df['brand'].map(dict2)

st.title('Car Worth Discovery: Find Out the Market Value of Your Vehicle')
selected_name=st.selectbox('Select the Model of Your Car',name)
selected_year=st.selectbox('Enter the Purchase Year of Car',year)
Entered_Km_driven=st.slider('Number of Kilometers Driven',0,600000,100)
selected_fuel_type=st.selectbox('Fuel type of your Vehicle',fuel)
selected_seller=st.selectbox('Enter the seller_type',seller_type)
selected_transmission=st.selectbox('Gearbox type',transmission)
selected_owner=st.selectbox('Owner Status',owner)

#code for applying the preprocessing steps on the dataset of the vehicle that is to be used for prediction
data = {
    'name': [selected_name],
    'year': [selected_year],
    'km_driven': [Entered_Km_driven],
    'fuel': [selected_fuel_type],
    'seller_type': [selected_seller],
    'transmission': [selected_transmission],
    'owner': [selected_owner]
}

d=pd.DataFrame(data)
d['age']=datetime.now().year-d.year
d['brand']=d.name.str.split(' ').str[0]
d['name']=d['name'].map(dict1)
d['brand']=d['brand'].map(dict2)
for col in ['fuel','seller_type','transmission','owner']:
    d[col]=le[col].transform(d[col])

#Training the model
model=RandomForestRegressor()
x=df.drop(columns=['selling_price'],axis=1)
y=df[['selling_price']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)



st.markdown(
    """
    <style>
    .stButton>button {
        width: 200px;
        margin: auto;
        display: block;
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green */
    }
    .stButton>button:active {
        background-color: #3e8e41; /* On click */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def display_output(prediction):
    st.markdown("<h1 style='text-align: center; color: silver;'>Your Vehicle's Resale Value is</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; font-size: 48px; color: silver;'>{prediction}</h2>", unsafe_allow_html=True)


if st.button('Click to Predict'):
    display_output(str(np.round(model.predict(d)[0],2)))
    st.snow()
    