import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


def age_feature_name(function_transformer, feature_names_in):
    return ['car_age']

def milage_feature_name(function_transformer, feature_names_in):
    return feature_names_in

def accident_feature_name(function_transformer, feature_names_in):
    return feature_names_in

def clean_title_feature_name(function_transformer, feature_names_in):
    return feature_names_in

def car_age(X: pd.DataFrame):
    return (datetime.now().year - X.iloc[:, 0]).to_frame()

def car_milage(X: pd.DataFrame):
    return X.iloc[:, 0].astype(str).str.strip('mi.').str.replace(',', '').astype(int).to_frame()

def car_accident(X: pd.DataFrame):
    return (X.iloc[:, 0] == 'None reported').astype(int).to_frame()

def car_title(X: pd.DataFrame):
    return (X.iloc[:, 0] == 'Yes').astype(int).to_frame()

def label_preprocessor(X):
    return X.apply(lambda x: x.lstrip('$').replace(',', '_')).astype(float)

age_calculator = FunctionTransformer(
    func=car_age,
    feature_names_out=age_feature_name
)
milage_transformer = FunctionTransformer(
    func=car_milage ,
    feature_names_out=milage_feature_name
)
accident_transformer = FunctionTransformer(
    func=car_accident,
    feature_names_out=accident_feature_name
)
clean_title_transformer = FunctionTransformer(
    func=car_title,
    feature_names_out=clean_title_feature_name
)

ohe_col = ['brand', 'fuel_type', 'transmission']


ohe_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False))
])

age_pipeline = Pipeline(steps=[
    ('age_transform', age_calculator),
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('car_age', age_pipeline, ['model_year']),
        ('car_mileage', milage_transformer, ['milage']),
        ('accident_record', accident_transformer, ['accident']),
        ('car_clean_title', clean_title_transformer, ['clean_title']),
        ('ohe', ohe_pipeline, ohe_col)
    ],
    remainder='drop')


from extra_data import car_brands, fuel_types, transmission_types, accident_records

def load_model():
    model = joblib.load('used_car_price_predictor.pkl')
    return model

model = load_model()

st.title("Car Price Prediction Model")
st.write("Enter the feature values to get a prediction:")

brands = car_brands
fuel_type = fuel_types
transmission = transmission_types
accident_record = accident_records


feature1 = st.selectbox('Brand', brands, placeholder="Select car Brand")

model_feature = 'fill-up data'

feature2 = st.date_input('Model Year', date.today(), min_value=date(1900, 1, 1), max_value=date.today()).year

feature3 = st.number_input('What is the Milage of the car?', min_value=1, max_value=1000000, step=1, placeholder="Enter the milage in miles")

feature4 = st.selectbox('Fuel Type', fuel_type, placeholder="Select the fuel type")

engine_feature = 'fill-up data'

feature5 = st.selectbox('Transmission', transmission, placeholder="Select the transmission type")

ext_col_feature = 'fill-up data'

int_col_feature = 'fill-up data'

feature6 = st.selectbox('Accident History', accident_record, placeholder="Tell the Accident History")

feature7 = st.selectbox('Clean Title', ['Yes', 'No'], placeholder="Does the car have a clean title?")


if st.button("Predict"):
    input_dict = {
        'brand': feature1,
        'model': model_feature,
        'model_year': feature2,
        'milage': feature3,
        'fuel_type': feature4,
        'engine': engine_feature,
        'transmission': feature5,
        'ext_col': ext_col_feature,
        'int_col': int_col_feature,
        'accident': feature6,
        'clean_title': feature7
    }

    input_data = pd.DataFrame([input_dict])

    prediction = model.predict(input_data)

    st.success(f"Predicted Value: ${prediction[0]:,.2f} ± 15,263.36")
