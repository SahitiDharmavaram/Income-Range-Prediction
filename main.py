import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from PIL import Image

data = pd.read_csv('updated_dataset.csv')

# Separate the features and target variable
X = data.drop('income', axis=1)
y = data['income']

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Load the pre-trained model
pickle_in = open("E:\VSCode_workshop\python_workshop\HackStory\model.pkl", "rb")
model = pickle.load(pickle_in)

classes = {0:'Income lower than 50k', 1:'Income greater than or equal to 50K'}

class_labels = list(classes.values())

st.title('Income Range Predictor')

st.markdown('This is the user interface of an ML model of a machine learning algorithm to predict job applicants income range based on their demographic and professional information.')
st.markdown('Please enter your details')

def predict_income(age, workclass, fnlwgt, education, marital_status, occupation,
                   relationship, race, gender, hours_per_week, native_country):
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country],
    })  
    # Encode the input data
    input_encoded = input_data.apply(label_encoder.transform)

    # Make prediction
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)
    print(prediction)
    return prediction[0], prediction_proba[0]

def main():
    age = st.number_input('Age', min_value=0, max_value=100)
    workclass = st.selectbox('Workclass', X['workclass'].unique())
    education = st.selectbox('Education', X['education'].unique())
    marital_status = st.selectbox('Marital Status', X['marital-status'].unique())
    occupation = st.selectbox('Occupation', X['occupation'].unique())
    relationship = st.selectbox('Relationship', X['relationship'].unique())
    race = st.selectbox('Race', X['race'].unique())
    gender = st.selectbox('Gender', X['gender'].unique())
    hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=100)
    native_country = st.selectbox('Native Country', X['native-country'].unique())

    if st.button('Predict'):
        prediction, prediction_proba = predict_income(age, workclass, education, marital_status, occupation, relationship, race, 
                                                      gender, hours_per_week, native_country) #?

        # Display the prediction
        st.header('Prediction')
        st.write(f'Income class: {prediction}')
        st.write('Prediction probabilities:')
        st.write(prediction_proba)

if __name__ == '__main__':
    main()