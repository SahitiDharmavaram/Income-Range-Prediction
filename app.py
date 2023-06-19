import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv('encoded_dataset.csv')

loaded_model = pickle.load(open("trained_model.sav", 'rb'))

classes = {0: 'Income lower than 50k', 1: 'Income greater than or equal to 50K'}

class_labels = list(classes.values())

@st.cache_data
def predict_income(user_choices):
    input_data = {
        'age': [user_choices['age']],
        'workclass': [user_choices['workclass']],
        'fnlwgt': [1],  # Set fnlwgt value to 1
        'education': [user_choices['education']],
        'educational-num': [user_choices['educational-num']],
        'marital-status': [user_choices['marital-status']],
        'occupation': [user_choices['occupation']],
        'relationship': [user_choices['relationship']],
        'gender': [user_choices['gender']],
        'hours-per-week': [user_choices['hours-per-week']],
        'native-country': [user_choices['native-country']]
    }

    test_df = pd.DataFrame(input_data)

    predicted_income = loaded_model.predict(test_df)

    return predicted_income


def main():
    st.title("Income Range Predictor")

    mappings = {
        'age': None,
        'workclass': {'Private': 0, 'Self-emp-not-inc': 1, 'Local-gov': 2, 'State-gov': 3, 'Self-emp-inc': 4, 'Federal-gov': 5, 'Without-pay': 6, 'Never-worked': 7},
        'education': {'HS-grad': 0, 'Some-college': 1, 'Bachelors': 2, 'Masters': 3, 'Assoc-voc': 4, '11th': 5, 'Assoc-acdm': 6, '10th': 7, '7th-8th': 8, 'Prof-school': 9, '9th': 10, '12th': 11, 'Doctorate': 12, '5th-6th': 13, '1st-4th': 14, 'Preschool': 15},
        'educational-num': None,
        'marital-status': {'Married-civ-spouse': 0, 'Never-married': 1, 'Divorced': 2, 'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6},
        'occupation': {'Craft-repair': 0, 'Prof-specialty': 1, 'Exec-managerial': 2, 'Adm-clerical': 3, 'Sales': 4, 'Other-service': 5, 'Machine-op-inspct': 6, 'Transport-moving': 7, 'Handlers-cleaners': 8, 'Farming-fishing': 9, 'Tech-support': 10, 'Protective-serv': 11, 'Priv-house-serv': 12, 'Armed-Forces': 13},
        'relationship': {'Husband': 0, 'Not-in-family': 1, 'Own-child': 2, 'Unmarried': 3, 'Wife': 4, 'Other-relative': 5},
        'gender': {'Male': 0, 'Female': 1},
        'hours-per-week': None,
        'native-country': {'United-States': 0, 'Other': 1},
    } 

    user_choices = {}

    for question, mapping in mappings.items():
        if mapping is None:
            if question == 'age':
                default_value = int(df['age'].mean())
                user_choice = st.number_input(question, value=default_value, min_value=18)
            elif question == 'educational-num':
                default_value = int(df['educational-num'].mean())
                user_choice = st.number_input(question, value=default_value)
            elif question == 'hours-per-week':
                default_value = int(df['hours-per-week'].mean())
                user_choice = st.number_input(question, value=default_value)
        else:
            options = list(mapping.keys())
            default_option = [k for k, v in mapping.items() if v == 0][0]
            selected_option = st.selectbox(question, options, index=options.index(default_option), key=question)
            user_choice = mapping.get(selected_option)
        user_choices[question] = user_choice
        

    if st.button("Predict"):
        
        predicted_income = predict_income(user_choices)

        st.write("Prediction:")
        if predicted_income[0] == [0]:
            st.write("Income range: <=50k")
        elif predicted_income[0] == [1]:
            st.write("Income range: >50k")


if __name__ == '__main__':
    main()