import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

rawdata = pd.read_csv('income_train.csv')

rawdata = rawdata.drop(['signature_id', 'capital-gain', 'capital-loss'], axis=1)

rawdata.replace('?', np.nan, inplace=True)
for column in rawdata.columns:
    if rawdata[column].isnull().sum() > 0:
        if rawdata[column].dtype == 'object':
            mode = rawdata[column].mode()[0]
            rawdata[column].fillna(mode, inplace=True)
        else:
            mean = rawdata[column].mean()
            rawdata[column].fillna(mean, inplace=True)


rawdata.to_csv('updated_dataset.csv', index=False)

data = pd.read_csv('updated_dataset.csv')

# Biases Analysis - Race
race_counts = data['race'].value_counts()
race_percentages = race_counts / len(data) * 100
print("Race Distribution:")
print(race_percentages)

# Biases Analysis - Gender
gender_counts = data['gender'].value_counts()
gender_percentages = gender_counts / len(data) * 100
print("\nGender Distribution:")
print(gender_percentages)

# Replace letters with NaN and then Average
data['educational-num'] = pd.to_numeric(
    data['educational-num'], errors='coerce')

average = data['educational-num'].mean()
round_avg = round(average)

data['educational-num'].fillna(int(round_avg), inplace=True)

data.to_csv('updated_dataset.csv', index=False)

data_updt = pd.read_csv('updated_dataset.csv')

threshold = 0.1
total_count = len(data_updt)
count_threshold = total_count * threshold
country_distribution = data_updt['native-country'].value_counts() / total_count
rare_categories = country_distribution[country_distribution < threshold].index.tolist()
data_updt['native-country'] = data_updt['native-country'].apply(lambda x: 'Other' if x in rare_categories else x)

data_updt.to_csv('updated_dataset.csv', index=False)

label_encoder = LabelEncoder()

mappings = {}

for column in data_updt.columns:
    if data_updt[column].dtype == 'object':
        value_counts = data_updt[column].value_counts()
        sorted_values = value_counts.index.tolist()
        encoded_values = label_encoder.fit_transform(sorted_values)
        
        # Create the mapping dictionary with keys in descending order of frequencies and values in ascending order
        mapping = dict(zip(sorted_values, range(len(sorted_values))))
        
        data_updt[column] = data_updt[column].map(mapping)
        mappings[column] = mapping

data_updt.to_csv('encoded_dataset.csv', index=False)

df = pd.read_csv('encoded_dataset.csv')

X = df.drop(['income', 'race'], axis=1)
y = df['income']

# Compute weights based on 'fnlwgt'
weights_fnlwgt = X['fnlwgt'] / X['fnlwgt'].sum()

# Calculate weights based on 'gender' column
gender_counts = X['gender'].value_counts()
total_samples = len(X)
weights_gender = total_samples / (2 * gender_counts)

weights_combined = []

for index, row in X.iterrows():
    if row['gender'] == 0:
        weight = weights_fnlwgt[index] * weights_gender[0]
    elif row['gender'] == 1:
        weight = weights_fnlwgt[index] * weights_gender[1]
    weights_combined.append(weight)

weights_combined = pd.DataFrame(weights_combined, columns=['weights'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
sample_weights_resampled, _ = smote.fit_resample(weights_combined, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
sample_weights_train, sample_weights_test = train_test_split(sample_weights_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight = 'balanced')
model.fit(X_train, y_train, sample_weight=sample_weights_train.values.ravel())

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.4f}%')

classes = {0:'Income lower than 50k', 1:'Income greater than or equal to 50K'}

class_labels = list(classes.values())

test_input = {
    'age': [43],
    'workclass': [3],
    'fnlwgt': [1],
    'education': [15],
    'educational-num': [13.0],
    'marital-status': [2],
    'occupation': [11],
    'relationship': [0],
    'gender': [1],
    'hours-per-week': [45],
    'native-country': [38]
}

# Test input dictionary to a DataFrame
test_df = pd.DataFrame(test_input)  # Test input to be replaced with user input

predicted_income = model.predict(test_df)

if predicted_income == [0]:
    print("\nIncome range for test input: <=50k")
elif predicted_income == [1]:
    print("\nIncome range for test input: >50k")

import pickle
filename = "trained_model.sav"
pickle.dump(model, open(filename, 'wb'))