import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')

categories = df['Category'].drop_duplicates().values

label_encoder = LabelEncoder()

label_encoder.fit(categories)

def transform_category_to_number(category):
    return label_encoder.transform([category])[0]

def transform_number_to_category(number):
    return label_encoder.inverse_transform([number])[0]
