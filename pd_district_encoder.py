import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')

districts = df['PdDistrict'].drop_duplicates().values

label_encoder = LabelEncoder()

label_encoder.fit(districts)

def transform_district_to_number(district):
    return label_encoder.transform([district])[0]

def transform_number_to_district(number):
    return label_encoder.inverse_transform([number])[0]
