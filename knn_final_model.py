import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def columns():
    return ['X', 'Y', 'Intersection', 'Night', 'Year', 'Month', 'Day', 'Hour']

train_df = pd.read_csv('training-set.csv')
test_df = pd.read_csv('transformed_test.csv')

columns = columns()
class_to_predict = ['Category']

train_data = train_df.as_matrix(columns)
train_targets = train_df.as_matrix(class_to_predict)

test_data = test_df.as_matrix(columns)

scaler = StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

knn = KNeighborsClassifier(n_neighbors=600, weights='distance')

knn.fit(train_data, train_targets.ravel())

def predict(test_data):
    return knn.predict(test_data)