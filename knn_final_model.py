import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def columns():
    return ['X', 'Y', 'Intersection', 'Night', 'Year', 'Month', 'Day', 'Hour']

# Read transformed training set
df = pd.read_csv('training-set.csv')

# Split in training and test set, where testsize is 5% of the entire set
train_df, test_df = train_test_split(df, test_size = 0.05)

# Use the columns specified in the columns function
columns = columns()
class_to_predict = ['Category']

# Transform to matrix for scikit learn
train_data = train_df.as_matrix(columns)
train_targets = train_df.as_matrix(class_to_predict)

test_data = test_df.as_matrix(columns)
test_targets = test_df.as_matrix(class_to_predict)

# Feature scaling
scaler = StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Initiate the classifier
knn = KNeighborsClassifier(n_neighbors=600, weights='distance')

# Train the classifier
knn.fit(train_data, train_targets.ravel())

# Get the accuracy of the classifier
print knn.score(test_data, test_targets.ravel())

def predict(test_data):
    test_data = scaler.transform(test_data)
    return knn.predict(test_data)
