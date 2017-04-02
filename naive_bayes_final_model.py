import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read transformed training set
df = pd.read_csv('training-set.csv')

# Split in training and test set, where testsize is 5% of the entire set
train_df, test_df = train_test_split(df, test_size = 0.05)

# Attributes used
all_columns = ['Night', 'Weekend', 'X', 'Y', 'Intersection', 'Year', 'Month', 'Day', 'Hour', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

columns = all_columns

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

weights = 'distance'

# Initiate BernoulliNB
bayes = BernoulliNB()

# Train classifier
bayes.fit(train_data, train_targets.ravel())

# Print the score of the classifier
print bayes.score(test_data, test_targets.ravel())
