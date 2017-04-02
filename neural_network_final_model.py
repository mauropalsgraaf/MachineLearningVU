import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read transformed training-set
df = pd.read_csv('training-set.csv')

# Split in training and test set, where testsize is 5% of the entire set
train_df, test_df = train_test_split(df, test_size = 0.05)

# Attributes used to classify
all_columns = ['X', 'Y', 'Night', 'Intersection']

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

# Instantiate classifier
rnn = MLPClassifier(activation="logistic", solver="adam", alpha=1e-4, hidden_layer_sizes=5, random_state=6)

# Train classifier
rnn.fit(train_data, train_targets.ravel())

# Get the accuracy of the classifier
rnn_score = rnn.score(test_data, test_targets.ravel())

print rnn_score
