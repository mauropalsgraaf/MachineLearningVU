import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('training-set.csv')

train_df, test_df = train_test_split(df, test_size = 0.05)
# , ,'Year', 'Month', 'Day', 'Hour',
# 'Weekend', 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'
all_columns = ['X', 'Y', 'Night', 'Intersection']

columns = all_columns
print str(columns)
class_to_predict = ['Category']

train_data = train_df.as_matrix(columns)
train_targets = train_df.as_matrix(class_to_predict)

test_data = test_df.as_matrix(columns)
test_targets = test_df.as_matrix(class_to_predict)

scaler = StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

weights = 'distance'

print 'before create instance'

rnn = MLPClassifier(activation="logistic", solver="adam", alpha=1e-4, hidden_layer_sizes=5, random_state=6)

print 'before fitting'
rnn.fit(train_data, train_targets.ravel())
print 'before scoring'
rnn_score = rnn.score(test_data, test_targets.ravel())
print 'before writing to file'
rnn_results_file = open("results_neuralnet.txt", "a")
rnn_results_file.write('columns: ' + str(columns) + '; hn_fst ' + str(5) + '; hn_snd ' + str(5) + '; random_state ' + str(6) + '; alpha ' + str(1e-4) + '; score: ' + str(rnn_score) + '\n')

rnn_results_file.close()
