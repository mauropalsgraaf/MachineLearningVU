import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('training-set.csv')

train_df, test_df = train_test_split(df, test_size = 0.05)
#
all_columns = ['Night', 'Weekend', 'X', 'Y', 'Intersection', 'Year', 'Month', 'Day', 'Hour', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

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

bayes = BernoulliNB()

bayes.fit(train_data, train_targets.ravel())

bayes_score = bayes.score(test_data, test_targets.ravel())

bayes_results_file = open("results_bayes.txt", "a")
bayes_results_file.write('columns: ' + str(columns) + '; score: ' + str(bayes_score) + '\n')

bayes_results_file.close()
