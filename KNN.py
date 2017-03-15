import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('training-set.csv')

train_df, test_df = train_test_split(df, test_size = 0.2)

columns = ['X', 'Y']
class_to_predict = ['Category']

train_data = train_df.as_matrix(columns)
train_targets = train_df.as_matrix(class_to_predict)

test_data = test_df.as_matrix(columns)
test_targets = test_df.as_matrix(class_to_predict)

knn = KNeighborsClassifier(n_neighbors=6)

print 'before_fit'
knn.fit(train_data, train_targets.ravel())

print 'after_fit'
knn_score = knn.score(test_data, test_targets.ravel())
print 'after_score'

print knn_score
