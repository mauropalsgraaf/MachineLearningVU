import numpy as np
import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('training-set.csv')

train_df, test_df = train_test_split(df, test_size = 0.01)

all_columns = ['X', 'Y', 'Intersection', 'Night']

for radius in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    columns = all_columns
    print str(columns) + ' ' + str(radius)
    class_to_predict = ['Category']

    train_data = train_df.as_matrix(columns)
    train_targets = train_df.as_matrix(class_to_predict)

    test_data = test_df.as_matrix(columns)
    test_targets = test_df.as_matrix(class_to_predict)

    weights = 'distance'

    print 'before create instance'
    rnn = RadiusNeighborsClassifier(radius=radius, weights=weights)

    print 'before fitting'
    rnn.fit(train_data, train_targets.ravel())
    print 'before scoring'
    rnn_score = rnn.score(test_data, test_targets.ravel())
    print 'before writing to file'
    rnn_results_file = open("results.txt", "a")
    rnn_results_file.write('columns: ' + str(columns) + '; number of neighbors ' + str(radius) + '; weights ' + weights + '; score: ' + str(rnn_score) + '\n')

    rnn_results_file.close()
