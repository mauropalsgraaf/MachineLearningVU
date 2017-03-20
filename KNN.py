import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('training-set.csv')

train_df, test_df = train_test_split(df, test_size = 0.05)

all_columns = ['X', 'Y', 'Intersection', 'Night']

for number_of_neighbors in [50]:
    columns = all_columns
    print str(columns) + ' ' + str(number_of_neighbors)
    class_to_predict = ['Category']

    train_data = train_df.as_matrix(columns)
    train_targets = train_df.as_matrix(class_to_predict)

    test_data = test_df.as_matrix(columns)
    test_targets = test_df.as_matrix(class_to_predict)

    weights = 'distance'

    knn = KNeighborsClassifier(n_neighbors=number_of_neighbors, weights=weights)

    knn.fit(train_data, train_targets.ravel())

    knn_score = knn.score(test_data, test_targets.ravel())

    knn_results_file = open("results.txt", "a")
    knn_results_file.write('columns: ' + str(columns) + '; number of neighbors ' + str(number_of_neighbors) + '; weights ' + weights + '; score: ' + str(knn_score) + '\n')

    knn_results_file.close()
