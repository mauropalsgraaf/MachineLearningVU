import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('training-set.csv')

train_df, test_df = train_test_split(df, test_size = 0.05)
# , 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'
all_columns = ['X', 'Y', 'Intersection', 'Night', 'Year', 'Month', 'Day', 'Hour', 'Weekend']

for number_of_neighbors in [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    for nr_of_columns in [8]:
        columns = all_columns[:nr_of_columns]
        print str(columns) + ' ' + str(number_of_neighbors)
        class_to_predict = ['Category']

        train_data = train_df.as_matrix(columns)
        train_targets = train_df.as_matrix(class_to_predict)

        test_data = test_df.as_matrix(columns)
        test_targets = test_df.as_matrix(class_to_predict)

        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        weights = 'distance'

        knn = KNeighborsClassifier(n_neighbors=number_of_neighbors, weights=weights)

        knn.fit(train_data, train_targets.ravel())

        knn_score = knn.score(test_data, test_targets.ravel())

        knn_results_file = open("results.txt", "a")
        knn_results_file.write('columns: ' + str(columns) + '; number of neighbors ' + str(number_of_neighbors) + '; weights ' + weights + '; score: ' + str(knn_score) + '\n')

        knn_results_file.close()
