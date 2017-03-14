import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

train_df = pd.read_csv('training-set.csv')
test_df = pd.read_csv('test-set.csv')

columns = ["X", "Y"]
labels = train_df["Category"].values

train_features = train_df[list(columns)].values
test_features = test_df[list(columns)].values

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(train_features, labels)

knn_score = knn.score(labels, test_features)

print("{0} -> ET: {1})".format(columns, knn_score))
