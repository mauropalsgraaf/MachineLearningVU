import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


## Read Dataframes
train_data = pd.read_csv('training-set.csv')
test_df = pd.read_csv('test.csv')

## Select target and train columns
target = train_data.iloc[:,:1]
train = train_data.iloc[:,1:]

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target.values.ravel())
print(rf.score(train,target.values.ravel()))



######
#lf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, test.iloc[:,:1], test.iloc[:,1:])
#print(scores.mean())


#scores = cross_val_score(clf, test, y)
#scores.mean()


#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y)
#scores.mean() > 0.999
