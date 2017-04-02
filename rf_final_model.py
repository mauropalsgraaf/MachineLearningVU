import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

## Read Dataframes
train_df = pd.read_csv('training-set.csv')

print "loading in df"
## Select target and train columns
target = train_df.iloc[:,:1]
train = train_df.iloc[:,1:]

print "before training"
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target.values.ravel())

print "after training"
def predict(test_data):
    return rf.predict(test_data)
