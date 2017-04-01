import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


## Read Dataframes
train_data = pd.read_csv('training-set.csv')


train_df, test_df = train_test_split(train_data, test_size = 0.05)

## Select target and train columns
target = train_df.iloc[:,:1]
train = train_df.iloc[:,1:]


test_df_target = test_df.iloc[:,:1]
test_df_set = test_df.iloc[:,1:]

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target.values.ravel())
print(rf.score(test_df_set,test_df_target.values .ravel()))
