import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

train_data = pd.read_csv('training-set.csv')
test_df = pd.read_csv('test.csv')


train_df, test_df = train_test_split(train_data, test_size = 0.05)

## Select target and train columns
target = train_df.iloc[:,:1]
train = train_df.iloc[:,1:]


test_df_target = test_df.iloc[:,:1]
test_df_set = test_df.iloc[:,1:]

gnb = GaussianNB()
y_pred = gnb.fit(train, target.values.ravel()).predict(train)

diff = y_pred == target
print(diff)
