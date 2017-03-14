import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

# In the dataset, every element
print df.query('X != -120.5 & Y != 90').groupby("Category").mean()
