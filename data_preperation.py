import pandas as pd
import numpy as np

def is_night(pandas_time):
    return not pandas_time.hour >= 6 and pandas_time.hour < 18

df = pd.read_csv('train.csv')

df['Dates'] = df['Dates'].apply (lambda time: pd.to_datetime(time))

df['Year'] = df.apply (lambda row: row['Dates'].year, axis=1)
df['Month'] = df.apply (lambda row: row['Dates'].month, axis=1)
df['Day'] = df.apply (lambda row: row['Dates'].day, axis=1)
df['Hour'] = df.apply (lambda row: row['Dates'].hour, axis=1)

df['Night'] = df.apply (lambda row: 1 if is_night(row['Dates']) else 0 , axis=1)
df['Intersection'] = df.apply (lambda row: 1 if "/" in row['Address'] else 0, axis=1)

df.to_csv('transformed.csv', index=False)
