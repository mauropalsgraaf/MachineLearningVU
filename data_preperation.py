import pandas as pd
import numpy as np

def is_night(pandas_time):
    if pandas_time.hour < 6 or pandas_time.hour > 18:
        return True

    if pandas_time.hour > 6 and pandas_time.hour < 18:
        return False

    if pandas_time.hour == 6 and pandas_time.minute < 36:
        return True

    if pandas_time.hour == 18 and pandas_time.minute > 42:
        return True

df = pd.read_csv('train.csv')

df['Dates'] = df['Dates'].apply (lambda time: pd.to_datetime(time))
df['Year'] = df.apply (lambda row: row['Dates'].year, axis=1)
df['Month'] = df.apply (lambda row: row['Dates'].month, axis=1)
df['Day'] = df.apply (lambda row: row['Dates'].day, axis=1)
df['Hour'] = df.apply (lambda row: row['Dates'].hour, axis=1)
df['Night'] = df.apply (lambda row: 1 if is_night(row['Dates']) else 0 , axis=1)
df['Intersection'] = df.apply (lambda row: 1 if "/" in row['Address'] else 0, axis=1)
df = pd.merge(df, pd.get_dummies(df['DayOfWeek']), left_index=True, right_index=True)

df = df.drop('Resolution', 1)
df = df.drop('Dates', 1)
df = df.drop('Descript', 1)
df = df.drop('DayOfWeek', 1)
df = df.drop('Address', 1)

df.to_csv('transformed.csv', index=False)
