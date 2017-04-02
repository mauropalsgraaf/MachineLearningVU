import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
training_df = pd.read_csv('train.csv')

categories = training_df.groupby("Category").Category.unique()

for category in categories:
    print "iteration"
    category = category[0]
    df[category] = df.apply (lambda row: 1 if row["Category"] == category else 0, axis=1)

# df = pd.merge(df, ), left_index=True, right_index=True)

df = df.drop('Category', 1)
df = df.drop('Dates', 1)
df = df.drop('DayOfWeek', 1)
df = df.drop('PdDistrict', 1)
df = df.drop('Address', 1)
df = df.drop('X', 1)
df = df.drop('Y', 1)

df.to_csv(sys.argv[2], index=False)
