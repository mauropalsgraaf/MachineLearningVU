import pandas as pd
import sys

# Read in csv file
df = pd.read_csv(sys.argv[1])

# Read in original train.csv for categories
training_df = pd.read_csv('train.csv')

# Get all unique categories
categories = training_df.groupby("Category").Category.unique()

# Make all categories binary and check if that is the one to predict
for category in categories:

    category = category[0]
    df[category] = df.apply (lambda row: 1 if row["Category"] == category else 0, axis=1)


# Drop unused columns (needed for kaggle)
df = df.drop('Category', 1)
df = df.drop('Dates', 1)
df = df.drop('DayOfWeek', 1)
df = df.drop('PdDistrict', 1)
df = df.drop('Address', 1)
df = df.drop('X', 1)
df = df.drop('Y', 1)

# Dump to csv
df.to_csv(sys.argv[2], index=False)
