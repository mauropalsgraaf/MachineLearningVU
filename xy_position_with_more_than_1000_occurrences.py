import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

print df.groupby(['X', 'Y'])
        .filter(lambda x: len(x) > 1000)
        .groupby(['X', 'Y'])
        .size()
