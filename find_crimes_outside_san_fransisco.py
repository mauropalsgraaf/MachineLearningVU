# position of san fransisco top left corner 37.821167, -122.537166
# position of san fransisco bottom right corner 37.696310, -122.328426

import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

print df.query('X < -122.537166 | X > -122.328426 | Y < 37.696310 | Y > 37.821167')
