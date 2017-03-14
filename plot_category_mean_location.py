import gmplot
import pandas as pd
import numpy as np

gmap = gmplot.GoogleMapPlotter.from_geocode("San Francisco")

df = pd.read_csv('train.csv')

grouping = df.query('X != -120.5 & Y != 90').groupby("Category").mean()

lon = grouping.X.tolist()
lan = grouping.Y.tolist()

gmap.scatter(lan, lon, '#000', size=80, marker=False)
gmap.draw("plot_category_mean_location.html")
