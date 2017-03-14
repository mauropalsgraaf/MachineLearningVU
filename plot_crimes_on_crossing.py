import gmplot
import pandas as pd
import numpy as np

gmap = gmplot.GoogleMapPlotter.from_geocode("San Francisco")

df = pd.read_csv('train.csv')
dataFrame = df[df.Address.str.contains("/")]

lon = dataFrame.X.drop_duplicates().values.tolist()
lan = dataFrame.Y.drop_duplicates().values.tolist()

gmap.scatter(lan, lon, '#000', size=40, marker=False)
gmap.draw("plot_crimes_on_crossing.html")
