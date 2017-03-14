import gmplot
import pandas as pd
import numpy as np

gmap = gmplot.GoogleMapPlotter.from_geocode("San Francisco")

df = pd.read_csv('train.csv')

grouping = df.query('X != -120.5 & Y != 90')

lon = grouping.X.tolist()
lan = grouping.Y.tolist()

gmap.heatmap(lan, lon, 5, 10, None, 0.9, True)
gmap.draw("plot_heatmap_crimes.html")
