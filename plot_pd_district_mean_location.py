import gmplot
import pandas as pd
import numpy as np

gmap = gmplot.GoogleMapPlotter.from_geocode("San Francisco")

df = pd.read_csv('train.csv')

grouping = df.groupby('PdDistrict').mean()

lon = grouping.X.tolist()
lan = grouping.Y.tolist()

gmap.scatter(lan, lon, '#000', size=150, marker=False)
gmap.draw("plot_pd_district_mean_location.html")
