import folium
from folium.plugins import HeatMap
import pandas as pd


file = pd.read_csv('Final.csv')

file = file.dropna()


xmm = (file.Latitude.min()+file.Latitude.max())/2
ymm = (file.Longitude.min()+file.Longitude.max())/2

map = folium.Map(location=[ymm,xmm],zoom_start=5)

file.apply(lambda row: folium.CircleMarker(location=[row['Longitude'],row['Latitude']],radius=3).add_to(map),axis=1)

map.save('Final_map.html')

hmap=folium.Map(location=[ymm,xmm],zoom_start=5)

hm_wide=HeatMap(list(zip(file.Latitude.values,file.Longitude.values)),
                min_opacity=0.2,
                radius=17,
                blur=15,
                max_zoom=1)
hmap.add_child(hm_wide)

hmap.save('Final_hmap.html')
