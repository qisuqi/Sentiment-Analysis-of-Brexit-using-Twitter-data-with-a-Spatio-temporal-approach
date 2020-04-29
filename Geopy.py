import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="VA")

data['Co_ord'] = data['Location'].apply(geolocator.geocode)

Latitude  = data['Co_ord'].apply(lambda x: (x.latitude))
Longitude = data['Co_ord'].apply(lambda x: (x.longitude))

data.insert(3,'Latitude',Latitude)
data.insert(4,'Longitude',Longitude)

data.to_csv('xxx.csv')


