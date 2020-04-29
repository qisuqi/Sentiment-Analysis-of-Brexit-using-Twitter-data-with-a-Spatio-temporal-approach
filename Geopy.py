import pandas as pd
from geopy.geocoders import Nominatim

file = pd.read_csv('Location.csv')

file.columns=['Location','Location1','Number of Tweets']

geolocator = Nominatim(user_agent="VA")

data1 = file.iloc[135:150]
#data1['Location'] = data1['Location'].str.replace('deuropetschland','germany')
#data1['Location'] = data1['Location'].str.replace('europe 🇪🇺','europe')
#data1['Location'] = data1['Location'].str.replace('hefordshire','herefordshire')
#data1['Location'] = data1['Location'].str.replace('noh carolina','north carolina')
#data1['Location'] = data1['Location'].str.replace('noh london','north london')
#data1['Location'] = data1['Location'].str.replace('south east','south east england ')
#data1['Location'] = data1['Location'].str.replace('south west','south west england ')

#data1['Location'] = data1['Location'].str.replace('europe 🇪🇺','europe')
#data1['Location'] = data1['Location'].str.replace('hefordshire','herefordshire')
#data1['Location'] = data1['Location'].str.replace('noh carolina','north carolina')
#data1['Location'] = data1['Location'].str.replace('noh london','north london')

print(data1)


data1['Co_ord'] = data1['Location'].apply(geolocator.geocode)

Latitude  = data1['Co_ord'].apply(lambda x: (x.latitude))
Longitude = data1['Co_ord'].apply(lambda x: (x.longitude))

data1.insert(3,'Latitude',Latitude)
data1.insert(4,'Longitude',Longitude)

data1.to_csv('r9.csv')


