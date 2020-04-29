import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from geopy.distance import great_circle
import geopandas


file = pd.read_csv('Final.csv')

X = file[['Polarity','Subjectivity','Latitude','Longitude']]
X = X.dropna()
X = X.sample(n=150000)

neutral = X[X['Polarity']==0]
positive = X[X['Polarity']>0]
negative = X[X['Polarity']<0]

data = pd.concat([neutral,positive,negative])
sentiment = data['Polarity']

coords = X.as_matrix(columns=['Latitude','Longitude'])


kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian

db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
centermost_points = clusters.map(get_centermost_point)

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'Longitude':lons, 'Latitude':lats})

fig, ax = plt.subplots(figsize=[10, 6])
base = world.plot(ax=ax,color='white', edgecolor='black')
rs_scatter = ax.scatter(rep_points['Longitude'], rep_points['Latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(data['Longitude'], data['Latitude'], c=sentiment, alpha=0.9, s=3)
ax.set_title('Sample data set vs DBSCAN reduced set')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['Sample set', 'Reduced set'], loc='upper right')
#plt.ylim(50,62)
#plt.xlim(-8,2)
cb = plt.colorbar(df_scatter,ticks=range(3),label='Sentiment Value')
cb.set_clim(vmin=-1,vmax=1)
plt.show()
