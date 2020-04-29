import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium as fm
from shapely.geometry import Polygon
from math import radians, asin, sqrt, sin, cos, log, log10
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import Counter
from IPython.core.display import Markdown, display, HTML

file = pd.read_csv('xxx.csv')

file = file.dropna()

datetime_series = pd.to_datetime(file['Time'].astype(str), format='%d/%m/%Y %H:%M')
datetime_index = pd.DatetimeIndex(datetime_series.values)

file['Minute of the day'] = datetime_index.hour*60

#print(file.describe())
#print(file[['Latitude','Longitude']].max())
#print(file[['Latitude','Longitude']].min())

lon_range = (-123.5,153.5)
lon_cell  = 360

lat_range = (-38,65)
lat_cell  = 360

mod_range = (0,1440)
mod_step  = 24

lon_incr = (lon_range[1] - lon_range[0]) / lon_cell
lat_incr = (lat_range[1] - lat_range[0]) / lat_cell
x0, y0 = lon_range[0], lat_range[0]

cell_ids = []
grid_cells = []

for c in range(lon_cell):
    x1 = x0 + lon_incr
    for r in range(lat_cell):
        y1 = y0 + lat_incr
        grid_cells.append(Polygon([(x0,y0),(x0,y1),(x1,y1),(x1,y0)]))
        cell_ids.append('{:02d}_{:02d}'.format(c, r))
        y0 = y1
    x0 = x1
    y0 = lat_range[0]

file['Grid_x'] = np.floor((file['Longitude'] - lon_range[0]) / (lon_range[1] - lon_range[0]) * lon_cell).astype(int)
file['Grid_y'] = np.floor((file['Latitude'] - lat_range[0]) / (lat_range[1] - lat_range[0]) * lat_cell).astype(int)

file['Cell_id']  = file[['Grid_x','Grid_y']].apply(lambda x: '{:02d}_{:02d}'.format(x.Grid_x, x.Grid_y), axis=1)
file['Interval'] = np.floor((file['Minute of the day'] - mod_range[0]) / (mod_range[1] - mod_range[0]) * mod_step).astype(int)
# Note this results in 25 intervals as timestamps==17:30:00 are inside interval [17:30:00, 17:39:59]

#print(file.head())

# Derive possible time series by aggregating over cells x intervals
agg_func = {
    'Grid_x':'first',
    'Grid_y':'first',
    'ID':'count',
    'Polarity':['mean','median','std'],
    'Subjectivity':['median','mean','std']
}
st_aggregates = file.reset_index(drop=False)[['Cell_id','Grid_x','Grid_y','Interval','ID','Subjectivity','Polarity']].groupby(['Cell_id','Interval']).agg(agg_func)

# Flatten hierarchical column names
st_aggregates.columns = ["_".join(x) for x in st_aggregates.columns.ravel()]

# Rename columns to be more expressive
st_aggregates.rename(columns={'ID_count':'Count'}, inplace=True)

#print(st_aggregates.head())

# Reshape the times_series data frame to give us a dataframe with the time series for one selected target attribute per grid cell and time interval (i.e., each grid cell on a row, number of columns equal to number of time intervals)
target_col = 'Count'

# This unstack command retains the first level of our index (cell_id) and pivots the values of the target column (Count) into columns defined by values of the second index level ('interval', zero-based level index)
time_series = st_aggregates[target_col].unstack(level=1)

# Some cleanup - Reshaping will create null values in columns for empty groups (which here are our time intervals).
# For time series of lightning strike counts per time interval, replacing by 0 (no lightning strikes were registered) is appropriate.  Additionally, coerce to int after filling null values.
time_series = time_series.fillna(0).astype(int)

#print(time_series.head())

# Gather some statistics over all grid cells, per timestamp. We will use these for visualization.
interval_min = time_series.min(skipna=True)
interval_max = time_series.max(skipna=True)
interval_mean = time_series.mean(skipna=True)
interval_median = time_series.median(skipna=True)

quantile_borders = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
int_quantiles = time_series.quantile(quantile_borders).T

#print(int_quantiles.head())

fig, ax = plt.subplots()

for index, row in time_series.iterrows():
    ax.plot(time_series.columns, row, color='#00000080', label=row.name)

ax.plot(time_series.columns, interval_median, color='red')

ax.yaxis.grid(color='#D0D0D0', linestyle='--')

ax.set_xticks(time_series.columns)
# for tick in ax.get_xticklabels():
#    tick.set_rotation(45)

ax.xaxis.grid(color='#D0D0D0', linestyle='--')
plt.title('Sentiment Values Counts Over Time for All Places')
plt.show()

##############
# Tweak default output of pyplots
screen_dpi = plt.rcParams['figure.dpi']

quant_color_dark  = '#B0B0B0FF'
quant_color_light = '#F0F0F0FF'
quant_range_colors = [quant_color_light, quant_color_dark]

fig, axs = plt.subplots(1, 2, figsize=(1000/screen_dpi,400/screen_dpi))

###### First (left) subplot
ax = axs[0]
for i in range(1, len(quantile_borders)):
    ax.fill_between(int_quantiles.index, int_quantiles[quantile_borders[i-1]], int_quantiles[quantile_borders[i]], facecolor=quant_range_colors[i%2])

ax.plot(interval_min.index, interval_min, color=quant_color_dark)
ax.plot(interval_max.index, interval_max, color=quant_color_dark)

legend_handles = [None, None] # Manually collect relevant legend handles, so we can suppress that every single (auto-labeled) decentile element gets crammed into the legend
legend_handles[0], = ax.plot(interval_mean.index, interval_mean, color='blue')
legend_handles[1], = ax.plot(interval_median.index, interval_median, color='red')

# Draw a subtle reference grid
ax.yaxis.grid(color='#D0D0D0', linestyle='--')
ax.xaxis.grid(color='#D0D0D0', linestyle='--')

# X-axis tick labels might need a little bit of help to look nicely
ax.set_xticks(int_quantiles.index)
#for i, tick in enumerate(ax.get_xticklabels()):
#    tick.set_label(str(int_quantiles.index))
#    tick.set_rotation(45)

ax.set_title('Per-cell Time Series')

ax.set_xlim(0, mod_step)
ax.set_ylim(0)

###### Second (right) subplot
ax = axs[1]
for i in range(1, len(quantile_borders)):
    ax.fill_between(int_quantiles.index, int_quantiles[quantile_borders[i - 1]], int_quantiles[quantile_borders[i]],
                    facecolor=quant_range_colors[i % 2])

ax.plot(interval_min.index, interval_min, color=quant_color_dark)
ax.plot(interval_max.index, interval_max, color=quant_color_dark)

legend_handles = [None, None]  # Manually collect relevant legend handles, so we can suppress that every single (auto-labeled) decentile element gets crammed into the legend
legend_handles[0], = ax.plot(interval_mean.index, interval_mean, color='blue')
legend_handles[1], = ax.plot(interval_median.index, interval_median, color='red')

# Draw a subtle reference grid
ax.yaxis.grid(color='#D0D0D0', linestyle='--')
ax.xaxis.grid(color='#D0D0D0', linestyle='--')

# X-axis tick labels might need a little bit of help to look nicely
ax.set_xticks(int_quantiles.index)
# for i, tick in enumerate(ax.get_xticklabels()):
#    tick.set_label(str(int_quantiles.index))
#    tick.set_rotation(45)

ax.set_title('Low-value Range Details')

ax.set_xlim(0, mod_step)
# Zoom the plot on the low values range for more details
ax.set_ylim(0, 800)

###### Joint legend
labels = ['Mean', 'Median']
ax.legend(handles=legend_handles, title='Reference Time Series',
          labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()

####Geovisualisation
# 9-step continuous ordered color scale "OrdRd", courtesy of www.colorbrewer2.org
cb2_ordrd = ['#fff7ec','#fee8c8','#fdd49e','#fdbb84','#fc8d59','#ef6548','#d7301f','#b30000','#7f0000']
# Same color scale, as rgb tuples for easier interpolation
cb2_ordrd_rgb = [(255,247,236), (254,232,200), (253,212,158), (253,187,132), (252,141,89), (239,101,72), (215,48,31), (179,0,0), (127,0,0)]

all_cell_ids = ['{:02d}_{:02d}'.format(c,r) for c in range(lon_cell) for r in range(lat_cell)]
max_counts = time_series.max(axis=1, skipna=True).reindex(index=all_cell_ids).fillna(0).astype(int)

#print(max_counts.head())

# Set up the base map
m = fm.Map(tiles='cartodbdark_matter', width='100%',height='100%')
# If you adjusted the notebook display width to be as wide as your screen, the map might get very big.
# Adjust size as desired.
m.fit_bounds([[lat_range[0], lon_range[0]], [lat_range[1], lon_range[1]]])

min_val = max_counts.min()
max_val = max_counts.max()

ncol = len(cb2_ordrd_rgb)
col_range = 1 / (ncol - 1)

for c in range(lon_cell):
    for r in range(lat_cell):
        cell_id = '{:02d}_{:02d}'.format(c, r)

        val = max_counts[cell_id]
        nval = (val - min_val) / (max_val - min_val)

        center_x = lon_range[0] + (c + 0.5) * lon_incr
        center_y = lat_range[0] + (r + 0.5) * lat_incr

        # Determine interpolated color
        col_int = divmod(nval, col_range)
        ci = int(col_int[0])
        r = col_int[1] / col_range
        if (r > 0.0 and ci < ncol):
            c0 = cb2_ordrd_rgb[ci]
            c1 = cb2_ordrd_rgb[ci + 1]
            col = (
                int((1.0 - r) * c0[0] + r * c1[0]),
                int((1.0 - r) * c0[1] + r * c1[1]),
                int((1.0 - r) * c0[2] + r * c1[2])
            )
            c_string = '#{:02x}{:02x}{:02x}'.format(col[0], col[1], col[2])

        if (val > 0):
            fm.Circle((center_y, center_x), radius=15000, color=c_string, fill_color=c_string, fill=True, weight=0,
                      fill_opacity=0.25 + (nval * 0.5), popup='Maximum count: {}'.format(val)).add_to(m)

#m.save('spatio-temporal.html')

k = 12
cluster_id_col_name = 'Cluster ID(k={})'.format(k)

kmeans = KMeans(n_clusters=k,random_state=42)
clus = kmeans.fit(time_series)

time_series[cluster_id_col_name] = clus.labels_

centroid = pd.DataFrame(data=clus.cluster_centers_,columns=time_series.drop(cluster_id_col_name,axis=1).columns)
cluster_size = Counter(clus.labels_)

for cid, cnt in cluster_size.items():
    cluster_size[cid] = (cnt,log10(cnt)+1)

all_cell_ids = ['{:02d}_{:02d}'.format(c,r) for c in range(lon_cell) for r in range(lat_cell)]
time_series = time_series.reindex(index=all_cell_ids)
time_series[cluster_id_col_name] = time_series[cluster_id_col_name].fillna(k).astype(int)
time_series = time_series.fillna(0).astype(int)

clust_colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

m1 = fm.Map(tiles='cartodbdark_matter', width='100%', height='100%')
# If you adjusted the notebook display width to be as wide as your screen, the map might get very big. Adjust size as desired.
m1.fit_bounds([[lat_range[0], lon_range[0]], [lat_range[1], lon_range[1]]])

for c in range(lon_cell):
    for r in range(lat_cell):
        cell_id = '{:02d}_{:02d}'.format(c,r)
        data = time_series.loc[cell_id]
        center_x = lon_range[0] + (c + 0.5)*lon_incr
        center_y = lat_range[0] + (r + 0.5)*lat_incr
        cluster_id = data[cluster_id_col_name]
        if (cluster_id < k):
            fm.Circle((center_y, center_x), radius=8000, weight=1, color=clust_colors[cluster_id],
                      fill_color=clust_colors[cluster_id], fill=True, fill_opacity=0.7,
                      popup='Cell: {}_{}; Cluster: {}'.format(c,r,cluster_id)).add_to(m1)

#m1.save('KMeansClustering.html')

fig, ax = plt.subplots()

# Select subset of clusters for detail comparison
#selected_clusters = [0, 3, 5]
# or use the next statement to display all clusters simultaneously
selected_clusters = [x for x in range(k)]
x_vals = [x for x in range(mod_step+1)]

for index, row in time_series.iterrows():
    cluster_id = mod_step
    if (cluster_id < k and cluster_id in selected_clusters):
        ax.plot(x_vals, row[0:mod_step+1], color=clust_colors[cluster_id], alpha= 0.1, label=index)

legend_handles = []

for cluster_id, cent in centroid.iterrows():
    if (cluster_id in selected_clusters):
        centroid_handle, = ax.plot(cent.index, cent, color=clust_colors[cluster_id],
                                   alpha= 1.0, linewidth=2, label=cluster_id)
        legend_handles.append(centroid_handle)

ax.set_ylabel(target_col)
ax.yaxis.grid(color='#D0D0D0', linestyle='--')
ax.xaxis.grid(color='#D0D0D0', linestyle='--')

ax.set_ylim(0, max_val)

plt.title('Temporal Cluster Variance')

labels = [None] * len(selected_clusters)
for i, cid in enumerate(selected_clusters):
    labels[i] = '{:>2} ({:>4})'.format(cid, cluster_size[cid][0])

plt.legend(handles=legend_handles, title='Cluster ID (Cluster size)',
           labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()
