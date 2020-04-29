import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = pd.read_csv('Final.csv')

location_count = file['Location'].value_counts()
location_count = location_count[1:10,]
plt.figure(figsize=(15,6))
sns.barplot(location_count.index,location_count.values,palette='Set2')
plt.title('Top 10 Location Cleaned')
plt.xlabel('Location')
plt.ylabel('Number of Locations')
plt.xticks(rotation=65)
plt.tight_layout()
#plt.show()

print(len(file[file['Location']=='none']))
print(len(file))
print(len(file[file['Latitude']=='']))