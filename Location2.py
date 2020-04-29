import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

file = pd.read_csv('Twitter_data.csv')
file1 = pd.read_csv('Twitter_data_after.csv')

file2 = pd.concat([file,file1],sort=False).reset_index()

file2[['Location']] = file2['Location'].str.replace("'","")

file2 = file2.replace('\n', ' ',regex=True)

def preprocess(Features):
    Features = Features.str.replace('http\S+|www.\S+|@\S+', '', case=False)
    Features = Features.str.replace("rt'",'',case=False)
    Features = Features.str.replace("rt",'',case=False)
    Features = Features.str.replace("'rt'",'',case=False)
    Features = Features.str.replace('RT', ' ')
    Features = Features.str.lower()
    Features = Features.str.replace('^\w', '')
    Features = Features.str.replace("(<br/>)", "")
    Features = Features.str.replace('(<a).*(>).*(</a>)', '')
    Features = Features.str.replace('(&amp)', '')
    Features = Features.str.replace('(&gt)', '')
    Features = Features.str.replace('(&lt)', '')
    Features = Features.str.replace('(\xa0)', ' ')

    return Features

file2['Location'] = preprocess(file2['Location'])

location_count = file2['Location'].value_counts()
location_count = location_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(location_count.index,location_count.values,palette='Set2')
plt.xlabel('Location')
plt.ylabel('Number of Locations')
plt.show()

