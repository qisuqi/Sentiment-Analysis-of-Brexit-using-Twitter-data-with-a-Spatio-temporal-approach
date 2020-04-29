import pandas as pd
from nltk.tokenize import TweetTokenizer,word_tokenize
import string
import matplotlib.pyplot as plt
import seaborn as sns

#Import the dataset
file = pd.read_csv('xxx.csv')

#Drop missing values
file = file.dropna()

#Double check there isnt anymore missing values in the dataset
#print(file['Text'].isnull().sum())

print(file.shape)

file[['Location']] = file['Location'].str.replace("'","")
file['ID'] = file['ID'].str.replace("'","")
file['Name'] = file['Name'].str.replace("'","")
file['Screen_name'] = file['Screen_name'].str.replace("'","")
file['Time'] = file['Time'].str[0:19]

file = file.replace('\n', ' ',regex=True)

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
    Features = Features.apply(lambda x:''.join([i for i in x
                                                  if i not in string.punctuation]))
    return Features


file['Clean_tweets'] = preprocess(file['Text'])
file['Location'] = preprocess(file['Location'])

Tokens = file['Clean_tweets'].apply(word_tokenize)

def identify_tokens(row):
    line = row['Clean_tweets']
    tokens = word_tokenize(line)
    token_words = [word for word in tokens if word.isalpha()]
    return token_words

file['Token_tweets'] = file.apply(identify_tokens,axis=1)

file = file.rename(columns={'Text':'Tweets'})
file = file[['Time','ID','Name','Screen_name','Tweets','Language','Location','Clean_tweets','Token_tweets']]

#datetime_series = pd.to_datetime(file['Time'].astype(str), format='%Y-%m-%d %H:%M:%S')
#datetime_index = pd.DatetimeIndex(datetime_series.values)
#file1 = file.set_index(datetime_index)
#file1 = file1.drop('Time',axis=1)

location_count = file['Location'].value_counts()
location_count = location_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(location_count.index,location_count.values,palette='Set2')
plt.title('Top 10 Location')
plt.xlabel('Location')
plt.ylabel('Number of Locations')
plt.show()
