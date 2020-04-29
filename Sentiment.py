import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

file = pd.read_csv('xxx.csv')

filtered_tweets = str(file['Clean_tweets'])

blob = TextBlob(filtered_tweets)

file[['Polarity', 'Subjectivity']] = file['Clean_tweets'].apply(lambda Text: pd.Series(TextBlob(str(Text)).sentiment))

file = file.drop(columns=['Unnamed: 0','Unnamed: 0.1'])

#for col in file.columns:
    #print(col)

plt.subplot(1,2,1)
plt.hist(file2['Polarity'],color='mediumpurple')
plt.title('Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Number of tweets')
plt.ylim(0,250000)

plt.subplot(1,2,2)
plt.hist(file2['Subjectivity'],color='mediumpurple')
plt.title('Subjectivity Distribution')
plt.xlabel('Subjectivity')
plt.ylabel('Number of tweets')
plt.tight_layout()
plt.show()
