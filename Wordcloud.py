import pandas as pd
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

file = pd.read_csv('Final.csv')

file['Clean_tweets'] = file['Clean_tweets'].str.replace('_','')

stopwords = set(STOPWORDS)

#uk_mask = np.array(Image.open('united-kingdom-303323_960_720.png'))

file3 = file[file['Polarity']==0]
tweets = ','.join(str(tweet) for tweet in file3['Clean_tweets'])
wordcloud = (WordCloud(background_color='white',max_words=300,stopwords=stopwords,colormap='plasma').generate(tweets))

wordcloud.to_file('Wordcloud.png')

file1 = file[file['Polarity']>0]
tweets1 = ','.join(str(tweet) for tweet in file1['Clean_tweets'])
wordcloud1 = (WordCloud(background_color='white',max_words=300,stopwords=stopwords,colormap='plasma').generate(tweets1))

wordcloud1.to_file('Wordcloud_positive.png')

file2 = file[file['Polarity']<0]
tweets2 = ','.join(str(tweet) for tweet in file2['Clean_tweets'])
wordcloud2 = (WordCloud(background_color='white',max_words=300,stopwords=stopwords,colormap='plasma').generate(tweets2))

wordcloud2.to_file('Wordcloud_negative.png')

plt.subplot(1,3,1)
plt.imshow(wordcloud,interpolation='bilinear')
plt.title('Neutral Opinion',fontsize=11)
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(wordcloud1,interpolation='bilinear')
plt.title('Positive Opinion',fontsize=11)
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(wordcloud2,interpolation='bilinear')
plt.title('Negative Opinion',fontsize=11)
plt.axis('off')
plt.show()
