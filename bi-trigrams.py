import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud,STOPWORDS

file = pd.read_csv('xxx.csv')

def get_top_k_n_gram(corpus, k=None, n=None):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:k]

common_bigrams = get_top_k_n_gram(file['Clean_tweets'].values.astype(str), 25,2)

file1 = pd.DataFrame(common_bigrams, columns = ['Clean_tweets' , 'count'])

file1.groupby('Clean_tweets').sum()['count'].sort_values(ascending=True).plot(
    kind='barh', title='Top 25 Bigrams in Tweets after Removing Stop Words')
plt.xlabel('Number of Tweets')
plt.show()

stopwords = set(STOPWORDS)

tweets = ','.join(str(tweet) for tweet in file1['Clean_tweets'])
tweets = tweets.replace('"','')
wordcloud = (WordCloud(background_color='white',max_words=300,stopwords=stopwords,colormap='plasma').generate(tweets))

wordcloud.to_file('Wordcloud.png')

common_trigrams = get_top_k_n_gram(file['Clean_tweets'].values.astype(str), 25,3)

file2 = pd.DataFrame(common_trigrams, columns = ['Clean_tweets' , 'count'])

file2.groupby('Clean_tweets').sum()['count'].sort_values(ascending=True).plot(
    kind='barh', title='Top 25 Trigrams in Tweets after Removing Stop Words')
plt.xlabel('Number of Tweets')
plt.show()
