from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import CW_Credential
import json
import pandas as pd

Keyword = ['Brexit','Referendum','General Election','Election Result 2019']
tweets_to_capture = xxxx
File = 'Twitter_after9.json'

auth = OAuthHandler(CW_Credential.Consumer_key,CW_Credential.Consumer_secret)
auth.set_access_token(CW_Credential.Access_token,CW_Credential.Access_token_secret)

class StreamListener(tweepy.StreamListener):
    def __init__(self,api=None):
        super(StreamListener,self).__init__()
        self.num_tweets = 0
        self.file = open(File,'w')

    def on_status(self, status):
        tweets = status._json
        self.file.write(json.dumps(tweets) + '\n')
        self.num_tweets += 1

        if self.num_tweets <= tweets_to_capture:
            if self.num_tweets % 100 == 0:
                print('Number of tweets captured: {}'.format(self.num_tweets))
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status_code):
        print(status_code)

l = StreamListener()
stream = Stream(auth,l)
stream.filter(track=Keyword)

tweets_data = []

with open(File,'r') as tweets_file:
    for line in tweets_file:
        tweet = json.loads(line)
        tweets_data.append(tweet)


data = pd.DataFrame(tweets_data, columns=['created_at','lang','text','user'])

data.to_csv('xxx.csv')


