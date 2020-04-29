from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd

file = pd.read_csv('Final.csv')

train = file['Clean_tweets'].sample(n=10)
test  = file['Clean_tweets'].sample(n=5)

training = [('what we saw last night is there is no desire to change anything   so we need to get some strong new and fresh voices','neg'),
            ('years too late tories forgot about the uk due to preoccupation with stupid brexit','neg'),
            ('google bans eight different tory election adves as disinformation concerns mount ','neg'),
            ('its scary how coordinated the entire status quo is to make sure brexit happens no matter the cost','neg'),
            ('turns out  has pulled out of two television debates this week after the shocker with mr neil','pos'),
            ('greens  kicks off bbc 7way debate with a zinger about johnson on brexit  his ovenready meal is made','pos'),
            ('we offered proof the tories were trying to sell the nhs  we offered proof they were lowering health  safety standards  we ','neg'),
            ('my greatest fear is that the election of a johnson government and the hard brexit that will follow would lead to corruption oï¿½ï¿½_ï¿½ï¿½_','neg'),
            ('at last a proper nightï¿½Ûªs sleep the loony corbynistas destroyed brexit secured and a nice cup of tea all is well in the woï¿½ï¿½_','pos'),
            ('brexit closure  johnson wins commanding victory in uk election ','pos')]

testing = [('the democratic pay fulfills its true function during the primary election not the general once it succeeds in crushing','pos'),
           ('two years ago days before the general election there was a terrorist attack in london bridge borough market now two weeks','neg'),
           ('an annual reminder that both our nonpropoional voting system and split left vote are utter trash garbage  144m for','neg'),
           ('o your kids know everyone thinks youï¿½Ûªre a degenerate','neg'),
           ('talking to residents in bulwell market today where the messages were clear ï¿½ï¿½_ï¿½ï¿½_ï¿½ï¿½_get brexit doneï¿½ï¿½_ï¿½ and ï¿½ï¿½_ï¿½ï¿½_ï¿½ï¿½_we canï¿½ï¿½_ï¿½t vote for cï¿½ï¿½_ï¿½ï¿½__','pos')]


cl = NaiveBayesClassifier(training)

print(cl.accuracy(testing))
print(cl.show_informative_features(5))