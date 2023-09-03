#Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string
import csv

#Read CSV
data = pd.read_csv("filename.csv")
print(data.head())

#Let’s have a quick look at all the column names of the dataset:
print(data.columns)

#We only need three columns for this task (username, tweet, and language); I will only select these columns and move forward:
data = data[["username", "tweet", "language"]]

#Let’s have a look at whether any of these columns contains any null values or not:
print(data.isnull().sum())

#So none of the columns has null values, let’s have a quick look at how many tweets are posted in which language:
print(data["language"].value_counts())

#So most of the tweets are in English.
#Let’s prepare this data for the task of sentiment analysis.
#Here I will remove all the links, punctuation, symbols and other language errors from the tweets:
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

#let's make our text cleaner function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

# now apply it to the tweet data
data["tweet"] = data["tweet"].apply(clean)

#Now let’s have a look at the wordcloud of the tweets,
#which will show the most frequently used words in the tweets by people sharing
#their feelings and updates about the Ukraine and Russia war:
text = " ".join(i for i in data.tweet)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Now I will add three more columns in this dataset as Positive,
#Negative, and Neutral by calculating the sentiment scores of the tweets:
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
data = data[["tweet", "Positive", "Negative", "Neutral"]]

#print it out
print(data.head())

#write the dataframe to a csv file for our github review
data.head().to_csv('Sentiment-Rus-Vs-Ukr.csv', encoding='utf-8')

#Now let’s have a look at the most frequent words used by people with positive sentiments:
positive =' '.join([i for i in data['tweet'][data['Positive'] > data["Negative"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Now let’s have a look at the most frequent words used by people with negative sentiments:
negative =' '.join([i for i in data['tweet'][data['Negative'] > data["Positive"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(negative)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

