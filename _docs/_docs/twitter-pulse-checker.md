---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region colab_type="text" id="xNdTFUOmBWM0" -->
# Twitter Pulse Checker

- toc: true
- badges: true
- comments: true
- categories: [twitter, scraping, trend]
<!-- #endregion -->

<!-- #region colab_type="text" id="7x39hXNFaJ4Q" -->
![preview](https://cdn.pixabay.com/photo/2013/06/07/09/53/twitter-117595_960_720.png)
<!-- #endregion -->

<!-- #region colab_type="text" id="VDNM2RrQBapg" -->
This is a quick and dirty way to get a sense of what's trending on Twitter related to a particular Topic. For my use case, I am focusing on the city of Seattle but you can easily apply this to any topic.

**Use the GPU for this notebook to speed things up:** select the menu option "Runtime" -> "Change runtime type", select "Hardware Accelerator" -> "GPU" and click "SAVE".

The code in this notebook does the following things:


*   Scrapes Tweets related to the Topic you are interested in.
*   Extracts relevant Tags from the text (NER: Named Entity Recognition).
*   Does Sentiment Analysis on those Tweets.
*   Provides some visualizations in an interactive format to get a 'pulse' of what's happening.

We use Tweepy to scrape Twitter data and Flair to do NER / Sentiment Analysis. We use Seaborn for visualizations and all of this is possible because of the wonderful, free and fast (with GPU) Google Colab.

**A bit about NER (Named Entity Recognition)** 

This is the process of extracting labels form text. 

So, take an example sentence: 'George Washington went to Washington'. NER will allow us to extract labels such as Person for 'George Washington' and Location for 'Washington (state)'. It is one of the most common and useful applications in NLP and, using it, we can extract labels from Tweets and do analysis on them.

**A bit about Sentiment Analysis** 

Most commonly, this is the process of getting a sense of whether some text is Positive or Negative. More generally, you can apply it to any label of your choosing (Spam/No Spam etc.).

So, 'I hated this movie' would be classified as a negative statement but 'I loved this movie' would be classified as positive. Again - it is a very useful application as it allows us to get a sense of people's opinions about something (Twitter topics, Movie reviews etc). 

To learn more about these applications, check out the Flair Github homepage and Tutorials: https://github.com/zalandoresearch/flair


Note: You will need Twitter API keys (and of course a Twitter account) to make this work. You can get those by signing up here: https://developer.twitter.com/en/apps
<!-- #endregion -->

<!-- #region colab_type="text" id="7f9m2ucbDH8a" -->
To get up and running, we need to import a bunch of stuff and install Flair. Run through the next 3 cells.
<!-- #endregion -->

```python colab={} colab_type="code" id="TGc4FbSqCJDg"
# import lots of stuff
import sys
import os
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from IPython.display import clear_output
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
```

```python colab={} colab_type="code" id="2YaIwapFC7Yi"
# install Flair
!pip install flair

clear_output()
```

```python colab={} colab_type="code" id="CN7bPwceC77g"
# import Flair stuff
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')

clear_output()
```

```python colab={} colab_type="code" id="LhUwHI1zDDs_"
#import Flair Classifier
from flair.models import TextClassifier

classifier = TextClassifier.load('en-sentiment')

clear_output()
```

<!-- #region colab_type="text" id="LPfBYe-zqxme" -->
### Authenticate with Twitter API
<!-- #endregion -->

```python cellView="form" colab={} colab_type="code" id="D82o9BhxA0tq"
#@title Enter Twitter Credentials
TWITTER_KEY = '' #@param {type:"string"}
TWITTER_SECRET_KEY = '' #@param {type:"string"}
```

```python colab={} colab_type="code" id="MOxCv5dKBkVz"
# Authenticate
auth = tweepy.AppAuthHandler(TWITTER_KEY, TWITTER_SECRET_KEY)

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

```

<!-- #region colab_type="text" id="_0rweWLHXo1v" -->
###Lets start scraping!
<!-- #endregion -->

<!-- #region colab_type="text" id="T8oyLAkVYp4k" -->
The Twitter scrape code here was taken from: https://bhaskarvk.github.io/2015/01/how-to-use-twitters-search-rest-api-most-effectively.

My thanks to the author.

We need to provide a Search term and a Max Tweet count. Twitter lets you to request 45,000 tweets every 15 minutes  so setting something below that works.
<!-- #endregion -->

```python colab={} colab_type="code" id="As_PRtb-Bklo"
#@title Twitter Search API Inputs
#@markdown ### Enter Search Query:
searchQuery = 'Seattle' #@param {type:"string"}
#@markdown ### Enter Max Tweets To Scrape:
#@markdown #### The Twitter API Rate Limit (currently) is 45,000 tweets every 15 minutes.
maxTweets = 1000 #@param {type:"slider", min:0, max:45000, step:100}
Filter_Retweets = True #@param {type:"boolean"}

tweetsPerQry = 100  # this is the max the API permits
tweet_lst = []

if Filter_Retweets:
  searchQuery = searchQuery + ' -filter:retweets'  # to exclude retweets

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -10000000000

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
while tweetCount < maxTweets:
    try:
        if (max_id <= 0):
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en")
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", since_id=sinceId)
        else:
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", max_id=str(max_id - 1))
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", max_id=str(max_id - 1),
                                        since_id=sinceId)
        if not new_tweets:
            print("No more tweets found")
            break
        for tweet in new_tweets:
          if hasattr(tweet, 'reply_count'):
            reply_count = tweet.reply_count
          else:
            reply_count = 0
          if hasattr(tweet, 'retweeted'):
            retweeted = tweet.retweeted
          else:
            retweeted = "NA"
            
          # fixup search query to get topic
          topic = searchQuery[:searchQuery.find('-')].capitalize().strip()
          
          # fixup date
          tweetDate = tweet.created_at.date()
          
          tweet_lst.append([tweetDate, topic, 
                      tweet.id, tweet.user.screen_name, tweet.user.name, tweet.text, tweet.favorite_count, 
                      reply_count, tweet.retweet_count, retweeted])

        tweetCount += len(new_tweets)
        print("Downloaded {0} tweets".format(tweetCount))
        max_id = new_tweets[-1].id
    except tweepy.TweepError as e:
        # Just exit if any error
        print("some error : " + str(e))
        break

clear_output()
print("Downloaded {0} tweets".format(tweetCount))
```

<!-- #region colab_type="text" id="UVsHZlEroRQY" -->
##Data Sciencing
<!-- #endregion -->

<!-- #region colab_type="text" id="CC0Lz66Jn48L" -->
Let's load the tweet data into a Pandas Dataframe so we can do Data Science to it. 

The data is also saved down in a tweets.csv file in case you want to download it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 289} colab_type="code" executionInfo={"elapsed": 2764, "status": "ok", "timestamp": 1561106573593, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="Bu7qN8q6Bkn9" outputId="e548591b-3cb2-4ece-f8f6-cae988462158"
pd.set_option('display.max_colwidth', -1)

# load it into a pandas dataframe
tweet_df = pd.DataFrame(tweet_lst, columns=['tweet_dt', 'topic', 'id', 'username', 'name', 'tweet', 'like_count', 'reply_count', 'retweet_count', 'retweeted'])
tweet_df.to_csv('tweets.csv')
tweet_df.head()
```

<!-- #region colab_type="text" id="9lJ8UlW3ZIsH" -->
Unfortunately Twitter does not let you filter by date when you request tweets. However, we can do this at this stage. I have set it up to pull yesterday + todays Tweets by default.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 1794, "status": "ok", "timestamp": 1561106573594, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="pf_cZXTHBkqC" outputId="669fec29-ba09-4efb-fc88-c103837d3f23"
#@title Filter By Date Range
today = datetime.now().date()
yesterday = today - timedelta(1)

start_dt = '' #@param {type:"date"}
end_dt = '' #@param {type:"date"}

if start_dt == '':
  start_dt = yesterday
else:
  start_dt = datetime.strptime(start_dt, '%Y-%m-%d').date()

if end_dt == '':
  end_dt = today
else:
  end_dt = datetime.strptime(end_dt, '%Y-%m-%d').date()


tweet_df = tweet_df[(tweet_df['tweet_dt'] >= start_dt) 
                    & (tweet_df['tweet_dt'] <= end_dt)]
tweet_df.shape
```

<!-- #region colab_type="text" id="fC-cQNXwafbt" -->
## NER and Sentiment Analysis

Now let's do some NER / Sentiment Analysis. We will use the Flair library: https://github.com/zalandoresearch/flair

###NER

Previosuly, we extracted, and then appended the Tags as separate rows in our dataframe. This helps us later on to Group by Tags.

We also create a new 'Hashtag' Tag as Flair does not recognize it and it's a big one in this context.

### Sentiment Analysis

We use the Flair Classifier to get Polarity and Result and add those fields to our dataframe.

**Warning:** This can be slow if you have lots of tweets.
<!-- #endregion -->

```python colab={} colab_type="code" id="AOKbfZlzBksW"
# predict NER
nerlst = []

for index, row in tqdm(tweet_df.iterrows(), total=tweet_df.shape[0]):
  cleanedTweet = row['tweet'].replace("#", "")
  sentence = Sentence(cleanedTweet, use_tokenizer=True)
  
  # predict NER tags
  tagger.predict(sentence)

  # get ner
  ners = sentence.to_dict(tag_type='ner')['entities']
  
  # predict sentiment
  classifier.predict(sentence)
  
  label = sentence.labels[0]
  response = {'result': label.value, 'polarity':label.score}
  
  # get hashtags
  hashtags = re.findall(r'#\w+', row['tweet'])
  if len(hashtags) >= 1:
    for hashtag in hashtags:
      ners.append({ 'type': 'Hashtag', 'text': hashtag })
  
  for ner in ners:
    adj_polarity = response['polarity']
    if response['result'] == 'NEGATIVE':
      adj_polarity = response['polarity'] * -1
    nerlst.append([ row['tweet_dt'], row['topic'], row['id'], row['username'], 
                   row['name'], row['tweet'], ner['type'], ner['text'], response['result'], 
                   response['polarity'], adj_polarity, row['like_count'], row['reply_count'], 
                  row['retweet_count'] ])

clear_output()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 510} colab_type="code" executionInfo={"elapsed": 188896, "status": "ok", "timestamp": 1561106761154, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="VfZVjXldBkuc" outputId="0ca77d2c-68d3-4168-fd6e-fb5d41dba993"
df_ner = pd.DataFrame(nerlst, columns=['tweet_dt', 'topic', 'id', 'username', 'name', 'tweet', 'tag_type', 'tag', 'sentiment', 'polarity', 
                                       'adj_polarity','like_count', 'reply_count', 'retweet_count'])
df_ner.head()
```

<!-- #region colab_type="text" id="ETnIczIIyN_B" -->
Let's filter out obvious tags like #Seattle that would show up for this search. You can comment this portion out or use different Tags for your list.
<!-- #endregion -->

```python cellView="both" colab={} colab_type="code" id="tzwXUKUwBkzM"
# filter out obvious tags
banned_words = ['Seattle', 'WA', '#Seattle', '#seattle', 'Washington', 'SEATTLE', 'WASHINGTON',
                'seattle', 'Seattle WA', 'seattle wa','Seattle, WA', 'Seattle WA USA', 
                'Seattle, Washington', 'Seattle Washington', 'Wa', 'wa', '#Wa',
               '#wa', '#washington', '#Washington', '#WA', '#PNW', '#pnw', '#northwest']

df_ner = df_ner[~df_ner['tag'].isin(banned_words)]
```

<!-- #region colab_type="text" id="ajYB9VAC4-Ca" -->
Calculate Frequency, Likes, Replies, Retweets and Average Polarity per Tag.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 187783, "status": "ok", "timestamp": 1561106761159, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="xA3E8UTwBkw6" outputId="4361bf18-cad1-4980-82af-b2b3ee5f66a3"
ner_groups = df_ner.groupby(['tag', 'tag_type']).agg({'tag': "count", 'adj_polarity': "mean",
                                                     'like_count': 'sum', 'reply_count': 'sum',
                                                     'retweet_count': 'sum'})
ner_groups = ner_groups.rename(columns={
    "tag": "Frequency",
    "adj_polarity": "Avg_Polarity",
    "like_count": "Total_Likes",
    "reply_count": "Total_Replies",
    "retweet_count": "Total_Retweets"
})
ner_groups = ner_groups.sort_values(['Frequency'], ascending=False)
ner_groups = ner_groups.reset_index()
ner_groups.head()
```

<!-- #region colab_type="text" id="inLWlkSh8IW_" -->
Create an overall Sentiment column based on the Average Polarity of the Tag.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 187348, "status": "ok", "timestamp": 1561106761160, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="MeBq0NeO5H3P" outputId="f6e4943c-9d8c-4a60-832f-4bcd5f8189f1"
ner_groups['Sentiment'] = np.where(ner_groups['Avg_Polarity']>=0, 'POSITIVE', 'NEGATIVE')
ner_groups.head()
```

<!-- #region colab_type="text" id="jLzD6bwyauz9" -->
## Visualize!
<!-- #endregion -->

<!-- #region colab_type="text" id="bbhxVDawfaEQ" -->
We can get some bar plots for the Tags based on the following metrics:



*   Most Popular Tweets
*   Most Liked Tweets
*   Most Replied Tweets
*   Most Retweeted Tweets

By default, we do the analysis on all the Tags but we can also filter by Tag by checking the Filter_TAG box. 
This way we can further drill down into the metrics for Hashtags, Persons, Locations & Organizations.

We cut the plots by Sentiment i.e. the color of the bars tells us if the overall Sentiment was Positive or Negative.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 644} colab_type="code" executionInfo={"elapsed": 1555, "status": "ok", "timestamp": 1561107538982, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="D6lMQm32Bk1f" outputId="56f95c05-d925-4d31-b85d-0a470641c536"
#@title Visualize Top TAGs
Filter_TAG = False #@param {type:"boolean"}
TAG = 'Person' #@param ["Hashtag", "Person", "Location", "Organization"]
#@markdown ###Pick how many tags to display per chart:
Top_N = 10 #@param {type:"integer"}

# get TAG value
if TAG != 'Hashtag':
  TAG = TAG[:3].upper()

if Filter_TAG:
  filtered_group = ner_groups[(ner_groups['tag_type'] == TAG)]
else:
  filtered_group = ner_groups

# plot the figures
fig = plt.figure(figsize=(20, 16))
fig.subplots_adjust(hspace=0.2, wspace=0.5)

ax1 = fig.add_subplot(321)
sns.barplot(x="Frequency", y="tag", data=filtered_group[:Top_N], hue="Sentiment")
ax2 = fig.add_subplot(322)
filtered_group = filtered_group.sort_values(['Total_Likes'], ascending=False)
sns.barplot(x="Total_Likes", y="tag", data=filtered_group[:Top_N], hue="Sentiment")
ax3 = fig.add_subplot(323)
filtered_group = filtered_group.sort_values(['Total_Replies'], ascending=False)
sns.barplot(x="Total_Replies", y="tag", data=filtered_group[:Top_N], hue="Sentiment")
ax4 = fig.add_subplot(324)
filtered_group = filtered_group.sort_values(['Total_Retweets'], ascending=False)
sns.barplot(x="Total_Retweets", y="tag", data=filtered_group[:Top_N], hue="Sentiment")

ax1.title.set_text('Most Popular')
ax2.title.set_text('Most Liked')
ax3.title.set_text('Most Replied')
ax4.title.set_text('Most Retweeted')

ax1.set_ylabel('')    
ax1.set_xlabel('')
ax2.set_ylabel('')    
ax2.set_xlabel('')
ax3.set_ylabel('')    
ax3.set_xlabel('')
ax4.set_ylabel('')    
ax4.set_xlabel('')
```

<!-- #region colab_type="text" id="BybFabE9QyUv" -->
###Get the Average Polarity Distribution.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 409} colab_type="code" executionInfo={"elapsed": 606, "status": "ok", "timestamp": 1561107548548, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="LRdgAEbsDLyc" outputId="cc6bb6c6-0609-45d0-bbe1-7067ed09551d"
fig = plt.figure(figsize=(12, 6))
sns.distplot(filtered_group['Avg_Polarity'], hist=False, kde_kws={"shade": True})
```

<!-- #region colab_type="text" id="ipaVFUOPiJrk" -->
## Word Cloud

Let's build a Word Cloud based on these metrics. 

Since I am interested in Seattle, I am going to use overlay the Seattle city skyline view over my Word Cloud. 
You can change this by selecting a different Mask option from the drop down.

Images for Masks can be found at:

http://clipart-library.com/clipart/2099977.htm

https://needpix.com
<!-- #endregion -->

```python colab={} colab_type="code" id="rfYNVV1upjbL"
# download mask images
!wget http://clipart-library.com/img/2099977.jpg -O seattle.jpg
!wget https://storage.needpix.com/rsynced_images/trotting-horse-silhouette.jpg -O horse.jpg
!wget https://storage.needpix.com/rsynced_images/black-balloon.jpg -O balloon.jpg
  
clear_output()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 652} colab_type="code" executionInfo={"elapsed": 2256, "status": "ok", "timestamp": 1561107694057, "user": {"displayName": "Bilal Tahir", "photoUrl": "https://lh4.googleusercontent.com/-NChnXr2QFqQ/AAAAAAAAAAI/AAAAAAAAB3s/8QcWmRJWnIY/s64/photo.jpg", "userId": "00825885512617308384"}, "user_tz": 420} id="timxX2vKBk3-" outputId="ee7c4c13-60ae-47ad-bbfc-235a03378329"
#@title Build Word Cloud For Top TAGs
Metric = 'Most Popular' #@param ["Most Popular", "Most Liked", "Most Replied", "Most Retweeted"]
#@markdown
Filter_TAG = False #@param {type:"boolean"}
##@markdown
TAG = 'Location' #@param ["Hashtag", "Person", "Location", "Organization"]
Mask = 'Seattle' #@param ["Rectangle", "Seattle", "Balloon", "Horse"]

# get correct Metric value
if Metric == 'Most Popular':
   Metric = 'Frequency'
elif Metric == 'Most Liked':
   Metric = 'Total_Likes'
elif Metric == 'Most Replied':
   Metric = 'Total_Replies'
elif Metric == 'Most Retweeted':
   Metric = 'Total_Retweets'    

# get TAG value
if TAG != 'Hashtag':
  TAG = TAG[:3].upper()

if Filter_TAG:
  filtered_group = ner_groups[(ner_groups['tag_type'] == TAG)]
else:
  filtered_group = ner_groups

countDict = {}

for index, row in filtered_group.iterrows():
  if row[Metric] == 0:
    row[Metric] = 1
  countDict.update( {row['tag'] : row[Metric]} )
  
if Mask == 'Seattle':
  Mask = np.array(Image.open("seattle.jpg"))
elif Mask == 'Rectangle':
  Mask = np.array(Image.new('RGB', (800,600), (0, 0, 0)))
elif Mask == 'Horse':
  Mask = np.array(Image.open("horse.png"))
elif Mask == 'Balloon':
  Mask = np.array(Image.open("balloon.jpg"))

clear_output()

# Generate Word Cloud
wordcloud = WordCloud(
    max_words=100,
#     max_font_size=50,
    height=300,
    width=800,
    background_color = 'white',
    mask=Mask,
    contour_width=1,
    contour_color='steelblue',
    stopwords = STOPWORDS).generate_from_frequencies(countDict)
fig = plt.figure(
    figsize = (18, 18),
    )
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
```

```python colab={} colab_type="code" id="LS9tvmFIjUC9"

```
