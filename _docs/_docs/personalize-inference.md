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
    name: python3
---

<!-- #region id="xEMHTa9wbwxO" -->
# Amazon Personalize Generic Module - Inference Layer
<!-- #endregion -->

<!-- #region id="AOFXYzLFcOVl" -->
## Environment Setup
<!-- #endregion -->

```python id="MBht8kn3ZL8M"
!pip install -U -q dvc dvc[gdrive]
!dvc pull
```

```python id="u4uPYtbnt_4w"
import sys
sys.path.insert(0,'./code')
```

```python id="C4xjNqJZcw2a"
!pip install -q boto3
```

```python id="1oZPIuJFcXN1"
!mkdir -p ~/.aws && cp /content/drive/MyDrive/AWS/d01_admin/* ~/.aws
```

```python id="oYGIEVWhuUJ5"
%reload_ext autoreload
%autoreload 2
```

<!-- #region id="DiquObwDISkI" -->
---
<!-- #endregion -->

```python id="ZYOsV8qPYljX"
import time
from time import sleep
import json
from datetime import datetime
import uuid
import random

import boto3
import pandas as pd
```

```python id="Qyw92_6WZJby"
personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
personalize_events = boto3.client(service_name='personalize-events')
```

```python id="GmKxPzrhZTA0"
import pickle
from generic_modules.import_model import personalize_model
```

```python colab={"base_uri": "https://localhost:8080/"} id="4cffQ-Smae9k" executionInfo={"status": "ok", "timestamp": 1630182007202, "user_tz": -330, "elapsed": 910, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="10df6170-bd75-490c-f5eb-4744366d1db4"
with open('./artifacts/etc/personalize_model_sims.pkl', 'rb') as outp:
    sims_model = pickle.load(outp)

sims_model.__dict__
```

```python colab={"base_uri": "https://localhost:8080/"} id="6ZDRQmGAa2C9" executionInfo={"status": "ok", "timestamp": 1630182025046, "user_tz": -330, "elapsed": 635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f12e326-cd3e-4d49-cdb0-0a34c64fced2"
with open('./artifacts/etc/personalize_model_pers.pkl', 'rb') as outp:
    pers_model = pickle.load(outp)

pers_model.__dict__
```

<!-- #region id="W49e10_ZXu1J" -->
## Interact with Campaigns
<!-- #endregion -->

<!-- #region id="fhxFylkBXsBd" -->
First, let's create a supporting function to help make sense of the results returned by a Personalize campaign. Personalize returns only an item_id. This is great for keeping data compact, but it means you need to query a database or lookup table to get a human-readable result for the notebooks. We will create a helper function to return a human-readable result from the Movielens dataset.
<!-- #endregion -->

<!-- #region id="pM3NNLY2YU92" -->
Start by loading in the dataset which we can use for our lookup table.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="bL6_BmDLYQtL" executionInfo={"status": "ok", "timestamp": 1630181422163, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d950959-fc3d-40ae-8496-269011d2e7a9"
# Create a dataframe for the items by reading in the correct source CSV
items_df = pd.read_csv('./data/bronze/ml-latest-small/movies.csv',
                       sep=',', usecols=[0,1], encoding='latin-1',
                       dtype={'movieId': "object", 'title': "str"},
                       index_col=0)

# Render some sample data
items_df.head(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Fr2PgDB3YiCB" executionInfo={"status": "ok", "timestamp": 1630181443276, "user_tz": -330, "elapsed": 664, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="379a46a5-2ebe-4665-fe53-c5ef90ac3adc"
movie_id_example = 589
title = items_df.loc[movie_id_example]['title']
print(title)
```

```python id="3FHDEfqdYsFZ"
def get_movie_by_id(movie_id, movie_df=items_df):
    try:
        return movie_df.loc[int(movie_id)]['title']
    except:
        return "Error obtaining title"
```

```python colab={"base_uri": "https://localhost:8080/"} id="fljRvlZaY2GA" executionInfo={"status": "ok", "timestamp": 1630181497380, "user_tz": -330, "elapsed": 749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="86f3925d-7568-4bb9-c658-bb333796f94b"
# A known good id (The Princess Bride)
print(get_movie_by_id(movie_id="1197"))
# A bad type of value
print(get_movie_by_id(movie_id="987.9393939"))
# Really bad values
print(get_movie_by_id(movie_id="Steve"))
```

<!-- #region id="d0F_WwcQZF1u" -->
### SIMS
SIMS requires just an item as input, and it will return items which users interact with in similar ways to their interaction with the input item. In this particular case the item is a movie.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="m5fwi3NOY5Q_" executionInfo={"status": "ok", "timestamp": 1630182052075, "user_tz": -330, "elapsed": 485, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="85602b5b-24ed-4ed5-fe4d-401f45d73ce4"
get_recommendations_response = personalize_runtime.get_recommendations(
    campaignArn = sims_model.campaign_arn,
    itemId = str(589),
)

item_list = get_recommendations_response['itemList']
for item in item_list:
    print(get_movie_by_id(movie_id=item['itemId']))
```

```python id="EERjdbWHbAxO"
# Update DF rendering
pd.set_option('display.max_rows', 30)

def get_new_recommendations_df(recommendations_df, movie_ID):
    # Get the movie name
    movie_name = get_movie_by_id(movie_ID)
    # Get the recommendations
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = sims_model.campaign_arn,
        itemId = str(movie_ID),
    )
    # Build a new dataframe of recommendations
    item_list = get_recommendations_response['itemList']
    recommendation_list = []
    for item in item_list:
        movie = get_movie_by_id(item['itemId'])
        recommendation_list.append(movie)
    new_rec_DF = pd.DataFrame(recommendation_list, columns = [movie_name])
    # Add this dataframe to the old one
    recommendations_df = pd.concat([recommendations_df, new_rec_DF], axis=1)
    return recommendations_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="m8PDLLr5bJ2b" executionInfo={"status": "ok", "timestamp": 1630182132619, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40c66395-17a5-4a57-a23f-ce9f28ec94b1"
samples = items_df.sample(5)
samples
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SCj7Prm1bL8X" executionInfo={"status": "ok", "timestamp": 1630182133769, "user_tz": -330, "elapsed": 650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e341963c-0f90-4a5f-8bce-6ca30587e3b2"
sims_recommendations_df = pd.DataFrame()
movies = samples.index.tolist()

for movie in movies:
    sims_recommendations_df = get_new_recommendations_df(sims_recommendations_df, movie)

sims_recommendations_df
```

<!-- #region id="ZewYVpSmbN6X" -->
This is a good time to think about the hyperparameters of the Personalize recipes. The SIMS recipe has a popularity_discount_factor hyperparameter (see documentation). Leveraging this hyperparameter allows you to control the nuance you see in the results. This parameter and its behavior will be unique to every dataset you encounter, and depends on the goals of the business. You can iterate on the value of this hyperparameter until you are satisfied with the results, or you can start by leveraging Personalize's hyperparameter optimization (HPO) feature. For more information on hyperparameters and HPO tuning, see the documentation.
<!-- #endregion -->

<!-- #region id="LchIu0o6be44" -->
### User Personalization Model

HRNN is one of the more advanced algorithms provided by Amazon Personalize. It supports personalization of the items for a specific user based on their past behavior and can intake real time events in order to alter recommendations for a user without retraining.

Since HRNN relies on having a sampling of users, let's load the data we need for that and select 3 random users. Since Movielens does not include user data, we will select 3 random numbers from the range of user id's in the dataset.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tWG00gShbiXc" executionInfo={"status": "ok", "timestamp": 1630182249052, "user_tz": -330, "elapsed": 644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="68918cc3-c1ea-4f7a-8713-034a7d400ebf"
users = random.sample(range(1, 600), 3)
users
```

```python id="aPHdj0OebwvD"
# Update DF rendering
pd.set_option('display.max_rows', 30)

def get_new_recommendations_df_users(recommendations_df, user_id):
    # Get the movie name
    #movie_name = get_movie_by_id(artist_ID)
    # Get the recommendations
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = pers_model.campaign_arn,
        userId = str(user_id),
    )
    # Build a new dataframe of recommendations
    item_list = get_recommendations_response['itemList']
    recommendation_list = []
    for item in item_list:
        movie = get_movie_by_id(item['itemId'])
        recommendation_list.append(movie)
    new_rec_DF = pd.DataFrame(recommendation_list, columns = [user_id])
    # Add this dataframe to the old one
    recommendations_df = pd.concat([recommendations_df, new_rec_DF], axis=1)
    return recommendations_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 909} id="j7iuj3Ldb0g2" executionInfo={"status": "ok", "timestamp": 1630182285380, "user_tz": -330, "elapsed": 492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f1a50190-8385-4c8f-f4eb-2f86d8cc8ce6"
recommendations_df_users = pd.DataFrame()
#users = users_df.sample(3).index.tolist()

for user in users:
    recommendations_df_users = get_new_recommendations_df_users(recommendations_df_users, user)

recommendations_df_users
```

<!-- #region id="GXTi4G_Zb5dJ" -->
## Static and Dynamic Filters
Lets interact with the static filters we created in the previous notebook, and utilize dynamic filters in realtime.

A few common use cases for dynamic filters in Video On Demand are:

Categorical filters based on Item Metadata (that arent range based) - Often your item metadata will have information about the title such as Genre, Keyword, Year, Director, Actor etc. Filtering on these can provide recommendations within that data, such as action movies, Steven Spielberg movies, Movies from 1995 etc.

Events - you may want to filter out certain events and provide results based on those events, such as moving a title from a "suggestions to watch" recommendation to a "watch again" recommendations.

Now lets apply item filters to see recommendations for one of these users within each decade of our static filters.
<!-- #endregion -->

```python id="4Ny2eicccL37"
def get_new_recommendations_df_by_static_filter(recommendations_df, user_id, filter_arn):
    # Get the movie name
    #movie_name = get_movie_by_id(artist_ID)
    # Get the recommendations
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = pers_model.campaign_arn,
        userId = str(user_id),
        filterArn = filter_arn
    )
    # Build a new dataframe of recommendations
    item_list = get_recommendations_response['itemList']
    recommendation_list = []
    for item in item_list:
        movie = get_movie_by_id(item['itemId'])
        recommendation_list.append(movie)
    #print(recommendation_list)
    filter_name = filter_arn.split('/')[1]
    new_rec_DF = pd.DataFrame(recommendation_list, columns = [filter_name])
    # Add this dataframe to the old one
    recommendations_df = pd.concat([recommendations_df, new_rec_DF], axis=1)
    return recommendations_df
```

```python id="ySEvn8xrcRpY"
def get_new_recommendations_df_by_dynamicfilter(recommendations_df, user_id, genre_filter_arn, filter_values):
    # Get the movie name
    #movie_name = get_movie_by_id(artist_ID)
    # Get the recommendations
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = pers_model.campaign_arn,
        userId = str(user_id),
        filterArn = genre_filter_arn,
        filterValues = { "GENRE": "\"" + filter_values + "\""}
    )
    # Build a new dataframe of recommendations
    item_list = get_recommendations_response['itemList']
    recommendation_list = []
    for item in item_list:
        movie = get_movie_by_id(item['itemId'])
        recommendation_list.append(movie)
    filter_name = genre_filter_arn.split('/')[1]
    new_rec_DF = pd.DataFrame(recommendation_list, columns = [filter_values])
    # Add this dataframe to the old one
    recommendations_df = pd.concat([recommendations_df, new_rec_DF], axis=1)
    return recommendations_df
```

```python colab={"base_uri": "https://localhost:8080/"} id="WFTj7OZPcfIQ" executionInfo={"status": "ok", "timestamp": 1630182471779, "user_tz": -330, "elapsed": 566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="305cf85e-a76e-41d8-f015-fb5fc274cbdd"
pers_model.filter_arns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 824} id="CqaEw014cZzw" executionInfo={"status": "ok", "timestamp": 1630182506671, "user_tz": -330, "elapsed": 609, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70ace112-5f13-4f02-db3f-d800cc486f1f"
recommendations_df_decade_shelves = pd.DataFrame()
for filter_arn in pers_model.filter_arns:
    recommendations_df_decade_shelves = get_new_recommendations_df_by_static_filter(recommendations_df_decade_shelves, user, filter_arn)

recommendations_df_decade_shelves
```

<!-- #region id="E_T5HCUHcvpZ" -->
## Real-Time Events
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kru4tlXnc_Qo" executionInfo={"status": "ok", "timestamp": 1630182605804, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="65a426f1-c2a5-457e-e8b4-91759317002b"
# Start by creating an event tracker that is attached to the campaign.

response = personalize.create_event_tracker(
    name='MovieTracker',
    datasetGroupArn=pers_model.dataset_group_arn
)

print(response['eventTrackerArn'])
print(response['trackingId'])

TRACKING_ID = response['trackingId']
event_tracker_arn = response['eventTrackerArn']
```

<!-- #region id="UKJl4-8FdH9X" -->
We will create some code that simulates a user interacting with a particular item. After running this code, you will get recommendations that differ from the results above.

We start by creating some methods for the simulation of real time events.
<!-- #endregion -->

```python id="8IyY-nV0dR4m"
session_dict = {}

def send_movie_click(USER_ID, ITEM_ID, EVENT_TYPE):
    """
    Simulates a click as an envent
    to send an event to Amazon Personalize's Event Tracker
    """
    # Configure Session
    try:
        session_ID = session_dict[str(USER_ID)]
    except:
        session_dict[str(USER_ID)] = str(uuid.uuid1())
        session_ID = session_dict[str(USER_ID)]
        
    # Configure Properties:
    event = {
    "itemId": str(ITEM_ID),
    }
    event_json = json.dumps(event)
        
    # Make Call
    
    personalize_events.put_events(
    trackingId = TRACKING_ID,
    userId= str(USER_ID),
    sessionId = session_ID,
    eventList = [{
        'sentAt': int(time.time()),
        'eventType': str(EVENT_TYPE),
        'properties': event_json
        }]
    )

def get_new_recommendations_df_users_real_time(recommendations_df, user_id, item_id, event_type):
    # Get the artist name (header of column)
    movie_name = get_movie_by_id(item_id)
    # Interact with different movies
    print('sending event ' + event_type + ' for ' + get_movie_by_id(item_id))
    send_movie_click(USER_ID=user_id, ITEM_ID=item_id, EVENT_TYPE=event_type)
    # Get the recommendations (note you should have a base recommendation DF created before)
    get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = pers_model.campaign_arn,
        userId = str(user_id),
    )
    # Build a new dataframe of recommendations
    item_list = get_recommendations_response['itemList']
    recommendation_list = []
    for item in item_list:
        artist = get_movie_by_id(item['itemId'])
        recommendation_list.append(artist)
    new_rec_DF = pd.DataFrame(recommendation_list, columns = [movie_name])
    # Add this dataframe to the old one
    #recommendations_df = recommendations_df.join(new_rec_DF)
    recommendations_df = pd.concat([recommendations_df, new_rec_DF], axis=1)
    return recommendations_df
```

<!-- #region id="vGn2-eJWdlsb" -->
At this point, we haven't generated any real-time events yet; we have only set up the code. To compare the recommendations before and after the real-time events, let's pick one user and generate the original recommendations for them.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 824} id="Lc8lPDDGdoDn" executionInfo={"status": "ok", "timestamp": 1630182770952, "user_tz": -330, "elapsed": 610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="396914b4-115a-4dd3-e003-53204aae09fc"
# First pick a user
user_id = user

# Get recommendations for the user
get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = pers_model.campaign_arn,
        userId = str(user_id),
    )

# Build a new dataframe for the recommendations
item_list = get_recommendations_response['itemList']
recommendation_list = []
for item in item_list:
    artist = get_movie_by_id(item['itemId'])
    recommendation_list.append(artist)
user_recommendations_df = pd.DataFrame(recommendation_list, columns = [user_id])
user_recommendations_df
```

<!-- #region id="_8jkQ1KBd0wM" -->
Ok, so now we have a list of recommendations for this user before we have applied any real-time events. Now let's pick 3 random artists which we will simulate our user interacting with, and then see how this changes the recommendations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Qm7YjgwjdwNU" executionInfo={"status": "ok", "timestamp": 1630182815022, "user_tz": -330, "elapsed": 15880, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="de938567-aa55-4686-9f92-6cb923b73ad0"
# Next generate 3 random movies
movies = items_df.sample(3).index.tolist()
# Note this will take about 15 seconds to complete due to the sleeps
for movie in movies:
    user_recommendations_df = get_new_recommendations_df_users_real_time(user_recommendations_df, user_id, movie,'click')
    time.sleep(5)
```

<!-- #region id="G1PR2jd2d3O6" -->
Now we can look at how the click events changed the recommendations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="FTG7Zbc3d5Jp" executionInfo={"status": "ok", "timestamp": 1630182818957, "user_tz": -330, "elapsed": 535, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="28d186f8-c0b6-41ff-a5cf-5675bcb2827c"
user_recommendations_df
```

<!-- #region id="XnEy8Q6GoHau" -->
## Generic Module
<!-- #endregion -->

```python id="YBUO768ud7-_"
import boto3
import json
import time


class personalize_inference:
    def __init__(self,
                 dataset_group_arn = None,
                 campaign_arn = None,
                 event_tracker_arn = None,
                 role_arn=None,
                 solution_version_arn=None,
                 batch_job_arn=None
                 ):
        self.personalize = None
        self.personalize_runtime = None
        self.personalize_events = None
        self.dataset_group_arn = dataset_group_arn
        self.campaign_arn = campaign_arn
        self.event_tracker_arn = event_tracker_arn
        self.event_tracker_id = event_tracker_id
        self.role_arn = role_arn
        self.solution_version_arn = solution_version_arn
        self.batch_job_arn = batch_job_arn

    def setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            self.personalize_runtime = boto3.client('personalize-runtime')
            self.personalize_events = boto3.client(service_name='personalize-events')
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")

    def get_recommendations(self, itemid=None, userid=None, k=5,
                            filter_arn=None, filter_values=None):
        get_recommendations_response = self.personalize_runtime.get_recommendations(
            campaignArn = self.campaign_arn,
            itemId = str(itemid),
            userId = str(userid),
            filterArn = filter_arn,
            filterValues = filter_values,
            numResults = k
            )
        
    def get_rankings(self, userid=None, inputlist=None):
        get_recommendations_response = self.personalize_runtime.get_personalized_ranking(
            campaignArn = self.campaign_arn,
            userId = str(userid),
            inputList = inputlist
            )
        
    def create_event_tracker(self, name=None):
        response = self.personalize.create_event_tracker(
            name=name,
            datasetGroupArn=self.dataset_group_arn
            )
        self.event_tracker_arn = response['eventTrackerArn']
        self.event_tracker_id = response['trackingId']
    
    def put_events(self, userid=None, sessionid=None, eventlist=None):
        self.personalize_events.put_events(
            trackingId = self.event_tracker_id,
            userId = userid, 
            sessionId = sessionid,
            eventList = eventlist
            )
        
    def create_batch_job(self, jobname=None, input_path=None, output_path=None):
        response = self.personalize.create_batch_inference_job(
            solutionVersionArn = self.solution_version_arn,
            jobName = jobname,
            roleArn = self.role_arn,
            jobInput = {"s3DataSource": {"path": input_path}},
            jobOutput = {"s3DataDestination": {"path": output_path}}
            )
        self.batch_job_arn = response['batchInferenceJobArn']
    
    def check_batch_job_status(self):
        batch_job_response = self.personalize.describe_batch_inference_job(
        batchInferenceJobArn = self.batch_job_arn
        )
        status = batch_job_response["batchInferenceJob"]['status']
        return status

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['personalize']
        del attributes['personalize_runtime']
        del attributes['personalize_events']
        return attributes
```
