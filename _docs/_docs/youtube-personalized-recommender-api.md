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

<!-- #region id="xoGgm_pIqxgD" -->
# How I personalized my YouTube recommendation using YT API
> How to utilize most of YouTube’s API?

- toc: true
- badges: true
- comments: true
- categories: [video]
- image: 
<!-- #endregion -->

<!-- #region id="xA0COmyerKC8" -->
![yt_anime](https://miro.medium.com/max/513/0*RjycuvCHOmNiI8fd.gif)
<!-- #endregion -->

<!-- #region id="1VXVMbg6rVtg" -->
Source: https://pub.towardsai.net/how-i-personalized-my-youtube-recommendation-using-yt-api-d20f6174bdaa
<!-- #endregion -->

```python id="sIt-dy9jvAfa"
pip install google-api-python-client
```

```python id="RxhyZwk4vEV5"
from datetime import datetime, timedelta
import pandas as pd

start_time = datetime(year=2020, month=10, day=1).strftime('%Y-%m-%dT%H:%M:%SZ')
end_time = datetime(year=2021, month=5, day=11).strftime('%Y-%m-%dT%H:%M:%SZ')

from apiclient.discovery import build
api_key = 'AIzaSyCjHWHTmed0fhMZJDRdedQDku5qJv12xkY' # Enter your own API key – this one won’t work

youtube = build('youtube', 'v3', developerKey=api_key)

results = youtube.search().list(q="Twenty One Pilots", part="snippet", type="video", order="viewCount",publishedAfter=start_time,
                            publishedBefore=end_time, maxResults=5).execute()
```

```python colab={"base_uri": "https://localhost:8080/"} id="jfai6fVmyBHp" outputId="b7dd0cd1-1afb-4677-b2d2-42dcd5e742d8"
results
```

```python colab={"base_uri": "https://localhost:8080/"} id="qwRQzsgdDM8w" outputId="00ecc699-f26a-4931-e44f-a145ec0ea423"
#Get statistical attributes with video ID geW09OOqieU
video_statistics = youtube.videos().list(id='geW09OOqieU',
                                        part='statistics').execute()
video_statistics
```

```python id="6WpLO0VFyMSG" colab={"base_uri": "https://localhost:8080/"} outputId="4bf73250-4c2f-472b-b4c3-17e2692f40e4"
#Printing just Title, time and videoID
for item in sorted(results['items'], key=lambda x:x['snippet']['publishedAt']):
    print(item['snippet']['title'], item['snippet']['publishedAt'], item['id']['videoId'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="yIdoU5byzt2D" outputId="0e316138-5604-4d7d-a687-572a14c297ed"
#Converting time frame into generic time format as described in YT API Documentation
def get_start_date_string(search_period_days):
    """Returns string for date at start of search period."""
    search_start_date = datetime.today() - timedelta(search_period_days)
    date_string = datetime(year=search_start_date.year,month=search_start_date.month,
                           day=search_start_date.day).strftime('%Y-%m-%dT%H:%M:%SZ')
    return date_string
get_start_date_string(30)
```

```python id="XIC3ky0F0vg-"
def find_title(item):
    title = item['snippet']['title']
    return title

def find_video_url(item):
    video_id = item['id']['videoId']
    video_url = "https://www.youtube.com/watch?v=" + video_id
    return video_url

def find_viewcount(item, youtube):
    video_id = item['id']['videoId']
    video_statistics = youtube.videos().list(id=video_id,
                                        part='statistics').execute()
    viewcount = int(video_statistics['items'][0]['statistics']['viewCount'])
    return viewcount

def find_likecount(item, youtube):
    video_id = item['id']['videoId']
    video_statistics = youtube.videos().list(id=video_id,
                                        part='statistics').execute()
    likecount = int(video_statistics['items'][0]['statistics']['likeCount'])
    return likecount

def find_dislikecount(item, youtube):
    video_id = item['id']['videoId']
    video_statistics = youtube.videos().list(id=video_id,
                                        part='statistics').execute()
    dislikecount = int(video_statistics['items'][0]['statistics']['dislikeCount'])
    return dislikecount

def find_channel_id(item):
    channel_id = item['snippet']['channelId']
    return channel_id

def find_channel_url(item):
    channel_id = item['snippet']['channelId']
    channel_url = "https://www.youtube.com/channel/" + channel_id
    return channel_url

def find_channel_title(channel_id, youtube):
    channel_search = youtube.channels().list(id=channel_id,
                                            part='brandingSettings').execute()
    channel_name = channel_search['items'][0]\
                                    ['brandingSettings']['channel']['title']
    return channel_name

def find_num_subscribers(channel_id, youtube):
    subs_search = youtube.channels().list(id=channel_id,
                                            part='statistics').execute()
    if subs_search['items'][0]['statistics']['hiddenSubscriberCount']:
        num_subscribers = 1000000
    else:
        num_subscribers = int(subs_search['items'][0]\
                                    ['statistics']['subscriberCount'])
    return num_subscribers

def view_to_sub_ratio(viewcount, num_subscribers):
    if num_subscribers == 0:
        return 0
    else:
        ratio = viewcount / num_subscribers
        return ratio

def how_old(item):
    when_published = item['snippet']['publishedAt']
    when_published_datetime_object = datetime.strptime(when_published,
                                                        '%Y-%m-%dT%H:%M:%SZ')
    today_date = datetime.today()
    days_since_published = int((today_date - when_published_datetime_object).days)
    if days_since_published == 0:
        days_since_published = 1
    return days_since_published

def find_count(q, item):
  if q in item['snippet']['title'] and item['snippet']['description']: 
    count+=1
  return count

def custom_score(likecount, dislikecount, viewcount, ratio, days_since_published):
    ratio = min(ratio, 10) 
    score = (viewcount * ratio) / days_since_published
    return score + (likecount/dislikecount) + count
```

```python id="IkGgO6XZ2LBd"
def search_each_term(search_terms, api_key, uploaded_since,
                        views_threshold=10000, num_to_print=5):
    """Uses search term list to execute API calls and print results."""
    if type(search_terms) == str:
        search_terms = [search_terms]

    list_of_dfs = []
    for index, search_term in enumerate(search_terms):
        df = find_videos(search_terms[index], api_key, views_threshold=views_threshold,
                         uploaded_since = uploaded_since)
        df = df.sort_values(['Custom_Score'], ascending=[0])
        list_of_dfs.append(df)

    # 1 - concatenate them all
    full_df = pd.concat((list_of_dfs),axis=0)
    full_df = full_df.sort_values(['Custom_Score'], ascending=[0])
    print("THE TOP VIDEOS OVERALL ARE:")
    print_featured_videos(full_df, num_to_print)
    print("==========================\n")

    # 2 - in total
    for index, search_term in enumerate(search_terms):
        results_df = list_of_dfs[index]
        print("THE TOP VIDEOS FOR SEARCH TERM '{}':".format(search_terms[index]))
        print_featured_videos(results_df, num_to_print)

    results_df_dict = dict(zip(search_terms, list_of_dfs))
    results_df_dict['top_videos'] = full_df

    return results_df_dict


def find_videos(search_terms, api_key, views_threshold, uploaded_since):
    """Calls other functions (below) to find results and populate dataframe."""

    # Initialise results dataframe
    dataframe = pd.DataFrame(columns=('Title', 'Video URL', 'Custom_Score',
                            'Views', 'Channel Name','Num_subscribers',
                            'View-Subscriber Ratio','Channel URL'))

    # Run search
    search_results, youtube_api = search_api(search_terms, api_key,
                                                        uploaded_since)

    results_df = fill_dataframe(search_results, youtube_api, dataframe,
                                                        views_threshold)

    return results_df


def search_api(search_terms, api_key, uploaded_since):
    """Executes search through API and returns result."""

    # Initialise API call
    youtube_api = build('youtube', 'v3', developerKey = api_key)

    #Make the search
    results = youtube_api.search().list(q=search_terms, part='snippet',
                                type='video', order='viewCount', maxResults=50,
                                publishedAfter=uploaded_since).execute()

    return results, youtube_api


def fill_dataframe(results, youtube_api, df, views_threshold):
    """Extracts relevant information and puts into dataframe"""
    # Loop over search results and add key information to dataframe
    i = 1
    for item in results['items']:
        viewcount = find_viewcount(item, youtube_api)
        #likecount = find_likecount(item, youtube_api)
        #dislikecount = find_dislikecount(item, youtube_api)
        if viewcount > views_threshold:
            title = find_title(item)
            video_url = find_video_url(item)
            channel_url = find_channel_url(item)
            channel_id = find_channel_id(item)
            channel_name = find_channel_title(channel_id, youtube_api)
            num_subs = find_num_subscribers(channel_id, youtube_api)
            ratio = view_to_sub_ratio(viewcount, num_subs)
            days_since_published = how_old(item)
            likecount = find_likecount(item, youtube)
            dislikecount = find_dislikecount(item, youtube)
            score = custom_score(likecount, dislikecount, viewcount, ratio, days_since_published)
            df.loc[i] = [title, video_url, score, viewcount, channel_name,\
                                    num_subs, ratio, channel_url]
        i += 1
    return df


def print_featured_videos(df, num_to_print):
    """Prints top videos to console, with details and link to video."""
    if len(df) < num_to_print:
        num_to_print = len(df)
    if num_to_print == 0:
        print("NO RESULTS")
    else:
        for i in range(num_to_print):
            video = df.iloc[i]
            title = video['Title']
            views = video['Views']
            subs = video['Num_subscribers']
            link = video['Video URL']
            print("Video #{}:\nThe video '{}' has {} views, from a channel \
with {} subscribers and can be viewed here: {}\n"\
                                        .format(i+1, title, views, subs, link))
            print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n")

```

```python id="idAkpZFS-7ot" colab={"base_uri": "https://localhost:8080/"} outputId="bf94515a-efe1-4ab1-c011-60ab832c6dbe"
search_each_term("Data Science", api_key, '2021-01-11T00:00:00Z' )
```
