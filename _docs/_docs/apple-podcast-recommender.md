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

<!-- #region id="mDQxJNiBl3VE" -->
# Apple Podcast Recommender
> Scraping, cleaning, exploring and building content-based recommender system to recommend podcasts

- toc: true
- badges: true
- comments: true
- categories: [Visualization, Music, Scraping, TFIDF]
- author: "<a href='https://github.com/Peter-Chong/Podcast-Recommendation-System'>Peter Chong</a>"
- image:
<!-- #endregion -->

<!-- #region id="H4h1i5D8cuii" -->
## Scraping
<!-- #endregion -->

<!-- #region id="ctK9U9UncwHJ" -->
Theese are the scraping scripts:

1. [podcast_ep_scraper.py](https://github.com/sparsh-ai/Podcast-Recommendation-System/blob/main/scraper/podcast_ep_scraper.py)
2. [podcast_info_scraper.py](https://github.com/sparsh-ai/Podcast-Recommendation-System/blob/main/scraper/podcast_info_scraper.py)
3. [podcast_review_scraper.py](https://github.com/sparsh-ai/Podcast-Recommendation-System/blob/main/scraper/podcast_review_scraper.py)
4. [podcast_subs_scraper.py](https://github.com/sparsh-ai/Podcast-Recommendation-System/blob/main/scraper/podcast_subs_scraper.py)
5. [podcast_url_scraper.py](https://github.com/sparsh-ai/Podcast-Recommendation-System/blob/main/scraper/podcast_url_scraper.py)
<!-- #endregion -->

<!-- #region id="3vBE5Hend_u9" -->
These scripts scraped the data from Apple Podcasts using BeautifulSoup (BS4) and stored as JSON. I ran these scripts to validate and they all are working correctly. It would take at least 5 hrs to finish.
<!-- #endregion -->

<!-- #region id="7kxrTCV3erOQ" -->
## Cleaning
<!-- #endregion -->

```python id="pSMV1eDPbQZM"
import re
import json
import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', None)
```

```python id="7cLFtlGjge7z"
!git clone https://github.com/Peter-Chong/Podcast-Recommendation-System.git
!mv /content/Podcast-Recommendation-System/data/json/*.json /content
```

<!-- #region id="sJxu_8pLfvt_" -->
### Merge podcast_info datasets
<!-- #endregion -->

```python id="zICBRJ3AbW_m"
with open('podcast_info.json') as file:
  podcast_1 = json.load(file)
```

```python id="cqFcn2_sfPx1"
with open('podcast_info_add.json') as file:
  podcast_2 = json.load(file)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ncuRy2DsfiLd" outputId="dcd9c40e-0d42-4e38-9997-5915a0338abb"
df1 = pd.DataFrame(podcast_1)
df2 = pd.DataFrame(podcast_2)
print(df1.shape)
print(df2.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="VptXUCypflZF" outputId="18a83e24-ab2c-49d4-bb42-ba6b6b482b2d"
df1.drop(df1[df1.title == ""].index, inplace=True)
print(df1.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="94OWlnYzflv3" outputId="60a489a0-bb9d-4a8f-d6ba-53bc695d98da"
podcast_info = df1.append(df2, ignore_index=True, sort=False)
print(podcast_info.shape)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 502} id="Z6_mzppjfqho" outputId="324303cd-b796-40d4-90d4-52474a82b00e"
podcast_info.head(2)
```

<!-- #region id="jawuz0g3fylX" -->
### Merge podcast_eps datasets
<!-- #endregion -->

```python id="KpDw6T_mfr8J"
with open('podcast_eps.json') as file:
  podcast_1 = json.load(file)
```

```python id="txsGXuWugB56"
with open('podcast_eps_add.json') as file:
  podcast_2 = json.load(file)
```

```python colab={"base_uri": "https://localhost:8080/"} id="HShvCG5ygxQr" outputId="817f936f-821c-42c1-d5da-a292e67d6a5e"
df1 = pd.DataFrame(podcast_1)
df2 = pd.DataFrame(podcast_2)
print(df1.shape)
print(df2.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9EWGyqlPg1tc" outputId="31dbcdb6-805b-4510-ba72-36a0a8ccfce2"
df1.drop(df1[df1.episodes == ""].index, inplace=True)
print(df1.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="fFYbIyTsg16Y" outputId="331e71e8-d027-4cf4-90e3-6a9ab0a520be"
podcast_ep = df1.append(df2, ignore_index=True, sort=False)
print(podcast_ep.shape)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="CAbMhgS0g3La" outputId="039e91e0-ede1-4b9c-e8bc-497d31481c37"
podcast_ep.head(2)
```

<!-- #region id="lxa0wqZ7g5yw" -->
### Merge podcast_reviews datasets
<!-- #endregion -->

```python id="g1SUjQDgg4eo"
with open('podcast_reviews.json') as file:
  podcast_1 = json.load(file)
```

```python id="i9LcC_lSg9AE"
with open('podcast_reviews_add.json') as file:
  podcast_2 = json.load(file)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ctu1cBkYhAWD" outputId="921ab5b6-4a68-4862-b67e-b72d67803b37"
df1 = pd.DataFrame(podcast_1)
df2 = pd.DataFrame(podcast_2)
print(df1.shape)
print(df2.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UjaAOUMAhDDC" outputId="fe2d8b7f-eb8e-4add-fd02-3350483e2bd7"
df1.drop(df1[df1.reviews == ""].index, inplace=True)
print(df1.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RT6iXvkuhEPm" outputId="2034bc75-f53b-4d52-a1f3-784ec45b63b4"
podcast_reviews = df1.append(df2, ignore_index=True, sort=False)
print(podcast_reviews.shape)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 740} id="UNRLfW63hFqI" outputId="a3192ad5-9c35-49a1-c646-9a776211ac8f"
podcast_reviews.head(2)
```

<!-- #region id="1Rb1XWG8hMjF" -->
### Clean podcast_subs dataset
<!-- #endregion -->

```python id="1E1dRBbKhMjF"
with open('podcast_subs.json') as file:
  podcast_subs = pd.DataFrame(json.load(file))
```

```python id="ZnpcdFnuhMjG"
titles = list(podcast_subs.title)
```

```python id="GpRBBtQbhMjG"
for index, row in podcast_subs.iterrows():
    new = []
    for i in row.subs:
        if i in titles:
            new.append(i)
    podcast_subs.at[index, 'subs'] = new
```

```python id="3Um5nOiwhMjH" colab={"base_uri": "https://localhost:8080/"} outputId="25fd4851-ac80-4837-b4ea-b62e4b46cd7d"
print(podcast_subs.shape)
```

```python id="VtwKDYlihMjI" colab={"base_uri": "https://localhost:8080/", "height": 145} outputId="8eb6fdca-ac18-46d3-80d8-a73e961ffdc1"
podcast_subs.head(2)
```

<!-- #region id="ZIpFsoLThMjI" -->
### Merge all datasets
<!-- #endregion -->

```python id="URyyPnQNhMjJ"
df1 = pd.merge(podcast_info, podcast_ep, on="title", how="inner")
```

```python id="XpfWAGOphMjJ"
df2 = pd.merge(df1, podcast_reviews, on="title", how="inner")
```

```python id="PW3zdlUlhMjK"
df = pd.merge(df2, podcast_subs, on="title", how="inner")
```

```python id="agDk01A8hMjK" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="e8154f2e-3423-42fa-e815-daf493859d73"
df.head(2)
```

```python id="6JgH9FJnhMjL" colab={"base_uri": "https://localhost:8080/"} outputId="fd2936ee-9e04-432c-bf01-79dacfec1d71"
df.shape
```

<!-- #region id="q50_cJcmhMjM" -->
### Duplicates
<!-- #endregion -->

```python id="DnFvDqt4hMjN" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="c85ed2dc-3f62-4460-f367-409f8dcea840"
df[df.duplicated(subset=['title'], keep=False)]
```

```python id="VjNaMXgOhMjN"
df.drop([50, 512, 1403, 1718, 2124, 2198, 2371, 2688, 3628, 4369], inplace=True)
```

```python id="zFaYyg3ihMjO" colab={"base_uri": "https://localhost:8080/", "height": 49} outputId="1223b41a-7e5d-4427-ff77-a307cde735cc"
df[df.duplicated(subset=['title'], keep=False)]
```

```python id="4mEoIYHyhMjP" colab={"base_uri": "https://localhost:8080/"} outputId="fdd5a6dd-38c5-4ca0-993e-9a85fbc92cab"
df.shape
```

<!-- #region id="4-A8Xy7zhMjP" -->
### English titles
<!-- #endregion -->

```python id="ARsw6vE9hMjQ"
titles = list(df['title'])
titles = [title.replace(" ", "") for title in titles]
titles = [re.sub(r'[^\w\s]', '', title) for title in titles]
is_english = [bool(re.match("^[A-Za-z0-9]*$", title)) for title in titles]
```

```python id="pOdw8O1_hMjQ"
df = df.loc[is_english, :]
```

```python id="qeSRJhpxhMjR" colab={"base_uri": "https://localhost:8080/"} outputId="f71c9038-c860-4dc9-d0eb-4c982d9ed036"
df.shape
```

```python id="Hev9M3Q2hMjS"
df.reset_index(drop=True, inplace=True)
```

```python id="X8U3eioIhyDd"
def clean_description(desc):
    desc = re.sub(r"http\S+", "", desc)
    desc = re.sub(r"www\S+", "", desc)
    desc = re.sub(r"\S+\.com\S+", "", desc)
    return desc
```

```python id="V6wP6u2ahyDd"
df['description'] = df['description'].map(clean_description)
df['episodes'] = df['episodes'].map(clean_description)
```

```python id="fNAKppDfhyDe"
def clean_reviews(review):
    review = re.sub(r"http\S+", "", review)
    review = re.sub(r"www\S+", "", review)
    review = re.sub(r"\S+\.com\S+", "", review)
    review = review.replace('\n',' ')
    return review
```

```python id="bIWDpoNqhyDf"
df['reviews'] = df['reviews'].map(clean_reviews)
```

```python id="UFIqKn5phyDg" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="aa92f12f-9843-439e-cdf1-0632260a13fd"
df.sample(2)
```

```python id="J4ivl_wjhyDj"
df.to_pickle('podcasts.pkl')
```

<!-- #region id="2WeqpzFeiE5Y" -->
## EDA
<!-- #endregion -->

```python id="EySoUnXPiMNa"
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objs as go
```

```python id="5TjFK65siMNg"
df = pd.read_pickle('podcasts.pkl')
```

```python id="rdpkiB10iMNh" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="1d382320-a48d-4010-8c3d-35c6fcaf6a63"
df.sample(2)
```

```python id="a1TNX1GXiMNk" colab={"base_uri": "https://localhost:8080/"} outputId="4787854e-210e-43cc-cecb-bbb826faab89"
print("Total # of podcasts:", df.shape[0])
```

<!-- #region id="HEcaYJUwiMNl" -->
### Genre
<!-- #endregion -->

```python id="8xR9omZViMNm" colab={"base_uri": "https://localhost:8080/"} outputId="3e65e5e7-6ee1-4126-c5eb-9609da80d9d8"
df.genre.value_counts()
```

<!-- #region id="SmRQfV3uiMNn" -->
Since Self-Improvement, Natural Sciences and Relationships have only 1 podcast each, we change these genre into the Science, Society & Culture and Religion & Spirituality respectively
<!-- #endregion -->

```python id="tURhrHaGiMNp"
df.loc[df['genre'] == "Natural Sciences", 'genre'] = 'Science'
df.loc[df['genre'] == "Relationships", 'genre'] = 'Society & Culture'
df.loc[df['genre'] == "Self-Improvement", 'genre'] = 'Religion & Spirituality'
```

```python id="mgL4R9vhiMNq"
genre_count = df.genre.value_counts().rename_axis('Genres').reset_index(name='Count')
```

```python id="mmsFWUu8iMNq" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="f596c4af-ab32-4b58-ae90-614ea567ddea"
fig = px.bar(genre_count, x='Genres', y='Count', text='Count', title="Number of podcasts by Genre")
fig.update_traces(textposition='outside')
fig.update_layout(xaxis_tickangle=45, yaxis_range=[0,300])
fig.show()
```

```python id="wYKXDScBiMNv"
genre = df[['genre', 'rating']]
genre_rating = genre.groupby('genre').mean().sort_values(by=['rating'], ascending=False).reset_index()
```

```python id="2UpZH_kWiMNv" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="c0f8a571-760f-4a96-d5cf-70210d27c4c4"
fig = px.bar(genre_rating, x='genre', y='rating', title="Average rating by Genre")
fig.update_layout(xaxis_tickangle=45)
fig.show()
```

```python id="ESK5__5hiMNw" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="7bd59f4e-8bf7-41c3-a588-ad7fdec71196"
fig = px.box(df, x="genre", y="num_ratings", labels={
                     "genre": "Genre",
                     "num_ratings": "Number of Ratings"
                 }, title="Boxplot of Number of Ratings")
fig.show()
```

<!-- #region id="ZKUQ2k9ziMNx" -->
We can see that there are some outliers in the number of ratings, hence we will be using median for the next graph instead of mean.
<!-- #endregion -->

```python id="q401mpiDiMNx"
genre = df[['genre', 'num_ratings']]
genre_num_ratings = genre.groupby('genre').median().sort_values(by=['num_ratings'], ascending=False).reset_index()
```

```python id="Tn0w0WtbiMNy" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="89bce045-6a5e-4f40-a707-70bcb6d9ee18"
fig = px.bar(genre_num_ratings, x='genre', y='num_ratings',labels={
                     "genre": "Genre",
                     "num_ratings": "Number of Ratings"
                 }, title="Median number of rating by Genre")
fig.update_layout(xaxis_tickangle=45, yaxis_range=[0,4000])
fig.show()
```

```python id="OSCw3McMiMNy" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="2536dbc6-c107-4aa1-e87b-7e64ebeea1ea"
fig = px.scatter(df, x="num_episodes", y="rating", labels={
                     "rating": "Rating",
                     "num_episodes": "Number of Episodes"
                 }, title="Ratings by number of episodes")
fig.show()
```

<!-- #region id="dYMuRAEuiMNz" -->
I was wondering if the higher number of episode a podcast have, the higher the rating it will be. The above graph shows that I am wrong. There are podcasts with very little number of episodes but still have very high rating.
<!-- #endregion -->

<!-- #region id="gfbs9oI6iMN0" -->
### Network Graph
<!-- #endregion -->

<!-- #region id="s3OUFGiqiMN0" -->
Given a podcast, we know the what other subscribers of that podcast subscribes to. Hence, we can make a network visualizing it. However, since our data is too big, we will just look into one of the genre. Let's look into the Education genre.
<!-- #endregion -->

```python id="tjlRLS1miMN0" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="ec859072-941e-4aac-d430-3c549c085d46"
# Deep copy df with Education genre
df_edu = df[df['genre'] == 'Education'].copy()

# Clean the subs column
titles = list(df_edu.title)
for index, row in df_edu.iterrows():
    new = []
    for i in row.subs:
        if i in titles:
            new.append(i)
    df_edu.at[index, 'subs'] = new

# Initialize a network graph
G = nx.Graph()

# Add nodes and edges into the network graph
node_list = set()
for index,row in df_edu.iterrows():
    if len(row.subs) == 0:
        continue
    node_list.add(row.title)
    for i in row.subs:
        node_list.add(i)
        G.add_edges_from([(row.title, i)])
for i in node_list:
    G.add_node(i)

# Extract the coordinates of nodes
pos = nx.spring_layout(G, k=0.5)

# Adding coordinates of the nodes to the graph
for name, pos in pos.items():
    G.nodes[name]['pos'] = pos

# Adding nodes to the network
node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=1))
    
# Adding  edges to the network
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Add colours to the nodes
node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(adjacencies[0] + ' # of connections: '+str(len(adjacencies[1])))
node_trace.marker.color = node_adjacencies
node_trace.text = node_text

# Plot the network
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Education subscribers also subscribes to',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
```

<!-- #region id="hYtLhBdmiMN2" -->
### Word Cloud
<!-- #endregion -->

```python id="m-xb5a1DiMN4"
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
```

```python id="TQxKACfhiMN4"
def wordcloud(genre):
    text = ' '.join(df[df.genre == genre].description)
    tokenized_text = nltk.word_tokenize(text)
    stop = set(nltk.corpus.stopwords.words('english'))
    stop.update(['podcast', 'podcasts', 'every', 'new', 'weekly', 'week',
                'stories', 'story', 'episode', 'episodes', 'listen', 'us',
                'host', 'hosted', 'join', "'s"])
    texts = []
    for i in tokenized_text:
        if i.lower() not in stop and len(i) != 1:
            texts.append(i)
    texts = ' '.join(texts)
    
    wordcloud = WordCloud(
        background_color='white',
        max_font_size=60, 
        scale=2,
        random_state=123
    ).generate(texts)
    fig = plt.figure(1, figsize=(12, 12))
    plt.title(genre, loc='left', fontsize=20)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="xvyxQBWjkQIo" outputId="0ce070be-8bf0-4c18-82e6-279179668aef"
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

```python id="FHFqZhcziMN5" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="1722c2ab-21a3-4234-83f9-b2e57ebb88cf"
for i in df.genre.unique():
    wordcloud(i)
```

<!-- #region id="-rvjMV0Rkbpe" -->
## Recommender Engine
<!-- #endregion -->

```python id="axarlBnembIM"
!pip install -q umap-learn
```

```python id="7MJ5UG5tkbpZ"
import re
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer

import umap.umap_ as umap
import plotly.express as px
```

```python id="Lxq4lmaykbpf"
df = pd.read_pickle('podcasts.pkl')
```

```python id="odgUdJQykbph"
df['text'] = df[['title', 'producer', 'genre', 'description', 'episodes', 'reviews']].agg(' '.join, axis=1)
```

```python id="gFH3JpYVkbph"
df.drop(columns=['producer', 'rating', 'num_ratings', 'num_episodes', 'description',
                 'link', 'episodes', 'reviews'], inplace=True)
```

```python id="LR46TxE1kbpi"
df['subs_len'] = df.apply(lambda row: len(row.subs), axis=1)
```

```python id="EcctSkaTkbpj" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="3bd54c1a-ef43-4971-8ba7-2d62d0038609"
df.head(2)
```

```python id="rJpkMZCVkbpm" colab={"base_uri": "https://localhost:8080/"} outputId="e3c84436-e457-4675-9f91-07f7863b445f"
df.shape
```

```python id="fipP6MMLkbpn"
titleswsubs = list(df[df.subs_len != 0].title)
```

```python id="kh2Eh4Z_kbpo"
subsset = set()
for i in df.subs:
    for j in i:
        subsset.add(j)
```

<!-- #region id="xEmWC30lkbpo" -->
### Preprocessing
<!-- #endregion -->

```python id="Fc7Icjvukbpp"
stopwords = set(nltk.corpus.stopwords.words('english'))
add_stops = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
             'january', 'february', 'march', 'april', 'may', 'june', 'im', 'ive',
             'july', 'august', 'september', 'october', 'november', 'december',
             'nan', 'podcast', 'podcasts', 'every', 'new', 'weekly', 'week', 
             'stories', 'story', 'episode', 'episodes', 'listen', 'us', "'s", 'host', 'hosted', 'join']
for i in add_stops:
    stopwords.add(i)
```

```python id="_guwmQOLkbpq"
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\s\w]+', '', text)
    text = re.sub(r"\S+\.org\S+", "", text)
    text = re.sub(r"\S+\.net\S+", "", text)
    text = re.sub(r"\S+\.edu\S+", "", text)
    text = re.sub(r"\S+\.gov\S+", "", text)
    tokenized_text = nltk.word_tokenize(text)
    new_tokenized = []
    for i in tokenized_text:
        if i not in stopwords and len(i) != 1:
            new_tokenized.append(lemmatizer.lemmatize(i))
    return(' '.join(new_tokenized))
```

```python id="tzdB73o4kbpr"
df.text = df.text.map(preprocess_text)
```

```python id="sXK_-_Cvkbpr" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="5982bff0-e96a-4da1-b5e4-cd4529659181"
df.head(2)
```

```python id="fuzTLckDkbps"
classes = {
    'Arts':0, 'Business':1, 'Comedy':2, 'TV & Film':3, 'Education':4,
    'News':5, 'Self-Improvement':6, 'Fiction':7, 'Government':8, 'Health & Fitness':9,
    'History':10, 'Society & Culture':11, 'Kids & Family':12, 'Leisure':13, 'Music':14,
    'Religion & Spirituality':15, 'Science':16, 'Natural Sciences':17, 'Relationships':18,
    'Sports':19, 'Technology':20, 'True Crime':21
}
```

```python id="Ro8g1q2akbpt"
y_label = df.loc[:,'genre'].values
y = np.zeros((len(y_label),))
for i in range(len(y)):
    y[i] = classes[y_label[i]]
```

<!-- #region id="IF3rWB5Bkbpu" -->
### Modelling
<!-- #endregion -->

```python id="3qUyZILTkbpu"
tests = list(df[df.subs_len >= 5].sample(3).title)
```

```python id="vfD-7AqAkbpv"
def get_recommendations(matrix):
    for i in tests:
        print('\033[1m' + "Given:" + '\033[0m', i)
        index = df.loc[df.title == i].index[0]
        print('\033[1m' + "Given genre:" + '\033[0m', df.iloc[index]['genre'])
        array = list(enumerate(matrix[index]))
        sorted_array = sorted(array, key=lambda x:x[1], reverse=True)
        recs = []
        genres = []
        for j in sorted_array:
            rec_title = df.iloc[j[0]]['title']
            rec_genre = df.iloc[j[0]]['genre']
            if rec_title == i or rec_title not in subsset:
                continue
            recs.append(rec_title)
            genres.append(rec_genre)
            if len(recs) == 5:
                break
        print('\033[1m' + "Top 5 recommendations:" + '\033[0m')
        print(recs)
        print('\033[1m' + "Top 5 recommendations' genre:" + '\033[0m')
        print(genres)
        print('\033[1m' + "Subscribers also subscribes to according to Apple Podcasts:" + '\033[0m')
        for k in df.loc[df.title == i].subs:
            substo = k
        print(substo)
        correct  = 0
        for l in recs:
            correct = correct + 1 if l in substo else correct
        print('\033[1m', correct , "out of 5 are accurate" + '\033[0m'+ "\n")
```

```python id="PbFIbEHGkbpw"
def accuracy(matrix):
    num_titles = len(titleswsubs)
    acc = 0
    for i in titleswsubs:
        index = df.loc[df.title == i].index[0]
        array = list(enumerate(matrix[index]))
        sorted_array = sorted(array, key=lambda x:x[1], reverse=True)
        recs = []
        for j in sorted_array:
            rec_title = df.iloc[j[0]]['title']
            if rec_title == i or rec_title not in subsset:
                continue
            recs.append(rec_title)
            if len(recs) == 5:
                break
        for k in df.loc[df.title == i].subs:
            substo = k
        correct = 0
        for l in recs:
            correct = correct + 1 if l in substo else correct
        if correct >= len(substo)//2 or correct == 5:
            acc += 1
    return round(acc/num_titles,5)
```

```python id="VK8Z2WmSkbpx"
def showUMAP(matrix, title):
    mat_df = pd.DataFrame(matrix.toarray())
    x = mat_df.values
    embedding = umap.UMAP(n_components = 2).fit_transform(x, y=y)
    plot = pd.DataFrame(embedding)
    plot.columns = ['UMAP1', 'UMAP2']
    plot['labels'] = y_label
    fig = px.scatter(plot, x='UMAP1', y='UMAP2', color = 'labels', title=title)
    fig.update_traces(marker = dict(size=4))
    fig.show()
```

<!-- #region id="GYehklIQkbpy" -->
### CountVectorizer (Bag-of-words) + Cosine Similarity
<!-- #endregion -->

```python id="rQxmmQZ9kbpz"
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
```

```python id="skGPolD6kbpz"
bow = CountVectorizer(min_df=3)
bow_matrix = bow.fit_transform(df.text)
bow_cos_sim = cosine_similarity(bow_matrix)
```

```python id="hyEh6RcLkbp0" colab={"base_uri": "https://localhost:8080/"} outputId="003d546b-6055-46ee-d789-f2378c4d4f43"
bow_matrix.shape
```

```python id="ke8m8NThkbp0" colab={"base_uri": "https://localhost:8080/"} outputId="59f92720-494e-4d9f-b9fd-6dd8f9951af0"
get_recommendations(bow_cos_sim)
```

```python id="YmASO2cxkbp1" colab={"base_uri": "https://localhost:8080/"} outputId="94c261c9-c878-4327-f167-ca2e93863ec5"
accuracy(bow_cos_sim)
```

```python id="gmo9Z6Dakbp1" colab={"base_uri": "https://localhost:8080/", "height": 630} outputId="8530e659-2343-4025-c31e-ed5804fbf793"
showUMAP(bow_matrix, "CountVectorizer")
```

<!-- #region id="_90xq868kbp5" -->
### TF-IDF + Cosine Similarity
<!-- #endregion -->

```python id="PqetLPQckbp6"
from sklearn.feature_extraction.text import TfidfVectorizer
```

```python id="BvstpFBEkbp7"
tf = TfidfVectorizer(min_df=3)
tf_idf_matrix = tf.fit_transform(df.text)
tf_idf_cos_sim = cosine_similarity(tf_idf_matrix)
```

```python id="WgjjU1H0kbp7" colab={"base_uri": "https://localhost:8080/"} outputId="c0ad63fc-8cea-4833-8704-f5e8a161760e"
tf_idf_matrix.shape
```

```python id="0HgUJPVqkbp7" colab={"base_uri": "https://localhost:8080/"} outputId="23a4d994-c7c0-4a50-acc9-dd5858660ac1"
get_recommendations(tf_idf_cos_sim)
```

```python id="IBBVsuyAkbp8" colab={"base_uri": "https://localhost:8080/"} outputId="818af124-bf6d-493f-8bff-2eb9f894fd9e"
accuracy(tf_idf_cos_sim)
```

```python id="XB93jH5lkbp8" colab={"base_uri": "https://localhost:8080/", "height": 542} outputId="7c53aebc-373b-423a-8fd8-ebc875c7753c"
showUMAP(tf_idf_matrix, "TF-IDF")
```
