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

<!-- #region id="ho1_dZhI7lvx" -->
# Udemy Course Recommender
> Fetching Business course data from Udemy using official Udemy API 2.0, cleaning the data, exploring and building content-based course recommendation system

- toc: true
- badges: true
- comments: true
- categories: [Education, Scraping, Visualization, KMeans, PCA]
- author: "<a href='https://github.com/adrij/Udemy-recommender-system/tree/master/'>Adrienn Kitti Juhasz</a>"
- image:
<!-- #endregion -->

<!-- #region id="u_Y-zOshpyuE" -->
Udemy.com is an online learning platform with more than 100.000 courses and over 30 million students all over the world. The platform offers courses in different categories e.g. Business, Design or Marketing. With all the available options it is very hard to choose the proper course, since everyone has a different taste. A recommender system helps students choose the next course, without spending hours reading different course descriptions. It does not only spare time for the user, but helps to find something interesting based on their previous course choices.
<!-- #endregion -->

<!-- #region id="qvpL4Rd5NfP6" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8BPJcz4vqxw-" outputId="52906784-9f82-4e90-f819-1b1c71a15e38"
!pip install squarify
```

```python id="XtO6XPFSp98z"
#Import allrelevant libraries
import pandas as pd
import numpy as np

import scipy.stats as st

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import requests
import os

import requests
import ast
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from operator import itemgetter
from collections import Counter
import matplotlib
import squarify
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import datetime
import scipy.stats as st
import ast
import re 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
```

```python id="zKpppI71qp_7" cellView="form"
#hide
#@markdown Utils
#get the data from the Udemy 
def get_data(url, username, pw):
    r = requests.get(url, auth=(username,pw))
    data_json=r.json()
    return data_json

#transform dataframe columns with dict/list values 
def transform_col(col, col_key=None):
	if col_key:
		return col.apply(lambda x: ast.literal_eval(x).get(col_key))
	else:
		return col.apply(lambda x: ast.literal_eval(x))

def get_float(text):
    r=re.search('\d+\.*\d*', text)
    if r:
        return float(r.group(0))
    else:
        return np.nan

def remove_tags(text):
    tag_re = re.compile(r'<[^>]+>')
    if text==text:
        return tag_re.sub('', text).replace('\n',' ').replace('\xa0',' ').replace('\t',' ')

#functions for stemming
def tokenize(text):
    stemmer=SnowballStemmer('english')
    return [stemmer.stem(word) for word in word_tokenize(text.lower())]

def tokenize_only(text):
    return [word for word in word_tokenize(text.lower())]

def combine_list(l):
    new_str=""
    for item in l:
        new_str=new_str+' '+str(item) 
    return new_str

def vocab_stem(text):
    stemmer=SnowballStemmer('english')
    total_stemmed = []
    total_tokenized = []
    for i in text:
        obj_stemmed = tokenize(i) 
        total_stemmed.extend(obj_stemmed) 
        obj_tokenized = tokenize_only(i)
        total_tokenized.extend(obj_tokenized)
    vocab_frame = pd.DataFrame({'words': total_tokenized}, index = total_stemmed)
    #print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vocab_frame 

def drop_words(vocab_frame):
    vocab_frame=vocab_frame.reset_index()
    vocab_frame.columns = ['index','words']
    vocab_frame=vocab_frame.drop_duplicates(subset='index', keep='first').set_index('index')
    #print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vocab_frame

def get_words_count(text, StopWords, vocab_frame):
    list_words=[tokenize(items) for items in text]

    word=[word for words in list_words for word in words if word not in StopWords]
    words_dict=Counter(word)
    if vocab_frame is None:
        return sorted(words_dict.items(), key=itemgetter(1), reverse=True)
    else:
        words_dict_new={}
        for k,v in words_dict.items():
            word_new=vocab_frame.loc[k].values.tolist()[0]
            words_dict_new[word_new]=v
        return sorted(words_dict_new.items(), key=itemgetter(1), reverse=True)

def top_words_graph(df_courses, attribute, comb_list, kind, StopWords, vocab_frame):
    for subcat in df_courses['primary_subcategory'].unique():
        temp=df_courses[df_courses['primary_subcategory']==subcat]
        if comb_list:
            text=temp[attribute].apply(combine_list).values
        else:
            text=temp[attribute].values
        top_words=get_words_count(text, StopWords, vocab_frame)[:25]
        plt.subplots(figsize=(10,8))
        if kind=='bar':
            plt.barh(range(len(top_words)), [val[1] for val in top_words], align='center')
            plt.yticks(range(len(top_words)), [val[0] for val in top_words])
            plt.xlabel("Number of occurences")    
        else:
            top_words=dict(top_words)
            wordcloud = WordCloud(width=900,height=500, margin=0).generate_from_frequencies(top_words)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
        plt.title(subcat)

def get_label(index, vocab_frame, word_features):
    return vocab_frame.loc[word_features[index]].values.tolist()[0]


def get_common_words(model, count_words):
    count_words_new=count_words*(-1)-1
    common_words = model.cluster_centers_.argsort()[:,-1:count_words_new:-1]
    return common_words

def print_common_words(common_words, word_features, vocab_frame, print_list=True):
    dict_cluster={}
    for num, centroid in enumerate(common_words):
        dict_cluster[num]=[get_label(word, vocab_frame, word_features) for word in centroid]
        if print_list:
            print(str(num) + ' : ' + ', '.join(dict_cluster[num]))
    if print_list==False:
        return dict_cluster
    
def plot_common_words(model, n_words, word_features, vocab_frame, df_courses, cluster_name):
    common_words=get_common_words(model, n_words)
    dict_cluster=print_common_words(common_words, word_features, vocab_frame, False)
    fig, ax=plt.subplots(figsize=(12,5))
    keys=df_courses[cluster_name].value_counts().sort_index().index
    values=df_courses[cluster_name].value_counts().sort_index().values
    colors=['b', 'g', 'y','r', 'k', 'grey', 'purple','orange', 'pink', 'brown']
    for j in range(len(keys)):
        ax.bar(keys[j], values[j], width=0.8, bottom=0.0, align='center', color=colors[j], alpha=0.4, label=dict_cluster[j]) 
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(['cluster '+str(k) for k in keys])
    ax.set_ylabel('Number of courses')
    ax.set_title('Distribution of clusters with the top ' + str(n_words) + ' words')
    plt.legend(fontsize=13)
    
def squarify_words(common_words, word_features, vocab_frame):
    colormaps=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds','Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
           'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    for num, centroid in enumerate(common_words):
        sizes=np.arange(10,10+len(centroid))
        cmap_name=colormaps[num]
        cmap = plt.get_cmap(cmap_name)
        labels=[get_label(word, vocab_frame, word_features) for word in centroid]
        mini=min(sizes)
        maxi=max(sizes)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in sizes]
        squarify.plot(sizes=sizes, label=labels,alpha=0.6, color=colors)
        plt.title("Most frequent words in cluster "+str(num))
        plt.show()

def heatmap_categories_cluster(cluster_name, df_courses, cmap ):
    clusters = df_courses.groupby([cluster_name, 'primary_subcategory']).size()
    fig, ax = plt.subplots(figsize = (30, 15))
    sns.heatmap(clusters.unstack(level = 'primary_subcategory'), ax = ax, cmap = cmap)
    ax.set_xlabel('primary_subcategory', fontdict = {'weight': 'bold', 'size': 24})
    ax.set_ylabel(cluster_name, fontdict = {'weight': 'bold', 'size': 24})
    for label in ax.get_xticklabels():
        label.set_size(16)
        label.set_weight("bold")
    for label in ax.get_yticklabels():
        label.set_size(16)
        label.set_weight("bold")   

def get_inertia(data, nClusterRange):
    inertias = np.zeros(len(nClusterRange))
    for i in range(len(nClusterRange)):
        model = KMeans(n_clusters=i+1, init='k-means++', random_state=1234).fit(data)
        inertias[i] = model.inertia_
    return inertias

def plot_inertia(kRange, inertia_Kmean):
    plt.figure(figsize=(10,8))
    plt.plot(kRange, inertia_Kmean, 'o-', color='seagreen', linewidth=3)
    #plt.plot([6], [testKmean[5]], 'o--', color='dimgray', linewidth=3)
    #plt.plot([1,6,11], [8520, 8170,7820], '--', color='k', linewidth=1)
    #plt.annotate("Let's try k=6", xy=(6, testKmean[5]), xytext=(6,7700),
             #size=14, weight='bold', color='dimgray',
             #arrowprops=dict(facecolor='dimgray', shrink=0.05))
    plt.xlabel('k [# of clusters]', size=18)
    plt.ylabel('Inertia', size=14)
    plt.title('Inertia vs KMean Parameter', size=14)

def print_titles_cluster(n_title, df_courses, cluster_name):
    for i in df_courses[cluster_name].unique():
        temp=df_courses[df_courses[cluster_name]==i]
        print(temp['published_title'].values[:n_title])

#functions for hierarchical clustering:
def get_linkage(X ):
    dist=pdist(X.todense(), metric='euclidean')
    z = linkage(dist, 'ward')
    return z

def plot_dendrogram(z, last_p_show, line_dist=None):
    # lastp is telling the algorithm to truncate using the number of clusters we set
    plt.figure(figsize=(20,10))
    plt.title('Dendrogram for attribute objectives')
    plt.xlabel('Data Index')
    plt.ylabel('Distance (ward)')
    dendrogram(z, orientation='top', leaf_rotation=90, p=last_p_show, truncate_mode='lastp', show_contracted=True);
    if line_dist!=None:
        plt.axhline(line_dist, color='k')

def plot_with_pca (X, labels, plot_n_sample):
    pca=PCA(n_components=2)
    X_2d=pca.fit_transform(X.todense())
    print('The explained variance through the first 2 principal comonent is {}.'
          . format(round(pca.explained_variance_ratio_.sum(),4)))
    df = pd.DataFrame(dict(x=X_2d[:,0], y=X_2d[:,1], label=labels)) 
    df_sample=df.sample(plot_n_sample)
    groups = df_sample.groupby('label')
    cluster_colors=['b', 'g', 'y','r', 'k', 'grey', 'purple','orange', 'pink', 'brown']
    fig, ax = plt.subplots(figsize=(17, 9)) 
    for name in np.arange(len(df_sample['label'].unique())):
        temp=df_sample[df_sample['label']==name]
        ax.plot(temp.x, temp.y, marker='o', linestyle='', ms=12, 
            label='cluster '+str(name), 
            color=cluster_colors[name], 
            mec='none', alpha=0.6)
        ax.set_aspect('auto')
        ax.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis= 'y', which='both', bottom='off', top='off', labelbottom='off')
    ax.legend(numpoints=1) 
    plt.title('Courses with PCA decompostion')

#functions for the recommender system
def normalize_features(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = StandardScaler().fit_transform(df_norm[col].values.reshape(-1, 1))
    return df_norm

def recommend_courses(id, n_courses, df_courses, df_norm):
    n_courses=n_courses+1
    id_=df_courses[df_courses['id']==id].index.values
    title=df_courses[df_courses['id']==id]['published_title']
    X = df_norm.values
    Y = df_norm.loc[id_].values.reshape(1, -1)
    cos_sim = cosine_similarity(X, Y)
    df_sorted=df_courses.copy()
    df_sorted['cosine_similarity'] = cos_sim
    df_sorted=df_sorted.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)

    return title, df_sorted.iloc[1:n_courses][['published_title', 'cosine_similarity']]

def recommend_for_user(user_name, n_courses, df_reviews, df_courses, df_norm):
    list_courses=df_reviews[df_reviews['user_name']==user_name]['id'].values
    len_courses=len(list_courses)
    index_courses=df_courses[df_courses['id'].isin(list_courses)].index
    for id in list_courses:
        title, df_recommend= recommend_courses(id, n_courses, df_courses, df_norm)
        print('The following courses are recommended after taking the course {} with the id {}:'
          .format(title.values[0],id))
        print(df_recommend)
        print()
    if len_courses>1:
        n_courses=n_courses+1
        df_temp=df_courses.copy()
        for i, id in enumerate(list_courses):
            id_=df_courses[df_courses['id']==id].index.values
            X = df_norm.values
            Y = df_norm.loc[id_].values.reshape(1, -1)
            cos_sim = cosine_similarity(X, Y)
            df_temp[i] = cos_sim
        temp_avg=df_temp.iloc[:,-len_courses:].mean(axis=1).values
        df_temp['avg_cos_sim']=temp_avg
        df_temp.drop(index=index_courses, inplace=True)
        df_temp=df_temp.sort_values('avg_cos_sim', ascending=False).reset_index(drop=True)
        print('The following courses are recommended after all taken courses:')
        print(df_temp.iloc[1:n_courses][['published_title', 'avg_cos_sim']])
```

<!-- #region id="r7ZLOQKVqACd" -->
## Data Import
<!-- #endregion -->

<!-- #region id="qxpE94BpNciH" -->
### Import the courses

Import the Business courses from the Udemy API  - limit is 10.000 
<!-- #endregion -->

<!-- #region id="j3FTHHlRrqLG" -->
> Tip: You can go to Udemy's site and request for *Affiliate API* access. For me, it took 7 hours to get approval.
<!-- #endregion -->

```python id="-94ZixuGrCcB"
#hide-output
username='oZAbbHY1iopJmQlRBmUOvepJVAmEadBfLARYO42N'
pw = os.environ['udemy_client_secret']

list_json=[]
url='https://www.udemy.com/api-2.0/courses/?fields[course]=@all&page=1&category=Business'

global_counter = 0
local_counter = 0

while url!=None:
  if not os.path.exists(f"./courses_{global_counter+10}.txt"):
    try:
        local_counter+=1
        data_json=get_data(url, username, pw)
        url=data_json['next']
        list_json.extend(data_json['results'])
        if local_counter%10==0:
          local_counter = 0
          global_counter+=10
          with open(f"./courses_{global_counter}.txt", "wb") as fp:
            pickle.dump(list_json, fp)
          list_json = []
          print("Stored {} results!".format(global_counter))
    except:
        print(global_counter)
        continue
  else:
    global_counter+=10
    print("Stored {} results!".format(global_counter))
```

<!-- #region id="0tERIpVGFuE8" -->
<!-- #endregion -->

```python id="Lvy_aRANFHy6"
import glob
all_chunks = glob.glob("./*.txt")

list_json = []

for chunk in all_chunks:
  with open(chunk, "rb") as fp:
    list_json.extend(pickle.load(fp))
```

```python colab={"base_uri": "https://localhost:8080/"} id="l5yi8E2yHnJf" outputId="0991ca55-f647-437b-da14-4278e2b488f7"
len(list_json)
```

```python id="wMVzfXQ574QF"
#Save the result in a dataframe and export to csv file
df_courses = pd.DataFrame.from_dict(list_json)
df_courses.to_csv('df_courses.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 763} id="giQTxKalRZXY" outputId="f21c7502-78ad-493f-a2b2-4fd5cae83c7f"
df_courses.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 307} id="V3fdJBOkvUzA" outputId="f15c4f20-2015-4918-98ba-0a6a094bcb46"
df_courses.describe(include=['O'])
```

<!-- #region id="ybywcEUjrFuf" -->
### Import the reviews

For these courses, I downloaded the available reviews. The maximum number of available reviews for a course is 10.000.
<!-- #endregion -->

```python id="b2GlNeYNpz3i"
#hide-output
local_counter = 0
global_counter = 0

for j, id_ in enumerate(df_courses['id'].values):    
    url="https://www.udemy.com/api-2.0/courses/{}/reviews/?page=1&page_size=100".format(id_)
    list_json_review=[]
    while url!=None:
        try:
          data_json=get_data(url, username, pw)
          url=data_json['next']
          list_json_review.extend(data_json['results'])
          if local_counter%100==0:
            local_counter = 0
            global_counter+=100
            with open(f"reviews_{global_counter}.txt", "wb") as fp:
              pickle.dump(list_json_review, fp)
            list_json_review = []
            print("Stored {} results!".format(global_counter))
        except:
        	continue
    if j==0:
        df_review= pd.DataFrame.from_dict(list_json_review)
        df_review['id']=id_
    else:
        df_review_unique = pd.DataFrame.from_dict(list_json_review)
        df_review_unique['id']=id_
        df_review = pd.concat([df_review, df_review_unique])
```

<!-- #region id="y0lEXXkWKD-1" -->
<!-- #endregion -->

```python id="Zb9RKBzUJZPK"
import glob
all_chunks = glob.glob("./Udemyreviews/*.txt")

list_json_review = []

for chunk in all_chunks:
  with open(chunk, "rb") as fp:
    list_json_review.extend(pickle.load(fp))
```

```python id="6tn0SEHoI1of"
df_review = pd.DataFrame.from_dict(list_json_review)
df_review.to_csv('df_review.csv')
```

<!-- #region id="yV5nELIM3ViP" -->
### Data Persistence

We will store these 2 datasets in parquet copressed format for efficient and quick loading in future.
<!-- #endregion -->

```python id="OnbmU4zf3rcz"
df_courses.to_parquet('udemy_courses.parquet.gzip', compression='gzip')
df_review.to_parquet('udemy_reviews.parquet.gzip', compression='gzip')
```

<!-- #region id="SXR0R8mvsx5y" -->
## Data Cleaning
<!-- #endregion -->

<!-- #region id="dVuceBC7s5Fy" -->
Through the data cleaning process I did the following operations on the raw dataset:
1. import the raw data
2. transform the relevant columns
3. filter the dataset
4. keep only the relevant columns
5. drop the duplicates
6. treat the missing values
7. save the cleaned data
<!-- #endregion -->

<!-- #region id="-QKuUbcbtJQI" -->
### Cleaning the course data
<!-- #endregion -->

```python id="pYsYCOTVtNHV"
# 1. import the raw data
df_courses = pd.read_parquet("https://github.com/sparsh-ai/reco-data/raw/master/udemy/udemy_courses.parquet.gzip")
```

```python colab={"base_uri": "https://localhost:8080/"} id="xjy-fJpPPUFO" outputId="e15afb16-7393-4da2-ac23-8db421621e32"
df_courses.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 763} id="G5GUuOEPOz8j" outputId="0b2b188a-682b-41e2-e3a6-e57016ea46c4"
df_courses.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="RZ5VLfKVPGpN" outputId="f9b42f93-3589-40b0-d4bd-69f5d12e99fc"
df_courses.columns.tolist()
```

```python id="0P9YM67AtSWg"
# 2. transform the relevant columns
df = df_courses.copy()
df['primary_category']=transform_col(df['primary_category'], 'title')
df['primary_subcategory']=transform_col(df['primary_subcategory'], 'title')
df['content_info']=df['content_info'].apply(get_float)
df['price']=df['price'].apply(get_float)
df['published_time']=pd.to_datetime(df['published_time']).dt.tz_convert(None)
df['published_since_month']=(datetime.datetime.now()-df['published_time']).apply(lambda x: int(x.days/30))
df['objectives']=transform_col(df['objectives'])
df['description'] = df['description'].fillna('description not available')
df['description_text']=df['description'].apply(remove_tags)
```

```python id="bfs0cmxVtSUS"
#transform the rating distribution 
rating_orig=[]
rating_rel=[]
for i, rating in enumerate(df['rating_distribution'].values):
    total=0
    temp={}
    temp_rel={}
    if rating:
        rating=ast.literal_eval(rating)
        for rating_j in rating:
            j=rating_j['rating']
            count_j=rating_j['count']
            total+=count_j
            temp[j]=count_j
        rating_orig.append(temp)
        if total>0:
            for k,v in temp.items():
                temp_rel[k]=round(v*1.0/total,3)
            rating_rel.append(temp_rel)
        else:
            rating_rel.append({1:0, 2:0, 3:0, 4:0, 5:0})
    else:
        rating_rel.append({1:0, 2:0, 3:0, 4:0, 5:0})
        rating_orig.append({1:0, 2:0, 3:0, 4:0, 5:0})
df_rating=pd.DataFrame(rating_rel)
df_rating.columns=['rating_1', 'rating_2', 'rating_3', 'rating_4','rating_5']
df=pd.concat([df, df_rating], axis=1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="b5b75x8YPy7j" outputId="f5baac49-e8cc-45b2-a47f-29ec4ae9bab8"
df.shape
```

```python id="mp5EVIQrtSSC"
# 3. filter the dataset
df=df[(df['is_published']== True ) & (df['status_label']== 'Live')]
#drop the columns that are transformed or not relevant any more
df.drop(columns=['published_time','rating_distribution','status_label', 'is_published', 'rating', 'description' ], axis=1, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="DD_PfEJXP2WB" outputId="5c148d07-ce29-4fa5-9f07-f166f8b905c1"
df.shape
```

```python id="u6M95-yDtSQA"
# 4. keep the relevant columns
cols=['avg_rating', 'avg_rating_recent', 'description_text', 'has_certificate',  'is_paid',
      'id', 'instructional_level', 'is_enrollable_on_mobile', 'is_owned_by_instructor_team', 'is_practice_test_course', 
      'num_article_assets' , 'num_curriculum_items','num_lectures', 'num_practice_tests', 'num_quizzes',
      'num_subscribers', 'num_reviews', 'objectives', 'price','published_title', 'relevancy_score','rating_1', 
      'rating_2', 'rating_3', 'rating_4','rating_5', 'published_since_month', 'primary_category', 'primary_subcategory' ]
df=df[cols]
```

```python colab={"base_uri": "https://localhost:8080/"} id="yg7Oup8QP9I6" outputId="5d98ea87-d152-412b-84e6-d3d10dd02ca4"
df.shape
```

```python id="CBV-OH61tSIG"
# 5. drop the duplicates
df=df.drop_duplicates(subset='id', keep='first')
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZhCIdbvFP_VW" outputId="62a46b1c-8f85-4595-8991-f0eb61f77d22"
df.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="xGnaobarQFP6" outputId="b684b521-7c1a-4f4d-f304-8c44a0bbbe5f"
df.isnull().sum()
```

```python id="U6_f6h7jta61"
# 6. check the missing values
#The free courses are labeled as free -> change price for these courses: 0 
df['price']=df['price'].fillna(0)

# drop relevancy_score
df = df.drop('relevancy_score', axis=1)

#drop the missings
df.dropna(how='any', inplace=True)

#in the objectives, there are empty lists 
index_to_drop=df[df['objectives'].apply(lambda x: x==list([]))].index
df.drop(index=index_to_drop, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="JpnCqT4RQX0u" outputId="d26e9b1b-d76b-4541-cc37-32e4354483f1"
df.shape
```

```python id="w6NO4DWEtayo"
# 7. save the cleaned dataset
df.to_csv('df_courses.csv', sep=' ')
```

<!-- #region id="KbP6HdlgtzRT" -->
### Clean the review data
<!-- #endregion -->

```python id="CzN8nX05t7Kq"
# 1. import the raw data
df_review_raw = pd.read_parquet("https://github.com/sparsh-ai/reco-data/raw/master/udemy/udemy_reviews.parquet.gzip")
```

```python colab={"base_uri": "https://localhost:8080/"} id="qCWG6kuJQlZN" outputId="7a89676d-53c3-46b8-b21a-96b85588c6e1"
df_review_raw.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="d9zPfHo3QjGz" outputId="eaa5dc50-e41d-4abd-e5e9-f37011229a82"
df_review_raw.head()
```

```python id="9xtiwXHqt7H_"
# 2. transform the relevant columns
df_review=df_review_raw.loc[:, ~df_review_raw.columns.str.match('Unnamed')]
df_review['user_name']=transform_col(df_review['user'], 'display_name')
df_review['user_title']=transform_col(df_review['user'], 'title')
```

```python id="QweAo5V3t7Fl"
# 3. filter the dataset from anonymized users (3 types)
df_review=df_review[~df_review['user_name'].isin(['Anonymized User', 'Private Udemy For Business User', 'Udemy User'])]
```

```python id="Mz6b_Xs7t7Cw"
# 4. keep only the relevant columns
cols=['id', 'created', 'rating', 'user_name']
df_review=df_review[cols]
```

```python id="WppN5_Flt6-5"
# 5. drop the duplicates
#the user names in the reviews data are not unique, it is impossible to build a recommender system based on the user ratings
df_review.drop_duplicates(inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="AtswX5TFRC6Z" outputId="a135573e-b30f-43a7-bcb7-a74660d08420"
df_review.isnull().sum()
```

```python id="cQ0Sa8knt68c"
# 6. treat the missing values
#no missing values
```

```python id="RcMjofb1t65s"
# 7. save the cleaned data
df_review.to_csv('df_reviews.csv')
```

<!-- #region id="XRAXkEfjNTyi" -->
## EDA
<!-- #endregion -->

<!-- #region id="QpJRaHWxNVCT" -->
Most important findings on the course dataset:
- there are courses with no reviews/ratings, but most of them are between rating 4 and 4.5
- The price ranges between 0 and 199 EUR
- There are some really popular courses with a lot of subscribers. The top 3 are:
        - machinelearning	with more than 300T subscribers
        - python-for-data-science-and-machine-learning-bootcamp with 192T subscribers
        - an-entire-mba-in-1-courseaward-winning-business-school-prof with 187T subscribers
- Most courses don't have any quizzes or practice tests
- The number of lectures mostly vary between 13 and 37 (IQR)
- The average age of a course is 26 months (since it was published). There are more recently published courses than older ones.
- The majority of the courses is for all levels. Only a few courses requires an advanced level.
- The courses are divided into 16 subcategories, whereas the two most significant are Finance and Entrepreneurship. 
        - Two subcategories have an average price higher than 100 dollars : The subcategory Data & analytics with 112, and Project Management with 104
        - The total earning on the courses is the highest in the subcategory for Data & Analytics and the second is in Entrepreneurship.
        - The total number of subscribers are the highest in the category of Entrepreneurship (1.) and in Data & Analytics (2.)
        - There is not much difference between the average ratings of the courses in each subcategory. The highest average ratings are in the subcategories Media and Communications. 
- I investigated the top words in each subcategories in the attributes objectives and description separately. E.g. in the subcategory Data& Analytics, the top 5 words are:
    - data, use, model, understand, create

After the univariate analysis I also executed multivariate analysis:
- There is a positive correlation between the number of reviews/number of subscribers and the average rating - students normally give good ratings for courses they liked
- As expected, there is a positive correlation between number of subscribers and number of reviews
- There is also a positive correlation between published since and the average rating -> older courses have better ratings. This seems logic, since I would expect that courses which aren't popular won't stay long on the sortiment
- The price doesn't have an effect on the average ratings or on the number of subsribers
<!-- #endregion -->

<!-- #region id="PJgsZJFqNVCW" -->
Most important findings on the reviews dataset: 
- The users are unfortunately not unique. Because if this reason, it is not possible to build a recommender system on the user ratings. 
- - Most users (more than 600.000) gave only one review, but there are couple user_names, who have plenty of reviews: the most common username is David with more than 400 reviews.
- Most courses have very few reviews


<!-- #endregion -->

```python id="9P4CpYe-NVCW"
import pandas as pd
import numpy as np
import ast
import scipy.stats as st

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

#for the text attributes
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from operator import itemgetter
from nltk.stem import SnowballStemmer
```

<!-- #region id="Hswz7I03NVCY" -->
### EDA of the course dataset
<!-- #endregion -->

```python id="I5Rm16xhNVCY" colab={"base_uri": "https://localhost:8080/", "height": 581} outputId="c9db7e41-a1a8-458a-9026-f5375be45278"
df_courses = pd.read_csv('df_courses.csv', index_col=0, sep=' ', converters={"objectives": ast.literal_eval})
df_courses.head()
```

<!-- #region id="HxpHJ_zqNVCa" -->
#### Numerical columns
<!-- #endregion -->

```python id="2upYobXFNVCb" colab={"base_uri": "https://localhost:8080/", "height": 317} outputId="d58dcf71-e8cf-49ed-ccb2-b4655e23efe7"
df_courses.describe()
```

<!-- #region id="OAorKtuvNVCb" -->
There are around 900 courses with no reviews/ratings, but most of the ratings are between rating 4 and 4.5.
<!-- #endregion -->

```python id="GolCxllJNVCc" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="cda4081c-5885-4590-c198-8cd7a2bee9b6"
plt.hist(df_courses['avg_rating'], bins=50)
plt.xlabel('Number of courses')
plt.ylabel('Average rating')
plt.title('Distribution of average rating')
plt.savefig('avg_rating.png')
```

<!-- #region id="qz-BNeEPNVCd" -->
The price ranges between 0 and 199 EUR. Most courses cost eiter 19.99 or 199.99 $.
<!-- #endregion -->

```python id="9fBc5NZhNVCd" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="c3bb55fe-f9c0-434d-a087-b8319e09a56f"
plt.hist(df_courses['price'], bins=50)
plt.xlabel('Number of courses')
plt.ylabel('Price')
plt.title('Distribution of price')
plt.savefig('price.png')
```

<!-- #region id="U3t22I0hNVCe" -->
I checked which courses are most visited. Courses with the top 10 most subscribers can be seen below:
<!-- #endregion -->

```python id="_BylP4bzNVCf" colab={"base_uri": "https://localhost:8080/"} outputId="3ce5e95f-3db8-44ee-89a6-acdb7270c74b"
#most popular courses
top10_courses= df_courses.sort_values('num_subscribers', ascending=False)[['published_title', 'num_subscribers']].head(10)
for i, row in top10_courses.iterrows():
    print('The course {} has {} subscribers.'.format(row['published_title'],row['num_subscribers']))
```

```python id="rxNnOhh2NVCg" colab={"base_uri": "https://localhost:8080/", "height": 350} outputId="7ae2e220-3a32-4515-9e51-eba19de54225"
fig, ax= plt.subplots(figsize=(8,5))
ax.barh(np.arange(len(top10_courses)), top10_courses['num_subscribers'], alpha=0.6)
plt.yticks(np.arange(len(top10_courses)), top10_courses['published_title'])
plt.title('Top 10 courses with most subscribers')
ax.set_xlabel('Number of subscribers')
plt.savefig('top10courses.png')
```

<!-- #region id="gPhOg2QrNVCi" -->
I plotted a histogram and a boxplot from each numerical attribute. Some of the features has outliers, and the distribution is skewed.
<!-- #endregion -->

```python id="LpldcJ1aNVCi" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="f8b0b78e-4dd5-42c0-a872-77f36f084072"
var_num=['avg_rating', 'avg_rating_recent','num_article_assets' , 'num_curriculum_items',
         'num_lectures', 'num_practice_tests', 'num_quizzes','num_subscribers', 'num_reviews', 'price', 
         'published_since_month', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']

for col in var_num:
    fig, ax= plt.subplots(1,2,figsize=(16,4))
    ax[0].hist(df_courses[col], bins=20)
    ax[1].boxplot(df_courses[col])
    ax[0].set_title('Distribution of '+ str(col))
    ax[1].set_title('Boxplot of '+ str(col))
    print('Number of 0 values of attribute {} is {}.'.format(col, len(df_courses[df_courses[col]==0])))
```

<!-- #region id="l0eSruFnNVCj" -->
I defined all data points, whose distance from the mean is more than 3*standard deviation, as outliers. I checked the distribution without these outliers. I didn't excluded these outliers from the data, I only excluded them to have a better understanding of the distribution of the features.
<!-- #endregion -->

```python id="F5C-MAaoNVCk" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="556ffe6d-623c-4e18-a1b5-cc01280aa5e4"
#There are some outliers in the dataset - I filtered out the data, where the value was smaller/larger then mean+-3*std
var_num=['avg_rating', 'avg_rating_recent','num_article_assets' , 'num_curriculum_items','num_lectures', 
         'num_practice_tests', 'num_quizzes','num_subscribers', 'num_reviews', 'price',  'published_since_month']
excluded_all=[]

for col in var_num:
    mean=df_courses[col].mean()
    std=df_courses[col].std()
    temp=df_courses[(df_courses[col]>mean-3*std) & (df_courses[col]<mean+3*std)]
    excluded_all.extend(list(set(df_courses.index)-set(temp.index)))
    fig, ax= plt.subplots(1,2,figsize=(16,4))
    ax[0].hist(temp[col], bins=20)
    ax[1].boxplot(temp[col])
    ax[0].set_title('Distribution of '+ str(col))
    ax[1].set_title('Boxplot of '+ str(col))
    print('Number of dropped values of attribute {} is {}.'.format(col, len(df_courses)-len(temp)))
excluded=set(excluded_all)
```

```python id="FgzjaOJ-NVCl" colab={"base_uri": "https://localhost:8080/", "height": 557} outputId="4744df21-79e0-4333-81ae-b243f25c6c6b"
corr = df_courses[var_num].corr()
fig, ax=plt.subplots(figsize=(10,7))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
```

```python id="zYsWw0cTNVCm" colab={"base_uri": "https://localhost:8080/", "height": 920} outputId="3e6609bd-afc6-4f34-c198-22bc59821a87"
cols=['avg_rating','num_subscribers','published_since_month', 'num_reviews', 'price']
temp=df_courses[~df_courses.index.isin(excluded)]
sns.pairplot(temp[cols], plot_kws= {'alpha': 0.2})
```

<!-- #region id="m-gdYjwFNVCm" -->
The followings can be seen from the pairplot above:
- There is a positive correlation between the number of reviews/number of subscribers and the average rating - students normally give better ratings
- As expected, there is a positive correlation between number of subscribers and number of reviews
- There is also a positive correlation between published since and the average rating -> older courses have better ratings. This seems logic, since I would expect that courses which aren't popular won't stay long on the sortiment
- The price doesn't have an effect on the average ratings 
<!-- #endregion -->

```python id="WM9pWUECNVCn" colab={"base_uri": "https://localhost:8080/", "height": 920} outputId="0c42941c-8084-4b4c-e42d-ff69210f3cc2"
sns.pairplot(df_courses[['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']], markers="+",  plot_kws= {'alpha': 0.2})
```

<!-- #region id="xy7d-OPuNVCn" -->
#### Discrete variables
<!-- #endregion -->

```python id="GPoshjDQNVCo" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="153b0d63-9e00-4a72-982a-aed0d1116bf8"
#barchart
var_char=['has_certificate', 'is_paid', 'instructional_level', 'is_enrollable_on_mobile','is_owned_by_instructor_team',
          'is_practice_test_course', 'primary_category', 'primary_subcategory' ]
for col in var_char:
    temp=df_courses[col].value_counts()
    x_labels=temp.index
    plt.figure(figsize=(8, 4))
    ax = temp.plot(kind='bar', alpha=0.4)
    ax.set_title(col)
    ax.set_ylabel('Number of courses')
    ax.set_xticklabels(x_labels)
  
    rects = ax.patches
    labels = list(temp.values/temp.values.sum()*100)
    labels=[str(round(l,0))+'%' for l in labels]
    #for rect, label in zip(rects, labels):
        #height = rect.get_height()
        #ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
plt.show()
```

```python id="UUBFKCaENVCp" colab={"base_uri": "https://localhost:8080/", "height": 545} outputId="4e8815e7-fc55-40d1-dd8a-62f3c456317a"
#number of subscribers in the categories
df_courses['earnings']=df_courses['price']*df_courses['num_subscribers']
df_subcategories=df_courses.groupby('primary_subcategory').agg({'num_subscribers':'sum',
                                               'avg_rating': 'mean',
                                               'price': 'mean',
                                               'earnings': 'sum'})    
df_subcategories
```

```python id="dVMTuXr1NVCp" colab={"base_uri": "https://localhost:8080/", "height": 778} outputId="daffe861-d02d-4d86-c2f8-0a5dd7c22340"
titles=['Total number of subscribers', 'Average rating of courses', 'Total earning on courses', 'Average price of courses']
fig, ax= plt.subplots(2,2,figsize=(16,10))
num=0
for i, col in enumerate(df_subcategories.columns):
    num+=1
    ax= plt.subplot(2,2, num)
    df_subcategories[col].plot(kind='bar', ax=ax, alpha=0.5)
    plt.title(titles[i])
    if num in range(3) :
        plt.tick_params(labelbottom='off')
plt.show()
```

<!-- #region id="W0KwiZH8NVCq" -->
- Two subcategories have an average price higher than 100 dollars : The subcategory Data & analytics with 112, and Project Management with 104
- The total earning on the courses is the highest in the subcategory for Data & Analytics and the second is in Entrepreneurship.
- The total number of subscribers are the highest in the category of Entrepreneurship and in Data & Analytics
<!-- #endregion -->

<!-- #region id="9coJNAtoNVCq" -->
#### Attribute Objectives
<!-- #endregion -->

<!-- #region id="suNC0VvDNVCr" -->
I will analyse the attribute objectives of the courses to get a better understanding about the courses. 
At first I needed to transform the list of objectives into one string, and then investigate the frequencies of each word.
I also implemented stemming: for that, I created a dataframe, where thee indexes are the stemmed words, and the values are the words which were stemmed. I needed it, to transform back the stemmed words. By means of the stemming similar words were counted as the same word (e.g. the words learn and learning are treated as one word). 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iS6ffw67Tc46" outputId="ca05733d-b3d0-45e2-f49c-64945bd00306"
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

```python id="6C6ax2hkNVCr" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="5f2a70ac-e53b-427c-d7d7-60f87a2667ae"
objectives_text=df_courses['objectives'].apply(combine_list)
vocab_frame_orig=vocab_stem(objectives_text)
vocab_frame_orig.head()
```

```python id="7LzcNt6INVCs"
#drop duplicates from the dataframe with stemmed words
vocab_frame=drop_words(vocab_frame_orig)
```

```python id="EjUM9H9ANVCs"
StopWords=set(stopwords.words('english')+list(punctuation)+["’", "n't", "'s", "--", "-", "...", "``", "''", "“", "039"])
```

```python id="fBlFMv_INVCt" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="c1be2570-bc74-4e80-e554-735a65a7c573"
top_words_graph(df_courses, 'objectives', True, 'bar', StopWords, vocab_frame)
```

```python id="6tBPYiLINVCt" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="798633c7-0579-4500-df9f-9684c21bf80a"
top_words_graph(df_courses, 'objectives', True, 'wordcloud', StopWords, vocab_frame)
```

<!-- #region id="kUnjemnHNVCu" -->
#### Attribute Description
<!-- #endregion -->

```python id="kxxx1jEHNVCu" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="f0c7d37f-e9fe-4d2d-8d26-b01c9b31cc7e"
vocab_frame_descr=vocab_stem(df_courses['description_text'])
vocab_frame_descr.head()
```

```python id="fIxgGxmONVCv"
vocab_frame_descr=drop_words(vocab_frame_descr)
```

```python id="Ptz2BLRONVCv" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="597fc442-7fc8-4474-d090-9291dbd2f6cf"
top_words_graph(df_courses, 'description_text', False, 'bar', StopWords, vocab_frame_descr)
```

```python id="j1d5GfWFNVCw" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="340d9e87-119d-4e02-bd12-7a6a7e42ccfa"
top_words_graph(df_courses, 'description_text', False, 'wordcloud', StopWords, vocab_frame_descr)
```

<!-- #region id="NxgSZbtkNVCw" -->
### EDA of the reviews dataset
<!-- #endregion -->

```python id="j4Njt36BNVCw" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="3805c2c8-dacb-43c9-f020-382bb791acff"
df_reviews=pd.read_csv('df_reviews.csv', index_col=0)
df_reviews.head()
```

```python id="kFJBW_XvNVCx"
#Number of reviews per user_name (User_name is not unique!)
nr_user=df_reviews['user_name'].value_counts()
unique, counts = np.unique(nr_user, return_counts=True)
#dict(zip(unique, counts))
```

```python id="DU3h0tqsNVCy" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="7bf155e0-e93c-475e-f481-a15d0483c9f2"
#plot the first 20 value of the most common number of reviews per user
#most users (more than 600000) have only 1 review
ax, fig= plt.subplots(figsize=(10,4))

plt.bar(np.arange(len(counts[:20])), counts[:20], align='center')
plt.xticks(np.arange(len(counts[:20])), unique[:20])
plt.xlabel('number of reviews per user')
plt.ylabel('number of users')
plt.title('Number of reviews per user')
plt.show()
```

```python id="zRhaPUJ8NVCy" colab={"base_uri": "https://localhost:8080/", "height": 354} outputId="ecccb190-9687-49d8-fa85-15ea2e61de9b"
#plot the usernames with the most reviews (Username is not unique)
ax, fig= plt.subplots(figsize=(10,4))
nr_user[:20].plot(kind='bar', alpha=0.4)
```

<!-- #region id="2Qpvjb1CV5l0" -->
## Clustering and Recommender system
<!-- #endregion -->

<!-- #region id="C7ULCyU3V5l5" -->
In this section, I cluster the courses and based on the new clusters and other course features, I build a recommender system.

For the clustering I investigated the attributes OBJECTIVES and DESCRIPTION. After the preparation of these two attributes, the first part of the notebook tries to cluster the courses based on the attribute OBJECTIVES, while in the second part I build the clusters by means of the course DESCRIPTIONs. After comparing the results, I used the the clustering algorithm based on the description field. The last part of the notebook shows the recommender system,  that helps the user to find similar courses to the previously taken ones.
<!-- #endregion -->

```python id="A-9r9m_yV5l7"
import pandas as pd
import numpy as np
import ast
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pickle
```

<!-- #region id="Zo-8xBLyV5l-" -->
### Prepare the data
<!-- #endregion -->

```python id="IMF38JI2V5l_" colab={"base_uri": "https://localhost:8080/", "height": 581} outputId="e77c34f9-5422-478a-9eee-66c25dfeaff1"
df_courses=pd.read_csv('df_courses.csv', index_col=0, sep=' ', converters={"objectives": ast.literal_eval})
df_courses.head()
```

```python id="Mv0t1rF_V5mB" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="673dc4c3-943b-42f3-b04f-71f6ef1bca3d"
df_reviews=pd.read_csv('df_reviews.csv', index_col=0)
df_reviews.head()
```

<!-- #region id="zHmmcj2sV5mD" -->
### Pepare the attribute OBJECTIVES
<!-- #endregion -->

<!-- #region id="n8JFaoVtV5mD" -->
The feature Objectives is a list of course objectives. At first I make a string from the list items by means of the function combine_list. 

For the stemming I saved all words with their stemmed correspondence in the dataframe vocab_frame. Since I am interested only in the stemmed words, I dropped all the duplicates from this dataframe (e.g. I treat learn and learning as the same words). This dataframe will be used to transform back the stemmed words. 

I defined the StopWords which contains all the expression that shouldn't be considered from the texts. 

Finally I applied the TfidfVectorizer on the objectives attribute: This transformator builds feature vectors from text documents so, that it helps to identify words which are frequent in the text but rare in the corpus.
<!-- #endregion -->

```python id="peLBS7VZV5mG"
objectives_text=df_courses['objectives'].apply(combine_list)
```

```python id="Vx2QXDyvV5mH"
vocab_frame_orig=vocab_stem(objectives_text)
#drop duplicates from the dataframe with stemmed words
vocab_frame=drop_words(vocab_frame_orig)
```

```python id="iXG_v9CYV5mJ"
StopWords=set(stopwords.words('english')+list(punctuation)+["’", "n't", "'s", "--", "-", "...", "``", "''", "“", "039"])
```

```python id="7K7i1NmrV5mK" colab={"base_uri": "https://localhost:8080/"} outputId="72d66059-a95e-47c4-e4c6-fb5cfa93fff2"
#use TfidfVectorizer 
vectorizer= TfidfVectorizer(stop_words=StopWords, tokenizer=tokenize, max_features=1000, max_df=0.8)
X=vectorizer.fit_transform(objectives_text)
X.shape
```

```python id="pE0fFX50V5mM" colab={"base_uri": "https://localhost:8080/"} outputId="998daaf7-bba9-4a7f-fc84-a02f4700937a"
word_features = vectorizer.get_feature_names()
word_features[50:55]
```

<!-- #region id="TV24kYnAV5mN" -->
### Prepare the attribute DESCRIPTION
<!-- #endregion -->

<!-- #region id="G8ree0OdV5mO" -->
I executed the same steps as by the attribute Objectives except the combine_list functions: the attribute Description is alredy a string and not a list.
<!-- #endregion -->

```python id="CuQDJqXWV5mQ"
vocab_frame_descr=vocab_stem(df_courses['description_text'])
vocab_frame_descr=drop_words(vocab_frame_descr)
```

```python id="5oZgKW9cV5mS"
StopWords=set(stopwords.words('english')+list(punctuation)+["’", "n't", "'s", "--", "-", "...", "``", "''", "“", "039"])
```

```python id="4w9O35_GV5mT" colab={"base_uri": "https://localhost:8080/"} outputId="fe99a08c-f6cd-40cf-c64b-bc673288da78"
#use TfidfVectorizer
vectorizer_descr= TfidfVectorizer(stop_words=StopWords, tokenizer=tokenize, max_features=1000, max_df=0.8)
X_descr=vectorizer_descr.fit_transform(df_courses['description_text'])
X_descr.shape
```

```python id="x7y31zYUV5mU" colab={"base_uri": "https://localhost:8080/"} outputId="7dca63ff-1b7b-4a02-e4a3-2fa2ef4664af"
word_features_descr = vectorizer_descr.get_feature_names()
word_features_descr[50:55]
```

<!-- #region id="5xrsN9IGV5mV" -->
## Clustering with the OBJECTIVES
<!-- #endregion -->

<!-- #region id="on-p5_h1V5mV" -->
### K-Means Clustering - with k=15 clusters
<!-- #endregion -->

<!-- #region id="snk2vY2GV5mW" -->
At first I tried to create 15 clusters - there are 16 subcategories, but no need for category 'others'.
<!-- #endregion -->

```python id="qdz4norXV5mW" colab={"base_uri": "https://localhost:8080/"} outputId="e529afb9-7786-40a4-ea95-f8e680d11244"
kmeans = KMeans(n_clusters = 15, n_init = 10, n_jobs = -1, random_state=1234)
kmeans.fit(X)
```

```python id="zY4JQsH7V5mY" colab={"base_uri": "https://localhost:8080/"} outputId="750e26d1-9aef-451d-8090-0853f9798f26"
common_words=get_common_words(kmeans, 10)
print_common_words(common_words, word_features, vocab_frame)
```

```python id="3Oxma2TJV5mZ" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="69625d8a-4953-404c-e0d9-b8ff830f6dbd"
#plot the top10 most frequent words in each cluster
squarify_words(common_words, word_features, vocab_frame)
```

```python id="Swgq58VFV5ma" colab={"base_uri": "https://localhost:8080/", "height": 725} outputId="765143b7-6d20-4330-e5d4-10ce0d40418c"
#heatmap of the new clusters with the subcategories
df_courses['cluster']=kmeans.labels_
heatmap_categories_cluster('cluster', df_courses, 'Reds' )
```

<!-- #region id="_nN0Bqw5V5mb" -->
### Relationship between number of clusters (k) and inertia
<!-- #endregion -->

<!-- #region id="e7YmTPDWV5mc" -->
I investigated the relationship between the number of clusters and the inertia (within-cluster sum-of-squares ) to find to optimal number of clusters. According to the elbow method, the line is an arm and the "elbow" on the arm is the value of k that is the best. 
<!-- #endregion -->

```python id="CrRzRnQaV5mc"
kRange = range(1,30)
inertia_Kmean = get_inertia(X, kRange)
```

```python id="W8nmvJUYV5md" colab={"base_uri": "https://localhost:8080/", "height": 539} outputId="29061a1d-75d1-49ac-bd28-0fdca729a401"
plot_inertia(kRange, inertia_Kmean)
plt.plot([6], [inertia_Kmean[5]], 'o--', color='dimgray', linewidth=3)
plt.plot([1,6,11], [8520, 8170,7820], '--', color='k', linewidth=1)
plt.annotate("Let's try k=6", xy=(6, inertia_Kmean[5]), xytext=(6,7700),
             size=14, weight='bold', color='dimgray',
             arrowprops=dict(facecolor='dimgray', shrink=0.05))
```

<!-- #region id="EZK8H4h0V5me" -->
### K-Means with k=6 clusters
<!-- #endregion -->

<!-- #region id="tSgm467sV5me" -->
It is hard to tell what is the optimal number of clusters from the graph. I tried several number of clusters and finally created 6 clusters with k-Means algorithm
<!-- #endregion -->

```python id="OFnTMTHyV5mf" colab={"base_uri": "https://localhost:8080/"} outputId="e417169f-bc2c-44f1-bc34-7f4a198c3e98"
kmeans = KMeans(n_clusters = 6, n_init = 10, n_jobs = -1, random_state=1234)
kmeans.fit(X)
```

```python id="TnQE655tV5mf" colab={"base_uri": "https://localhost:8080/"} outputId="8faa8a19-edbd-4c82-fc6b-35630617485e"
common_words=get_common_words(kmeans, 10)
print_common_words(common_words, word_features, vocab_frame)
```

```python id="z0J9tL2dV5mg" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="ba043bd6-a5b2-4c5e-ae7c-7a53992849ed"
squarify_words(common_words, word_features, vocab_frame)
```

```python id="T6NpDM7IV5mh" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="da318256-cc2c-44d2-fcbd-86b2d7ee0b97"
df_courses['cluster']=kmeans.labels_
heatmap_categories_cluster('cluster', df_courses, 'Reds')
```

```python id="4V_FZlKyV5mh" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="7938ec0d-e8c8-4117-e05f-fdfdfb78bcb1"
#plot the cluster distribution with the top5 words.More than half of the courses in the last cluster.
plot_common_words(kmeans, 5, word_features, vocab_frame, df_courses, 'cluster')
```

```python id="c5B6-Us_V5mj" colab={"base_uri": "https://localhost:8080/"} outputId="0a96c5c2-8719-4625-9d11-836efecbca57"
#print out course titles in each cluster
print_titles_cluster(5, df_courses, 'cluster')
```

<!-- #region id="TOIzspBUV5mj" -->
### Hierarchical Clustering
<!-- #endregion -->

<!-- #region id="dvcODsvlV5mk" -->
In this section I used hierarchical clustering. This method suppose that at the beginning the items have their own clusters.
The algorithm starts to merge the individual clusters on by one. 
I created a dendrogram, which shows the distances between the clusters. I plotted the last 16 merges of the hierarchical clustering algorithm.

<!-- #endregion -->

```python id="IUwp7itcV5mk" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="9ab49cd3-78bd-475c-ba5f-d5ed5aa1cdf2"
z=get_linkage(X )
plot_dendrogram(z, 16, line_dist=7.8)
#let's cut the dendrogrm at 7.8
```

```python id="yLPRYwi_V5ml" colab={"base_uri": "https://localhost:8080/"} outputId="6288f1ae-6471-4e2d-c09a-799e48df5b04"
#according to the dendrogram, I would cut the graph at 7.8 and get 8 clusters with the following distirbution.
df_courses['cluster_hier']=fcluster(Z=z, t=7.8, criterion='distance')
df_courses['cluster_hier'].value_counts()
```

```python id="6tEsJ7aeV5ml" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="da288ea0-f215-4cfc-d7ba-392df00e775d"
heatmap_categories_cluster('cluster_hier', df_courses, 'Reds' )
```

<!-- #region id="biWYX3VoV5mm" -->
The distribution of the clusters through hierarchical clustering is very unproportional.

<!-- #endregion -->

<!-- #region id="PofTJtWsV5mn" -->
### PCA for plotting the courses
<!-- #endregion -->

<!-- #region id="nQguUocyV5mo" -->
I will do a simple PCA analysis and keep the first 2 principal components int order to plot the courses in 2D. I will use the results ofthe kmeans clustering (with 6 groups), since the hierarchical clustering resulted in an overproportional group.
<!-- #endregion -->

```python id="Xp8W55NDV5mo" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="8a343654-54c4-46d5-dac4-16f7ed37cede"
plot_with_pca (X, df_courses['cluster'], 500)
```

<!-- #region id="Zfiip9idV5mp" -->
In the 2-D plot, almost all the clusters are well separated from each other. Cluster0 and cluster5 overlap each other - in cluster 5 the majority of the courses can be found.
<!-- #endregion -->

<!-- #region id="3VuJ1P7IV5mp" -->
## Clustering with the description
<!-- #endregion -->

<!-- #region id="zHRxzxX5V5mq" -->
After building clusters with the objectives attribute, I investigated the course descriptions as the basis of the clustering algorithmns. I executed the same analyses and got better distributed clusters by means of the description feature.  
<!-- #endregion -->

<!-- #region id="U3DeSg5YV5mq" -->
### K-Means clustering - k=15 clusters
<!-- #endregion -->

<!-- #region id="2KFNCbE-V5mq" -->
At first I tried to create 15 clusters, similar to the previous clusterings witht the attribute objective. There are clusters with only a few courses, so I tried to optimize the number of clusters to build (k). 
<!-- #endregion -->

```python id="pVi4g6zBV5mr" colab={"base_uri": "https://localhost:8080/"} outputId="dc92fb0f-96c7-4409-d212-efa821fbfd2e"
kmeans_descr = KMeans(n_clusters = 15, n_init = 10, n_jobs = -1, random_state=1234)
kmeans_descr.fit(X_descr)
```

```python id="6tZjftBoV5ms" colab={"base_uri": "https://localhost:8080/"} outputId="94b8bca7-e51f-4293-ed97-500e3eda08ce"
#top 10 words in each cluster
common_words=get_common_words(kmeans_descr, 10)
print_common_words(common_words, word_features_descr, vocab_frame_descr)
```

```python id="8zlHBBw0V5ms" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="a7c36b55-2059-42cf-fbe7-e2efb8d572d2"
squarify_words(common_words, word_features_descr, vocab_frame_descr)
```

```python id="S3k5xFAWV5mt" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="5ee4e187-5b48-47b3-a7b9-7ca1ab0c1620"
df_courses['cluster_descr']=kmeans_descr.labels_
heatmap_categories_cluster('cluster_descr', df_courses, 'Reds')
```

```python id="BzOm1216V5mu" colab={"base_uri": "https://localhost:8080/"} outputId="cdc17a51-3b29-4dc2-e762-85a634c602a9"
#most of the courses are in cluster 4
df_courses['cluster_descr'].value_counts()
```

<!-- #region id="kwlvvoe4V5mu" -->
### Relationship between number of clusters and inertia by the describtion attribute
<!-- #endregion -->

```python id="ORLAYsw-V5mu"
kRange = range(1,30)
inertia_Kmean = get_inertia(X_descr, kRange)
```

```python id="RzZIzSgiV5mv" colab={"base_uri": "https://localhost:8080/", "height": 539} outputId="38801bee-ab24-4fc0-9141-dfc79d3c5f78"
plot_inertia(kRange, inertia_Kmean)
plt.plot([8], [inertia_Kmean[7]], 'o--', color='dimgray', linewidth=3)
plt.plot([1,8,15], [8050, 7580,7110], '--', color='k', linewidth=1)
plt.annotate("Let's try k=8", xy=(8, inertia_Kmean[7]), xytext=(9,7800),
             size=14, weight='bold', color='dimgray',
             arrowprops=dict(facecolor='dimgray', shrink=0.05))
```

<!-- #region id="9ZkU_2IhV5mv" -->
### K-Means with k=8 clusters
<!-- #endregion -->

```python id="KMdwHn_DV5mw" colab={"base_uri": "https://localhost:8080/"} outputId="197b6599-b19c-47cb-89e2-7db9131f86bf"
kmeans_descr = KMeans(n_clusters = 8, n_init = 10, n_jobs = -1, random_state=123456)
kmeans_descr.fit(X_descr)
```

```python id="D9KTvc9uV5mx" colab={"base_uri": "https://localhost:8080/"} outputId="bcc5ae4f-3263-482e-a0dc-9c6363fa7488"
#top 10 words in each cluster
common_words=get_common_words(kmeans_descr, 10)
print_common_words(common_words, word_features_descr, vocab_frame_descr)
```

```python id="HHu3a4rsV5my" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="3fbbb94f-5f48-4ff1-8f23-ac7fc4ba8bdb"
squarify_words(common_words, word_features_descr, vocab_frame_descr)
```

```python id="apvhMHg_V5my" colab={"base_uri": "https://localhost:8080/", "height": 725} outputId="7da7059a-b8b4-464b-8bfc-5331547dd9cd"
df_courses['cluster_descr']=kmeans_descr.labels_
heatmap_categories_cluster('cluster_descr', df_courses, 'Reds')
```

```python id="eH0H1veWV5mz" colab={"base_uri": "https://localhost:8080/", "height": 336} outputId="2bdb7b8c-7860-40db-e7e2-c5c3ee2f6a3d"
plot_common_words(kmeans_descr, 5, word_features_descr, vocab_frame_descr, df_courses, 'cluster_descr')
```

```python id="BKmYcEbhV5m0" colab={"base_uri": "https://localhost:8080/"} outputId="e9285496-618c-44db-f57b-f602d74e127b"
print_titles_cluster(3, df_courses, 'cluster_descr')
```

<!-- #region id="33JtpZLSV5m2" -->
### Hierarchical clustering
<!-- #endregion -->

```python id="h_PZaR-QV5m2" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="0167bf51-7c3a-4273-b6bc-b7394023465a"
z_descr=get_linkage(X_descr )
plot_dendrogram(z, 16, line_dist=7.8)
#let's cut the dendrogrm at 7.8
```

```python id="ZcCnyCM6V5m3" colab={"base_uri": "https://localhost:8080/"} outputId="6d714db6-f73f-4137-d4c0-117e4182a008"
#according to the dendrogram, I would cut the graph at 9.6
df_courses['cluster_hier_descr']=fcluster(Z=z_descr, t=9.6, criterion='distance')
df_courses['cluster_hier_descr'].value_counts()
```

```python id="-OAs0n4dV5m4" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="8f2c7d0b-ba40-4a8e-a126-941e4a6aa43e"
heatmap_categories_cluster('cluster_hier', df_courses, 'Reds' )
```

<!-- #region id="79WZDfy_V5m5" -->
The distribution of the clusters through hierarchical clustering is very unproportional with the attribute description as well. In the further analysis I will use results of the k-means clustering with k=8.
<!-- #endregion -->

<!-- #region id="ffHHRZeaV5m6" -->
### PCA
<!-- #endregion -->

```python id="BCeMq8N_V5m6" colab={"base_uri": "https://localhost:8080/", "height": 0} outputId="ce190227-87b7-4228-fe1d-072d7f6ef361"
plot_with_pca(X_descr, df_courses['cluster_descr'], 1000)
```

<!-- #region id="_qDB5R0FV5m7" -->
### Export the clustering algorithm
<!-- #endregion -->

```python id="FXD75omnV5m8"
filename ='kmeans8.sav'
pickle.dump(kmeans_descr, open(filename, 'wb'))
```

```python id="iunuJRjbV5m9" colab={"base_uri": "https://localhost:8080/"} outputId="d26339de-17b3-4980-c194-c3f16385c165"
model_kmeans=pickle.load(open('kmeans8.sav', 'rb')) 
model_kmeans
```

```python id="TpFQeZ3dV5m9"
values=model_kmeans.predict(X_descr)
```

```python id="w_nlVHzDV5m-" colab={"base_uri": "https://localhost:8080/"} outputId="6bb2ca7d-6919-4ba7-e52a-ea754cefc7a3"
(values==df_courses['cluster_descr']).sum()
```

<!-- #region id="xz20JokIV5m_" -->
There are clusters, which are close to each other, e.g. cluster 3 is between clusters 0 and 8. Clusters 7, 4 and 1 are also adjacent. It is important to remember that I kept only 2 pricipal components, that explain 4% of the total variance (which is plotted on the graph). In contrast, the clusters are not reduced, they contain all the informations.
<!-- #endregion -->

<!-- #region id="NLQxnNKaV5m_" -->
## Building the recommender system
<!-- #endregion -->

<!-- #region id="188Q3r2qV5nA" -->
For the recommender system I use the course features together with the result of the k-means clustering with k=8. 
I transformed the course dataset into a features matrix by keeping only the relevant features (e.g. no need for course id) For the categorical variables I introduced dummy variables. The clusters were also transformed into dummy variables, since the order of the clusters doesn't have any meaning (cluster 0 is not better or worse than cluster 1).
As the last step of the preparation I normalized the feature matrix, since the features have different scales. 
I used the cosine similarity to compare the courses which each other. 

There are 2 functions, which can be used to recommend courses:
 - Function recommend_for_user recommends courses for the user based on his/her previous courses. This function takes the user as input.
 - Function recommend_courses recommends courses based on another course_id. This function takes the course_id as input and looks for the courses that are similar to the original course.
<!-- #endregion -->

<!-- #region id="Mh9oJAV3V5nB" -->
### Keeping the relevant features and prepare the dataframe
<!-- #endregion -->

```python id="K6rJ1eBPV5nB"
rel_cols=['avg_rating',  'has_certificate',  'instructional_level', 'num_lectures','num_quizzes',
          'num_practice_tests','is_practice_test_course', 'num_article_assets', 'num_curriculum_items',
          'num_subscribers','num_reviews',  'price', 'primary_subcategory','cluster_descr']
df_rel=df_courses[rel_cols]
```

```python id="D1RoOIPMV5nB" outputId="a2ba18a3-6d22-45bb-9f6e-cd4b7b417dfc"
df_rel['has_certificate']=df_rel['has_certificate'].astype(int)
df_rel['cluster_descr']=df_rel['cluster_descr'].astype(str)
dummies=pd.get_dummies(df_rel[['primary_subcategory', 'instructional_level','cluster_descr']], prefix=['subcat', 'level', 'cluster'])
df_rel.drop(columns=['primary_subcategory', 'instructional_level', 'cluster_descr'], inplace=True)
df_rel=pd.concat([df_rel,dummies], axis=1)
df_rel.head()
```

```python id="PK06kVEqV5nC"
df_norm=normalize_features(df_rel)
```

```python id="vxmp6L5YV5nD"
nr_user=df_reviews['user_name'].value_counts()
unique, counts = np.unique(nr_user, return_counts=True)
#dict(zip(unique, counts))
#recommend_for_user(user_name)
```

```python id="e69uzgsJV5nD" outputId="f5ad81fc-bd95-42ae-f3b2-370804185ec9"
nr_user.sort_values()[:10]
```

```python id="c4nPEgkbV5nE" outputId="a5f59e4b-d8b0-4568-92bb-903838968ee6"
recommend_for_user('DEEPAK IYER', 5, df_reviews, df_courses, df_norm)
```

```python id="cmUx0zcoV5nE" outputId="20f18639-001b-4fdd-ed03-0c8b115143e9"
recommend_for_user('Henk Bergsma', 5,df_reviews, df_courses, df_norm)
```
