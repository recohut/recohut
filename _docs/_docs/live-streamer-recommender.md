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

<!-- #region id="I4Vn1AjK_V_v" -->
# Live Streamer Recommender with Implicit feedback
> A recommendation system based on implicit data (watch duration) of users to different streamers. Contains also clustering of these users based from the behavior/relationship to the different streamers

- toc: true
- badges: true
- comments: true
- categories: [Implicit, Optuna, Clustering, Surprise, SVD]
- author: "<a href='https://github.com/karlountalan/recommendation-system-implicit'>Karlo Untalan</a>"
- image:
<!-- #endregion -->

<!-- #region id="P7l5By3dRVHG" -->
## **Importing the initial needed libraries**
<!-- #endregion -->

```python id="y6EHw7Yg7xk5" colab={"base_uri": "https://localhost:8080/"} outputId="456688c7-e32d-4c48-a989-eea53229d4c2"
import pandas as pd
import numpy as np
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.express as px

import random

from sklearn.model_selection import train_test_split as train_test_split_sk
from sklearn.cluster import KMeans

!pip install scikit-surprise
from surprise.model_selection.search import GridSearchCV
from surprise import Reader,Dataset, accuracy,SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate, train_test_split

!pip install optuna
import optuna

!pip install implicit
import implicit

import scipy.sparse as sparse

```

<!-- #region id="-AtUfNVowZas" -->
## **Importing the datasets downloaded from Google BigQuery**
<!-- #endregion -->

```python id="I6w_az0R75uZ" colab={"base_uri": "https://localhost:8080/"} outputId="c20e3383-f822-4827-b0ae-a8014cb74f0b"
#Individual Channel streams information dataset
df_channel = pd.read_csv('/content/drive/MyDrive/kumu_data/kumu_channel_db.csv')

#Watchlog for both gameshow and livestreams of the users dataset
df_gameshow = pd.read_csv('/content/drive/MyDrive/kumu_data/kumu_gameshow_watchlog_db.csv')
df_ls1 = pd.read_csv('/content/drive/MyDrive/kumu_data/kumu_livestream_watchlog_db_p1.csv')
df_ls2 = pd.read_csv('/content/drive/MyDrive/kumu_data/kumu_livestream_watchlog_db_p2.csv')

#Relationship dataset between users and streamers
df_rel = pd.read_csv('/content/drive/MyDrive/kumu_data/kumu_reco_dataset.csv')


```

<!-- #region id="0qjvxM1GpoIA" -->
## **Some Feature Engineering and EDA (Part 1 and Part 2)**
<!-- #endregion -->

```python id="3UOztZEESiQm"
#Concatenating the two separated livestream watchlog because of the separation due to file size contraints
df_livestream = pd.concat([df_ls1,df_ls2],axis=0)
df_livestream = df_livestream.rename(columns={'viewer_id':'user_id'})

#Concatenating the livestream watchlog and gameshow watchlog
df_all = pd.concat([df_livestream,df_gameshow],axis=0)

#Summing up those entries with the same User_id and channel_ID
df_all = df_all.groupby(['user_id','channel_id']).agg({'duration':sum}).reset_index()

```

```python colab={"base_uri": "https://localhost:8080/", "height": 202} id="OgV15WlNwIq_" outputId="0a8638d8-1b9d-4c58-8b3d-658459c39ae2"
df_all.head(5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="ESgJ5xOuFFV_" outputId="b6d0f841-8e3e-44e9-a6dd-69a4c860a324"
df_channel.head(5)
```

<!-- #region id="zpGkWD9DUlsk" -->
Getting the day of the week using the live_start_time column...
<!-- #endregion -->

```python id="y9n53f5GDcAw"
df_channel['live_day'] = df_channel['live_start_time'].apply(lambda x: pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S %Z').strftime('%A'))
```

```python id="Ilj2L9jvxYDO"
#joining the channel df to get the details of every channel_id
df = pd.merge(df_all,df_channel,how='left',on='channel_id')
df = df.rename(columns={'duration_x':'user_duration','duration_y':'total_stream_duration'})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="Edt5ptvJTvTF" outputId="811cac7e-bf4b-4777-d684-b661faeab74c"
df.head()
```

<!-- #region id="c88bbpljT4MQ" -->
**Plotting the Distribution Plot of the Watch Duration of Users**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 585} id="C_ycyIZqxlcP" outputId="84a7a303-9277-4855-9f3e-abeb2600b949"
plt.figure(figsize=(16,9))
sns.set_style('darkgrid')
sns.kdeplot(df['user_duration'], shade=True, color='r')
plt.title('Distribution Plot of the Watch Durations of the Users')
plt.xlabel('user_watch_duration (s)')
```

```python colab={"base_uri": "https://localhost:8080/"} id="UuNXmbQG0_TY" outputId="e8bab514-9d0c-49b9-d480-b4932844735f"
df['user_duration'].describe()
```

<!-- #region id="sjo1YtbzUSsI" -->
The skewness signifies that the distribution of the user_duration is heavily right skewed

The Kurtosis signifies that the distribution is too peaked
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TlW3MzSWx6rZ" outputId="e9d59706-4a4c-4ef6-864f-9f1bf464c5a2"
#Skewness and kurtosis of the watch durations
print('Skewness: %f' % df['user_duration'].skew())
print('Kurtosis: %f' % df['user_duration'].kurtosis())
```

<!-- #region id="VLqlxy-_Ur5I" -->
Getting the summary of the data per day of the week...
<!-- #endregion -->

```python id="DmsSjoan46A9"
df_day = df.groupby(['live_day']).agg({'user_id': lambda x: x.nunique(),'streamer_id': lambda y: y.nunique(),'user_duration':sum}).reset_index()
df_day = df_day.rename(columns={'user_id':'total_unique_viewers','streamer_id':'total_unique_streamers','user_duration':'total_watch_duration'})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 263} id="O5wn7DGCLFbB" outputId="693d53b0-78f2-4df6-dc17-098d7f19ee22"
df_day
```

```python colab={"base_uri": "https://localhost:8080/", "height": 566} id="TTKvj82v46Fh" outputId="d3a345a4-cc6d-46f6-a298-238e8c540d72"
g, axes = plt.subplots(1,3, figsize=(20,8))
sns.set_style('dark')
sns.barplot(x='live_day', y='total_unique_viewers', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=df_day,ax=axes[0])
label1 = axes[0].set_xticklabels(axes[0].get_xticklabels(),rotation=45)

sns.barplot(x='live_day', y='total_unique_streamers', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=df_day,ax=axes[1])
label2 = axes[1].set_xticklabels(axes[1].get_xticklabels(),rotation=45)
sns.barplot(x='live_day', y='total_watch_duration', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],data=df_day,ax=axes[2])
label3 = axes[2].set_xticklabels(axes[2].get_xticklabels(),rotation=45)


```

<!-- #region id="ZNU7TwNNUyd8" -->
The graph and table signifies that there are no significant difference in the number of total unique streamers and viewers during any day of the week. Same goes with the total watch duration.

However based from the graph, we can see that **Friday** is the day where the number of viewers peaked and then **decreases** on **Sunday and Monday**

The number of streamers is at peak on **Tuesdays, Wednesdays, and Saturdays**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 388} id="irzcYsMY8S3E" outputId="579fab55-b36b-4230-e051-dad28dcd1417"
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.scatterplot(x='like_count',y='comment_count',data=df_channel,hue='live_day')
```

<!-- #region id="hb52EiF7WmlJ" -->
The like count and comment count doesn't show a linear relationship as expected. There are streams with few likes but high comments and vice versa. We will make a new column and add these two variable and we'll call it "**engagement count"**
<!-- #endregion -->

```python id="ESr8stQmc0dg"
df_channel['engagement_count']=df_channel['like_count']+df_channel['comment_count']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 852} id="zaaiecL6c3x3" outputId="4ae2e6a5-f303-4a2b-ace4-ae47892e21bb"
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
g, axes = plt.subplots(1,2, figsize=(16,14))
sns.set_style('darkgrid')
sns.boxplot(x='live_day',y='engagement_count',data=df_channel,order=days,ax=axes[0])
sns.boxplot(x='live_day',y='diamonds',data=df_channel,order=days,ax=axes[1])
```

<!-- #region id="166bkACSeO3l" -->
The value of the 1st quartile and 3rd quartile of all the days are similar but we can see a very high value of engagement count outlier during the days of **Friday, Saturday, Sunday**

Same for the diamonds received per day of week. All of the days are similar.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 525} id="GyGhSEkmnee2" outputId="6442e729-648c-49c7-882c-ebae707e98b6"
g, axes = plt.subplots(1,3, figsize=(20,8))
sns.set_style('darkgrid')
sns.scatterplot(x='duration',y='engagement_count',data=df_channel,hue='live_day',ax=axes[0])
sns.scatterplot(x='duration',y='diamonds',data=df_channel,hue='live_day',ax=axes[1])
sns.scatterplot(x='engagement_count',y='diamonds',data=df_channel,hue='live_day',ax=axes[2])
```

<!-- #region id="_g3aQfree4L3" -->
There is a linear relationship somehow between the stream duration and the engagement count

There is no linear relationship between the engagement_count and the diamonds received.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 526} id="uKEVFdCCfKrt" outputId="fb98ca2a-12ae-49c5-de2a-fc5c34d9d121"
g, axes = plt.subplots(1,2, figsize=(20,8))
sns.set_style('darkgrid')
sns.scatterplot(x='total_viewer',y='engagement_count',data=df_channel,hue='live_day',ax=axes[0])
sns.scatterplot(x='total_viewer',y='diamonds',data=df_channel,hue='live_day',ax=axes[1])
```

<!-- #region id="UYuW8Xuq8ADw" -->
The graph above is the relationship of the total viewer to engagement_count and diamonds
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 525} id="3On50Zd9-7fw" outputId="914f8114-0aa5-44c3-db06-d4b1c6bbff4d"
plt.figure(figsize=(12,8))
sns.heatmap(df_channel[['duration','diamonds','total_viewer','engagement_count']].corr(), cmap='viridis', annot=True)
```

<!-- #region id="wZ-rCg7x_nRo" -->
**Based from the heatmap above, there are no strong correlation between the variables in the stream_channels data.**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 202} id="HG65t72pAiy-" outputId="e639f5b5-1aba-4841-ea35-910e18a38d84"
df_rel.head(5)
```

<!-- #region id="_iZOB71hAkSs" -->
I'm not sure how is the engagement factor is derived (not also stated in the schema of the table), for the sake of the analysis, I will remove this feature from the data.
<!-- #endregion -->

```python id="eBT7e1soAtvG"
df_rel.drop('engagement',axis=1,inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 524} id="nrCbxAHJA6DG" outputId="98185c08-a136-4355-c75d-4056e8adbcf2"
g, axes = plt.subplots(1,3, figsize=(20,8))
sns.set_style('darkgrid')
sns.scatterplot(x='view_count',y='comment_count',data=df_rel,ax=axes[0])
sns.scatterplot(x='view_count',y='coin_count',data=df_rel,ax=axes[1])
sns.scatterplot(x='comment_count',y='coin_count',data=df_rel,ax=axes[2])
```

<!-- #region id="rDRfJwdVBVNc" -->
The **view count and comment count** somehow possess a **linear relationship**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 505} id="f5e9TYyg_Ds9" outputId="5e454856-8cbc-4041-d1a1-e345801effbd"
plt.figure(figsize=(12,8))
sns.heatmap(df_rel.corr(), cmap='viridis', annot=True)
```

<!-- #region id="Gc8M_UN7Bddw" -->
Based from the heatmap above and the previous scatterplot, we can really see a strong linear relationship between view_count and comment_count in the relationship data table between users and streamers

The correlation of the coin_count to both the view_count and comment_count are almost the same.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 514} id="miuS4istPVT_" outputId="848d3094-a00f-41e1-beb5-a03a50b9bcd6"
df_users = pd.DataFrame(df['user_id'].value_counts()).reset_index()
df_users.columns = ['user_id','times_watched']
ax = plt.figure(figsize=(16,8))
sns.set_style('darkgrid')
total = df_users['user_id'].nunique()
ax = sns.countplot(x='times_watched',data=df_users[df_users['times_watched']<=30])
tot_pct = 0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 1,
            '{:1.2f}%'.format(height*100/total),
            ha="center")
    tot_pct+= height*100/total
tit = ax.set_title('Distribution of Times Watched per User (Clipped at 50)')
    
```

<!-- #region id="4jMW8tcnVqv3" -->
**Almost 20% of the users only watched ONCE in the dataset**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 514} id="2yDg5bwdRdMb" outputId="94a2bcf3-04ef-487a-e3a2-7d94e2005837"
df_streamers = pd.DataFrame(df['streamer_id'].value_counts()).reset_index()
df_streamers.columns = ['streamer_id','times_streamed']
ax = plt.figure(figsize=(16,8))
sns.set_style('darkgrid')
total = df_streamers['streamer_id'].nunique()
ax = sns.countplot(x='times_streamed',data=df_streamers[df_streamers['times_streamed']<=30])
tot_pct = 0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 1,
            '{:1.2f}%'.format(height*100/total),
            ha="center")
    tot_pct+= height*100/total
tit = ax.set_title('Distribution of Times Streamed per Streamer (Clipped at 30)')
    
```

<!-- #region id="boHUvLJMV0KH" -->
**Almost 16% of the streamers only streamed ONCE in the dataset**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 514} id="uQWBuuhGWN8t" outputId="51f1f582-835a-450f-957e-d53c974690f9"
df['pct_watched'] = df['user_duration']*100/df['total_stream_duration']
ranges = [0,10,20,30,40,50,60,70,80,90,100]
labels=['0-10','11-20','21-30',
        '31-40','41-50','51-60','61-70','71-80','81-90','91-100']
df['pct_watched_bin'] = pd.cut(df.pct_watched, ranges,labels=labels)

ax = plt.figure(figsize=(16,8))
sns.set_style('darkgrid')
total = len(df)
ax = sns.countplot(x='pct_watched_bin',data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 1,
            '{:1.2f}%'.format(height*100/total),
            ha="center")
tit = ax.set_title('Percent buckets of the % of their watch time duration to the total stream duration')
```

<!-- #region id="UUx1RSXAc62S" -->
**Only 7% of the viewers watched more than 50% of the entire stream of a streamer in this data set**
<!-- #endregion -->

<!-- #region id="u9VuTwizF5o_" -->
## **Method 1 - Collaborative Filtering Using Model-Based CF (Part 1)**
<!-- #endregion -->

<!-- #region id="vWOz7GsFeQsG" -->
### **Model Selection for the Recommender Model Using Surprise SVD (Part 1)**
<!-- #endregion -->

<!-- #region id="bUtZJT-3as94" -->
Preparing the train data...
<!-- #endregion -->

```python id="q8TmdIlqbF8Q"
df_cf = df.groupby(['user_id','streamer_id']).agg({'user_duration':np.mean}).reset_index()
```

<!-- #region id="CBOPgKo_fETx" -->
Since one of our models in our model selection is SVD, we will scale our target variable...
<!-- #endregion -->

```python id="TmVswUhFiGMu" colab={"base_uri": "https://localhost:8080/", "height": 202} outputId="d531e557-d6bd-4250-e9ab-3bd507092f60"
df_cf['scaled_user_duration'] = np.log(df_cf['user_duration'])
df_cf.head(5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 386} id="fXSepQLFXpXw" outputId="365e72b1-3ea1-41cb-8b90-e2020487a127"
g = sns.distplot(df_cf['scaled_user_duration'])
tit = g.set_title('Distribution Plot for the Scaled Watch Duration for each User')
```

```python id="WAYif-2bZjUU"
#Sampling the dataset for the initial model algorithm selection
sampled_df = df_cf.sample(n=100000)
reader = Reader(rating_scale=(0,df_cf['scaled_user_duration'].max()))
data = Dataset.load_from_df(sampled_df[['user_id','streamer_id','scaled_user_duration']], reader)
raw_duration = data.raw_ratings

#Shuffling the data
random.shuffle(raw_duration)

#Manually splitting train and test set after shuffling the data
# Train = 80% of the data, Test = 20% of the data
threshold = int(.8 * len(raw_duration))
train_raw_duration = raw_duration[:threshold]
test_raw_duration = raw_duration[threshold:]

data.raw_ratings = train_raw_duration # data is now the train set
```

```python id="iru_h-bPDph_"
benchmark = []

#We will use SVD, SVDpp, NMF, and KNNWithMeans as our model choices
for algorithm in [SVD(), SVDpp(), NMF(), KNNWithMeans()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=5, verbose=False,n_jobs=1)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 202} id="GFVNHCVOi4Gp" outputId="f6044dbf-d9fe-4e6e-ea0f-011b08833776"
results
```

<!-- #region id="beiThIHngeRP" -->
Based from the Benchmarking above, using the defaults parameter values, SVD performs best in our dataset...
<!-- #endregion -->

<!-- #region id="eqeHGzpyhAaL" -->
Checking if the model overfitted in our data by fitting it in our training set and testing it out on our hold-out set
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JtRzw9lmgQez" outputId="9352db42-133c-4342-bdc3-0cbea3e88648"
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)
testset = data.construct_testset(test_raw_duration)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

<!-- #region id="1RnVBm7MhXXJ" -->
**The RMSE is almost the same as the RMSE loss with the cross validation... It means that the model didn't overfit with the data. We will proceed with SVD.**
<!-- #endregion -->

<!-- #region id="oAXreRMAjNWp" -->
### **Hyperparameter Optimization using Optuna**
<!-- #endregion -->

<!-- #region id="9bAvqVWXkNpK" -->
**Getting the WHOLE dataset since we sampled the dataset earlier with 100k samples and building new train and test sets**
<!-- #endregion -->

```python id="NinXNFgWkNFF"
data = Dataset.load_from_df(df_cf[['user_id','streamer_id','scaled_user_duration']], reader)
raw_duration = data.raw_ratings

# shuffle ratings if you want
random.shuffle(raw_duration)

# Train = 80% of the data, Test = 20% of the data
threshold = int(.8 * len(raw_duration))
train_raw_duration = raw_duration[:threshold]
test_raw_duration = raw_duration[threshold:]

data.raw_rating = train_raw_duration
trainset = data.build_full_trainset()
```

```python id="xXZC6MyIMUrn" outputId="1a9286be-a6c1-4977-82b6-1a112f3e9970"
import optuna

def objective(trial):
    n_factors = trial.suggest_int('n_factors', 70,300,10)
    n_epochs = trial.suggest_int('n_epochs', 6,16,2)
    reg_all = trial.suggest_uniform('reg_all',0.001,0.05)
    lr_all = trial.suggest_uniform('lr_all',0.00005,0.01)
    svd = SVD(n_factors=n_factors,n_epochs=n_epochs,reg_all=reg_all,lr_all=lr_all)

    return cross_validate(svd, data, measures=['RMSE'], cv=3, verbose=False,n_jobs=1)['test_rmse'].mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

<!-- #region id="_yofzsVXNcbi" -->
**Getting the BEST hyperparameter combination using the output of the study...**

**From 0.986 to 0.958 RMSE loss using the optimized hyperparameters**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LY7v4QyjNBKm" outputId="5bbd41f0-e3a3-496b-f906-183e3db5d0a3"
study.best_params
```

```python colab={"base_uri": "https://localhost:8080/"} id="vzIoR2BqNWgq" outputId="151f6431-7ed2-4a91-e9bd-a35fad43d4f1"
study.best_value
```

<!-- #region id="9XKE5MGRlljt" -->
**Fitting the model to the WHOLE train dataset**
<!-- #endregion -->

```python id="Q586fXh4jfP5"
params = {'lr_all': 0.001427237312606455,
 'n_epochs': 14,
 'n_factors': 70,
 'reg_all': 0.028138308089573162}
 
algo = SVD(n_factors = params['n_factors'],n_epochs=params['n_epochs'],reg_all=params['reg_all'],lr_all=params['lr_all'])

algo.fit(trainset)
testset = data.construct_testset(test_raw_duration)
```

<!-- #region id="-Bu84qdRlwIx" -->
**Checking for the performance on the WHOLE test set... Looks like the hyperparamter optimization really improved our model and it didn't overfit..**

**We got 0.9339 RMSE on our hold-out set...**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="i_5wPm55kh77" outputId="1e490a31-deef-4061-bc77-864abd419f8d"
testset = data.construct_testset(test_raw_duration)
predictions = algo.test(testset)
accuracy.rmse(predictions)

```

<!-- #region id="YAbWaXIBmjHJ" -->
Getting the RMSE in a non logarithmic space since we transformed our target variable into a log space
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8YcqBlNmmiUi" outputId="f0e6a211-98d8-4eef-9476-946f6f7ab500"
def get_real_rmse(preds):
  err2 = []
  for i in range(len(preds)):
    err = (np.exp(preds[i][2])-np.exp(preds[i][3]))**2
    err2.append(err)
  return (np.sqrt(np.mean(err2)))

get_real_rmse(predictions)
```

<!-- #region id="3-mHey6ujU4u" -->
### **Model in Action**
<!-- #endregion -->

<!-- #region id="UTQRKqXcQ5Ca" -->
Function definition that accepts a user_id and a streamer ID and outputs an estimate watch duration (in seconds).
<!-- #endregion -->

```python id="sLhCRr9zPmzc"
def get_est_duration(uid,iid):
  log_est = algo.predict(uid,iid)[3]
  est_dur = np.exp(log_est)
  return est_dur
```

```python colab={"base_uri": "https://localhost:8080/"} id="4fNXom06RBVF" outputId="381ca0f9-72ce-48ca-d781-04effe0b1c91"
get_est_duration('ffff2b91-308d-4ac8-8bad-c255f9bbd187','56c17686-ad3e-4ba8-a21a-402720862f14')
```

<!-- #region id="qiMnnXfrrhbo" -->
**Building a recommender function that will accept input of a user_id and output the top n streamer that he/she will likely like.**

**The "only_new" parameter is a boolean variable to identify whether to return only NEW streamers that the user haven't watched yet or no.**

**When a NEW user is inputted, it will return the top 10 streamer with highest watch time based from all the users.**

**Number of recommendations can be changed by changing the parameter num_reco. Default value is 10**
<!-- #endregion -->

```python id="lSUbvWfUmfLQ"
def recommend_streamer(uid,only_new=True,num_reco=10):
  if uid not in df_cf['user_id'].unique():
    df = df_cf.groupby(['streamer_id']).agg({'user_duration':sum}).reset_index().sort_values(['user_duration'],ascending=False).iloc[0:num_reco].reset_index(drop=True)
    df.columns = ['streamer_id','total_duration_from_other_users']
  else:
    if only_new:
      watched = df_cf[df_cf['user_id']==uid]['streamer_id'].values
      all_streamers = df_cf['streamer_id'].unique()
      streamers = [i for i in all_streamers if i not in watched]
    else:
      streamers = df_cf['streamer_id'].unique()
    df = pd.DataFrame(streamers,columns=['streamer_id'])
    df['est_duration'] = df['streamer_id'].apply(lambda x: np.exp(algo.predict(uid,x)[3]))
    df = df.sort_values(['est_duration'],ascending=False).iloc[0:num_reco,:].reset_index(drop=True)
  return df


```

```python colab={"base_uri": "https://localhost:8080/", "height": 355} id="V0rWhBzkqFFj" outputId="cd7c37c6-f6ef-4cf1-feae-773ad3034ba7"
recommend_streamer('unknown-user-id')
```

```python id="gmBTEDCLrOa-" colab={"base_uri": "https://localhost:8080/", "height": 355} outputId="f99aaf54-f7b3-4b4e-dd29-e4fc7cd82506"
recommend_streamer('00000a8a-b9de-44de-88d4-4adc741eb7d8')
```

<!-- #region id="01Lu3FstHHPa" -->
## **Method 2 - Collaborative Filtering Using ALS with Implicit Feedback Using Implicit Library (Part 1)**
<!-- #endregion -->

<!-- #region id="9I50chcB6d7G" -->
Based from the paper: http://yifanhu.net/PUB/cf.pdf
<!-- #endregion -->

```python id="bD73MokhII54"
#Preparing the Sparse Matrix that will be fitted to the ALS

sparse_streamer_user = sparse.csr_matrix((df_cf['user_duration'].astype(float), 
                                          (df_cf['streamer_id'].astype('category').cat.codes, df_cf['user_id'].astype('category').cat.codes)))
sparse_user_streamer = sparse.csr_matrix((df_cf['user_duration'].astype(float), 
                                          (df_cf['user_id'].astype('category').cat.codes, df_cf['streamer_id'].astype('category').cat.codes)))

users_dict = dict(zip(df_cf.user_id.astype('category'), df_cf.user_id.astype('category').cat.codes))
streamers_dict = dict(zip( df_cf.streamer_id.astype('category').cat.codes, df_cf.streamer_id.astype('category')))
```

```python id="qdCIrk9rRkXm"
# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(sparse_streamer_user)
```

```python id="39jwMQLuItwm"
def recommend_streamers(uid,user_streamers):
    recommendations = model.recommend(users_dict[uid],user_streamers,filter_already_liked_items=False)
    iid = []
    scores = []
    for i in recommendations:
        iid.append(streamers_dict[i[0]])
        scores.append(i[1])
    return pd.DataFrame(zip(iid,scores),columns=['streamer_id','score'])
```

<!-- #region id="rpbh14O9OK9L" -->
List of Streamers really watched by user **ffffe69c-b408-44e4-b611-def853fd19f6** based from his/her watchlog
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 141} id="H8I-1PVjKRln" outputId="a4e810a5-9967-452a-cd3b-fb1f7a5acae3"
df_cf[df_cf['user_id']=='ffffe69c-b408-44e4-b611-def853fd19f6'][['streamer_id','user_duration']].sort_values('user_duration',ascending=False)
```

<!-- #region id="Nhsak8NBOgRu" -->
List of Streamers recommended by our ALS engine to the user **ffffe69c-b408-44e4-b611-def853fd19f6**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 355} id="JqjZhUlbKL9y" outputId="37ccf16d-a76b-4071-d67f-c57be5e224ff"
recommend_streamers('ffffe69c-b408-44e4-b611-def853fd19f6',user_streamers)
```

<!-- #region id="7A6GvYmmO8xa" -->
**Based from the tables above, all of the streamers watched by user ffffe69c-b408-44e4-b611-def853fd19f6 ended up in our top recommendation for that user.**
<!-- #endregion -->

<!-- #region id="ULZjxhvg9jvE" -->
### **Evaluating the implicit ALS Recommender Engine**
<!-- #endregion -->

<!-- #region id="w2UFKA-a_HMH" -->
Based from the methodology in https://towardsdatascience.com/building-a-collaborative-filtering-recommender-system-with-clickstream-data-dffc86c8c65 and https://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/RecEngine_NB.ipynb



We basically **mask/remove some values in our sparse matrix in our training set** and then compare with the final recommendation if the masked values are included in the recommendation. The test set will contain the unmasked and complete data.
<!-- #endregion -->

```python id="1e22TmlGLIig"
def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
    training set for later comparison to the test set, which contains all of the original ratings. 
    
    returns:
    
    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
    compares with the actual interactions.
    
    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  
```

<!-- #region id="8hvXmay2J0fQ" -->
Splitting the data into training set and validation set. Also returns the user_ids that were masked for evaluation.
<!-- #endregion -->

```python id="2LJaV_FrJyuN"
train, test, users_altered = make_train(sparse_user_streamer, pct_test = 0.2)
```

```python id="gMKnZQ_XLIt4"
from sklearn import metrics
def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    
    parameters:
    
    - predictions: your prediction output
    
    - test: the actual target result you are comparing to
    
    returns:
    
    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)   
```

```python id="ZPIPXCPYJCa3"
def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 
    
    parameters:
    
    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model 
    
    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one. 
    
    altered_users - The indices of the users where at least one user/item pair was altered from make_train function
    
    test_set - The test set constucted earlier from make_train function
    
    
    
    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''
    
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for i,user in enumerate(altered_users):
        # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    score = float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))
    score = pd.DataFrame(score).T
    score.columns = ['recommender_AUC_score','popularity_AUC_score']
    
    return score  
```

<!-- #region id="gncKnWSCKNHk" -->
Implementing the ALS on the users and streamers data.

Also uses an alpha of 40 since the paper used this value of alpha
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 78} id="AlIA4skuM9b7" outputId="f66eeef0-c035-4570-c56f-f5085b73c069"
alpha = 40

user_vecs, item_vecs = implicit.alternating_least_squares((train*alpha).astype('double'), 
                                                          factors=20, 
                                                          regularization = 0.1, 
                                                         iterations = 50)

calc_mean_auc(train, users_altered, 
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], test)


```

<!-- #region id="sc-W3kygNX04" -->
**Our ALS recommender model performed great as it gets 0.921 AUC score. It means that streamers recommended by our recommender ended up really watched by our users based from their historical watchlog.**

**However, recommender using streamer popularity garnered a higher AUC score and outperforms our ALS recommender engine.**
<!-- #endregion -->

<!-- #region id="ndJS1VimpU9A" -->
## **Conclusion in the Recommendation Model**
<!-- #endregion -->

<!-- #region id="K7bDTsXFpcHD" -->
As we can see, the RMSE of the first method we did was ~3693s which is ***not a great score*** even though the model didn't overfit in our data. However, we got a promising evaluating score in our second method but it can't exactly predict the definite duration for a user-streamer pair but it's much better in recommending compared to the performance given by our first method.
<!-- #endregion -->

<!-- #region id="DLoYOUCNCpwf" -->
## **Clustering of the Users and Streamers (Part 2)**
<!-- #endregion -->

<!-- #region id="4Zo9cZ0TC260" -->
We will be clustering the users and streamers based from their **total streams viewed/streamed, total comments given/received, and total coins given/received.**
<!-- #endregion -->

```python id="SNxMBf3s8TDE"
df_viewer = df_rel.groupby(['viewer_id']).agg({'view_count':sum,'comment_count':sum,'coin_count':sum}).reset_index()
df_viewer.columns = ['viewer_id','total_streams_viewed','total_comments_given','total_coins_given']
df_streamer = df_rel.groupby(['streamer_id']).agg({'view_count':sum,'comment_count':sum,'coin_count':sum}).reset_index()
df_streamer.columns = ['streamer_id','total_streams','total_comments_received','total_coins_received']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 414} id="Dg5d8zTy8TEp" outputId="7786275a-45d4-4e3d-be69-49ef6137e3c5"
df_viewer
```

<!-- #region id="IyughHhzDH3d" -->
### **Using KMeans Algorithm in Clustering**
<!-- #endregion -->

<!-- #region id="UHrfD234DOeh" -->
We will use the **Elbow Method** in determining the optimal number of clusters.
<!-- #endregion -->

```python id="pnFtvJIG8m_d"
from sklearn.cluster import KMeans

def elbow_method(df):
  #Using Streams viewed/streamed, coin given/received, and comments given/received as the clustering factors...
  xcluster3d = df.values
  wcss = []
  for i in range(1,15):
      km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 42)
      km.fit(xcluster3d)
      wcss.append(km.inertia_)
  plt.figure(figsize=(12,8))
  sns.set_style('darkgrid')
  g = sns.lineplot(x=range(1,15),y=wcss,color='r')
  g.set_title('Elbow Method')
  g.set_xlabel("Clusters")
  g.set_ylabel("WCSS")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 513} id="dGVnTeSSJ6JY" outputId="000f067d-58ba-48a8-8ebe-721ee0dbda5d"
elbow_method(df_viewer[['total_streams_viewed','total_comments_given','total_coins_given']])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 513} id="plhBKsT0KX-m" outputId="7ce8862c-73bf-44a2-b3b7-a49c07c56609"
elbow_method(df_streamer[['total_streams','total_comments_received','total_coins_received']])
```

<!-- #region id="FQVXC_TSDZAf" -->
Our optimal clusters are **4** for **both streamers and viewers**...

<!-- #endregion -->

```python id="7WqwObrVA-7Z"
#Applying the optimal clusters to the kmeans
km3d_viewer = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 42)
km3d_streamer = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 42)
kcluster3d_viewer = km3d_viewer.fit_predict(df_viewer[['total_streams_viewed','total_comments_given','total_coins_given']])
kcluster3d_streamer = km3d_streamer.fit_predict(df_streamer[['total_streams','total_comments_received','total_coins_received']])
df_viewer['cluster']=kcluster3d_viewer
df_viewer['cluster'] = df_viewer['cluster'].apply(lambda x: 'Cluster '+str(x+1))
df_streamer['cluster']=kcluster3d_streamer
df_streamer['cluster'] = df_streamer['cluster'].apply(lambda x: 'Cluster '+str(x+1))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="rgOTugMBLqCP" outputId="75735dec-ddd0-452e-d81d-a131961a8bab"
fig = px.scatter_3d(df_streamer, x='total_streams', y='total_comments_received', z='total_coins_received',
              color='cluster')
fig.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="JtKzuXUGFXMe" outputId="72926048-5e9e-416a-cdab-93df983f7ea7"
import plotly.express as px
fig = px.scatter_3d(df_viewer, x='total_streams_viewed', y='total_comments_given', z='total_coins_given',
              color='cluster')
fig.show()
```

<!-- #region id="1cdYmcqBD1j8" -->
**Based from the two 3D graphs above for the clustering of streamers and viewers, the algorithm mainly clustered the data by how the user and streamer gives and receives coins, respectively.**
<!-- #endregion -->
