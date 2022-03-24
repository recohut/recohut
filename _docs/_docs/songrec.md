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

# Song Recommender

```python colab={"base_uri": "https://localhost:8080/"} id="1FobpPeUT1JU" outputId="e7a9d8aa-a992-4132-faf2-c2c98f488efe"
!pip install -q surprise
```

```python id="ttIbHs9jRPjP"
import pickle
import pandas as pd
import numpy as np
import warnings
import logging
import sys
import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from surprise import SVD, SlopeOne, NMF, KNNBaseline
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise import Dataset
from surprise import Reader

from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
rcParams.update({'font.size': 18})
%reload_ext autoreload
%autoreload 2
```

<!-- #region id="CkpER4magJ4o" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HAJEi7PxSr5w" outputId="fe013bc4-93f5-4919-dc0e-ac15ba4c7ac6"
# download data
!python ./code/load_data.py
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="RE32q3AhTJPG" outputId="02c27621-af42-4c84-d7f7-01bb3759e994"
data = pd.read_parquet('./data/bronze/interactions.parquet.gz')
data.columns = ['user_id', 'track_id','plays']
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="3YC1rn4_TU8h" outputId="fd61d689-181f-4d04-9cb0-1a87729a3255"
data.info()
```

```python id="R4zLi5SRTrD-"
# data_small = data.head(1000).copy()
```

<!-- #region id="Cy985iL8gNQq" -->
## Preprocessing
<!-- #endregion -->

```python id="OojYxl_JTXlh"
def tt_preprocess(tt):
    # Saturate top 1% of plays, where 24 is 99th percentile of plays
    tt.plays = tt.plays.transform(lambda x: 24 if x > 24 else x)
    # Create 'rating' column based on log(plays/max_plays) transformed to 1-5 scale
    tt['rating'] = np.log10(tt.plays)/np.log10(tt['plays'].max())*5+1
    # Include only top 250 tracks (11% of total data, ~4.8 millions entries)
    top_tracks = tt['track_id'].value_counts()[:250].index.tolist()
    tt = tt[tt['track_id'].isin(top_tracks)]
    # Drop plays
    tt = tt.drop(columns='plays')
    # Create dummy column, timestamp, to fit Surprise format
    tt['timestamp'] = 0
    return tt
```

```python id="JN6ASKtjVzVt"
# load preprocessed tt data
data = tt_preprocess(data)

# Split into train/validation data 
frac = 0.8
train = data.sample(frac=frac,random_state=0)
test = data.drop(train.index)
# test = test.set_index('user_id')

# Save training data
train_path = './data/silver/train.parquet.gz'
train.to_parquet(train_path, compression='gzip')

# Save testing data
test_path = './data/silver/test.parquet.gz'
test.to_parquet(test_path, compression='gzip')
```

<!-- #region id="hw35zEjIg8Ws" -->
## User Profile Recommender
<!-- #endregion -->

```python id="TYN96WOCUNps"
class SongRecommender():
    """Basic song recommender system."""
    
    def __init__(self, model_name='svd'):
        """Constructs a SongRecommender"""
        self.logger = logging.getLogger('reco-cs')
        self.model_name = model_name

    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.
        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user_id', 'track_id', 'likes', 'timestamp'
        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")
        # processing for Surprise
        ratings = ratings.sample(frac=1)
        ratings = Dataset.load_from_df(ratings[['user_id', 'track_id', 'rating']],
                                       reader=Reader(rating_scale = (1,5)))
        self.trainset = ratings.build_full_trainset()
        # Choose model class based on model_name
        if self.model_name == 'svd':
            self.algo = SVD(lr_all=0.001,n_epochs=125)
        elif self.model_name == 'slopeone':
            self.algo = SlopeOne()
        elif self.model_name == 'nmf':
            self.algo = NMF()
        elif self.model_name == 'knnbaseline':
            self.algo = KNNBaseline() 
        elif self.model_name == 'cocluster':
            self.algo = CoClustering() 
        elif self.model_name == 'baseline':
            self.algo = BaselineOnly()
        elif self.model_name == 'normal':
            self.algo = NormalPredictor()            
        self.algo.fit(self.trainset)
        self.logger.debug("finishing fit")
        return(self)

    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.
        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user_id', 'track_id', 'rating', 'timestamp'
        Returns
        -------
        dataframe : a pandas dataframe with columns 'user_id', 'track_id', 'rating'
                    column 'rating' contains the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))
        testset = Dataset(reader=Reader()).construct_testset(raw_testset = requests.values)
        predictions = self.algo.test(testset)
        pred_base = [(pred.uid,pred.iid,pred.est) for pred in predictions]
        predictions = pd.DataFrame(pred_base,columns=['user_id', 'track_id', 'rating'])
        self.logger.debug("finishing predict")
        return(predictions)
```

<!-- #region id="Y_-zAIB8hBtV" -->
## Model Training
<!-- #endregion -->

```python id="i1-0EG54cYhq"
# set list of models to train
model_name_ = ["svd", "cocluster", "nmf"]

# set TRAIN SET path
path_train_ = "./data/silver/train.parquet.gz"

# Reading TRAIN SET from input file into pandas
train_data = pd.read_parquet(path_train_)

for model_name in model_name_:
    # Creating an instance of your recommender with the right parameters
    reco = SongRecommender(model_name)
    # fits on training data, returns a SongRecommender object
    model = reco.fit(train_data)
    # save model to pickle file
    pickle.dump(model, open(f"./artifacts/models/model_{model_name}.p", "wb"))
```

<!-- #region id="tXJpsjxYigko" -->
## Model Inference
<!-- #endregion -->

```python id="7AP6ZAOpXqaS"
path_requests_ = "./data/silver/test.parquet.gz"
result_path_ = "./outputs/model_result_ensemble.csv"

model_paths_ = ['./artifacts/models/model_cocluster.p',
                './artifacts/models/model_svd.p',
                './artifacts/models/model_nmf.p']
```

```python id="eDKSkHY9Uplo"
# Reading REQUEST SET from input file into pandas
request_data = pd.read_parquet(path_requests_)

if model_paths_ == 'default':
    global_mean = train_data['rating'].mean()
    result_data = request_data.drop(columns=['timestamp'])
    result_data['rating'] = global_mean
else:
    # Creating an instance of your recommender with the right parameters
    # reco_instance = SongRecommender(model_paths_)
    if len(model_paths_)>1:
        result_dfs = []
        for path in model_paths_:
            model = pickle.load(open(path, "rb"))
            result_dfs.append(model.transform(request_data))
        # Designate weights based on val set RMSE relative to
        # global mean RMSE, and normalize
        global_mean_rmse = 1.3706
        cocluster_weight = 1.3706 - 1.255
        nmf_weight = 1.3706 - 1.2716
        svd_weight = 1.3706 - 1.1934
        result_dfs[0]['rating'] = \
            (result_dfs[0]['rating']*cocluster_weight + \
                result_dfs[1]['rating']*nmf_weight + \
                    result_dfs[2]['rating']*svd_weight)/ \
                        (cocluster_weight + nmf_weight + svd_weight)
        result_data = result_dfs[0]
    else:
        # load the model
        model = pickle.load(open(model_paths_[0], "rb"))
        # apply predict on request_data, returns a dataframe
        result_data = model.transform(request_data)

result_data.to_csv(result_path_, index=False)
```

```python id="XAzKEuxcv05-"
# !echo *.p >> .gitignore
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="fs4Og2Fowe6Z" outputId="ece45af9-fb6a-41c7-ec19-7c982b3431ea"
result_data.head()
```

<!-- #region id="_bJMtJwFf-12" -->
## Evaluation
<!-- #endregion -->

```python id="v_SeDs2Uiyy5"
def compute_score(predictions, actual):
    """Look at 5% of most highly predicted songs for each user.
    Return the average actual rating of those songs.
    """
    actual.drop(columns=['timestamp'],inplace=True)
    df = pd.merge(predictions, actual.rename(columns={'rating':'actualrating'}), on=['user_id','track_id']).fillna(1.0)
    # for each user
    g = df.groupby('user_id')
    # detect the top 5% songs as predicted by your algorithm
    top_5 = g.rating.transform(
        lambda x: x >= x.quantile(.95))
    # return the mean of the actual score on those
    return df.actualrating[top_5==1].mean()
```

```python id="an-9RrKEiz5V"
def compute_rmse(predictions, actual):
    # RMSE between predictions and actual ratings
    rmse = ((predictions.rating - actual.rating) ** 2).mean() ** .5
    return rmse
```

```python id="Igm2YT3HjgWS"
path_testing_ = "./data/silver/test.parquet.gz" # groundtruth
result_path_ = "./outputs/model_result_ensemble.csv" # predictions
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZAOT7lthf8j9" outputId="0a19039e-ee50-4abc-8473-72116f7b439e"
# Load predictions data
prediction_data = pd.read_csv(result_path_)

# Load actual validation data
actual_data = pd.read_parquet(path_testing_)

# Compute score based on mean of top 5% of each users song rankings
score = compute_score(prediction_data, actual_data)
print(score)

# Compute RMSE between prediction and actual data
rmse = compute_rmse(prediction_data, actual_data)
print(rmse)
```

```python id="IB4dlyq61UmV"
# Save results to csv file
fields=[result_path_,round(score,4),round(rmse,4)]
with open(r'./outputs/eval_results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
```

<!-- #region id="trxJL3Mej_R5" -->
## Results Figure
<!-- #endregion -->

```python id="XaJPGMQ7kAnG"
# # Load results.csv file
# results = pd.read_csv('./outputs/results.csv')

# # If hyperparameters = True, generate SVD hyperparameters bar plot
# # If hyperparameters = False, generate model comparison bar plot
# hyperparameters = False

# if hyperparameters == False:
#     model_labels = ['SVD', 'NMF','KNN',
#                         'Co-cluster','ALS Base','Normal','Global Mean']

#     models_names = results.name[results.name.str.contains(
#         'svd_default|nmf|knn|cocluster|baseline|normal|global_mean')]

#     xlabel = 'Model Name'
    
#     comparison = results[results.name.isin(models_names)]

#     comparison['label'] = model_labels
#     comparison.sort_values(by='rmse',ascending=False,inplace=True)

#     fig,ax = plt.subplots(figsize=(13,6))
#     plt.bar(comparison['label'],comparison['rmse'],color = sns.color_palette("husl", 7))
#     plt.xlabel('Model Type')
#     plt.ylabel('RMSE')

#     plt.show(block=False)
#     plt.savefig('./outputs/rmse_comparison_no_ensemble.jpg')
#     plt.close('all')

#     # comparison = comparison.iloc[:,[3,2]]
#     # comparison.to_csv('data/results/comparison_results.csv',index=False)
# else:
#     hyp_names = results.name[results.name.str.contains(
#         'svd|pred.csv')]
    
#     hyp_df = results[results.name.isin(hyp_names)]
    
#     hyp_df.drop(index=[9,10],inplace=True)
    
#     hyp_df['lr'] = [.01, .005, .001, .001, .001, .001, .005, .005, .005, .005]
#     hyp_df['epochs'] = [20, 20, 20, 50, 75, 125, 10, 50, 125, 75]
    
#     print(hyp_df)

#     fig,ax = plt.subplots(figsize=(8,5))
#     plt.scatter(x=hyp_df['lr'], y=hyp_df['epochs'], s=400, 
#                 c=hyp_df['rmse'], cmap=sns.color_palette('plasma', as_cmap=True))
#     plt.xlabel('Learning Rate')
#     plt.ylabel('Number of Epochs')

#     cb = plt.colorbar()
#     cb.ax.set_title('RMSE')
    
#     fig.tight_layout()
#     plt.show(block=False)
#     plt.savefig('./outputs/rmse_svd_hyp.jpg')
#     plt.close('all')

#     hyp_df = hyp_df.iloc[:,[3,4,2]]
#     hyp_df.to_csv('./outputs/results/hyp_results.csv',index=False)
```

<!-- #region id="M_n6lr7Rf5-b" -->
## Group Recommender
<!-- #endregion -->

```python id="htCfj2Hx4m_o"
with open('./artifacts/le_user.pkl', 'rb') as f:
    le_user = pickle.load(f)
```

```python id="Vucf-bjlRdh5"
class GroupRecommender():
    """Playlist recommender system for groups."""

    def __init__(self,user_ids,num_songs=5,ensemble=True):
        """Constructs a GroupRecommender"""
        self.user_ids = user_ids
        self.num_songs = num_songs
        self.ensemble = ensemble

    def score_for_users(self,train_path):
        """Loads a trained model and predicts ratings for all tracks for given users
        Args:
            train_path (str): path to training data csv file
        Returns:
            self: GroupRecommender class instance
        """        
        self.train = pd.read_parquet(train_path)
        self.train.loc[:,'user_id'] = le_user.inverse_transform(self.train.user_id.values)
        
        track_list = self.train['track_id'].unique()
        
        df = pd.DataFrame(columns = self.user_ids, index = track_list).reset_index()
        
        request_data = pd.melt(df, id_vars = 'index', value_vars=self.user_ids)
        request_data.rename(columns={'index': 'track_id', 'variable': 'user_id', 'value': 'rating'},inplace=True)
        request_data = request_data[['user_id', 'track_id', 'rating']]
        request_data['timestamp'] = 0
        
        if self.ensemble:
            # Load ensemble of models for predictions
            model_paths = ['./artifacts/models/model_cocluster.p',
                           './artifacts/models/model_svd.p',
                           './artifacts/models/model_nmf.p']
            result_dfs = []
            for path in model_paths:
                model = pickle.load( open( path, "rb" ) )
                result_dfs.append(model.transform(request_data))
                
            global_mean_rmse = 1.3706
            cocluster_weight = global_mean_rmse - 1.255
            nmf_weight = global_mean_rmse - 1.2716
            svd_weight = global_mean_rmse - 1.1934
            result_dfs[0]['rating'] = \
                (result_dfs[0]['rating']*cocluster_weight + \
                    result_dfs[1]['rating']*nmf_weight + \
                        result_dfs[2]['rating']*svd_weight)/ \
                            (cocluster_weight + nmf_weight + svd_weight)
                            
            self.predictions = result_dfs[0]
        else:
            # Use best SVD model
            model = pickle.load(open("./artifacts/models/model_svd.p", "rb"))

            # Predict for request_data, returns a dataframe
            self.predictions = model.transform(request_data)
        
        return self
        
    def impute_knowns(self):
        """Imputes actual ratings into prediction ratings for known entries
        Returns:
            self: GroupRecommender class instance
        """        
        updated = self.predictions.merge(self.train, how='left', on=['user_id', 'track_id'],
                            suffixes=('', '_new'))
        # updated.drop(columns = ['Unnamed: 0','timestamp'],inplace=True)

        updated['rating'] = np.where(pd.notnull(updated['rating_new']), updated['rating_new'], updated['rating'])
        # # # Modify here
        # updated['known'] = np.where(pd.notnull(updated['rating_new']), updated['rating_new'], updated['rating'])
        updated.drop('rating_new', axis=1, inplace=True)
        
        self.predictions = updated
        
        return self
        
    def create_rankings(self):
        """For each user, sorts the ratings column, and replaces it with indices
            indicating the ranking of the entries
        Returns:
            self: GroupRecommender class instance
        """        
        dfs = []
        for idx, user_id in enumerate(self.user_ids):
            sorted_by_rating = self.predictions[self.predictions['user_id'] == user_id].sort_values(
                by='rating',ascending=False).reset_index().drop(
                columns=['user_id','index']).reset_index().set_index('track_id')
            dfs.append(sorted_by_rating.rename(columns={'index':'rank'},inplace=True))
            if idx == 0:
                rankings = sorted_by_rating
            else:
                rankings = rankings.join(sorted_by_rating,rsuffix=str(idx+1))

        self.rankings = rankings
        
        return self
    
    # def track_artist_names(self,df):
    #     """Extract song_title and artist_name for tracks in given DataFrame
    #     Args:
    #         df (pandas DataFrame): DataFrame of top recommended songs
    #     Returns:
    #         pandas DataFrame: Initial DataFrame with additional song_title, artist_name columns
    #     """    
        
    #     # read in map of track_id to artist_name and song_title
    #     names = pd.read_csv('data/track_artist_names.txt',sep = '<SEP>',header=None)
        
    #     # drop unnecessary column and rename columns
    #     names.drop(columns = [0], inplace = True)
    #     names.rename(columns={1: "track_id", 2: "artist_name", 3: "song_title"},inplace=True)
        
    #     # Merge on track id to add new columns
    #     df = pd.merge(df, names, on = 'track_id',how='left')
        
    #     return df

    def rec_group_playlist(self,strategy='lm'):
        """Applies a strategy to generate group rankings using the individual rankings
        Args:
            strategy (str, optional): Name of group ranking strategy. Defaults to 'lm'.
        Returns:
            pandas DataFrame: DataFrame of recommended songs with rankings
        """        
        rankings = self.rankings
        
        rank_cols = [col for col in rankings.columns if 'rank' in col]

        if strategy == 'mp':
             # Most pleasure strategy
            rankings['best_rnk'] = rankings[rank_cols].min(axis=1)
            strat_col = 'best_rnk'
        elif strategy == 'avg':
            # Average rank strategy
            rankings['avg_rnk'] = rankings[rank_cols].mean(axis=1)
            strat_col = 'avg_rnk'
        else:
            # Least misery strategy
            rankings['worst_rnk'] = rankings[rank_cols].max(axis=1)
            strat_col = 'worst_rnk' # default is least misery strategy
        
        top_songs = rankings.sort_values(strat_col)[:self.num_songs]
        # top_songs = self.track_artist_names(top_songs)
        
        # Rearrange dataframe to be presentable
        # top_songs.insert(0, 'artist_name', top_songs.pop('artist_name'))
        # top_songs.insert(1, 'song_title', top_songs.pop('song_title'))
        # top_songs.drop(columns=['track_id'],inplace=True)
        # rating_cols = [col for col in rankings.columns if 'rating' in col]
        # top_songs.drop(columns=rating_cols,inplace=True)
        # char_lim = 30
        # top_songs['artist_name'] = top_songs['artist_name'].transform(lambda x: 
        #             x[:char_lim] + '...' 
        #             if len(x) > char_lim+1 else x)
        # top_songs['song_title'] = top_songs['song_title'].transform(lambda x: 
        #             x[:char_lim] + '...' 
        #             if len(x) > char_lim+1 else x)
        # print(top_songs)
        
        return top_songs
```

```python id="fdJtaBwEjUkD"
# Initial parameters
user_ids = ['d1ca8b3e78811238cf94ee7caa1868d7ae9e908a',
        '621659a10f52dc4f8b50f205ab85b6d6b7d1b0dc',
        '257fc9ff00cd0ac79f53c7d65510b2ebba0c6b8e']

num_songs = 5
train_path = './data/silver/train.parquet.gz'
strategy = 'avg'
save_path = './outputs/rankings_avg.csv'
ensemble = False
```

```python id="mkEKlobqOKOZ"
reco = GroupRecommender(user_ids,num_songs,ensemble)
reco.score_for_users(train_path)
reco.impute_knowns()
reco.create_rankings()
top_songs = reco.rec_group_playlist(strategy)
top_songs.to_csv(save_path,index='False')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="Bg0FveyN3bhr" outputId="acc00d8c-54f0-43eb-fe99-f642ddbfcfd7"
top_songs
```

```python id="P4QWPRDE-zjK"
!sudo apt install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="IEIqfuVG-8ca" outputId="865e3539-db76-4df4-8111-a86cc95e1900"
!tree -L 3
```

<!-- #region id="HdRM66_YOdo8" -->
## App
<!-- #endregion -->

```python id="I8Gvk7rpk-11"
# from flask import Flask, render_template, flash, request
# from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
# from group_rec import GroupRecommender

# DEBUG = True
# app = Flask(__name__)
# app.config.from_object(__name__)
# app.config['SECRET_KEY'] = '<redacted>'

# class ReusableForm(Form):
#     user1 = TextField('user1:', validators=[validators.required()])
#     user2 = TextField('user2:', validators=[validators.required()])
#     user3 = TextField('user3:', validators=[validators.required()])

# @app.route("/", methods=['GET', 'POST'])
# def hello():
#     # Generate playlist recommendations based on user ids
#     form = ReusableForm(request.form)

#     if request.method == 'POST':
#         user1=request.form['user1']
#         user2=request.form['user2']
#         user3=request.form['user3']

#         if form.validate():
#             flash('Generating playlist...')
#             user_ids = [user1,user2,user3]
        
#             num_songs = 5
#             train_path = 'data/train.csv'
#             strategy = 'lm'

#             ensemble = False
            
#             reco = GroupRecommender(user_ids,num_songs,ensemble)
            
#             reco.score_for_users(train_path)
            
#             reco.impute_knowns()

#             reco.create_rankings()
            
#             top_songs = reco.rec_group_playlist(strategy)
            
#             top_songs = top_songs.loc[:,['artist_name','song_title']]
            
#             top_songs.rename(columns = {'artist_name': 'Artist', 
#                                         'song_title':'Song Title'},inplace=True)
            
#             top_songs['Track Number'] = range(1,6)
#             top_songs.set_index('Track Number', drop=True, inplace=True)
#             top_songs.index.name = None
            
#             return render_template('view.html',tables=[top_songs.to_html(classes='songs')],titles=[''])
#         else:
#             flash('Error: All Fields are Required')

#     return render_template('index.html', form=form)

# if __name__ == "__main__":
#     app.run()
```
