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

```python id="y9BH6x2NfQyG" executionInfo={"status": "ok", "timestamp": 1629795076255, "user_tz": -330, "elapsed": 1388, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-aml"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="IDk0gUSSfatt"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="q71TF7O6fatx"
!git status
```

```python id="n6x2rPTifaty"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="nD65sfIhW5GY" executionInfo={"status": "ok", "timestamp": 1629795160657, "user_tz": -330, "elapsed": 61090, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f4b6b894-a441-40d4-b21c-b9c22aded01b"
!pip install -U -q dvc dvc[gdrive]
!dvc pull
```

<!-- #region id="QsZrsE9SUKKI" -->
---
<!-- #endregion -->

<!-- #region id="JnKB63tiWmSr" -->
In this simple scenario, we are building a recommendation system for movies based on a streaming service. I used a Kafka service that streamed data about movie files watched by users and movie ratings they submitted. The original data had ~1 million users and ~27 thousand movies. I streamed this data, parsed it, and saved it in a database.

In this scenario, we are going to use data regarding the user and movie to predict how the user would rate the movie on a scale of 1-5. This can be used in a recommendation service to sort the highest predicted ratings and recommend a movie.
<!-- #endregion -->

```python id="lWmR334gYo8m"
!pip install tpot xgboost
```

```python id="InJtmFcMXA1b" executionInfo={"status": "ok", "timestamp": 1629795499289, "user_tz": -330, "elapsed": 600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tpot import TPOTRegressor
```

<!-- #region id="OVsyqIESXZ5y" -->
## Data loading
<!-- #endregion -->

```python id="Ph8FQQSGXNWf" executionInfo={"status": "ok", "timestamp": 1629795353583, "user_tz": -330, "elapsed": 1274, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
movies_raw = pd.read_parquet('./data/bronze/ml-1m-movies.parquet.snappy')
users_raw = pd.read_parquet('./data/bronze/ml-1m-users.parquet.snappy')
ratings_raw = pd.read_parquet('./data/bronze/ml-1m-ratings.parquet.snappy')
```

```python colab={"base_uri": "https://localhost:8080/"} id="DnSg73QwX4Js" executionInfo={"status": "ok", "timestamp": 1629795394023, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ccef7529-ef21-4033-f295-913f334cf881"
movies_raw.shape, users_raw.shape, ratings_raw.shape
```

```python id="p0yCT15qXP6C" executionInfo={"status": "ok", "timestamp": 1629795401419, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
users_raw.set_index('user_id', inplace=True)
movies_raw.set_index('movie_id', inplace=True)
```

<!-- #region id="vok1UNDhql4g" -->
## Data Cleaning
Before we pass the data into TPOT we should do some basic cleaning of the data. Currently, TPOT works with numerical data although there is some work being done to add some auto [data cleaning](https://github.com/rhiever/datacleaner/issues/1). Therefore, we need to transform some of our data into a format that TPOT will understand. The best way to do this is with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [column transformers](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html). This makes the transformations a repeatable process, which is important because we are going to need to apply the same transformations when a making a prediction in our hypothetical production system.


<!-- #endregion -->

<!-- #region id="M8iyxmbrzxJV" -->
### Categorical Features
Some of our features are categorical, such as `genres`. I decided to turn the categorical features into binary features. For example, instead of `genres`, I would have `Action` with a value of `1` if the movie was an action movie and `0` if it was not an action movie.

To do this, I created a pipeline to apply scikit-learn's [MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html?highlight=multilabel%20binarizer#sklearn.preprocessing.MultiLabelBinarizer). First I needed to turn the cells of the columns into arrays instead of strings that resembled arrays. 
<!-- #endregion -->

```python id="mVenTsJItaap" executionInfo={"status": "ok", "timestamp": 1629795471387, "user_tz": -330, "elapsed": 490, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class MultiLabelStringToArray(BaseEstimator, TransformerMixin):
    """
    This shapes the data to be passed into the MultiLabelBinarizer. It takes
    columns that are array-like strings and turns them into arrays.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for column_name in df.columns:
            df[column_name] = self._transform_column_to_array(df[column_name])
        return df

    def _transform_column_to_array(self, pd_column):
        transformed_column = pd_column.copy()
        
        # replace null cells with empty array
        transformed_column.loc[transformed_column.isnull()] = transformed_column.loc[
            transformed_column.isnull()
        ].apply(lambda x: '[]')

        # parse string into array
        transformed_column = transformed_column.apply(self._parse_arraystr)
        return transformed_column

    def _parse_arraystr(self, str):
        """
        Applies a number of rules to turn an array looking string into an array
          - remove brackets
          - remove quotes
          - remove extra spaces
          - deliminate by comma
          - remove empty string entries in the array
        """
        str_without_brackets = str.replace("[","").replace("]","")
        str_without_quotes = str_without_brackets.replace("'","")
        str_without_spaces = str_without_quotes.replace(" ","")
        list_with_empties = str_without_spaces.split(',')
        if '' in list_with_empties:
            while("" in list_with_empties) : 
                list_with_empties.remove("") 
        return np.array(list_with_empties)
    
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    This tranformer creates a MultiLabelBinarizer for every column passed in.
    """
    def __init__(self):
        self.mlbs = {}
    def fit(self, X, y=None):
        """Fit the MultiLabelBinarizer to the data passed in"""
        df = X.copy()
        for column_name in df.columns:
            mlb = MultiLabelBinarizer()
            mlb.fit(df[column_name])
            # Uncomment the following line if you want to print out the values
            # that the MultiLabelbinarizer discovered.
            #print('Column: {NAME} Values: {VALUES}'.format(NAME=column_name, VALUES=mlb.classes_))
            self.mlbs[column_name] = mlb
        return self
    def transform(self, X, y=None):
        """
        Returns a dataframe with the binarized columns. When applied in a
        ColumnTransformer this will effectively remove the original column and 
        replace it with the binary columns
        """
        df = X.copy()
        binarized_cols = pd.DataFrame()
        for column_name in df.columns:
            mlb = self.mlbs.get(column_name)
            new_cols = pd.DataFrame(mlb.transform(df[column_name]),columns=mlb.classes_)
            binarized_cols = pd.concat([binarized_cols, new_cols], axis=1)
        return binarized_cols
```

<!-- #region id="fO3EidIKwtHZ" -->
### Date Features
We have a `release_date` feature; however, it is currently stored as a string so we want to extract meaningful data from the string. Since I do not believe that the day has much impact on how a user would rate something I am going to leave it out. I think the year could have some impact because users could be more excited by new movies or our streaming service might only contain very popular old movies. Also, I think the month could be helpful. It could discover that users are more likely to rate a movie highly if it is released during "Oscar season".
<!-- #endregion -->

```python id="I5Q2Nzr5xiJN" executionInfo={"status": "ok", "timestamp": 1629795491172, "user_tz": -330, "elapsed": 688, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class ExtractReleaseDateFeatures(BaseEstimator, TransformerMixin):
    """
    This transformer takes a column with a date string formatted as 
    'YYYY-mm-dd', extracts the year and month, and returns a DataFrame with
    those columns.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Returns a dataframe with the year and month as integer fields. When 
        applied in a ColumnTransformer this will effectively remove the
        original column and replace it with the new columns.
        """
        df = X.copy()

        # fill nulls values that wont show up in valid data
        df = df.fillna('0000-00-00') 

        df['year'] = df.iloc[:,0].apply(lambda x: str(x)[:4])
        df['month'] = df.iloc[:,0].apply(lambda x: str(x)[5:7])
        df = df.astype({'year':'int64', 'month':'int64'})

        return df.loc[:,['year','month']]
```

<!-- #region id="7mEjln8HzoyC" -->
### Column Transformation
Now let's combine all those transformations to create our pipeline. First, we create a pipeline to sequentially execute the steps for our categorical columns. Next, we define a `ColumnTransformer` which will apply the categorical transformations, date transformations, and will pass our other feature columns through into the final data. All other columns not specified here will be dropped.
<!-- #endregion -->

```python id="EQJu-3Djy-dm" executionInfo={"status": "ok", "timestamp": 1629795508078, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Pipeline to create binary columns
multilabel_binarizer_pipeline = Pipeline([
    ('multilabel_str_to_array',MultiLabelStringToArray()),
    ('binarizer', MultiLabelBinarizerTransformer()),
],verbose=True)

MULTILABEL_BINARIZER_COLUMNS = ['genres','production_countries', 'spoken_languages', 'gender', 'occupation']
RELEASE_DATE_COLUMNS = ['release_date']
PASSTHROUGH_COLUMNS = ['age','budget','popularity','revenue', 'runtime', 'vote_average', 'vote_count']

full_data_clean_pipeline = ColumnTransformer([
    ('multilabel_binarizer', multilabel_binarizer_pipeline, MULTILABEL_BINARIZER_COLUMNS),
    ('release_date', ExtractReleaseDateFeatures(), RELEASE_DATE_COLUMNS),
    ('passthrough_columns','passthrough', PASSTHROUGH_COLUMNS)
],remainder='drop',verbose=True)
```

<!-- #region id="1L5aoDhC0zP-" -->
## Training
With our pipeline setup, we are ready to try it out on some data. First, I combine all our raw data loaded from GitHub into a single DataFrame. Next, we sort the data based on `userid` because we want to train and test our model on different users, to see if what the model learned about one user's preferences apply to other users. Finally, I've decided to drop rows that contain nulls in columns that we are not applying transformations to. I chose to do this because there were only ~650 rows to which this applied. If there were more rows with null values I might consider a different approach because it could mean losing too much data. Another possibility is that there could be hidden meaning in the null values, such as null values being a proxy for old movies where that data could be harder to get. Either way, 650 rows is not even 1/100th of our data set so I'm not going to lose sleep over it.
<!-- #endregion -->

```python id="CX-gqq1w4YEM" colab={"base_uri": "https://localhost:8080/", "height": 564} executionInfo={"status": "ok", "timestamp": 1629795523399, "user_tz": -330, "elapsed": 1902, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4726f079-26f3-4606-f6b9-bf20fffdb790"
records = ratings_raw.join(users_raw, on='user_id', how='left')
records = records.join(movies_raw, on='movie_id', how='left')

records = records.sort_values(by=['user_id'])
records = records.dropna(subset=['budget','popularity','revenue','runtime','vote_average','vote_count'])
records.head()
```

<!-- #region id="Sma5FEDQKLKX" -->
Before we start playing around with TPOT we need to grab some train and test data. In this scenario, I'm going to put all my faith in TPOT to come up with the best model so I don't need to create a verification dataset. Let's start with a small amount of data just to see TPOT in action. First, we will fit and transform our dataset with the data cleaning pipeline we built then I'm going to select 10,000 records for both training and testing. We have a lot more data, but the more data there is the longer TPOT takes so, let's just start with 10k.
<!-- #endregion -->

```python id="XPLJ2HGg5RUb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629795669086, "user_tz": -330, "elapsed": 38561, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77013439-aaa1-4dac-d593-14d0154f09b5"
X_all = full_data_clean_pipeline.fit_transform(records)
y_all = records['rating']

X_train = X_all[:10000]
y_train = y_all[:10000]
X_test = X_all[10000:20000]
y_test = y_all[10000:20000]
```

<!-- #region id="ause_3b5GooK" -->
Now the time you've all been waiting for... afternoon tea (or whatever time of day you happen to be reading this). TPOT supports both regression and classification problems. I decided that this would be better as a regression problem because too many movies would tie for the top spot otherwise. 
Let's review some of the configuration options I chose:

* `generations` - This is the number of iterations of pipeline generation that 
TPOT will run for. Alternatively, you could specify a `max_time_minutes` to stop TPOT after a certain amount of time.

* `population_size` - This is the number of pipelines trained during each generation.

* `verbosity` - This just gives us some feedback to let us know that TPOT is boiling away. It can take a long time I find this reassuring to make sure nothing is frozen. 

* `random_state` - This ensures that if we run this a second time we start with the same seed.

* `template` - This describes how I want my pipeline to look. Since I have done little feature engineering I want to start with a Selector to find the best features, then transform those features and finally use a regressor. If I were to not specify a template TPOT would pick whatever combination worked best. In my trials, the shape of the pipeline would end up
`Regressor-Regresssor-Regressor`.

* `n_jobs` - The number of parallel processes to use for evaluation

* `warm_start` - This tells TPOT whether to reuse populations from the last call to fit. This is good if you want to stop and restart the fit process.

* `periodic_checkpoint_folder` - Where to intermittently save pipelines during the training. This can help make sure you get an output even if TPOT suddenly dies or you decide to stop the training early.

For a full list of TPOT's configurations checkout their [documentation](https://epistasislab.github.io/tpot/api/).

The configuration below will train 10,100 pipelines and compare them using 5-fold (another config options, but I just used the default) cross-validation and a negative mean squared error scoring function. It may not generate 10,100 unique pipelines; so, it will skip over any repeat pipelines that are generated. This example generates about 2,500 unique pipelines. 

**Warning:** This training takes about 6 hours to run. If you want to shorten the example you can change the `generations` and `population_size` to 10 and it will only generate 110 pipelines. This shorter process should take around 30 minutes to train.  
<!-- #endregion -->

```python id="Mava9Y6OZfpb" executionInfo={"status": "ok", "timestamp": 1629795782414, "user_tz": -330, "elapsed": 434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir -p ./extras/tpot/checkpoints
```

```python id="cEAVUfaB5x3J" colab={"base_uri": "https://localhost:8080/", "height": 377, "referenced_widgets": ["e368a6b6897f44eab08ebdf82ca033b8", "6ec270f418e24abeba033ca52d9200dd", "9be43b61843949008b663a1376526b9f", "edb6ffabccd445f6b6cae6ef7569c778", "1d7077e691644795a0633f5764741cbf", "a3dc72e38b474406bf22452a76cfff59", "4df27d9bd9f04e0baa8c3dcdf72ddad3", "61d5eab7e93947ae8740b3ca6d7c65a2", "811ae8843fd645778c2d4018bf034e03", "fb06e3c690af4160b6b90c02f51717e7", "76d4d86ce8fd4baf931c58bf0521d329"]} executionInfo={"status": "ok", "timestamp": 1629795972095, "user_tz": -330, "elapsed": 173428, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="84dac757-671c-4f35-b10d-7a8319df70bb"
pipeline_optimizer = TPOTRegressor(generations=5, population_size=5, verbosity=2, random_state=42,
                                   template='Selector-Transformer-Regressor', n_jobs=-1,
                                   warm_start=True, periodic_checkpoint_folder='./extras/tpot/checkpoints')
pipeline_optimizer.fit(X_train, y_train)
```

<!-- #region id="eVzjX8u1Qdux" -->
## Evaluation
Just like scikit-learn TPOT comes with a built-in evaluation mechanism. We can use the test data to evaluate our pipeline with the same scoring function that we used in training (we used the default which is negative mean squared error). We can see that our test data gives similar results as the cross validation scores seen during training. 

It looks like our model is off by almost a whole number in its predictions. This is likely in adequite for our scenario however, we would need to examine what sort of errors the model is making. For exmaple, if the model just estimates one point too low everytime then the model is perfect because we would recommend the correct movie; However, if the direction of error is variable it would cause some unfavorable movies to be recommended (at least personally the difference between a 3 and a 4 on a 5 point scale is enormous).

**Note**: I want to point out that this evaluation strategy is not really the most appropriate for our use case because we are not actually that concerned with the actual predicted value of a movie rating. We should be more concerned about whether the ranked order of predicted ratings resemble the users actual ratings. However, since this example is mostly evaluating the value of TPOT as a tool the negative mean squared error is a good evauluation of the model that is generated without considering the context that the model is being applied to.
<!-- #endregion -->

```python id="BtTJbRVTKnRR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629796033448, "user_tz": -330, "elapsed": 424, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d0414e3a-bb63-4da8-88de-2743f20e0721"
pipeline_optimizer.score(X_test, y_test)
```

<!-- #region id="KUBwraJ3OVXY" -->
## More Data is Better (maybe)
Since we have so much data and machine learning models are almost always better when trained on more data, let's use everything we've got. I'm going to split the data into 500k rows for training and the remaining ~240k for testings. 
<!-- #endregion -->

```python id="y7kFbftqS33p" executionInfo={"status": "ok", "timestamp": 1629796037346, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
X_train_large = X_all[:500000]
y_train_large = y_all[:500000]
X_test_large = X_all[500000:]
y_test_large = y_all[500000:]
```

<!-- #region id="SoRToIZJUHDS" -->
More is not always better. TPOT is already a timely process because there are so many pipelines generated and evaluated using k-fold cross-validation. The larger the dataset that the models are trained on, the longer this process is going to take. If you noticed I added one parameter to the configuration. `config_dict="TPOT light"` tells TPOT that I am using a large data set so it will limit the model search to only model features that are simpler and fast running. Therefore, it finds a pipeline that works well for large datasets.

**Warning**: This training takes somewhere between 12 and 20 hours to complete. Google Colab's runtime may timeout before you are complete. You may need to clone or fork my repo and use the [Jupyter Lab notebook](https://github.com/bialesdaniel/se4ai-i5-tpot/blob/master/notebooks/TPOT_Movies_Explained.ipynb) there to run this.
<!-- #endregion -->

```python id="sSFnYDA3SbOc" colab={"base_uri": "https://localhost:8080/", "height": 377, "referenced_widgets": ["47592c68e0c4436e8905a0bf1ded673e", "de9c2792569748658ed1e10c854d7067", "e00526f957054e959ad4f0b574b4314e", "d4f44f7e93d34b72969d5d036c77bedf", "6bfdfcaf8a074ab8ad265111eadd51ea", "fcf6ee7ba257479c870e780152e6479b", "665edf9f2227408298aa86f74b282c1a", "91c96a6ec9ca4b4698f397dc89e737c6", "53fbeebd64734f93a21e97f0ae43117f", "a426d81aedd54802a68be27b20698689", "0b9fa9ff2ec44622bc004172a0c41d1a"]} executionInfo={"status": "ok", "timestamp": 1629798553574, "user_tz": -330, "elapsed": 2498739, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4e259f07-1139-4f71-cc2c-8ba0aa485e2f"
pipeline_optimizer_large = TPOTRegressor(generations=5, population_size=5, verbosity=2, random_state=42,
                                      template='Selector-Transformer-Regressor', config_dict="TPOT light", n_jobs=-1,
                                     warm_start=True, periodic_checkpoint_folder='./extras/tpot/checkpoints_large/')
pipeline_optimizer_large.fit(X_train_large, y_train_large)
```

```python id="KPPJfcNUDcXg" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629798569840, "user_tz": -330, "elapsed": 1244, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5500cf8-28f3-4a7f-cd77-217df1f798cd"
pipeline_optimizer_large.score(X_test_large, y_test_large)
```

<!-- #region id="YHbyS-mIW3hg" -->
## Conclusions
As we have seen, TPOT is quite easy to use.  The autoML process can save some valuable time and effort in feature engineering and hyperparameter tuning. On the other hand, TPOT is slow. It can take a long time to generate the optimal pipeline. 

Sorry, these conclusions are rather shallow because this notebook is mostly focused on how to set up and use TPOT for our movie recommendation system. For more in-depth analysis of the tool please continue reading the [Medium article](https://medium.com/@daniel.biales/automl-taking-tpot-to-the-movies-cf7e6f67f876?sk=6737cdd9d4cf2ff3c7322ee25f80fe70).
<!-- #endregion -->
