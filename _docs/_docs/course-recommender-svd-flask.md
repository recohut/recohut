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

<!-- #region id="Uy5MoE99aESR" -->
# Course Recommender with SVD based similarity
> Applying SVD on education course dataset, storing in sqlite and wrapping in Flask REST api

- toc: true
- badges: true
- comments: true
- categories: [Flask, SQLite, Education, SVD]
- author: "<a href='https://github.com/Tsmith5151/user-recommender'>Trace Smith</a>"
- image:
<!-- #endregion -->

<!-- #region id="xBxkLnWf39V2" -->
## Setup
<!-- #endregion -->

```python id="LjlfOLxQDWk5"
import os
import yaml
import copy
import json
import sqlite3
import logging
import requests
import functools
import numpy as np
import pandas as pd
from time import time
from typing import List
from flask import request
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances

logging.getLogger().setLevel(logging.INFO)
```

```python id="OssJ94WBEkt2"
data_path = "."

# SQlite
env = "dev"
database = "recommender_dev.db"
username = "admin"
pwd = ""
hostname = "0.0.0.0"
port = 8081

similarity_metric = "cosine" # Similarity metric for pairwise distance measurement
weights = ['0.50','0.30','0.20'] # Weights for similarity matrix: interest,assessment,tags
results_table = "rank_matrix"  # SQLite3 table containing user similarity metrics
user_id = None # unique user id for scoring similarities

# Flask server
hostname = "0.0.0.0" # hostname for serving Flask application
port = 5000 # port for serving Flask application
```

<!-- #region id="56diW9dz32vs" -->
## Data ingestion
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GAB7Yq0kHx8t" outputId="5ee81ebb-05c5-49cf-f94c-6aa6b9da0953"
#hide-output
!wget https://github.com/sparsh-ai/user-recommender/raw/main/data/course_tags.csv
!wget https://github.com/sparsh-ai/user-recommender/raw/main/data/user_interests.csv
!wget https://github.com/sparsh-ai/user-recommender/raw/main/data/user_course_views.csv
!wget https://github.com/sparsh-ai/user-recommender/raw/main/data/user_assessment_scores.csv
```

```python id="fBqjXjpdKq1v"
def ingest_raw_data(env: str, data_dir: str = "data"):
    """Write .csv raw files to SQLite Database"""
    csv_files = [i for i in os.listdir(data_dir) if ".csv" in i]
    for f in csv_files:
        df = pd.read_csv(os.path.join(data_dir, f))
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        df.to_sql(name=f.split(".")[0], con=conn, if_exists="replace", index=False)
```

```python id="XHIiY3D7Kw6h"
ingest_raw_data(env, data_path)
```

<!-- #region id="JpJBwY9j3rm9" -->
## Load data from SQlite
<!-- #endregion -->

```python id="ggnYx7wTQ72v"
def read_table(env: str, query: str) -> pd.DataFrame:
    """Query Table from SQLite Database"""
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[column[0] for column in cur.description])
    return df
```

```python id="nwwcDfi2QkHb"
def load_data(env: str) -> dict:
    """Load Users and Content Data from SQLite"""

    df_course = read_table(env, f"select * from user_course_views")
    df_asmt = read_table(env, f"select * from user_assessment_scores")
    df_interest = read_table(env, f"select * from user_interests")
    df_tags = read_table(env, f"select * from course_tags")

    return {
        "course": df_course,
        "assessment": df_asmt,
        "interest": df_interest,
        "tags": df_tags,
    }
```

```python id="czO1yIuAQxQF"
# Load Users/Assessments/Course/Tags Data
data_raw = load_data(env)
```

<!-- #region id="wcOw1mQM3xAT" -->
## Summarize
<!-- #endregion -->

```python id="vA16sAQXRomz"
def data_summary(data: dict):
    """Print Summary Metrics of Data"""
    for name, df in data.items():
        logging.info(f"\nDataframe: {name.upper()} -- Shape: {df.shape}")
        for c in df.columns:
            unique = len(df[c].unique())
            is_null = df[df[c].isnull()].shape[0]
            logging.info(f"{c} -- Unique: {unique} -- Null: {is_null}")
    return
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hd5bQRA1Rdx-" outputId="24422b36-2a9d-4d53-eaed-33bb4f82fc61"
# Summary of Users/Assessments/Courses/Tags Data
data_summary(data_raw)
```

<!-- #region id="ih8a4TUL3mc4" -->
## Preprocessing
<!-- #endregion -->

```python id="DPdzp-6RSN8d"
def preprocess(data: dict) -> dict:
    """Preprocess input DataFrames"""
    prep = {}
    data = copy.deepcopy(data)
    for name, df in data.items():
        # drop null values
        df.dropna(axis=1, how="all", inplace=True)  # course tags table
        df.reset_index(drop=True, inplace=True)

        # rename columns in dataframe
        rename = {
            "interest_tag": "tag",
            "assessment_tag": "tag",
            "course_tags": "tag",
            "user_assessment_score": "score",
            "view_time_seconds": "view",
        }
        df.columns = [rename[i] if i in rename.keys() else i for i in df.columns]

        # discretize user assessment scores quantile buckets
        # if "score" in df.columns:
        #     df = df.replace({"score": {0:"low", 1:"medium", 2:"high"}})
        if any("score" in col for col in df.columns):
            df["score"] = pd.qcut(df["score"], q=3, labels=["high", "medium", "low"])

        # discretize user viewing time into quantile buckets
        # if "view" in df.columns:
        #     df = df.replace({"view": {0:"low",1:"high"}})
        if any("view" in col for col in df.columns):
            df["view"] = pd.qcut(df["view"], q=4, labels=["high", "medium", "low", "very low"])

        # encode categorical columns
        cat_cols = ["tag", "score", "view", "level"]
        for col in df.columns:
            if col in cat_cols:
                df[col] = pd.Categorical(df[col]).codes

        # save prep dataframe
        prep[name] = df

    # add key for max users -> used for initializing user-item matrix
    prep["max_users"] = max(
        [max(v["user_handle"]) for k, v in prep.items() if "user_handle" in v.columns]
    )

    # add key containing dataframe for merged course/tags
    prep["course_tags"] = pd.merge(
        prep["course"], prep["tags"], on="course_id", how="left"
    )
    return prep
```

```python id="vtOR7Zx2SPva"
# Preprocess Raw Data
data = preprocess(data_raw)
```

<!-- #region id="1Kk8JXy83en1" -->
## Calculating similarities and ranking
<!-- #endregion -->

<!-- #region id="mxgc7l5c3ZaL" -->
### Similarity
<!-- #endregion -->

```python id="8Q2ROQiKD69A"
class UserSimilarityMatrix:
    """Class for building and computing similar users"""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __repr__(self) -> str:
        return f"Dimensions of User-Items Matrix: {self.matrix.shape}"

    def build_user_item_matrix(self, max_users: str, item: str) -> None:
        """Build User/Item Interaction Matrix"""
        matrix = np.zeros(shape=(max_users, max(self.data[item])))
        for _, row in self.data.iterrows():
            matrix[row["user_handle"] - 1, row[item] - 1] = 1
        return matrix

    def get_user_item_matrix(self, max_users: int, features: List[str]):
        """Concatenate Features into One User-Items Matrix"""
        results = []
        for item in features:
            results.append(self.build_user_item_matrix(max_users, item))
        self.matrix = np.hstack(results)
      
    def _truncatedSVD(self, threshold: float = 0.90) -> np.ndarray:
        """Apply Truncated SVD to Explain 'n'% of total variance"""
        n_components = 2  # minimum components to begin
        ex_var = 0
        while ex_var < threshold:
            pc = TruncatedSVD(n_components=n_components)
            pc.fit_transform(self.matrix)
            ex_var = np.sum(pc.explained_variance_ratio_)
            n_components += 1
        logging.info(
            f"Total components {pc.n_components} with {ex_var:0.2f} variance explained"
        )
        self.matrix = pc.transform(self.matrix)

    def compute_similarity(self, metric: str = "cosine") -> np.ndarray:
        """Compute Similarity"""
        return pairwise_distances(self.matrix, metric=metric)
```

```python id="DU9UA1m1Wj3C"
def apply_similarity_calculation(name: str, features: List[str], metric: str) -> np.ndarray:
  """Compute User-Items Similarity Matrix
  Steps:
      - Construct User-Item Binary Vector for each input dataset
      - Apply truncatedSVD to determine 'n' components to explain m% of total variance
      - Compute cosine similarity
  """
  logging.info("=" * 50)
  logging.info(f"Computing USER-{name.upper()} Similarity Matrix...")
  logging.info(f"Input Features: {features}")
  SM = UserSimilarityMatrix(data[name])
  SM.get_user_item_matrix(data["max_users"], features)

  logging.info(f"Applying Truncated SVD: Input Shape: {SM.matrix.shape}...")
  SM._truncatedSVD()
  logging.info(f"Reduced User-Item Matrix Shape: {SM.matrix.shape}")

  # Compute pairwise user-similarity
  return SM.compute_similarity(metric=metric)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kXYUi1BcWvJq" outputId="c685368c-60a9-4a6a-8235-9e3001228060"
user_interest = apply_similarity_calculation("interest", ["tag"], similarity_metric)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4C6GRvulWvE-" outputId="5c9808b2-87ca-44c0-bba1-72ed248b97f4"
user_assessment = apply_similarity_calculation("assessment", ["tag", "score"], similarity_metric)
```

```python colab={"base_uri": "https://localhost:8080/"} id="A1slgCZJWu97" outputId="95ffbec2-21c4-41fa-91d0-056a94df5fc4"
%%time
user_courses = apply_similarity_calculation("course_tags", ["tag", "view"], similarity_metric)
```

<!-- #region id="bxHWZMi03TpZ" -->
### Weighted averaging
<!-- #endregion -->

```python id="cXKNW42YZgkg"
def compute_weighted_matrix(
    users: np.ndarray, assessments: np.ndarray, course: np.ndarray, weights: List[float]
) -> np.ndarray:
    """Compute Weighted Similarity Matrix where: weight_1 + weight_2 + weight_3 = 1"""
    return (
        (users * float(weights[0]))
        + (assessments * float(weights[1]))
        + (course * float(weights[2]))
    )
```

```python id="J3urPoKXX8Z_"
def apply_weighted_similarity(i: np.ndarray, a: np.ndarray, c: np.ndarray, weights: List[float]) -> np.ndarray:
  """Compute Interest/Assessment/Courses Weighted Matrix"""
  logging.info("=" * 50)
  logging.info("Computing Weighted Similarity Matrix...")
  return compute_weighted_matrix(i, a, c, weights)
```

```python colab={"base_uri": "https://localhost:8080/"} id="akKgRv4iX9zF" outputId="8bf50101-256b-4a9a-d0d9-f06643b38b74"
weighted_matrix = apply_weighted_similarity(user_interest, user_assessment, user_courses, weights)
```

<!-- #region id="_KO723Fb3RjG" -->
### Ranking
<!-- #endregion -->

```python id="ZXAewofzZq2G"
def rank_similar_users(X: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Apply Custom Pandas Function to Rank Top 'n' Users"""

    def custom_udf(X):
        """
        Custom Pandas function for using index/score to
        generate output results dataframe.
        """
        idx = np.argsort(X.values, axis=0)[::-1][1 : top_n + 1]
        return [
            str({"user": i, "score": X.astype(float).round(4).values[i]}) for i in idx
        ]

    # dimensions: users x top_n
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    ranking = X.apply(custom_udf).T
    ranking.columns = [f"{i+1}" for i in ranking.columns]
    ranking["user_handle"] = ranking.index
    logging.info(f"User Ranking Dataframe Shape: {ranking.shape}")
    return ranking
```

```python id="iWUFp4W9YLGJ"
def apply_user_ranking(df: pd.DataFrame) -> pd.DataFrame:
  """Rank Users based on Similarity Metric"""
  logging.info("=" * 50)
  logging.info("Computing Weighted Similarity Matrix...")
  return rank_similar_users(df)
```

```python colab={"base_uri": "https://localhost:8080/"} id="w7hAOwPWYIqV" outputId="167ba7df-9e60-424e-c2a9-a04f2e732199"
%%time
rank_matrix = apply_user_ranking(weighted_matrix)
```

<!-- #region id="iSYO881a3Jz8" -->
### Save the similarity matrix into database
<!-- #endregion -->

```python id="mIg0CugOZHRA"
def write_table(env: str, table: str, df: pd.DataFrame) -> None:
  """Write Table from SQLite Database"""
  conn = sqlite3.connect(database)
  cur = conn.cursor()
  df.to_sql(name=table, con=conn, if_exists="replace", index=False)
```

```python id="6jXLCb7bYS8F"
def save(results: pd.DataFrame) -> None:
  """Write Output Data to Table in SQLite Database"""
  logging.info("=" * 50)
  logging.info("Updating similarity matrix in SQLite Database...")
  write_table(env, results_table, results)
```

```python colab={"base_uri": "https://localhost:8080/"} id="akdX564OYV94" outputId="9852bbdc-ccbc-4938-f98f-54e82c604614"
save(rank_matrix)
```

<!-- #region id="6uTC57Fa3F1k" -->
## Adding Test sample
<!-- #endregion -->

```python id="nwCNoS_9a99f"
def read_table(env: str, query: str) -> pd.DataFrame:
  """Query Table from SQLite Database"""
  conn = sqlite3.connect(database)
  cur = conn.cursor()
  cur.execute(query)
  df = pd.DataFrame(
      cur.fetchall(), columns=[column[0] for column in cur.description]
  )
  return df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="NQbC8M1qa9zN" outputId="3ffc4987-045a-42b2-c876-f0b1da505d26"
df_check_tags = read_table('dev', f"select * from course_tags")
df_check_tags.head()
```

```python id="zZEaW1UIsgnm"
# Sample Data
df_test = pd.DataFrame({'user_handle':['110','110','111','111'],
                        'user_match': ['112','113','157','145'],
                        'similarity': ['80.2','20.8','52.0','48.0']})
```

```python id="1tcGg1P9tGPx"
# Write Similarty Results to Table
write_table('dev','test_table',df_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 111} id="o7XdpxENtF97" outputId="4269ec35-0247-4e70-e2bc-dd5271b1afe9"
# Read from Table
users = '110'
read_table('dev', f"select * from test_table where user_handle = {users}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="dlj9ilMKtUio" outputId="6071c881-9fe2-4690-faf8-f579ae8b2556"
# Add Index on Results Table (user_ranking)
conn = sqlite3.connect(database)
sql_table = f"""CREATE UNIQUE INDEX user_handle_index ON {results_table} (user_handle)"""
cur = conn.cursor()
cur.execute(sql_table)
```

<!-- #region id="C7i7ls7L4EiQ" -->
## Manual evaluation
<!-- #endregion -->

```python id="eyIzvj2o4L6h"
# User content
user_assesments = pd.read_csv('user_assessment_scores.csv')
user_interest = pd.read_csv('user_interests.csv')
user_course_views = pd.read_csv('user_course_views.csv')
course_tags = pd.read_csv('course_tags.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 97} id="ffcdSJpn4L20" outputId="8d609205-83ea-4429-e3c6-df77cf471e5f"
input_user  = 9
read_table('dev', f"select * from {results_table} where user_handle = {input_user}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="7YpiMrAv4Lyk" outputId="564afaa5-9fd0-4683-bb41-56d17c524b12"
user_interest[user_interest['user_handle'] == input_user]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="hU3mhL_i4Lve" outputId="ae2bc9f5-595e-4b7b-8df0-32972a4696f6"
user_interest[user_interest['user_handle'] == 9776].head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="wufmGLxh4Lr-" outputId="6ed9ff73-3207-4f50-ea2a-e7880eadf928"
user_assesments[user_assesments['user_handle'] == input_user]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="a2ssultC4-k-" outputId="22764c74-d630-483f-b640-41f1f5b45c22"
user_interest[user_interest['user_handle'] == 9776].head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 669} id="CKXbS1XR6FZv" outputId="6269a9fd-3c2f-40e4-dae7-915e7fb60b53"
user_course_views[user_course_views['user_handle'] == input_user]
```

<!-- #region id="kyFz981V2_g7" -->
## Deploy and Inference
<!-- #endregion -->

<!-- #region id="1s_I2W_s24pL" -->
### Build the API
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VoTbfCsqGNYi" outputId="1e5fdf81-7adc-486d-9e96-c38812e529fa"
%%writefile app.py

import os
import json
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify

DATABASE_ENV = "dev"
DATABASE_NAME = "recommender_dev.db"
TABLE = "rank_matrix"

app = Flask(__name__)


def read_table(env: str, query: str) -> pd.DataFrame:
  """Query Table from SQLite Database"""
  conn = sqlite3.connect(DATABASE_NAME)
  cur = conn.cursor()
  cur.execute(query)
  df = pd.DataFrame(
      cur.fetchall(), columns=[column[0] for column in cur.description]
  )
  return df


class SimilarUsers:
    def __init__(self, user):
        self.user = user

    def fetch_user_from_db(self):
        """Fetch User Record from SQLite Database"""
        query = f"select * from {TABLE} where user_handle = {self.user}"
        print("Table", TABLE)
        return read_table(DATABASE_ENV, query)

    def get_payload(self):
        """Return JSON Payload containing Input User and Top
        Similar Users with associated similarity scores"""
        data = self.fetch_user_from_db()
        if data.shape[0] == 0:
            return {self.user_id: "No records found!"}
        else:
            return {str(self.user): list(data.loc[0].values.flatten()[:-1])}


@app.route("/api/similarity/", methods=["POST", "GET"])
def get_user_similarity():
    user = json.loads(request.get_data())["user_handle"]
    SU = SimilarUsers(user)
    results = SU.get_payload()
    return results


if __name__ == '__main__':
  app.run(debug=True)
```

<!-- #region id="wJenoG7J2z2T" -->
### Run the server API
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="heWzh51NwqEf" outputId="bcbcf378-6b7f-4cb6-ed31-c39e5a351628"
!chmod +x app.py
!nohup python3 app.py > output.log &
```

```python colab={"base_uri": "https://localhost:8080/"} id="nRrNznADyKWn" outputId="648e6553-7328-44cc-f3df-0570d0848015"
!cat output.log
```

<!-- #region id="N8CWXAs42ss4" -->
### Make post request
<!-- #endregion -->

```python id="2lJ6mWqMGIc5"
def similarity(user_id: str, host: str = "0.0.0.0", port: int = 5000) -> json:

    """API call to flask app running on localhost
    and fetch top similar customers to the input customer(s)
    """
    url = f"http://{host}:{port}/api/similarity/"
    to_json = json.dumps({"user_handle": user_id})
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
    response = requests.post(url, data=to_json, headers=headers)
    print(response.text)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9Ry-RHzdt6CY" outputId="6b154bc9-1531-4608-f96e-86919ccb4fa4"
similarity(user_id='110')
```

<!-- #region id="PpsZ73fq2jiQ" -->
### Make post request using CURL from command line
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lpGmAbNZ0oiO" outputId="4c65cfdc-a60e-4393-c21c-4062efb3c948"
!curl -X GET -H "Content-type: application/json" -d "{\"user_handle\":\"110\"}" "http://0.0.0.0:5000/api/similarity/"
```
