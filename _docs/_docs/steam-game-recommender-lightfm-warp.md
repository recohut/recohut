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

<!-- #region id="mrmomdDfELyi" -->
# Video Game Recommendations
> Loading steam video game data and performing EDA, ETL and training an implicit model based on WARP loss using LightFM library

- toc: true
- badges: true
- comments: true
- categories: [EDA, BPR, Visualization, TSNE, Games, Games&Entertainment, LightFM]
- image:
<!-- #endregion -->

<!-- #region id="3pwf1Gc4CFOC" -->
## Introduction
<!-- #endregion -->

<!-- #region id="hQrnEqnBIx9n" -->
Gaming, being a fiercely competitive industry, your platform has to win its players with a different approach. We will build a system that upgrades your homepage or adjusts your app’s infinite scroll to individual tastes.
<!-- #endregion -->

<!-- #region id="7FRREaM5JGse" -->
<!-- #endregion -->

<!-- #region id="_GUtWgjwJHck" -->
A video game recommendation engine suggests games using on-site **behavioural targeting.** This studies the user’s interactions, and suggests games based on similarities, but can also keep the players inspired with games from different genres. The recommender **initiates higher excitement about new games,** which is translated into **stronger customer loyalty** and increased time spent on the platform - factors that push sites through the ranks.

These type of recommendation engine analyzes item properties such as **title, genre, description, minimal age viewer** or **tags,** or interactions like **playtime, rating,** or **purchase.** You may also inform your players about new games to further increase CTR with **personalization of newsletters, emails,** or **push notifications.**
<!-- #endregion -->

<!-- #region id="T2PnXRZExgO0" -->
## Setup
<!-- #endregion -->

```python id="UJ6SiQXj82VG"
!pip install lightfm
```

```python id="dsJKjrbNxPP4"
import pandas as pd
import numpy as np
import pickle 
import ast
import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from scipy import sparse
from scipy.spatial import distance

from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import textwrap

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
```

```python colab={"base_uri": "https://localhost:8080/"} id="XyJgnfD-COJY" outputId="0b5163c8-d0a8-453b-fa8d-67a01aa819dd"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d -p lightfm,gensim
```

<!-- #region id="INAbb2ZtTKy9" -->
## Data ingestion
<!-- #endregion -->

<!-- #region id="lbC0-5DETaWI" -->
Steam is the world's most popular PC Gaming hub, with over 6,000 games and a community of millions of gamers. With a massive collection that includes everything from AAA blockbusters to small indie titles, great discovery tools are a highly valuable asset for Steam.

**[Steam Video Game and Bundle Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)**

These datasets contain reviews from the Steam video game platform, and information about which games were bundled together.

**citation**

1. Self-attentive sequential recommendation. Wang-Cheng Kang, Julian McAuley. ICDM, 2018. [pdf](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)
2. Item recommendation on monotonic behavior chains. Mengting Wan, Julian McAuley. RecSys, 2018. [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf)
3. Generating and personalizing bundle recommendations on Steam. Apurva Pathak, Kshitiz Gupta, Julian McAuley. SIGIR, 2017. [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf)



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="UGQwzrkrUPLu" outputId="8b56d783-c021-4e27-df32-47484f914eb8"
!wget http://deepx.ucsd.edu/public/jmcauley/steam/australian_users_items.json.gz
!gzip -d australian_users_items.json.gz

!wget http://cseweb.ucsd.edu/~wckang/steam_games.json.gz
!gzip -d steam_games.json.gz

# !wget http://deepx.ucsd.edu/public/jmcauley/steam/australian_user_reviews.json.gz
# !gzip -d australian_user_reviews.json.gz
```

<!-- #region id="BA4WtzXjWAVA" -->
### User-item Interactions
<!-- #endregion -->

```python id="iVdLixwVS1tl" colab={"base_uri": "https://localhost:8080/", "height": 171} outputId="ec1dd683-179e-4249-d797-faeb3214e672"
with open('/content/australian_users_items.json') as f:
    lines = f.readlines()

lines[0]
```

```python id="Bx4GUJtZS1to" colab={"base_uri": "https://localhost:8080/"} outputId="c853df9f-ca2c-4f58-f262-9ede2f16a3a0"
len(lines)
```

<!-- #region id="sveDx4cTS1tp" -->
There are 88310 lines of data in this file, each representing a user.
<!-- #endregion -->

```python id="1rBeb6CDS1tq" colab={"base_uri": "https://localhost:8080/"} outputId="30714019-38e3-425b-b410-a642c4cbef6f"
# Evaluate the first line
j = ast.literal_eval(lines[0])
j
```

```python id="LvE5j_HZcAVO"
lines_list = []
for l in lines:
    _x = '[' + l + ']'
    _x =  ast.literal_eval(_x)
    lines_list.extend(_x)

with open(f'data.json', 'w') as json_file:
    json.dump(lines_list, json_file)
```

<!-- #region id="J4TtXKuGS1tv" -->
We now have a `.json` file that we can easily view as a Pandas DataFrame.
<!-- #endregion -->

```python id="hzki1CFiS1tv" colab={"base_uri": "https://localhost:8080/", "height": 289} outputId="e649cde8-1aa9-4027-eda1-e4b2c51fefb4"
df = pd.read_json("data.json")
df.head()
```

```python id="cLdOL8EaeVMp"
df.to_parquet('interactions.parquet.gzip', compression='gzip')
```

```python id="Zd3ohU-Oezl0"
!mv /content/interactions.parquet.gzip ./steam
!git status
!git add . && git commit -m 'commit' && git push origin steam
```

<!-- #region id="AMoJXYPTS1tw" -->
### Item Metadata
<!-- #endregion -->

```python id="DnNfvmWPS1tx" colab={"base_uri": "https://localhost:8080/", "height": 103} outputId="866849e6-9948-4d3d-f9f6-7bdffe5d2a1a"
with open('steam_games.json') as f:
    lines = f.readlines()

lines[0]
```

```python id="DgaUNBeWS1tz" colab={"base_uri": "https://localhost:8080/"} outputId="aeb04673-dc07-4a02-c8af-6356c6743281"
# Get number of lines
len(lines)
```

<!-- #region id="PmCuq_-9S1tz" -->
There are 32135 lines, each representing a different game.
<!-- #endregion -->

```python id="C0dk1MxHS1t0" colab={"base_uri": "https://localhost:8080/"} outputId="a5602a71-4bd6-454c-fee4-7c2bcda82637"
# evaluate the first string
j = ast.literal_eval(lines[0])
j
```

```python id="K2ZAkrPUfqza"
lines_list = []
for l in lines:
    _x = '[' + l + ']'
    _x =  ast.literal_eval(_x)
    lines_list.extend(_x)

with open(f'meta.json', 'w') as json_file:
    json.dump(lines_list, json_file)
```

<!-- #region id="UJ7hvee9fqzb" -->
We now have a `.json` file that we can easily view as a Pandas DataFrame.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 615} id="Tx2Mr5pCfqzc" outputId="0dc3481e-bfd1-4293-de7d-a88f81d96677"
df = pd.read_json("meta.json")
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6QHRZd1xgVqb" outputId="f865ba1b-428e-4ae3-d940-0cbeca3ac922"
df.dtypes
```

```python colab={"base_uri": "https://localhost:8080/"} id="vEDPL7aig45i" outputId="e3101b08-e17a-420a-f51d-06e36fb5a7d5"
df.price = df.price.astype('str')
df.metascore = df.metascore.astype('str')
df.dtypes
```

```python id="JP47e-gQfqzd"
df.to_parquet('items.parquet.gzip', compression='gzip')
```

```python id="d7ERdmI4fqze"
%cd /content/reco-data
!mv /content/items.parquet.gzip ./steam
!git status
!git add . && git commit -m 'commit' && git push origin steam
```

<!-- #region id="zN4jS6xuBFzP" -->
## Transformation
<!-- #endregion -->

<!-- #region id="BJpCKFGBA5KR" -->
### Data loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Aw0RsavDxpGF" outputId="2e6936e8-bada-4c97-b250-03de708e2261"
!wget https://github.com/sparsh-ai/reco-data/raw/steam/steam/items.parquet.gzip
!wget https://github.com/sparsh-ai/reco-data/raw/steam/steam/interactions.parquet.gzip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 615} id="e5GRUUfYxxRR" outputId="2cafbd78-e77f-4613-e51f-14b90a0392cd"
df = pd.read_parquet("items.parquet.gzip")
df.head()
```

```python id="kCUGaWmexPP_" colab={"base_uri": "https://localhost:8080/", "height": 289} outputId="4f0e088c-3f87-45de-97bf-1b7af2fcc153"
useritems = pd.read_parquet("interactions.parquet.gzip")
useritems.head()
```

```python id="ViDxQNaSxPQE" colab={"base_uri": "https://localhost:8080/", "height": 527} outputId="12e5c6b2-da62-4304-c810-6b91ac084111"
useritems['item_id'] = useritems['items'].apply(lambda x: [x [index]['item_id'] for index, _ in enumerate(x)])
useritems.head()
```

<!-- #region id="HL1NVXDxA_A8" -->
### Transformation
<!-- #endregion -->

```python id="6sj-ZUUtxPQG" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="7f4cf175-b3de-491d-da6b-d6514fc94295"
# Add a column with substitute user_id, counter
useritems['uid'] = np.arange(len(useritems))

# numgames dataframe for later use
numgames = useritems[['user_id', 'items_count']]

# Take relevant columns
useritems = useritems[['uid', 'item_id']]

# Check
useritems.head()
```

```python id="puH78y_txPQH" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="d81bb4a6-f926-4bd5-9b33-dcd590ac5d89"
# Explode item_ids into seperate rows
lst_col = 'item_id'
useritems = pd.DataFrame({col:np.repeat(useritems[col].values, useritems[lst_col].str.len())
                              for col in useritems.columns.difference([lst_col])
                        }).assign(**{lst_col:np.concatenate(useritems[lst_col].values)})[useritems.columns.tolist()]
useritems
```

```python id="nzkKQaYbxPQI" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="cad24edb-4e20-4baa-bfdc-1798a0a9c544"
# Add binary owned column
useritems['owned'] = np.ones(shape = useritems.shape[0])

# Change item_id to int
useritems['item_id'] = useritems['item_id'].astype(int)

# Rename column to match
useritems = useritems.rename(columns={'item_id': 'id'})

useritems.head()
```

```python id="7TpievtCxPQK" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="d2255d8e-83e1-454e-a457-85e9c5513c7e"
# Merge useritems and games data dataframes
alldata = pd.merge(useritems, df, on = 'id')

# Drop entries with no title
datawithnames = alldata.dropna(axis=0, subset=['title'])

# Get relevant columns for recommendation engine
recdata = datawithnames[['uid','id','owned']]
recdata.head()
```

<!-- #region id="mYri6y7_9yRc" -->
### Data checkpointing
<!-- #endregion -->

```python id="vcCdHbuU9248"
df.to_parquet('gamesdata.parquet.gzip', compression='gzip')
datawithnames.to_parquet('mergeddata.parquet.gzip', compression='gzip')
numgames.to_parquet('numgames.parquet.gzip', compression='gzip')
recdata.to_parquet('recdata.parquet.gzip', compression='gzip')
```

<!-- #region id="Wf0JvBZY-AGP" -->
## EDA
<!-- #endregion -->

<!-- #region id="OqVcu6EZ_dPe" -->
### Data loading
<!-- #endregion -->

```python id="sh9jIni00Ug3" colab={"base_uri": "https://localhost:8080/", "height": 615} outputId="172a2a46-2dc6-414e-b597-a6205dd73217"
gamesdata = pd.read_parquet('gamesdata.parquet.gzip')
gamesdata.head()
```

```python id="W1YxyZxE0Ug8" colab={"base_uri": "https://localhost:8080/", "height": 649} outputId="83ae412e-f329-4629-ecce-542d090fca60"
mergeddata = pd.read_parquet('mergeddata.parquet.gzip')
mergeddata.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="jJO1wXNS1dQz" outputId="7560337c-64d6-4bff-9952-0e839248bde5"
numgames = pd.read_parquet('numgames.parquet.gzip')
numgames.head()
```

```python id="A-6tWfzR0Ug_" outputId="a94c78ac-2558-4370-f5fd-14c5ffbb0e2a"
# Load numgames data
numgames = pd.read_csv('numgames.csv', index_col = 0)
numgames.head()
```

<!-- #region id="kSVmrsMtArWD" -->
### Release date
<!-- #endregion -->

```python id="Qr5RnBpn0UhG" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="a2537902-41ba-4b16-b4de-262ec8655859"
# Select entries where release date is not null
data = gamesdata[gamesdata['release_date'].notnull()]

# Replace strings which are not of the format xxxx-xx-xx with None
data['release_date'] = data['release_date'].map(lambda x : x if x[-3] == '-'else None)

# Select entries where release date is not null
data = data[data['release_date'].notnull()]

# Convert to DateTime 
data['release_date'] = pd.to_datetime(data['release_date'])

# Check 
data['release_date'].describe()

# Plot histogram of release date feat
data['release_date'].hist()
plt.title('Game Releases')
plt.ylabel('Number of Games')
plt.xlabel('Year')
plt.show()
```

<!-- #region id="Dh2oS-yd0UhJ" -->
We see that our data contains games ranging from 1970 up to predicted release date of December 2021.
<!-- #endregion -->

```python id="mFhE8sND0UhK" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="ac36a6b1-53f4-4c02-b26e-cdaef6cc442d"
# Focus on post 2010
recentgames = data[data['release_date'].dt.year > 2010]

recentgames['release_date'].hist()
plt.title('Game Releases post 2010')
plt.ylabel('Number of Games')
plt.xlabel('Year')
plt.show()
```

<!-- #region id="3bBDjfnl0UhM" -->
Let's see which months are most popular for new releases.
<!-- #endregion -->

```python id="Dw40soH80UhN" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="6ac85b06-a2d8-4c3e-fca9-a2c886cdeb43"
# Create month feature
data['release_month'] = data['release_date'].dt.month

# Plot countplot using Seaborn
sns.countplot(x = data['release_month'], data = data)
plt.title('Frequency of game releases per month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()
```

```python id="tPPUGqL00UhO" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="c773f6eb-84f9-4632-b3b7-4f6c65b1356f"
# Countplot of sale month

# define palette to highlight best months to buy house
custompalette = {release_month: "skyblue" if (release_month == 10 or release_month == 11 or release_month == 12 ) else "lightgrey" \
                 for release_month in data['release_month'].unique()}

with sns.axes_style("whitegrid"):
    sns.countplot(x = data['release_month'], palette = custompalette, data = data)
plt.title('Number of game releases per month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.savefig('month.pdf', bbox_inches = "tight")
```

<!-- #region id="1RkUF76L0UhQ" -->
We see that October, November and December have the highest number of game releases. Let's look at quarters now.
<!-- #endregion -->

```python id="eSBKtrm20UhQ"
# Define function to determine quarter
def quarter(month):
    ''' Returns quarter in which month falls'''
    if 1 <= month <= 3:
        quarter = 'Q1'
    elif 4 <= month <= 6:
        quarter = 'Q2'
    elif 7 <= month <= 9:
        quarter = 'Q3'
    else:
        quarter = 'Q4'
    return quarter
```

```python id="xF5xJGBv0UhR" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="10da622b-891f-4e39-80ad-e2a8569ab9ac"
# Create quarter feature
data['release_quarter'] = data['release_month'].apply(quarter)

# Plot countplot using Seaborn
sns.countplot(x = data['release_quarter'], data = data, 
              order = data['release_quarter'].value_counts().index)
plt.title('Frequency of game releases per quarter')
plt.xlabel('Quarter')
plt.ylabel('Count')
plt.show()
```

<!-- #region id="rqD36gpV0UhS" -->
**Recommendation:**

Q4 and in particular the month of October sees the most new games released. We would recommend ensuring advertisement deals are priced at a premium during this period.
<!-- #endregion -->

<!-- #region id="4_D1ZJTm0UhT" -->
Finally, let's look at release date for the user-item data.
<!-- #endregion -->

```python id="BaYDbfQg0UhU" colab={"base_uri": "https://localhost:8080/"} outputId="794b82e4-4f46-4985-c956-06f88f684120"
# Create copy to work with
releasedatedata = mergeddata.copy()

# Select entries where release date is not null
releasedatedata = releasedatedata[releasedatedata['release_date'].notnull()]

# Replace strings which are not of the format xxxx-xx-xx with None
releasedatedata['release_date'] = releasedatedata['release_date'].map(lambda x : x if x[-3] == '-'else None)

# Select entries where release date is not null
releasedatedata = releasedatedata[releasedatedata['release_date'].notnull()]

# Convert to DateTime 
releasedatedata['release_date'] = pd.to_datetime(releasedatedata['release_date'])

# Check 
releasedatedata['release_date'].describe()
```

<!-- #region id="l8rmy8fJ0UhV" -->
Of course, we now have plenty of duplicate entries. However we note that the games span 1983 to 2018.
<!-- #endregion -->

<!-- #region id="qM9JNd-40UhV" -->
### Game library size
<!-- #endregion -->

```python id="J4eQa-sV0UhW" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="9b8342a0-7084-4a53-e7c9-d2d60fad65a0"
# View head
numgames.head()
```

```python id="yu7cwWUp0UhX" colab={"base_uri": "https://localhost:8080/"} outputId="0412e28f-db40-4d99-a0d9-cd000a7c2885"
# Get summary statistics
numgames['items_count'].describe()
```

<!-- #region id="H0ZjrOIK0Uhb" -->
We have data for 88310 unique steam users. We note that the minimum number of games owned is 0 whereas the maximum is 7762. The average number of games owned is 58.
<!-- #endregion -->

```python id="HWWjG1No0Uhc" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="968faf35-8f92-4380-acc3-2b8fa647accb"
# Plot distribution of `items_count`
numgames['items_count'].hist()
plt.title('Distribution of number of games owned')
plt.xlabel('Number of games')
plt.ylabel('Number of users')
plt.show()
```

```python id="eTEVwnCb0Uhd" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="52f13158-6afe-4eec-caad-134c7aa9ee90"
# Plot distribution of items_count within 90% centile
numgames[numgames['items_count'] < numgames['items_count'].quantile(0.90)].hist()
plt.title('Distribution of number of games owned')
plt.xlabel('Number of games')
plt.ylabel('Number of users')
plt.show()
```

```python id="GCIUFstO0Uhe" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="61fa6394-ff55-4c28-d333-c4d1ba9af4a7"
# Plot distribution of items_count within 90% centile
numgames[numgames['items_count'] < numgames['items_count'].quantile(0.90)].hist()
plt.title('Distribution of number of games owned')
plt.xlabel('Number of games')
plt.ylabel('Number of users')
plt.savefig('numgames.pdf', bbox_inches = "tight")
plt.show()
```

<!-- #region id="JEDdNL4B0Uhf" -->
**Recommendation:**

Focus campaign on users who have below the average number of games of 58. These users are more likely to find games they do not own which appeal.
<!-- #endregion -->

<!-- #region id="2MJXJ2x20Uhg" -->
### Game Price
<!-- #endregion -->

```python id="qe1LbMq50Uhh" colab={"base_uri": "https://localhost:8080/"} outputId="578bd41a-75c2-4287-83e1-2e08d19206ab"
# Create a copy to work with
gamesprice = gamesdata.copy()

# Get statistics and type
gamesprice['price'].describe()
```

<!-- #region id="nqyEhVGV0Uhi" -->
We see that the values are of type `object`. 

From viewing the head above, we noticed the presence of the string `Free To Play`. Let us replace that value with 0. 

We will also iterate and replace all strings we find with 0.
<!-- #endregion -->

```python id="M7i-UDkh0Uhi"
gamesprice.price = gamesprice.price.replace(to_replace = 'Free To Play', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free to Play', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free Demo', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Play for Free!', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Install Now', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Play WARMACHINE: Tactics Demo', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free Mod', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Install Theme', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Third-party', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Play Now', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free HITMAN™ Holiday Pack', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Play the Demo', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Starting at $499.00', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Starting at $449.00', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free to Try', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free Movie', value = 0)
gamesprice.price = gamesprice.price.replace(to_replace = 'Free to Use', value = 0)
```

```python id="6NOIQ7r20Uhj"
# Convert to float
gamesprice['price'] = gamesprice['price'].astype(float)
```

```python id="pNBumagm0Uhk" colab={"base_uri": "https://localhost:8080/"} outputId="75b34a3e-5f9d-4ac6-f6b1-f5a8349e8956"
# Get summary statistics
gamesprice['price'].describe()
```

<!-- #region id="tRqTrEnT0Uhl" -->
We see that 75% of games are under $10! Looks like the majority of games are cheap.
<!-- #endregion -->

```python id="q6G_GnI60Uhl"
belowcentile = gamesprice[gamesprice['price'] < gamesprice['price'].quantile(0.99)]
```

```python id="AsKrTawD0Uhl" colab={"base_uri": "https://localhost:8080/"} outputId="eda5e73e-5762-446b-c829-1a3f0054af92"
belowcentile['price'].describe()
```

```python id="MiRJTV0p0Uhm" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="294de55b-2872-4d71-e6b8-3ce061ced3c2"
belowcentile['price'].hist()
plt.xlabel('Price in USD')
plt.title('Game Price Distribution')
plt.savefig('price.pdf', bbox_inches = "tight")
plt.show()
```

<!-- #region id="VU33QniW0Uhn" -->
**Recommendation**: Focus on volume of sales as the 75% of games are below $10. Highlights the importance of bundles for higher single transactions and where the user may not be interested in all games but still think it worthwhile.
<!-- #endregion -->

<!-- #region id="ZZ4PgzYd0Uhn" -->
### Game genre
<!-- #endregion -->

```python id="2hV8uerG0Uho" colab={"base_uri": "https://localhost:8080/", "height": 615} outputId="bce934d7-795d-46ea-b4d7-f3ac50c0238c"
gamesdata.head()
```

```python id="0JhpR9kz0Uho" colab={"base_uri": "https://localhost:8080/"} outputId="1f6523ff-bcc3-483f-e273-c985cf25d1fb"
# Create copy
gamegenres = gamesdata.copy()

# Drop NaN
gamegenres = gamegenres[gamegenres['genres'].notnull()]

# Get unique lists
genres = list(gamegenres['genres'].astype('str').unique())

# View first 5
genres[:5]
```

```python id="LjT8AeoI0Uhp" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="2155bd87-56c8-42c0-91c4-ae48e80863fa"
# Combine all strings
allgenres = ','.join(genres)

# Preview first 100 characters
allgenres[:100]
```

```python id="6KwtCDwL0Uhq" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="02c7c65a-1ef2-42d1-af82-40c268dea310"
# Replace chars
allgenres = allgenres.replace("' '",",").replace("\n","").replace("[","").replace("]", "").replace("'", "").replace(" ","")

# Check
allgenres[:100]
```

```python id="Zx6h8pcv0Uhq" colab={"base_uri": "https://localhost:8080/"} outputId="97e76097-7295-4208-8c41-35dd0747e809"
# Split
splitgenres = allgenres.split(',')
splitgenres[:5]
```

```python id="w-aW4pDG0Uhs" colab={"base_uri": "https://localhost:8080/"} outputId="b292e21e-7f47-44ae-fe5a-bb51d9715740"
# Use set to obtain unique values
uniquegenres = set(splitgenres)
uniquegenres
```

```python id="dX3sKQD40Uhs" colab={"base_uri": "https://localhost:8080/", "height": 301} outputId="cf8bb3e6-48d2-4846-bcd9-91b5c7bc220d"
# Create columns with genres
for genre in uniquegenres:
    gamegenres[genre] = 0
    
# Split genres in genres column
gamegenres['genres'] = gamegenres['genres'].astype('str').map(lambda x : x.replace("' '",",").replace("\n","").replace("[","").replace("]", "").replace("'", "").replace(" ","").split(','))

# Map to columns - set to 1 if genre applies
for index, genres in enumerate(gamegenres['genres']):
    for genre in genres:
        gamegenres.loc[index,genre] = 1
        
# Visuale the new columns
gamegenres.head(2)  
```

```python id="hFgiN7v10Uht" colab={"base_uri": "https://localhost:8080/"} outputId="322882d3-e832-4102-81db-5df569e2c092"
gamegenres.columns
```

```python id="HTtbZ7dU0Uhu" colab={"base_uri": "https://localhost:8080/"} outputId="2cda0c64-58b8-4ded-a5cd-0388df69ab4d"
# Start with empty dictionary
genredict = {}

# Get genre columns
genrecols = gamegenres.iloc[:, 16:].columns

# Go through each column and sum it
for col in genrecols:
    genredict[col] = gamegenres[col].sum()
        
# sort dictionary based on counts, ascending order so reverse = True    
sortedgenresdict = {keys: values for keys, values in \
                        sorted(genredict.items(), key = lambda item: item[1], reverse = True)}
sortedgenresdict = sorted(sortedgenresdict.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)  

sortedgenresdict
```

<!-- #region id="Fx70rPv70Uhv" -->
We see that Indie is the most popular genre, followed by Action. On the other end of the spectrum, there are few entries relating to Photo Editing and only 7 for Accounting. This makes sense as Steam is a gaming platform, and so photo editing or accounting software doesn't really belong.
<!-- #endregion -->

<!-- #region id="L6EjDuh10Uhw" -->
### Game tags
<!-- #endregion -->

```python id="yFrhaK640Uhw" colab={"base_uri": "https://localhost:8080/"} outputId="b0524e45-9ea2-499b-8e6d-37d536200faa"
# Create copy
gametags = gamesdata.copy()

# Drop NaN
gametags = gamegenres[gamegenres['tags'].notnull()]

# Get unique lists
tags = list(gametags['tags'].astype('str').unique())

# View first 5
tags[:5]
```

```python id="1GF0OsYp0Uhy" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="8d2266e6-b854-4f17-9c92-f9623f0503bf"
# Combine all strings
alltags = ','.join(tags)

# Preview first 100 characters
alltags[:100]
```

```python id="RKwYeKmQ0Uh0" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="aac53e48-4862-485c-be0b-68aeb5c1a243"
# Replace chars
alltags = alltags.replace("' '",",").replace("\n","").replace("["," ").replace("]", "").replace("'", "")

# Check
alltags[:100]
```

```python id="pSngy-Q30Uh1" colab={"base_uri": "https://localhost:8080/"} outputId="512f4a39-e909-4a4f-c1d1-d4fcb062dd6f"
# Split
splittags = alltags[1:].split(',')
splittags[:5]
```

```python id="YQhdl6Ns0Uh3" colab={"base_uri": "https://localhost:8080/"} outputId="7f8a8ff7-7e3a-465f-bdb8-6c2b2c666be6"
# Use set to obtain unique values
uniquetags = set(splittags)
len(uniquetags)
```

<!-- #region id="rsoHmYUC0Uh4" -->
### Top publishers
<!-- #endregion -->

```python id="4Th03PbW0Uh4"
# Select entries where publisher is non-null
data = gamesdata[gamesdata['publisher'].notnull()]
```

```python id="ocv7Z25U0Uh5"
# Create dictionary
game_publishers = {}
for publisher in list(data['publisher']):
    if not publisher in game_publishers:
        game_publishers[publisher] = 1
    else:
        game_publishers[publisher] += 1
```

```python id="kNh4ouju0Uh6" colab={"base_uri": "https://localhost:8080/"} outputId="cc71b64d-853d-47be-f839-ddc58d01daf7"
# Get top 10 publishers
top10_publishers = dict(Counter(game_publishers).most_common(10))
top10_publishers
```

```python id="McCDA-400Uh6" colab={"base_uri": "https://localhost:8080/", "height": 281} outputId="ee874751-b3a4-4450-a400-c1e162dae07c"
# Prepare for bar chart plot
top10_publishers = dict(sorted(Counter(game_publishers).most_common(10), key=lambda x:x[1]))

# Plots most popular publishers
fig = plt.figure(figsize = (8,4))
plt.barh(range(len(top10_publishers)), list(top10_publishers.values()), align='center')
plt.yticks(range(len(top10_publishers)), list(top10_publishers.keys()), fontsize=12)
plt.title("Most Popular Publishers", fontsize=12, fontweight= 22)
plt.show()
```

<!-- #region id="guhdFSW589jB" -->
## Modeling
<!-- #endregion -->

<!-- #region id="3QQyv_hm9MiR" -->
### Utils
<!-- #endregion -->

```python id="TkNCsMya8-X0"
def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Creates an interaction matrix DataFrame
    Arguments:
        df = Pandas DataFrame containing user-item interactions
        user_col = column name containing user's identifier
        item_col = column name containing item's identifier
        rating col = column name containing user rating on given item
        norm (optional) = True if a normalization of ratings is needed
        threshold (required if norm = True) = value above which the rating is favorable
    Returns:
        Pandas DataFrame with user-item interactions
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions

def create_user_dict(interactions):
    '''
    Creates a user dictionary based on their index and number in interaction dataset
    Arguments:
        interactions - DataFrame with user-item interactions
    Returns:
        user_dict - Dictionary containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict

def create_item_dict(df,id_col,name_col):
    '''
    Creates an item dictionary based on their item_id and item name
    Arguments: 
        - df = Pandas dataframe containing item information
        - id_col = column name containing unique identifier for an item
        - name_col = column name containing name of the item
    Returns:
        item_dict = Dictionary containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

def run_model(interactions, n_components=30, loss='warp', epoch=30, n_jobs = 4):
    '''
    Runs matrix-factorization model using LightFM
    Arguments:
        interactions = Pandas DataFrame containing user-item interactions
        n_components = number of desired embeddings to create to define item and user
        loss = loss function other options are logistic, brp
        epoch = number of epochs to run 
        n_jobs = number of cores used for execution 
    Returns:
        Model = Trained model
    '''
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss)
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model

def get_recs(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,num_items = 10, show_known = True, show_recs = True):
    '''
    Produces user recommendations
    Arguments:
        model = Trained matrix factorization model
        interactions = dataset used for training the model
        user_id = user ID for which we need to generate recommendation
        user_dict = Dictionary containing interaction_index as key and user_id as value
        item_dict = Dictionary containing item_id as key and item_name as value
        threshold = value above which the rating is favorable in new interaction matrix
        num_items = Number of recommendations to provide
        show_known (optional) - if True, prints known positives
        show_recs (optional) - if True, prints list of N recommended items  which user hopefully will be interested in
    Returns:
        list of titles user_id is predicted to be interested in 
    '''
    n_users, n_items = interactions.shape
    # Get value for user_id using dictionary
    user_x = user_dict[user_id]
    # Generate predictions
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    # Get top predictions
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    # Get list of known values
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    # Ensure predictions are not already known
    scores = [x for x in scores if x not in known_items]
    # Take required number of items from prediction list
    return_score_list = scores[0:num_items]
    # Convert from item id to item name using item_dict
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    
    if show_known == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1
            
    if show_recs == True:
        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return scores

def create_item_emdedding_matrix(model,interactions):
    '''
    Creates item-item distance embedding matrix
    Arguments:
        model = trained matrix factorization model
        interactions = dataset used for training the model
    Returns:
        Pandas dataframe containing cosine distance matrix between items
    '''
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_matrix = pd.DataFrame(similarities)
    item_emdedding_matrix.columns = interactions.columns
    item_emdedding_matrix.index = interactions.columns
    
    return item_emdedding_matrix

def get_item_recs(item_emdedding_matrix, item_id, 
                             item_dict, n_items = 10, show = True):
    '''
    Function to create item-item recommendation
    Arguments: 
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
        - item_id  = item ID for which we need to generate recommended items
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - n_items = Number of items needed as an output
    Returns:
        - recommended_items = List of recommended items
    '''
    recommended_items = list(pd.Series(item_emdedding_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    if show == True:
        print("Item of interest: {0}".format(item_dict[item_id]))
        print("Similar items:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' +  item_dict[i])
            counter+=1
    return recommended_items
```

<!-- #region id="WC6Y8D4iACP9" -->
### Data loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="NvesxWzq_RwO" outputId="141303e9-3ae3-4f77-bc04-a154c2ff9759"
recdata = pd.read_parquet("recdata.parquet.gzip")
recdata
```

```python id="puXQTNVl8bHL" colab={"base_uri": "https://localhost:8080/", "height": 615} outputId="e9913e23-06eb-4cf9-f800-2d829edd94c3"
gamesdata = pd.read_parquet("gamesdata.parquet.gzip")
gamesdata.head()
```

<!-- #region id="oMf8u6DK8bHO" -->
### Create interaction matrix
<!-- #endregion -->

<!-- #region id="NpYm9GXU8bHP" -->
We will create an interactions matrix using the user-item data. This is done using the `create_interaction_matrix` function, which can be found in `resources.py`.
<!-- #endregion -->

```python id="sRd_vilz8bHP"
# Use create_interaction_matrix function
interactions = create_interaction_matrix(df = recdata,
                                         user_col = 'uid',
                                         item_col = 'id',
                                         rating_col = 'owned')
interactions.shape
```

<!-- #region id="r8rLcK8W8bHQ" -->
From the shape, we note that we have 69277 unique users and 8791 different games represented.
<!-- #endregion -->

```python id="DqqHsbPG8bHR" outputId="eba6e0a1-9538-4a4b-f4a2-1317589af96b"
# Preview head
interactions.head()
```

<!-- #region id="Ztqms-xw8bHR" -->
### Train test split
<!-- #endregion -->

<!-- #region id="bgVgEA_t8bHS" -->
We will manually split the interactions matrix into a training set and test set for evaluation purposes.
<!-- #endregion -->

```python id="WNj1_RRN8bHS" outputId="4319e146-e5ba-4a1f-9417-3be856a4a58c"
# Get number of users
len(interactions)
```

<!-- #region id="hWWOqdKs8bHS" -->
We choose to have roughly 80% of our data as training and 20% as test.
<!-- #endregion -->

```python id="fJ6VqBaJ8bHT" outputId="d8ef52f3-9966-4a36-aa65-a9f4ef85ed78"
# Establish number of users in train/test sets

train_num = round((80/100)*len(interactions),0)
print(f'We desire {train_num} users in our training set.')

test_num = len(interactions)-train_num
print(f'We desire {test_num} users in our test set.')
```

```python id="XXDXQMqK8bHU"
# Define train and test sets
train = interactions[:55422]
test = interactions[55422:]
```

<!-- #region id="_goyJGaH8bHU" -->
### Create user dictionary
<!-- #endregion -->

<!-- #region id="bn5RnOE88bHU" -->
We will create a dictionary which matches users with another a counter `id`, by using the `create_user_dict` function, which can be found in `resources.py`.
<!-- #endregion -->

```python id="8us5BoZl8bHX"
# Create user dictionary using helper function
user_dict = create_user_dict(interactions=interactions)
```

<!-- #region id="6nhsw0CK8bHX" -->
### Create item dictionary
<!-- #endregion -->

<!-- #region id="g7zAQL8k8bHX" -->
We will create a dictionary which matches each game `id` with its `title`, by using the `create_item_dict`function, which can be found in `resources.py`.
<!-- #endregion -->

```python id="lV0IIwPH8bHY"
# Create game dictionary using helper function
games_dict = create_item_dict(df = gamesdata, id_col = 'id', name_col = 'title')
```

<!-- #region id="BjnEQW7k8bHY" -->
### Create sparse matrices
<!-- #endregion -->

<!-- #region id="p27WQt8H8bHZ" -->
We will transform the interaction into a sparse matrix, to make computations efficient.

For the trainset, we simply use the `sparse.csr_matrix()` function.

With the test set, due to a known issue, we need to add additional rows so that the number of rows matches the trains set.
<!-- #endregion -->

```python id="zjpm98w48bHZ"
# Create sparse matrices for evaluation 
train_sparse = sparse.csr_matrix(train.values)

#Add X users to Test so that the number of rows in Train match Test
N = train.shape[0] #Rows in Train set
n,m = test.shape #Rows & columns in Test set
z = np.zeros([(N-n),m]) #Create the necessary rows of zeros with m columns
#test = test.toarray() #Temporarily convert Test into a numpy array
test = np.vstack((test,z)) #Vertically stack Test on top of the blank users
test_sparse = sparse.csr_matrix(test) #Convert back to sparse
```

<!-- #region id="lFfNUS5R8bHa" -->
### Modelling using LightFM
<!-- #endregion -->

<!-- #region id="hkGJACtr8bHa" -->
To build our recommendation engine, we will make use of the LightFM library.
<!-- #endregion -->

<!-- #region id="sXIa40zu8bHa" -->
### WARP loss model
<!-- #endregion -->

<!-- #region id="d9Hy4izW8bHb" -->
For the first model, we will choose the loss function to be WARP.
<!-- #endregion -->

```python id="XU-fzxxA8bHb"
# Instantiate and fit model
mf_model_warp = run_model(interactions = train,
                 n_components = 30,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)
```

```python id="EDrGzaFt8bHc" outputId="af881c42-dde9-4f7e-82ed-71cd1f37be89"
# Get precision
train_precision = precision_at_k(mf_model_warp, train_sparse, k=10).mean()
test_precision = precision_at_k(mf_model_warp, test_sparse, k=10).mean()
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
```

<!-- #region id="faNAje9h8bHd" -->
We note that the precision score for the test is low, but this is expected due to the sheer volume of games available and the scarcity of interactions.
<!-- #endregion -->

```python id="PG5s0-XG8bHd" outputId="ccb97c9e-521f-4d83-c20e-3b89be1d1269"
# Get AUC
train_auc = auc_score(mf_model_warp, train_sparse).mean()
test_auc = auc_score(mf_model_warp, test_sparse).mean()
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
```

<!-- #region id="G5deY38Y8bHe" -->
The AUC score is very high, which is great.
<!-- #endregion -->

<!-- #region id="wdnc-T828bHe" -->
### BPR loss model
<!-- #endregion -->

<!-- #region id="YOJ4V3Af8bHf" -->
We will compare the model above with a model using BPR as the loss function.
<!-- #endregion -->

```python id="AZtchOvw8bHf"
# Instantiate and fit model
mf_model_bpr = run_model(interactions = train,
                 n_components = 30,
                 loss = 'bpr',
                 epoch = 30,
                 n_jobs = 4)
```

```python id="b1qmxvzT8bHf" outputId="f8f20d3d-bcb0-424b-96db-b9f6000b218a"
# Get precision
train_precision = precision_at_k(mf_model_bpr, train_sparse, k=10).mean()
test_precision = precision_at_k(mf_model_bpr, test_sparse, k=10).mean()
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
```

<!-- #region id="2JsS8g668bHg" -->
Again, we note that the precision score for the test is low, but this is expected due to the sheer volume of games available and the scarcity of interactions.
<!-- #endregion -->

```python id="irgZBXbX8bHg" outputId="ed289f72-4e52-48e7-8740-267520476c80"
# Get AUC
train_auc = auc_score(mf_model_bpr, train_sparse).mean()
test_auc = auc_score(mf_model_bpr, test_sparse).mean()
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
```

<!-- #region id="vAuvytcP8bHh" -->
Again, the AUC score is good, though significantly lower for the test set compare to the train set.
<!-- #endregion -->

<!-- #region id="9NqewAgk8bHh" -->
Based on these two models, we will keep WARP as the loss function due to better performance all round.
<!-- #endregion -->

<!-- #region id="1qCOCC598bHi" -->
### Adjust num of components
<!-- #endregion -->

<!-- #region id="tv0C7MK98bHi" -->
The `n_components` parameter controlls the number of embeddings (dimension of the features in the latent space.)

We will vary this number, lowering it to `10` first and then increasing it to `50` to see how this affects model performance.
<!-- #endregion -->

```python id="wy0CDAhX8bHi"
# Instantiate and fit model
mf_model_warp_2 = run_model(interactions = train,
                 n_components = 10,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)
```

```python id="j4cYimVI8bHj" outputId="d50aea69-0fd7-42ea-e029-5acd3c4246b8"
# Get precision
train_precision = precision_at_k(mf_model_warp_2, train_sparse, k=10).mean()
test_precision = precision_at_k(mf_model_warp_2, test_sparse, k=10).mean()
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
```

```python id="X_6enPTI8bHj" outputId="3f730d54-e582-46f0-94a7-551f5f3ef2f5"
# Get AUC
train_auc = auc_score(mf_model_warp_2, train_sparse).mean()
test_auc = auc_score(mf_model_warp_2, test_sparse).mean()
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
```

```python id="k5mZ8RqD8bHk"
# Instantiate and fit model
mf_model_warp_50 = run_model(interactions = train,
                 n_components = 50,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)
```

```python id="fchNgh9R8bHk" outputId="484d67fd-e491-475a-d849-36d32675a77e"
# Get precision
train_precision = precision_at_k(mf_model_warp_50, train_sparse, k=10).mean()
test_precision = precision_at_k(mf_model_warp_50, test_sparse, k=10).mean()
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
```

```python id="n9uLPEYj8bHk" outputId="c4352219-46d8-45d5-b85e-5e5093b5a932"
# Get AUC
train_auc = auc_score(mf_model_warp_50, train_sparse).mean()
test_auc = auc_score(mf_model_warp_50, test_sparse).mean()
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
```

<!-- #region id="0BqZee0i8bHl" -->
Overall, we see that varying the `n_components` feature has little impact on overall model performance.
<!-- #endregion -->

<!-- #region id="d_otYyNZ8bHl" -->
### Final model
<!-- #endregion -->

<!-- #region id="eE22_xqB8bHl" -->
We train the chosen model (WARP loss and 30 components) on the full interactions matrix.
<!-- #endregion -->

```python id="fpU5CD4s8bHm"
# Instantiate and fit model on full interactions set
mf_model = run_model(interactions = interactions,
                 n_components = 30,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)
```

<!-- #region id="6BFzE9Gk8bHm" -->
### Embeddings
<!-- #endregion -->

<!-- #region id="VUZymGUX8bHm" -->
Before using our model to provide recommendations, let us explore the embedding space.
<!-- #endregion -->

<!-- #region id="dbXU_OOA8bHp" -->
### Retrieve embeddings matrix
<!-- #endregion -->

```python id="LVDgjowC8bHq" outputId="59151632-a0d4-42c1-9b5a-a73e29877312"
# Get embeddings
embeddings = mf_model.item_embeddings
embeddings
```

```python id="m4ZdnAeA8bHr" outputId="4a18f88a-60a8-4001-ca91-2ad81bf53bcc"
# View shape
embeddings.shape
```

<!-- #region id="en99tG8T8bHs" -->
Our embeddings array has 8791 rows, with each row representing a game. Each row has 30 numbers, representing the components of our game embeddings. This is because we set `n_components` to be 30 in our model.
<!-- #endregion -->

<!-- #region id="ov6dYMCi8bHt" -->
Let's investigate a sample game vector.
<!-- #endregion -->

```python id="_TmCuqmJ8bHt" outputId="d039bf4d-48e6-424c-f7fe-c1695ca15e39"
embeddings[0]
```

<!-- #region id="YlDEknkr8bHu" -->
To retrieve the name of the game, we first look up the game id using our interactions matrix and then obtain the name using the games dictionary.
<!-- #endregion -->

```python id="6XVj2_QA8bHv" outputId="a66cdfbd-da74-41a2-dc93-47bb3b195adf"
firstgameid = interactions.columns[0]
games_dict[firstgameid]
```

<!-- #region id="_Dmtw-1b8bHv" -->
### Pair similarity
<!-- #endregion -->

<!-- #region id="lsLgnhWV8bHw" -->
Let us examine how close or distant pairs of games are in the embedding space. The embedding space can be interpreted as a distance metric and so we would expect similar games to have similar vectors.

Our first game will be `Counter-Strike` as seen above. 

We would expect it to be similar to `Left 4 Dead 2`.
<!-- #endregion -->

```python id="4WdM5Caf8bHx" outputId="f0273c27-06c7-439f-8798-d829e128ba35"
# Get data for both games
gamesdata[(gamesdata['title']=='Counter-Strike') | (gamesdata['title']=='Left 4 Dead 2') ]
```

<!-- #region id="GyxEuC3c8bHx" -->
Let us look at the vectors for these two games.
<!-- #endregion -->

```python id="459uGaly8bHy" outputId="c43b56cc-c250-4a8b-e30c-a731c4e45f84"
# Set index for Counter-Strike
cs_index = 0

# Obtain embeddings vector 
cs_vector = embeddings[cs_index]

cs_vector
```

```python id="DwqssFpa8bHz" outputId="c3bd9c9b-de2b-45ad-f85b-e121b95dd325"
# Retrieve game id for LFD2
lfd2_id = gamesdata[gamesdata['title']=='Left 4 Dead 2']['id'].values[0]

# Obtain index for Squad in interactions matrix
lfd2_index = list(interactions.columns).index(lfd2_id)

# Obtain embeddings vector
lfd2_vector = embeddings[lfd2_index]

lfd2_vector
```

<!-- #region id="x0AMbkcA8bH0" -->
To assign a single value to the similarity between these two vectors, we calculate the distance between them. Let us first compute the Euclidean distance.
<!-- #endregion -->

```python id="MMd-tsBt8bH0" outputId="d4bd7f49-d18a-49a2-a077-59d29a2198c8"
# Compute Euclidean distance
distance.euclidean(cs_vector, lfd2_vector)
```

<!-- #region id="QTZDpRg08bH1" -->
Let us compare this figure with a pair of games we believe to be very different.
<!-- #endregion -->

```python id="_df2Lun68bH1" outputId="197542d6-ae87-4f19-831c-54d7299efefb"
# Get data for both games
gamesdata[(gamesdata['title']=='Counter-Strike') | (gamesdata['title']=='The Room') ]
```

```python id="GoeKL8wQ8bH2" outputId="aad6aec4-1c00-4cf3-f6da-c2535de61bd7"
# Retrieve game id for The Room
room_id = gamesdata[gamesdata['title']=='The Room']['id'].values[0]

# Obtain index for Squad in interactions matrix
room_index = list(interactions.columns).index(room_id)

# Obtain embeddings vector
room_vector = embeddings[room_index]

room_vector
```

```python id="HwIqcXXU8bH3" outputId="20aedab0-04a6-41ac-f833-957dac541682"
# Compute Euclidean distance
distance.euclidean(cs_vector, room_vector)
```

<!-- #region id="dQDLSgxN8bH4" -->
When looking at the similarity of embeddings, it is more common to look at cosine similarity. Cosine similarity ranges between -1 and 1 based on the angle between the vectors. From this figure, cosine distance is defined as 1 minus the cosine similarity and therefore a value between 0 and 2.
<!-- #endregion -->

```python id="ZE19L4Ki8bH4" outputId="e9c2cacb-19eb-49f7-85ed-f29542c93188"
print(f'Cosine distance between Counter Strike and Left 4 Dead 2: {distance.cosine(cs_vector, lfd2_vector)}')
print(f'Cosine distance between Counter Strike and The Room: {distance.cosine(cs_vector, room_vector)}')
```

<!-- #region id="fDyftdQ38bH5" -->
### Exploring embeddings with Gensim
<!-- #endregion -->

```python id="FEDWRNKJ8bH6"
# Set embedding size
embedding_size = embeddings.shape[1]

# Create instance
kv = WordEmbeddingsKeyedVectors(embedding_size)

# Get list of game names in correct order to match embeddings
gameslist = []
for game_id in interactions.columns:
    name = games_dict[game_id]
    gameslist.append(name)
    
# Add to kv
kv.add(gameslist, embeddings )
```

<!-- #region id="jltlvJQZ8bH7" -->
Let us obtain the games closest to `Counter-Strike`.
<!-- #endregion -->

```python id="ix3EWdKU8bH7" outputId="1bcfe12a-13bd-4dac-f197-1413e4932545"
# Get games closest to Counter-Strike
kv.most_similar('Counter-Strike')
```

```python id="lonQ6_o-8bH8" outputId="0458d9cb-7856-4187-dbec-6e2db713ef93"
# Get games closest to Left 4 Dead 2
kv.most_similar('Left 4 Dead 2')
```

```python id="Mzse2Lp38bH9" outputId="ee94f204-897a-4f19-e307-6a6bead70787"
# Get games closest to The Room
kv.most_similar('The Room')
```

```python id="ChhQQ-ct8bH-" outputId="2c8a8437-03d8-47ad-f3fd-30dc1595819b"
# Get games closest to RollerCoaster Tycoon
kv.most_similar('RollerCoaster Tycoon®: Deluxe')
```

```python id="aagD9htv8bH-" outputId="a30f8953-c14f-4cb7-9769-b7988844bc49"
# Get games closest to Dishonored
kv.most_similar('Dishonored')
```

```python id="OXsVhzEd8bH_" outputId="085ed25a-f0d5-47a6-f33c-11e6a1fe4312"
# Get games closest to The Jackbox Party Pack
kv.most_similar('The Jackbox Party Pack')
```

```python id="UjUQSbhH8bIA" outputId="9422bd1a-31b9-4e0e-89be-ff53fc3442f8"
# Get games closest to American Truck Simulator
kv.most_similar('American Truck Simulator')
```

```python id="OlieSrhk8bIB"
def plot_similar(item, ax, topn=5):
    '''
    Plots a bar chart of similar items
    Arguments:
        - item, string
        - ax, axes on which to plot
        - topn (default = 5) number of similar items to plot
    '''
    sim = kv.most_similar(item, topn=topn)[::-1]
    y = np.arange(len(sim))
    w = [t[1] for t in sim]
    ax.barh(y, w)
    left = min(.6, min(w))
    ax.set_xlim(right=1.0, left=left)
    # Split long titles over multiple lines
    labels = [textwrap.fill(t[0] , width=24)
              for t in sim]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(item)   
```

```python id="wgwDb6bl8bIB" outputId="4792fd77-6039-43c4-c496-b4230f823605"
# Define list of games to visualise similar items for
games = ['Counter-Strike', 'The Room', 'RollerCoaster Tycoon®: Deluxe', 'Dishonored', 
         'The Jackbox Party Pack', 'American Truck Simulator']

# Set figure/axes to have 3 rows with 2 columns
fig, axes = plt.subplots(3, 2, figsize=(15, 9))

# Loop through games and use plot_similar function 
for game, ax in zip(games, axes.flatten()):
    plot_similar(game, ax)
    
fig.tight_layout()
```

<!-- #region id="fSXtWlOX8bIC" -->
### Visualizing embeddings with t-SNE
<!-- #endregion -->

<!-- #region id="5U8VMeEa8bID" -->
We will use the t-SNE algorithm to visualise embeddings, going from a 30-dimensional space (number of components) to a 2-dimensional space.
<!-- #endregion -->

```python id="XxyXJeuz8bID"
# Instantialte tsne, specify cosine metric
tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')

# Fit and transform
embeddings2d = tsne.fit_transform(embeddings)
```

<!-- #region id="Zb44OObB8bIE" -->
Let's create a DataFrame with the game name and 2d embeddings.
<!-- #endregion -->

```python id="6guCQ9wp8bIE" outputId="0f3a3575-7b63-493a-a291-127caff3e687"
# Create DF
embeddingsdf = pd.DataFrame()

# Add game names
embeddingsdf['game'] = gameslist

# Add x coordinate
embeddingsdf['x'] = embeddings2d[:,0]

# Add y coordinate
embeddingsdf['y'] = embeddings2d[:,1]

# Check
embeddingsdf.head()
```

<!-- #region id="xBJzN2Zc8bIH" -->
Let's draw a scatterplot of our games, using the 2D mapping we created.
<!-- #endregion -->

```python id="OEAqqr9O8bII" outputId="194b03d5-9317-4111-ae06-03cddf7f6c90"
# Set figsize
fig, ax = plt.subplots(figsize=(10,8))

# Scatter points, set alpha low to make points translucent
ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.1)

plt.title('Scatter plot of games using t-SNE')

plt.show()
```

<!-- #region id="RijGsbgU8bIJ" -->
It is hard to judge anything by the shape of the plot alone. However as a check, let us ensure that games we expect to be close are indeed close in this 2-dimensional space.
<!-- #endregion -->

<!-- #region id="CAjpCSZp8bIK" -->
Let's leave `Counter-Strike` alone and look for the `Roller Coaster Tycoon` games.
<!-- #endregion -->

```python id="gnhZiBvn8bIK" outputId="938eb635-c437-4b87-fbd6-48f7c4071b4f"
match = embeddingsdf[embeddingsdf.game.str.contains('RollerCoaster')]
match
```

<!-- #region id="Dpg-YRQn8bIL" -->
We will highlight the above games on our scatter plot.
<!-- #endregion -->

```python id="p7P-MpPM8bIL" outputId="b48df7e8-049b-4b18-8e76-e4668487ffcb"
# Set figsize
fig, ax = plt.subplots(figsize=(10,8))

Xlabeled = embeddings2d[match.index, 0]
Ylabeled = embeddings2d[match.index, 1]
labels = match['game'].values

# Scatter specific points
for x, y, label in zip(Xlabeled, Ylabeled, labels):
    ax.scatter(x, y, marker='1', label = label, s=90, color = 'red')

# Scatter points, set alpha low to make points translucent
ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.1)

plt.title('Scatter plot of games using t-SNE')
plt.legend()

plt.show()
```

<!-- #region id="HS4u-_ft8bIO" -->
### Recommendations for existing user
<!-- #endregion -->

```python id="4O5qGnu38bIP" outputId="15d70370-19e1-4dbb-d274-fe2d76dc4fc6"
# Get recommendations
rec_list_u12 = get_recs(model = mf_model, 
                    interactions = interactions, 
                    user_id = 5000, 
                    user_dict = user_dict,
                    item_dict = games_dict, 
                    threshold = 0,
                    num_items = 5,
                    show_known = True, 
                    show_recs = True)
```

<!-- #region id="Rf7APT488bIP" -->
These look reasonable, for instance we note the choice of sequels e.g. Portal 2 when user owns Portal.
<!-- #endregion -->

<!-- #region id="ThACZPyW8bIQ" -->
### Create item embedding matrix
<!-- #endregion -->

```python id="PbdaOrC-8bIR"
item_dist = create_item_emdedding_matrix(model = mf_model,interactions = interactions)
```

```python id="UkZPvlJ48bIR" outputId="8775cf64-bb65-41fe-ebe3-0935088e50f5"
item_dist.shape
```

```python id="E_-EKdPS8bIS" outputId="4752e5d9-0cb7-4230-e4a6-38dacff11bd2"
item_dist.head()
```

<!-- #region id="IgwTN_AB8bIS" -->
### Generate item recommendations
<!-- #endregion -->

```python id="TnBbLswu8bIT" outputId="818c8d75-9970-4076-c186-a6ae49b26cbd"
# Similar items to item_id 10 - Counter Strike
item_rec_list_10 = get_item_recs(item_emdedding_matrix = item_dist, 
                              item_id = 10, 
                              item_dict = games_dict, 
                              n_items = 5, 
                              show = True)
```

```python id="-dhbEBAi8bIU" outputId="8f5e48b8-336c-48d8-abf9-26969b016c0f"
gamesdata[gamesdata['title'] == 'The Witness']
```

```python id="TWGxTDxW8bIU" outputId="e0a433c3-6113-481b-9b99-55308866f8fb"
# Similar items to item_id 210970 The Witness
item_rec_list_210970 = get_item_recs(item_emdedding_matrix = item_dist, 
                              item_id = 210970, 
                              item_dict = games_dict, 
                              n_items = 5, 
                              show = True)
```

```python id="FsoHf4zZ8bIV" outputId="8e9783d7-dae1-4fe3-8341-dd2e3e5ba4fe"
gamesdata[gamesdata['title'] == 'American Truck Simulator']
```

```python id="QrgM0dN68bIW" outputId="14eac945-1f28-49ee-baf0-29942a76e57e"
# Similar items to item_id 270880 American Truck Simulator
item_rec_list_270880 = get_item_recs(item_emdedding_matrix = item_dist, 
                              item_id = 270880, 
                              item_dict = games_dict, 
                              n_items = 5, 
                              show = True)
```

<!-- #region id="9KUqgAMX8bIW" -->
These look reasonable.
<!-- #endregion -->

<!-- #region id="ylAF58NL8bIX" -->
### Recommendations for new user - Demo
<!-- #endregion -->

<!-- #region id="Okf5uZrY8bIX" -->
Example games owned:
* 210970 The Witness
* 288160 The Room
* 550 Left 4 Dead 2
<!-- #endregion -->

```python id="iuBnwQQ38bIY" outputId="92b302c8-32d1-40ac-ad1b-00f6b8de252c"
# Get list of owned games from user

# Create empty list to store game ids
ownedgames = []

# Create loop to prompt for game id and ask if want to continue
moretoadd = 'y'
while moretoadd == 'y':
    game = input('Please enter the game id: ')
    ownedgames.append(game)
    moretoadd = input('Do you have more games to add? y/n ')
    
# Print list of owned games
print(f"You own the following games: {ownedgames}")
```

<!-- #region id="0l1_uaReJvaS" -->
## References

1. [https://www.kaggle.com/tamber/steam-video-games/data](https://www.kaggle.com/tamber/steam-video-games/data)
2. [https://github.com/nadinezab/video-game-recs](https://github.com/nadinezab/video-game-recs)
3. [https://medium.com/web-mining-is688-spring-2021/video-game-recommendation-system-b9bcb306bf16](https://medium.com/web-mining-is688-spring-2021/video-game-recommendation-system-b9bcb306bf16)
4. [https://audreygermain.github.io/Game-Recommendation-System/](https://audreygermain.github.io/Game-Recommendation-System/)
5. [https://github.com/annelovepeace/Game-Recommendation-System](https://github.com/annelovepeace/Game-Recommendation-System)
6. [https://library.ucsd.edu/dc/object/bb5021836n/_3_1.pdf](https://library.ucsd.edu/dc/object/bb5021836n/_3_1.pdf)
7. [https://github.com/manandesai/game-recommendation-engine](https://github.com/manandesai/game-recommendation-engine)
8. [https://towardsdatascience.com/steam-recommendation-systems-4358917288eb](https://towardsdatascience.com/steam-recommendation-systems-4358917288eb)
9. [https://github.com/AlbertNightwind/Steam-recommendation-system](https://github.com/AlbertNightwind/Steam-recommendation-system)
10. [https://steffy-lo.github.io/Gameo/docs/en/](https://steffy-lo.github.io/Gameo/docs/en/)
11. [https://gitlab.com/recommend.games](https://gitlab.com/recommend.games)
12. [Personalized Bundle Recommendation in Online Games](https://dl.acm.org/doi/10.1145/3340531.3412734)
13. [A Machine-Learning Item Recommendation System for Video Games](https://ieeexplore.ieee.org/document/8490456)
14. [Content Based Player and Game Interaction Model for Game Recommendation in the Cold Start setting](https://arxiv.org/abs/2009.08947)
15. [Intelligent Game Recommendation System](https://aip.scitation.org/doi/epdf/10.1063/5.0042063)
16. [The MARS – A Multi-Agent Recommendation System for Games on Mobile Phones](https://link.springer.com/chapter/10.1007/978-3-642-30947-2_14)
17. [A Machine-Learning Item Recommendation System for Video Games](https://arxiv.org/pdf/1806.04900.pdf)
18. [HybridRank : A Hybrid Content-Based Approach To Mobile Game Recommendations](http://ceur-ws.org/Vol-1245/cbrecsys2014-paper02.pdf)
19. [A Recommender System for Mobile Applications of Google Play Store](https://thesai.org/Downloads/Volume11No9/Paper_6-A_Recommender_System_for_Mobile_Applications.pdf)
20. [https://www.recombee.com/domains/online-gaming.html](https://www.recombee.com/domains/online-gaming.html)
<!-- #endregion -->

<!-- #region id="6vLtkQnbBw5Y" -->
## Appendix
<!-- #endregion -->

<!-- #region id="eY5yBFGoJj3U" -->
### Expected features for a game recommendation system

1. **Adapting to your data:** A robust system that can utilize all data available to generate great recommendations for your users, including collaborative filtering and content-based models.
2. **Dynamically Retrained Models:** Real-time content personalization to meet the flourishing customer’s tastes and adaptation of newly added gaming content.
3. **Specific Functionalities to Online Gaming:** NLP recognizing texts in 80+ languages to analyze game attributes for an international clientele.
4. **AI-powered A/B Testing:** AutoML AI applied to keep maximal KPIs and advance the deep learning algorithm functions.
5. **Advanced Business Rules:** Boosters or filters to push forward desired games and easy to manipulate, adjustable rules for additional optimization of your content.
6. **Real AI Inside:** Reinforcement learning and other methods recommending desired games based on historical on-site behavior.
<!-- #endregion -->
