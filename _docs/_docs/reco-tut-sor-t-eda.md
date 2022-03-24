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

```python id="Eei0NuRj35QP"
import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UM90qwuS4K-k" executionInfo={"status": "ok", "timestamp": 1628747612895, "user_tz": -330, "elapsed": 3568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee14ae11-1e4e-434d-fab3-8c1c06416dde"
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

```python id="TE_AZh_X4K-p" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628763510307, "user_tz": -330, "elapsed": 861, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="24068815-3dde-4efc-de61-a8b874feaa8a"
!git status
```

```python id="sd_n1hSi4K-q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628763516504, "user_tz": -330, "elapsed": 2989, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0d9966dd-30ce-4552-dbc5-e78f1907a377"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="p-GuT3uF4O9n" -->
---
<!-- #endregion -->

<!-- #region id="KeB5VRUd4TMn" -->
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

The task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

The provided transactional data shows user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

Let's keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
<!-- #endregion -->

<!-- #region id="IGF0JS2V40y0" -->
## Dataset

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record
<!-- #endregion -->

```python id="QUFBWS-t9BAH"
import datetime
import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

```python id="qU6gMBox9ifm" executionInfo={"status": "ok", "timestamp": 1628750118062, "user_tz": -330, "elapsed": 2898, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# read in the json files
portfolio = pd.read_json('./data/bronze/portfolio.json', orient='records', lines=True)
profile = pd.read_json('./data/bronze/profile.json', orient='records', lines=True)
transcript = pd.read_json('./data/bronze/transcript.json', orient='records', lines=True)
```

<!-- #region id="nRv-d6ip9ovm" -->
## Portfolio
<!-- #endregion -->

<!-- #region id="SHGJsitj9twL" -->
| attribute | description |
| --------- | ----------- |
| id | offer id |
| offer_type | type of offer ie BOGO, discount, informational |
| difficulty | minimum required spend to complete an offer |
| reward | reward given for completing an offer |
| duration | time for offer to be open, in days |
| channels | email, web, mobile |
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="gqifGTIf9zCc" executionInfo={"status": "ok", "timestamp": 1628750118064, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5186a8af-4b20-4e06-c976-43a952a1309c"
portfolio
```

```python colab={"base_uri": "https://localhost:8080/"} id="qkvHvvlN927j" executionInfo={"status": "ok", "timestamp": 1628750118066, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ad6143de-42f3-4438-e334-e18ef6e72d48"
portfolio.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="Al5DTg35-bLx" executionInfo={"status": "ok", "timestamp": 1628750118067, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c889b883-f7dc-4e6e-de58-ffa09e1c0362"
portfolio.describe().round(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="ifOIVQUJ-wq0" executionInfo={"status": "ok", "timestamp": 1628750120532, "user_tz": -330, "elapsed": 2489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fff66368-5c7f-4817-cfe5-19e55e73d389"
fig, ax = plt.subplots(figsize=(12,7))
portfolio.hist(ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="EpckHS4n-eDM" executionInfo={"status": "ok", "timestamp": 1628750120533, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f288ad8-cb4a-4bfd-a3f7-e27f8dbf3044"
portfolio.describe(include='O')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 268} id="SlwAB0_o-vk9" executionInfo={"status": "ok", "timestamp": 1628750120534, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="75abf403-7017-43f8-e119-363c6b8af804"
portfolio.channels.astype('str').value_counts().plot(kind='barh');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 268} id="DpvRB8do_O5G" executionInfo={"status": "ok", "timestamp": 1628750121748, "user_tz": -330, "elapsed": 1228, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6d0978b-a7a7-4294-9ea9-3e33c61fd91c"
portfolio.offer_type.value_counts().plot(kind='barh');
```

<!-- #region id="I786e7NL_tIS" -->
## Transcript
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="VvxWS181AhwU" executionInfo={"status": "ok", "timestamp": 1628750121750, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e10d74ec-971a-4287-e799-5f7fb152625a"
transcript.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="b1Lbn69kAo5-" executionInfo={"status": "ok", "timestamp": 1628750121751, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a68eb2f1-22ed-4658-e3e2-91a7892d50d8"
transcript.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="4WfQIsd-Auyp" executionInfo={"status": "ok", "timestamp": 1628750121754, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="457990f9-2d34-4556-e007-7493b640adb4"
transcript.describe().round(1).T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="bwzEcw4VA5ta" executionInfo={"status": "ok", "timestamp": 1628750150390, "user_tz": -330, "elapsed": 28662, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8947797d-5783-4ad2-f516-be7479822488"
transcript.describe(include='O')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 268} id="SBp5_Zc_A8l1" executionInfo={"status": "ok", "timestamp": 1628750150391, "user_tz": -330, "elapsed": 50, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f8b67ee-d60d-4559-c3ba-74f6c3f10b90"
transcript.event.astype('str').value_counts().plot(kind='barh');
```

<!-- #region id="xF4Wp0F4BQno" -->
## Profile
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="l9uv6HQBBVco" executionInfo={"status": "ok", "timestamp": 1628750150393, "user_tz": -330, "elapsed": 48, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="54d7444b-de58-40c2-9d03-a2575559b6c7"
profile.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="MvHnlWDBBWrS" executionInfo={"status": "ok", "timestamp": 1628750150394, "user_tz": -330, "elapsed": 47, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fbcdb27f-5dec-47cc-89b2-ff423e46b4ff"
profile.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="Vv4P25jGBhHs" executionInfo={"status": "ok", "timestamp": 1628750150395, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8dfb95f1-a95a-438d-dea0-83ab7fdca153"
profile.describe().round(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="l2V8pZvJBouL" executionInfo={"status": "ok", "timestamp": 1628750152468, "user_tz": -330, "elapsed": 2106, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="47825892-4193-4e02-9c7d-b73e8d8cb800"
fig, ax = plt.subplots(figsize=(12,7))
profile.hist(ax=ax)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="7kd-Q3BrBy2D" executionInfo={"status": "ok", "timestamp": 1628750152470, "user_tz": -330, "elapsed": 39, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c771e9c7-f7fa-446f-ba17-50c4658e25c5"
profile.describe(include='O')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 268} id="We-kTthgBPgG" executionInfo={"status": "ok", "timestamp": 1628750152474, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12d58d66-93ec-4f70-fb5d-96d9a2f7ec93"
profile.gender.astype('str').value_counts(dropna=False).plot(kind='barh');
```

<!-- #region id="ml-mNxim9NxD" -->
## Cleaning the data and Feature Engineering

<!-- #endregion -->

```python id="ckGknxwb9NxD" colab={"base_uri": "https://localhost:8080/", "height": 398} executionInfo={"status": "ok", "timestamp": 1628750160479, "user_tz": -330, "elapsed": 8037, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a02c7492-9177-4fad-a6f7-76dbb439274c"
group_income = profile.groupby(['income', 'gender']).size().reset_index()
group_income.columns = ['income', 'gender', 'count']

sns.catplot(x="income", y="count", hue="gender", data=group_income,
                  kind="bar", palette="muted", height=5, aspect=12/5)
plt.xlabel('Income per year')
plt.ylabel('Count')
plt.title('Age/Income Distribution')
plt.savefig('./extras/images/income-age-dist-binned.png', dpi=fig.dpi)
```

```python id="IIBD1Q8V9NxE" executionInfo={"status": "ok", "timestamp": 1628750160490, "user_tz": -330, "elapsed": 41, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
portfolio['web'] = portfolio['channels'].apply(lambda x: 1 if 'web' in x else 0)
portfolio['email'] = portfolio['channels'].apply(lambda x: 1 if 'email' in x else 0)
portfolio['mobile'] = portfolio['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
portfolio['social'] = portfolio['channels'].apply(lambda x: 1 if 'social' in x else 0)
    
# apply one hot encoding to offer_type column
offer_type = pd.get_dummies(portfolio['offer_type'])

# drop the channels and offer_type column
portfolio.drop(['channels', 'offer_type'], axis=1, inplace=True)

# combine the portfolio and offer_type dataframe to form a cleaned dataframe
portfolio = pd.concat([portfolio, offer_type], axis=1, sort=False)
```

```python id="Pd5O4gL29NxF" executionInfo={"status": "ok", "timestamp": 1628750162041, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
profile['memberdays'] = datetime.datetime.today().date() - pd.to_datetime(profile['became_member_on'], format='%Y%m%d').dt.date
profile['memberdays'] = profile['memberdays'].dt.days
profile['income'] = profile['income'].fillna(0)

profile['gender'] = profile['gender'].fillna('X')
profile['gender'] = profile['gender'].map({'X':0,'O':1, 'M':2, 'F':3})
income_bins = [0, 20000, 35000, 50000, 60000, 70000, 90000, 100000, np.inf]
labels = [0,1,2,3,4,5,6,7]
profile['income'] = pd.cut(profile['income'], bins = income_bins, labels= labels, include_lowest=True)
```

```python id="5nzETyQD9NxH" colab={"base_uri": "https://localhost:8080/", "height": 398} executionInfo={"status": "ok", "timestamp": 1628750162043, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40b7beb6-2768-462b-f532-5cde85aa8f15"
# Let's plot the sama data and see if this provide us with better insights

group_income = profile.groupby(['income', 'gender']).size().reset_index()
group_income.columns = ['income', 'gender', 'count']

sns.catplot(x="income", y="count", hue="gender", data=group_income,
                  kind="bar", palette="muted", height=5, aspect=12/5)
plt.xlabel('Income per year')
plt.ylabel('Count')
plt.title('Age/Income Distribution')
plt.savefig('./extras/images/income-age-dist-binned.png', dpi=fig.dpi)
```

<!-- #region id="EzG3tNoAGQ6C" -->
## Joining the data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="MciojYTbFy-u" executionInfo={"status": "ok", "timestamp": 1628750655313, "user_tz": -330, "elapsed": 1403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6c3e8065-e664-4ae8-91a3-c8cdba54fc20"
transcript = transcript[transcript.person != None]
# extract ids for each offer
transcript['offer_id'] = transcript[transcript.event != 'transaction']['value'].apply(lambda x: 
                                                             dict(x).get('offer id') 
                                                             if dict(x).get('offer id') is not None 
                                                             else dict(x).get('offer_id') )

# transaction offers does not have offer id, so we filter them out next
joined_df = pd.merge(profile, transcript[transcript.event != 'transaction'], how='left', left_on=['id'], right_on=['person'])
joined_df['event'] = joined_df['event'].map({'offer received': 0, 'offer viewed': 1, 'offer completed': 2})

# rename column for ease of joining of dataframes
portfolio.rename({'id':'offer_id'}, inplace=True, axis=1)

# now all data can be joined together
df = pd.merge(joined_df, portfolio, how='inner', left_on=['offer_id'], right_on=['offer_id'])
df = df.drop(['person', 'value'], axis=1)

df.head()
```

<!-- #region id="fY3E91GI9NxM" -->
## Exploring correlations

Correlation is used to find which values are closely related with each other.
Now let's describe how values are correlated with each ther. For simplicity - the size of the output dot will define the correlation (the bigger - the closer).
<!-- #endregion -->

```python id="PhvN3NbG9NxM" colab={"base_uri": "https://localhost:8080/", "height": 783} executionInfo={"status": "ok", "timestamp": 1628750667280, "user_tz": -330, "elapsed": 4382, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b9dc92b1-07b1-4ed4-f275-6bc8ea004682"
#!mkdir images
def heatmap(x, y, size, figsize=(18,15), fig_name='temp.png'):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    fig.savefig(fig_name, dpi=fig.dpi)
    
offer_specs = ['difficulty', 'duration', 'reward', 'web',
       'email', 'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'became_member_on', 'gender', 'income', 'memberdays']

corr = df[offer_specs + user_specs + ['event']].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'event']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['event'].abs(),
    fig_name='./extras/images/heatmap-general.png'
)
```

<!-- #region id="fhVWP5y79NxO" -->
Correlation between features seems to be quite weak. However it can be noted that `bogo` is strongly related to `discount` and `reward` fields, while `mobile` channel is correlated with `difficulty` field. Which is quite expected.

Now let's see more closely into columns of our interest and define if this should be cleaned or changed.
<!-- #endregion -->

```python id="VV7gjhHe9NxP" colab={"base_uri": "https://localhost:8080/", "height": 293} executionInfo={"status": "ok", "timestamp": 1628750706093, "user_tz": -330, "elapsed": 715, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4701ba6b-ff2c-482e-a663-7bf222f191c4"
corr = df[['income', 'gender','event']].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs(),
    figsize=(4,4),
    fig_name='./extras/images/heatmap-event.png'
)
```

<!-- #region id="c5OpVnuW9Nxh" -->
## Building Recommendation matrix
<!-- #endregion -->

<!-- #region id="OyKg2V5v9Nxh" -->
At the moment data for each user has entries for each offer if it was received, viewed and responded to it.
To be able to give valid recommendations we leave only last user action on each offer (either viewed, responded or ignored).
<!-- #endregion -->

```python id="bNwVzZSl9Nxi" outputId="130e6558-8feb-4e4f-d40b-e6261b8e1c01"
df[(df.id == '68be06ca386d4c31939f3a4f0e3dd783') & (df.offer_id == '2906b810c7d4411798c6938adc9daaa5')]
```

```python id="Urpob3n19Nxi" colab={"base_uri": "https://localhost:8080/", "height": 69} executionInfo={"status": "ok", "timestamp": 1628751137489, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d538ba77-f7db-4b79-e032-444bff2b38db"
users = df['id'].unique()
offers = df['offer_id'].unique()
recommendation_df = pd.DataFrame(columns=df.columns)

recommendation_df.head()
```

```python id="iuX9uveY9Nxj" outputId="163622b4-3ba5-4447-c5c5-79ff00039454"
print("Number of known users: ", len(users))
print("Number of created offers: ", len(offers))
```

```python colab={"base_uri": "https://localhost:8080/"} id="QXB5L_QNIsZL" executionInfo={"status": "ok", "timestamp": 1628763250292, "user_tz": -330, "elapsed": 4711884, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45f14a9d-1013-44fa-934d-673cece65ad6"
for i, offer in enumerate(offers):
    for j, user in enumerate(users):
        offer_id_actions = df[(df.id == user) & (df.offer_id == offer)]
        # log progress 
        if j % 5000 == 0:
            print('Processing offer %s for user with index: %s' % (i, j))        
        if len(offer_id_actions) > 1:
            # user viewed or resonded to offer
            if offer_id_actions[offer_id_actions.event == 2]['event'].empty == False:
                # user has not completed an offer
                recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 2])
            elif offer_id_actions[offer_id_actions.event == 1]['event'].empty == False:
                # user only viewed offer
                recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 1])
            else:
                # Offer could be de received multiple times but ignored
                #print("Filter length", len())
                #print("No event were found in filtered data\n:", offer_id_actions)
                recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 0])
        else:
            # offer has been ignored
            recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="J3Hi7DojIvIN" executionInfo={"status": "ok", "timestamp": 1628763430408, "user_tz": -330, "elapsed": 599, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18bd75c9-ce34-4b7d-b7a5-f99f061a933a"
recommendation_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 444} id="iRyA8VysIyHe" executionInfo={"status": "ok", "timestamp": 1628763436289, "user_tz": -330, "elapsed": 1098, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bfff32bc-0611-44a0-fe3a-66c88140f0f3"
recommendation_df['event'][10000:50000].map({0:'offer received', 1: 'offer viewed', 2: 'offer completed'}).value_counts().plot.pie(figsize=(7, 7), 
                                       title="Event Pie Chart", 
                                       autopct='%1.1f%%', 
                                       legend=True)
```

```python id="pz-KQCZC9Nxl" colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"status": "ok", "timestamp": 1628750948335, "user_tz": -330, "elapsed": 593, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ef8e0ecd-687f-4c2c-add4-abb31a9ed39b"
gr = df.groupby(['id','offer_id'])
user_actions = pd.concat([gr.tail(1)]).reset_index(drop=True)
user_actions.head()
```

```python id="52KbNFvc9Nxn" colab={"base_uri": "https://localhost:8080/", "height": 162} executionInfo={"status": "ok", "timestamp": 1628750962887, "user_tz": -330, "elapsed": 816, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea8784a6-01c1-4190-d006-59cf9bab73e7"
user_actions[user_actions.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6']
```

```python id="eASXne249Nxo" colab={"base_uri": "https://localhost:8080/", "height": 444} executionInfo={"status": "ok", "timestamp": 1628750994232, "user_tz": -330, "elapsed": 2210, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f776e81c-efa9-4cf9-d895-85e989c92b53"
user_actions['event'][0:1000].map({0:'offer received', 1: 'offer viewed', 2: 'offer completed'}).value_counts().plot.pie(figsize=(7, 7), 
                                       title="Event Pie Chart", 
                                       autopct='%1.1f%%', 
                                       legend=True)
```

<!-- #region id="pvhE5lV29Nxo" -->
Final users/offers datasets look pretty good, however we still not able to extract some actions perfomed by users, especially with filtering duplicates. This might be caused by the fact when offer was received twice.

Let's filter them and explore once more.
<!-- #endregion -->

```python id="vMnOMMEB9Nxp" colab={"base_uri": "https://localhost:8080/", "height": 162} executionInfo={"status": "ok", "timestamp": 1628751080643, "user_tz": -330, "elapsed": 3336, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="39e5e26d-8270-4e3b-8e34-0d260b98b7f7"
user_actions.drop_duplicates(subset=['id', 'offer_id'], keep=False)

user_actions[user_actions.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6' ]
```

```python id="bBpeOHGX9Nxq" colab={"base_uri": "https://localhost:8080/", "height": 444} executionInfo={"status": "ok", "timestamp": 1628751083812, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c5a4b210-2e5c-4284-f4b6-732d59c7ef23"
user_actions['event'][0:1000].map({0:'offer received', 1: 'offer viewed', 2: 'offer completed'}).value_counts().plot.pie(figsize=(7, 7), 
                                       title="Event Pie Chart", 
                                       autopct='%1.1f%%', 
                                       legend=True)
```

<!-- #region id="dz5_Z8CN9Nxr" -->
Now the matrices look pretty similar and we are ready to build the Recommendation Engine.
<!-- #endregion -->

```python id="cExtk2qp9Nxr" executionInfo={"status": "ok", "timestamp": 1628763482452, "user_tz": -330, "elapsed": 640, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recommendation_df.to_csv('./data/silver/userdata.csv', index=False)
```

```python id="fjSwoCXb9Nxr" executionInfo={"status": "ok", "timestamp": 1628763483455, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
user_actions.to_csv('./data/silver/useractions.csv', index=False)
```

<!-- #region id="zspqs-Pt9NxQ" -->
If we look closely how event outcome is related to gender or income we can notice that correlation is quite weak, so other additional parameters should be definitely be taken into account.
<!-- #endregion -->
