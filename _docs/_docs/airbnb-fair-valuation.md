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

<!-- #region id="3MWwGITBcbQo" -->
# AirBnB Fair Valuation
<!-- #endregion -->

<!-- #region id="0aELHnRaw5E_" -->
Welcome to Airbnb Analysis Corp.! Your task is to set the competitive **daily accomodation rate** for a client's house in Bondi Beach. The owner currently charges $500. We have been tasked to estimate a **fair value** that the owner should be charging. The house has the following characteristics and constraints. While developing this model you came to realise that Airbnb can use your model to estimate the fair value of any property on their database, your are effectively creating a recommendation model for all prospective hosts!


1. The owner has been a host since **August 2010**
1. The location is **lon:151.274506, lat:33.889087**
1. The current review score rating **95.0**
1. Number of reviews **53**
1. Minimum nights **4**
1. The house can accomodate **10** people.
1. The owner currently charges a cleaning fee of **370**
1. The house has **3 bathrooms, 5 bedrooms, 7 beds**.
1. The house is available for **255 of the next 365 days**
1. The client is **verified**, and they are a **superhost**.
1. The cancelation policy is **strict with a 14 days grace period**.
1. The host requires a security deposit of **$1,500**


*All values strictly apply to the month of July 2018*
<!-- #endregion -->

```python id="aKJYHFgSw5FB"
from dateutil import parser
dict_client = {}

dict_client["city"] = "Bondi Beach"
dict_client["longitude"] = 151.274506
dict_client["latitude"] = -33.889087
dict_client["review_scores_rating"] = 95
dict_client["number_of_reviews"] = 53
dict_client["minimum_nights"] = 4
dict_client["accommodates"] = 10
dict_client["bathrooms"] = 3
dict_client["bedrooms"] = 5
dict_client["beds"] = 7
dict_client["security_deposit"] = 1500
dict_client["cleaning_fee"] = 370
dict_client["property_type"] = "House"
dict_client["room_type"] = "Entire home/apt"
dict_client["availability_365"] = 255
dict_client["host_identity_verified"] = 1  ## 1 for yes, 0 for no
dict_client["host_is_superhost"] = 1
dict_client["cancellation_policy"] = "strict_14_with_grace_period"
dict_client["host_since"] = parser.parse("01-08-2010")


```

<!-- #region id="0NqmpQhkw5FL" -->
## Setup
<!-- #endregion -->

<!-- #region id="h8BPo5jdw5FN" -->
First, let's make sure this notebook works well in both python 2 and 3, import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures:
<!-- #endregion -->

```python id="CZK0NbLRw5FQ"
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    try:
        plt.savefig(path, format=fig_extension, dpi=resolution)
    except:
        plt.savefig(fig_id + "." + fig_extension, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
pd.options.display.max_columns = None
```

<!-- #region id="8R-FkkOFw5FY" -->
## Get the data
<!-- #endregion -->

```python id="JY4PmIA-w5Fa" outputId="1fef65d8-7e83-471f-a4af-5b63b2c387c1" executionInfo={"status": "ok", "timestamp": 1563918102375, "user_tz": -60, "elapsed": 7183, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 86}
## This is simply a bit of importing logic that you don't have ..
## .. to concern yourself with for now. 

from pathlib import Path

github_p = "https://raw.githubusercontent.com/Finance-781/FinML/master/Lecture%202%20-%20End-to-End%20ML%20Project%20/Practice/"

my_file = Path("datasets/sydney_airbnb.csv") # Defines path
if my_file.is_file():              # See if file exists
    print("Local file found")      
    df = pd.read_csv('datasets/sydney_airbnb.csv')
else:
    print("Be patient: loading from github (2 minutes)")
    df = pd.read_csv(github_p+'datasets/sydney_airbnb.csv')
    print("Done")
```

```python id="Y8GZ8Mv4w5Fy"
### To make this project easier, I will select only a small number of features
```

```python id="nzXpxiqDw5F7"
incl = ["price","city","longitude","latitude","review_scores_rating","number_of_reviews","minimum_nights","security_deposit","cleaning_fee","accommodates","bathrooms","bedrooms","beds","property_type","room_type","availability_365" ,"host_identity_verified", "host_is_superhost","host_since","cancellation_policy"] 
```

```python id="I0oHakekw5GB"
df = df[incl]
```

<!-- #region id="5MFo1B34w5GN" -->
Lets reformat the price to floats, it is currently a string (object). And lets makes sure the date is in a datetime format.
<!-- #endregion -->

```python id="9KajhaZew5GO"
import re
price_list = ["price","cleaning_fee","security_deposit"]

for col in price_list:
    df[col] = df[col].fillna("0")
    df[col] = df[col].apply(lambda x: float(re.compile('[^0-9eE.]').sub('', x)) if len(x)>0 else 0)

df['host_since'] = pd.to_datetime(df['host_since'])
```

```python id="EDYefLGIw5Gc" outputId="cf9a8904-f76e-4140-a29c-afa117f39b99" executionInfo={"status": "ok", "timestamp": 1563918105805, "user_tz": -60, "elapsed": 10489, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 121}
df["price"].head()
```

```python id="D56JFFrVw5Gt" outputId="604c73a0-8cce-437f-9f93-2bfb11bf670f" executionInfo={"status": "ok", "timestamp": 1563918106202, "user_tz": -60, "elapsed": 10763, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 289}
## Winsorize for high price values, outliers.

df.boxplot(column="price")
```

```python id="z7JM4sHqw5G1" outputId="205dbe56-ee8b-4f2c-9216-fad11399ea89" executionInfo={"status": "ok", "timestamp": 1563918106207, "user_tz": -60, "elapsed": 10652, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
## this is high, because we have a price we expect it to be high.
## however, it shouldn't be much above 3. 
df["price"].skew()
```

```python id="AHr5vYEZw5G-" outputId="bca074b2-2737-4aba-a03c-491e899e435c" executionInfo={"status": "ok", "timestamp": 1563918106211, "user_tz": -60, "elapsed": 10537, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
## This value is still relatively high
df["price"].quantile(0.995) ## @99.5% 
```

```python id="5YfvE8BTw5HI"
df = df[df["price"]<df["price"].quantile(0.995)].reset_index(drop=True)
```

```python id="3ORYKN6Yw5HO" outputId="a877b942-49ea-437f-e0bf-8954df94f1f7" executionInfo={"status": "ok", "timestamp": 1563918106228, "user_tz": -60, "elapsed": 10487, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
## This would do for now, it might also be worth transforming ..
## .. the price with a log function at a later stage
df["price"].skew()
```

```python id="mjQ3evsZw5HY" outputId="1a503a26-fa57-48f3-c5d4-34b769285d8e" executionInfo={"status": "ok", "timestamp": 1563918106721, "user_tz": -60, "elapsed": 10891, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 381}
df.isnull().sum()
```

```python id="fCl6gg05w5Hh" outputId="a3c0b472-17f9-4916-ae8d-d9c21612ca63" executionInfo={"status": "ok", "timestamp": 1563918106733, "user_tz": -60, "elapsed": 10815, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 450}
df.info()
```

```python id="H4Uw7ou7w5Hz" outputId="66f2f279-30bf-4c70-a85b-57228a5348ca" executionInfo={"status": "ok", "timestamp": 1563918106736, "user_tz": -60, "elapsed": 10750, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 121}
df["cancellation_policy"].value_counts()
```

```python id="Z2B1YEXrw5H7" outputId="a2b5d58b-bc62-471a-de4b-59cad5792857" executionInfo={"status": "ok", "timestamp": 1563918106738, "user_tz": -60, "elapsed": 10600, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 294}
df.describe()
```

```python id="ysvjJRt1w5II" outputId="5e379b1e-56a1-4084-9f0d-2015e1c33817" executionInfo={"status": "ok", "timestamp": 1563918114964, "user_tz": -60, "elapsed": 18738, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 1000}
%matplotlib inline
import matplotlib.pyplot as plt

try:
    df.iloc[:,6:].hist(bins=50, figsize=(20,15))
    save_fig("attribute_histogram_plots")
    plt.show()
except AttributeError:
    pass

```

```python id="XsCwHu6Aw5IU" outputId="c724d93e-02ab-43d7-b5cb-9f03f0e6e7b7" executionInfo={"status": "ok", "timestamp": 1563918114967, "user_tz": -60, "elapsed": 18658, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 208}
## Even though our customer, sepecifcally wants information about..
## .. Bondi the addition of other areas will help the final prediction

df["city"].value_counts().head(10)
```

```python id="4wL-UHLfw5Ij"
## For this taks we will keep the top 20 Sydney locations

list_of_20 = list(df["city"].value_counts().head(10).index)
df = df[df["city"].isin(list_of_20)].reset_index(drop=True)
```

```python id="Onn2CeW5w5It" outputId="22f89ba0-cb1e-43cf-d8d1-c92964e26572" executionInfo={"status": "ok", "timestamp": 1563918114984, "user_tz": -60, "elapsed": 18573, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 485}
df["property_type"].value_counts()
```

```python id="pEA9rG38w5I6"
## Remove rare occurences in categories as is necessary for..
## .. the eventaul cross validation step, the below step is somewhat ..
## .. similar for what has been done with cities above

item_counts = df.groupby(['property_type']).size()
rare_items = list(item_counts.loc[item_counts <= 10].index.values)
```

```python id="jAfIGcpZw5JC"
df = df[~df["property_type"].isin(rare_items)].reset_index(drop=True)
```

```python id="sDhH3Dhdw5JG"
# to make this notebook's output identical at every run
np.random.seed(42)
```

```python id="xvy15Atdw5JL"
import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

```python id="ZOXU5zTnw5JX" outputId="8db72564-b9a6-4a03-d660-129f23d0caa6" executionInfo={"status": "ok", "timestamp": 1563918115042, "user_tz": -60, "elapsed": 18565, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
train_set, test_set = split_train_test(df, 0.2)
print(len(train_set), "train +", len(test_set), "test")
```

```python id="B0QW3H50w5Jj"
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

<!-- #region id="g3ZejQi1w5Jt" -->
The implementation of `test_set_check()` above works fine in both Python 2 and Python 3. In earlier releases, the following implementation was proposed, which supported any hash function, but was much slower and did not support Python 2:
<!-- #endregion -->

```python id="m0L9cj0Vw5Jz"
import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
```

<!-- #region id="IQvysHypw5J4" -->
If you want an implementation that supports any hash function and is compatible with both Python 2 and Python 3, here is one:
<!-- #endregion -->

```python id="fwp7MKunw5J7"
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio
```

```python id="Dlmvvvc4w5KF"
df_with_id = df.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(df_with_id, 0.2, "index")
```

```python id="VRRntF4Mw5KJ"
df_with_id["id"] = df["longitude"] * 1000 + df_with_id["latitude"]
train_set, test_set = split_train_test_by_id(df_with_id, 0.2, "id")
```

```python id="0awTDRvsw5KP" outputId="f777f25c-5b44-4a0c-e935-6ecdce173f53" executionInfo={"status": "ok", "timestamp": 1563918116077, "user_tz": -60, "elapsed": 19529, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 309}
test_set.head()
```

```python id="1-wPuzs1w5Kd"
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
```

```python id="sc7D0atAw5Km" outputId="fd6aa4e5-f3a2-4794-cdda-290342398c19" executionInfo={"status": "ok", "timestamp": 1563918116554, "user_tz": -60, "elapsed": 19948, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 309}
test_set.head()
```

<!-- #region id="3lEV2jN6w5K0" -->
The models that would be used in this project can't read textual data, thus we have to turn text categories into numeric categories. The code below will create city codes, this time for the purpose of statified sampeing. 

<!-- #endregion -->

```python id="P6GhLgIZw5K-"
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in ["city"]:
    df[col+"_code"] = le.fit_transform(df[col])

```

```python id="UW7POEbXw5LG"
##  Similar to above encoding, here we encode binary 1, 0 for t and f. 

df["host_identity_verified"] = df["host_identity_verified"].apply(lambda x: 1 if x=="t" else 0)
df["host_is_superhost"] = df["host_is_superhost"].apply(lambda x: 1 if x=="t" else 0)

```

```python id="mwIMKxThw5LM"
from sklearn.model_selection import StratifiedShuffleSplit

## we will stratify according to city

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["city_code"]):
    del df["city_code"]
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
```

```python id="fXrt7-6zw5LV" outputId="22c2f6e9-6fc8-4550-925e-902ac0123b0e" executionInfo={"status": "ok", "timestamp": 1563918116612, "user_tz": -60, "elapsed": 19944, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 225}
## Average price per area
strat_test_set.groupby("city")["price"].mean()
```

<!-- #region id="a-Rbqr4Lw5Ll" -->
## Discover and visualize the data to gain insights
<!-- #endregion -->

```python id="e-PE8Bmuw5Ln"
traval = strat_train_set.copy() ##traval - training and validation set
```

```python id="_SN0cbvQw5L5" outputId="eb5cfdd6-b915-4a82-b245-38f264da5e44" executionInfo={"status": "ok", "timestamp": 1563918117025, "user_tz": -60, "elapsed": 20316, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 314}
traval.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
```

```python id="1CPPYjTqw5MJ" outputId="a7d432cd-d7d2-4955-f007-fde9f18a984a" executionInfo={"status": "ok", "timestamp": 1563918117914, "user_tz": -60, "elapsed": 21185, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 314}
traval.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
```

<!-- #region id="GosBBdOdw5MT" -->
The argument `sharex=False` fixes a display bug (the x-axis values and legend were not displayed). This is a temporary fix (see: https://github.com/pandas-dev/pandas/issues/10611). Thanks to Wilmer Arellano for pointing it out.
<!-- #endregion -->

```python id="Hd4OBhY2w5Mc"
traval_co = traval[(traval["longitude"]>151.16)&(traval["latitude"]<-33.75)].reset_index(drop=True)

traval_co = traval_co[traval_co["latitude"]>-33.95].reset_index(drop=True)

traval_co = traval_co[traval_co["price"]<600].reset_index(drop=True)
```

```python id="E2r10b4ew5Mk" outputId="837c2fba-6643-40d6-ec2d-5ac911b07754" executionInfo={"status": "ok", "timestamp": 1563918121282, "user_tz": -60, "elapsed": 24530, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 530}
traval_co.plot(kind="scatter", x="longitude", y="latitude", alpha=0.5,
    s=traval_co["number_of_reviews"]/2, label="Reviews", figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
```

```python id="wpYvVA5Sw5Mt"
corr_matrix = traval.corr()
```

```python id="HHtMTAziw5My" outputId="2bdf7905-d92a-4bfc-f44b-5396cdda8ab5" executionInfo={"status": "ok", "timestamp": 1563918121307, "user_tz": -60, "elapsed": 24520, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 294}
corr_matrix["price"].sort_values(ascending=False)
```

```python id="v3Po_K3Lw5M7" outputId="a6b36635-7ed1-4d21-f2c8-a2d75b791a84" executionInfo={"status": "ok", "timestamp": 1563918130328, "user_tz": -60, "elapsed": 33514, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 602}
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["price", "accommodates", "bedrooms",
              "cleaning_fee","review_scores_rating"]
scatter_matrix(traval[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
```

```python id="YtjGL1F_w5NR" outputId="9df07af4-f89a-4ec9-b3bf-824db8513282" executionInfo={"status": "ok", "timestamp": 1563918131618, "user_tz": -60, "elapsed": 34759, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 314}
traval.plot(kind="scatter", x="accommodates", y="price",
             alpha=0.1)
save_fig("income_vs_house_value_scatterplot")
```

```python id="sOCF-KD_w5Nh" outputId="c629b1e3-be8a-47c5-80be-649b3ef2ee89" executionInfo={"status": "ok", "timestamp": 1563918131626, "user_tz": -60, "elapsed": 34747, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 309}
traval.head()
```

```python id="vdAainPuw5Nr"
#### Some Feature Engineering
```

```python id="4oF78gHzw5Nw"
traval["bedrooms_per_person"] = traval["bedrooms"]/traval["accommodates"]
traval["bathrooms_per_person"] = traval["bathrooms"]/traval["accommodates"]
traval['host_since'] = pd.to_datetime(traval['host_since'])
traval['days_on_airbnb'] = (pd.to_datetime('today') - traval['host_since']).dt.days
```

<!-- #region id="3cXZ9PZWw5N1" -->
## Prepare the data for Machine Learning algorithms
<!-- #endregion -->

```python id="7Dl2cqXBw5N4"
## Here I will forget about traval and use a more formal way of introducing...
## ..preprocessin using pipelines
```

```python id="gFyYERMgw5N9"
X  = traval.copy().drop("price", axis=1) # drop labels for training set
```

```python id="7etxdWRJw5OC" outputId="ff763fac-8530-407e-f367-8dda01876641" executionInfo={"status": "ok", "timestamp": 1563918131683, "user_tz": -60, "elapsed": 34768, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 326}
sample_incomplete_rows = X[X.isnull().any(axis=1)].head()
print(sample_incomplete_rows.shape)
sample_incomplete_rows
```

```python id="GtCPLD4Ew5OI" outputId="c5b695b4-6949-4864-a6c2-1b76c728b57b" executionInfo={"status": "ok", "timestamp": 1563918131686, "user_tz": -60, "elapsed": 34750, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 69}
# Rows Remove
sample_incomplete_rows.dropna(subset=["review_scores_rating"])    # option 1
```

```python id="ka6RIgWlw5OU" outputId="fd0d035e-9dc4-4bd1-f807-a314b3229fd5" executionInfo={"status": "ok", "timestamp": 1563918131689, "user_tz": -60, "elapsed": 34729, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 309}
# Columns Remove
sample_incomplete_rows.drop(["review_scores_rating"], axis=1)       # option 2
```

```python id="wJ4II0G-w5Oe" outputId="6567e4e7-e3db-474d-e932-a20aa826e6c7" executionInfo={"status": "ok", "timestamp": 1563918133109, "user_tz": -60, "elapsed": 36129, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 309}
median = X["review_scores_rating"].median()
sample_incomplete_rows["review_scores_rating"].fillna(median, inplace=True) # option 3

sample_incomplete_rows
```

```python id="5W1NjzUfw5Ok" outputId="b0bcba20-3199-4122-cb80-442a21805a83" executionInfo={"status": "ok", "timestamp": 1563918133114, "user_tz": -60, "elapsed": 36131, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 72}
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
```

<!-- #region id="GpQe6f4Kw5Os" -->
Remove the text attribute because median can only be calculated on numerical attributes:
<!-- #endregion -->

```python id="2s_PW2I3w5Ov"
cat_cols = ["city","cancellation_policy","host_since","room_type","property_type","host_since"]
X_num = X.drop(cat_cols, axis=1)
# alternatively: X_num = X.select_dtypes(include=[int, float])
```

```python id="Yf3va5snw5PD" outputId="16bca0ec-452a-4f76-8956-5ad127d6371f" executionInfo={"status": "ok", "timestamp": 1563918133155, "user_tz": -60, "elapsed": 36147, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
imputer.fit(X_num)
```

```python id="Z7BvFuaVw5PJ" outputId="2fbd1eda-96ea-4233-ef62-1236006ab5ca" executionInfo={"status": "ok", "timestamp": 1563918133161, "user_tz": -60, "elapsed": 36127, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 104}
imputer.statistics_
```

<!-- #region id="hzxuikicw5PW" -->
Check that this is the same as manually computing the median of each attribute:
<!-- #endregion -->

```python id="B5do-SLFw5PX" outputId="a28b556d-a0aa-425d-c276-73c72b0021cf" executionInfo={"status": "ok", "timestamp": 1563918133165, "user_tz": -60, "elapsed": 36107, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 104}
X_num.median().values
```

<!-- #region id="jhqi5OU9w5Pe" -->
Transform the training set:
<!-- #endregion -->

```python id="Lo700DUsw5Pg"
X_num_np = imputer.transform(X_num)
```

```python id="lIIwDuqpw5Pl"
X_num = pd.DataFrame(X_num_np, columns=X_num.columns,
                          index = list(X_num.index.values))
```

```python id="GZw1eaSlw5Pp" outputId="6b6877c7-a1fd-4cfc-d042-f3de64720612" executionInfo={"status": "ok", "timestamp": 1563918133197, "user_tz": -60, "elapsed": 36113, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 222}
X_num.loc[sample_incomplete_rows.index.values]
```

```python id="AdmidxW3w5P5" outputId="23e3c98f-995b-4a5a-dfcc-447d941465ea" executionInfo={"status": "ok", "timestamp": 1563918133203, "user_tz": -60, "elapsed": 36095, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
imputer.strategy
```

<!-- #region id="ZzxQqz0iw5QD" -->
Now let's preprocess the categorical input feature, `ocean_proximity`:
<!-- #endregion -->

```python id="ZwC3xBx2w5QE" outputId="05d1a505-75c9-4492-a658-eaa469949afe" executionInfo={"status": "ok", "timestamp": 1563918133207, "user_tz": -60, "elapsed": 36078, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 356}
X_cat = X.select_dtypes(include=[object])
X_cat.head(10)
```

```python id="s9bBi-cIw5QJ"
from sklearn.preprocessing import OrdinalEncoder
```

```python id="1k2uHTfww5QM" outputId="b65cae53-4248-4372-9412-3d40abba76d7" executionInfo={"status": "ok", "timestamp": 1563918133222, "user_tz": -60, "elapsed": 36071, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 202}
X_cat.head()
```

```python id="zNdyufW6w5QX" outputId="69ca0ffd-34a4-4491-f94d-96e2b8d1f402" executionInfo={"status": "ok", "timestamp": 1563918133231, "user_tz": -60, "elapsed": 36060, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 190}
ordinal_encoder = OrdinalEncoder()
X_cat_enc = ordinal_encoder.fit_transform(X_cat)
X_cat_enc[:10]
```

```python id="b47WGxuuw5Qh" outputId="36bf43e6-8828-4551-a685-5fa61d29ada7" executionInfo={"status": "ok", "timestamp": 1563918133235, "user_tz": -60, "elapsed": 36043, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 173}
ordinal_encoder.categories_
```

```python id="cG-V0NPaw5Qq" outputId="03979837-12f6-4147-904d-1302495047fe" executionInfo={"status": "ok", "timestamp": 1563918133243, "user_tz": -60, "elapsed": 36031, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 52}
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
X_cat_1hot = cat_encoder.fit_transform(X_cat)
X_cat_1hot
```

<!-- #region id="VSbE62pNw5Q8" -->
By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if needed by calling the `toarray()` method:
<!-- #endregion -->

```python id="wse7kfOJw5Q_" outputId="89ec3267-b4d7-4fbf-e460-8720e9762cb1" executionInfo={"status": "ok", "timestamp": 1563918133247, "user_tz": -60, "elapsed": 36014, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 138}
X_cat_1hot.toarray()
```

<!-- #region id="Czb3Sc58w5RP" -->
Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:
<!-- #endregion -->

```python id="MPvz38hvw5RQ" outputId="8b36f967-778d-4b03-f384-0077281aaa86" executionInfo={"status": "ok", "timestamp": 1563918133253, "user_tz": -60, "elapsed": 35996, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 138}
cat_encoder = OneHotEncoder(sparse=False)
X_cat_1hot = cat_encoder.fit_transform(X_cat)
X_cat_1hot
```

```python id="_RN9A0Cbw5RY" outputId="f40ad8f8-439b-4d7d-deab-5bc4f7604020" executionInfo={"status": "ok", "timestamp": 1563918133257, "user_tz": -60, "elapsed": 35975, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 173}
cat_encoder.categories_
```

<!-- #region id="E92-ZqWiw5Rn" -->
Let's create a custom transformer to add extra attributes:
<!-- #endregion -->

<!-- #region id="LDHO17Jrw5Ro" -->
**Now let's create a pipeline for preprocessing that is built on the techniques we used up and till now and introduce some new pipeline techniques.**
<!-- #endregion -->

```python id="BTVFcERtw5Rp"
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Receive numpy array, convert to pandas for features, convert back to array for output.

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, popularity = True, num_cols=[]): # no *args or **kargs
        self.popularity = popularity
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        
        ### Some feature engineering
        X = pd.DataFrame(X, columns=num_cols)
        X["bedrooms_per_person"] = X["bedrooms"]/X["accommodates"]
        X["bathrooms_per_person"] = X["bathrooms"]/X["accommodates"]
        
        global feats
        feats = ["bedrooms_per_person","bathrooms_per_person"]

        if self.popularity:
            X["past_and_future_popularity"]=X["number_of_reviews"]/(X["availability_365"]+1)
            feats.append("past_and_future_popularity")
            
            return X.values
        else:
            return X.values
        
```

```python id="7Z0PQD7jw5Rx" outputId="9a814272-1b69-4041-d5d6-dcb8ed683618" executionInfo={"status": "ok", "timestamp": 1563918133278, "user_tz": -60, "elapsed": 35990, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 72}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = strat_train_set.copy().drop("price",axis=1)
Y = strat_train_set["price"]

num_cols = list(X.select_dtypes(include=numerics).columns)
cat_cols = list(X.select_dtypes(include=[object]).columns)

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(num_cols=num_cols,popularity=True)),
        ('std_scaler', StandardScaler()),
    ])
```

<!-- #region id="3u34Imrxw5R5" -->
**Warning**: earlier versions of the book applied different transformations to different columns using a solution based on a `DataFrameSelector` transformer and a `FeatureUnion` (see below). It is now preferable to use the `ColumnTransformer` class that will is introduced in Scikit-Learn 0.20. 
<!-- #endregion -->

```python id="1ssxaRz7w5R6"
from sklearn.compose import ColumnTransformer
import itertools


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

mid_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", OneHotEncoder(),cat_cols ),
    ])
```

```python id="pQwKODGow5R9" outputId="ec30bd1b-0395-410f-cdb9-0befd71a8512" executionInfo={"status": "ok", "timestamp": 1563918135741, "user_tz": -60, "elapsed": 38448, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 106}
mid_pipeline.fit(X) # this one specifically has to be fitted for the cat names
cat_encoder = mid_pipeline.named_transformers_["cat"]
sublists = [list(bas) for bas in cat_encoder.categories_]
one_cols = list(itertools.chain(*sublists))

## In this class, I will be converting numpy back to pandas

class ToPandasDF(BaseEstimator, TransformerMixin):
    def __init__(self, fit_index = [] ): # no *args or **kargs
        self.fit_index = fit_index
    def fit(self, X_df, y=None):
        return self  # nothing else to do
    def transform(self, X_df, y=None):
        global cols
        cols = num_cols.copy()
        cols.extend(feats)
        cols.extend(one_cols) # one in place of cat
        X_df = pd.DataFrame(X_df, columns=cols,index=self.fit_index)

        return X_df

def pipe(inds):
    return Pipeline([
            ("mid", mid_pipeline),
            ("PD", ToPandasDF(inds)),
        ])
    
params = {"inds" : list(X.index)}

X_pr = pipe(**params).fit_transform(X) # Now we have done all the preprocessing instead of
                                   #.. doing it bit by bit. The pipeline becomes 
                                   #.. extremely handy in the cross-validation step.
```

<!-- #region id="jjNyCAI4w5SF" -->
## Select and train a model 
<!-- #endregion -->

```python id="bexu5Jxew5SG" outputId="42b92071-e6c3-472e-ad72-08d94291f086" executionInfo={"status": "ok", "timestamp": 1563918135751, "user_tz": -60, "elapsed": 38438, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
from sklearn.linear_model import LinearRegression
Y_pr = Y.copy() # just for naming convention, _pr for processed.

lin_reg = LinearRegression()
lin_reg.fit(X_pr, Y_pr)
```

```python id="nDUC_rAIw5SK" outputId="51eb6eb8-c295-4343-88e2-40c6594d86ca" executionInfo={"status": "ok", "timestamp": 1563918135759, "user_tz": -60, "elapsed": 38414, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
# let's try the full preprocessing pipeline on a few training instances
some_data = X.iloc[:5]
some_labels = Y.iloc[:5]
some_data_prepared = pipe(inds=list(some_data.index)).transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
```

<!-- #region id="c5A2Oz8yw5SQ" -->
Compare against the actual values:
<!-- #endregion -->

```python id="frLz_Tjxw5SR" outputId="7bb4346e-3b4c-49b1-f4b6-be65f7604691" executionInfo={"status": "ok", "timestamp": 1563918135762, "user_tz": -60, "elapsed": 38386, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
print("Labels:", list(some_labels))
```

```python id="F7kueKyIw5SY"
## Naturally, these metrics are not that fair, because it is insample.
## However the first model is linear so overfitting is less likley.
## We will deal with out of sample validation later on. 
```

```python id="ZibYqcsDw5Sb" outputId="6c01e616-fb3c-4d80-ff44-a6064d186968" executionInfo={"status": "ok", "timestamp": 1563918135783, "user_tz": -60, "elapsed": 38371, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_pred = lin_reg.predict(X_pr)
lin_mse = mean_squared_error(Y, X_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```

```python id="9CnEnTcnw5Sg" outputId="a8fdfb4b-7251-4c1a-a730-77e186462e0d" executionInfo={"status": "ok", "timestamp": 1563918135786, "user_tz": -60, "elapsed": 38354, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(Y, X_pred)
lin_mae
```

```python id="7DBl7-UGw5So" outputId="549ab15d-0bce-4c3a-9e89-105913266ae9" executionInfo={"status": "ok", "timestamp": 1563918135788, "user_tz": -60, "elapsed": 38331, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 104}
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_pr, Y)
```

```python id="-ls5NvMCw5Su" outputId="6e7dd468-c448-4b37-ff4a-add8d4f1e48f" executionInfo={"status": "ok", "timestamp": 1563918135796, "user_tz": -60, "elapsed": 38318, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
X_pred = tree_reg.predict(X_pr)
tree_mse = mean_squared_error(Y, X_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse  ## Model is complex and overfits.
```

<!-- #region id="H4E18Mu_w5S0" -->
## Fine-tune your model
<!-- #endregion -->

```python id="iYD-lPbNw5S1"
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeRegressor(random_state=42), X_pr, Y,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```

```python id="mY4F8kbow5S5" outputId="06fcc168-2b4a-4c73-a94d-e775022501af" executionInfo={"status": "ok", "timestamp": 1563918135831, "user_tz": -60, "elapsed": 38317, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 86}
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
```

```python id="Yr_AG4xDw5S-" outputId="ab98f70b-6545-40a1-9685-ab266b4eb0db" executionInfo={"status": "ok", "timestamp": 1563918136986, "user_tz": -60, "elapsed": 39421, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 86}
lin_scores = cross_val_score(LinearRegression(), X_pr, Y,
                             scoring="neg_mean_absolute_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
## bad performance, might need some regularisation.
```

```python id="49Khr1V3w5TE" outputId="23f05a47-e335-4965-bff6-2861067e0726" executionInfo={"status": "ok", "timestamp": 1563918136989, "user_tz": -60, "elapsed": 39378, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 173}
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_pr, Y)
```

```python id="TZRGtlqKw5TM" outputId="78fa7f0a-f48c-4cc6-dc96-908bc80b217f" executionInfo={"status": "ok", "timestamp": 1563918136992, "user_tz": -60, "elapsed": 39340, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
X_pred = forest_reg.predict(X_pr)
forest_mse = mean_squared_error(Y, X_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```

```python id="0-zM3FnYw5TQ" outputId="29e65b19-e2d1-44c8-e36b-7b9d69acb298" executionInfo={"status": "ok", "timestamp": 1563918139637, "user_tz": -60, "elapsed": 41936, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 86}
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, X_pr, Y,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

```python id="iInfXGrxw5Ta" outputId="80f73852-3385-4dea-d939-abf6ba7f1847" executionInfo={"status": "ok", "timestamp": 1563918139642, "user_tz": -60, "elapsed": 41893, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 173}
scores = cross_val_score(lin_reg,  X_pr, Y, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()
```

```python id="bO6AkZkfw5Th" outputId="91061449-4fe2-47f3-d6b9-93668bd138fa" executionInfo={"status": "ok", "timestamp": 1563918150861, "user_tz": -60, "elapsed": 53071, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit( X_pr, Y,)
X_pred = svm_reg.predict(X_pr)
svm_mse = mean_squared_error(Y, X_pred)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
```

```python id="ibHscUwGw5Ts" outputId="c682b5ee-1293-4edc-98a9-63cbb57b062b" executionInfo={"status": "ok", "timestamp": 1563918162156, "user_tz": -60, "elapsed": 64310, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 364}
## 30 Seconds to run this code block.
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit( X_pr, Y)
```

<!-- #region id="qx78BNO3w5Tw" -->
The best hyperparameter combination found:
<!-- #endregion -->

```python id="7nNoAm6tw5Tx" outputId="57c2826e-bc0f-4022-f21a-de15184d2f93" executionInfo={"status": "ok", "timestamp": 1563918162158, "user_tz": -60, "elapsed": 64248, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
grid_search.best_params_
```

```python id="Mnv6MP-2w5T_" outputId="f7e25d6f-bab1-45b7-cae4-b84ea1367665" executionInfo={"status": "ok", "timestamp": 1563918162161, "user_tz": -60, "elapsed": 64199, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 138}
grid_search.best_estimator_
```

<!-- #region id="W9WoG5q7w5UE" -->
Let's look at the score of each hyperparameter combination tested during the grid search:
<!-- #endregion -->

```python id="fCTtGGOtw5UM" outputId="aba548f1-0001-45f7-f9ff-c95e42fdb5fe" executionInfo={"status": "ok", "timestamp": 1563918162167, "user_tz": -60, "elapsed": 64163, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 364}
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
print("")
print("Best grid-search performance: ", np.sqrt(-cvres["mean_test_score"].max()))
```

```python id="CWQL-Wqyw5US" outputId="e01e42b6-12ff-499b-aaf0-b3287c830987" executionInfo={"status": "ok", "timestamp": 1563918162172, "user_tz": -60, "elapsed": 64127, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 482}
# Top five results as presented in a dataframe
pd.DataFrame(grid_search.cv_results_).head(5)

```

```python id="heRyEXOKw5UX" outputId="e35901fc-ca86-422f-c54c-a37b34e85411" executionInfo={"status": "ok", "timestamp": 1563918179563, "user_tz": -60, "elapsed": 81472, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 381}
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit( X_pr, Y)
```

```python id="hq13w9Ebw5Ug" outputId="1e0776b3-7a78-4264-ec8a-e3ade35ad726" executionInfo={"status": "ok", "timestamp": 1563918179568, "user_tz": -60, "elapsed": 81433, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 121}
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
print("Best grid-search performance: ", np.sqrt(-cvres["mean_test_score"].max()))
```

```python id="D9LAn5JHw5Uu" outputId="cec2bc2c-26f8-4532-a79f-bfa60b89ac3d" executionInfo={"status": "ok", "timestamp": 1563918179572, "user_tz": -60, "elapsed": 81393, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 225}
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```

```python id="7v-RO30Pw5Uz"
feats = pd.DataFrame()
feats["Name"] = list(X_pr.columns)
feats["Score"] = feature_importances
```

```python id="Vcl5kjafw5U6" outputId="f95410ae-d55a-4c0a-b919-79a23c03dae6" executionInfo={"status": "ok", "timestamp": 1563918179595, "user_tz": -60, "elapsed": 81370, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 662}
feats.sort_values("Score",ascending=False).round(5).head(20)
```

```python id="iL5nHXrKw5VV" outputId="91afae38-5dd8-4d4e-bfab-c2a202c22e87" executionInfo={"status": "ok", "timestamp": 1563918180368, "user_tz": -60, "elapsed": 82094, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 309}
strat_test_set.head()
```

```python id="wFUjb6tjw5Vf"
### Now we can test the out of sample performance. 

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("price", axis=1)
y_test = strat_test_set["price"].copy()

X_test_prepared = pipe(list(X_test.index)).transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

```python id="TwCDU_rYw5Vi" outputId="8d85d46c-faf1-4a2f-c920-29a214f89156" executionInfo={"status": "ok", "timestamp": 1563918180398, "user_tz": -60, "elapsed": 82085, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
final_rmse
```

```python id="EgEMS8TXw5Vo"
final_mae = mean_absolute_error(y_test, final_predictions)
```

```python id="9224Lkj7w5Vw" outputId="fed9cd0b-7019-4732-ed81-22e70f9f8837" executionInfo={"status": "ok", "timestamp": 1563918180423, "user_tz": -60, "elapsed": 82067, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
final_mae ## not too bad
```

```python id="4s5JAt_Ww5V2"
## Value Estimation for Client
```

```python id="1vo1oYm3w5V_"
df_client = pd.DataFrame.from_dict(dict_client, orient='index').T
```

```python id="FtYlxBNxw5WE" outputId="6ed87b82-fe35-4630-eb83-ad93a649182e" executionInfo={"status": "ok", "timestamp": 1563918180482, "user_tz": -60, "elapsed": 82077, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 117}
df_client
```

```python id="F-4FydXgw5WN"
df_client = pipe(list(df_client.index)).transform(df_client)
```

```python id="ihA0mreKw5WU"
client_pred = final_model.predict(df_client)
```

```python id="KMQ5Uyr5w5WW" outputId="de3cb903-623f-485a-9141-41478b74616d" executionInfo={"status": "ok", "timestamp": 1563918180516, "user_tz": -60, "elapsed": 81877, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 69}
### Client should be charging about $150 more. 
print('\x1b[1;31m'+str(client_pred[0])+'\x1b[0m')
print('\x1b[1;31m'+str(-500)+'\x1b[0m')
print('\x1b[1;31m'+"= "+str(client_pred[0]-500)+'\x1b[0m')
```

<!-- #region id="rqGxrkjUw5Wi" -->
We can compute a 95% confidence interval for the test RMSE:
<!-- #endregion -->

```python id="LCTH5U14w5Wi"
from scipy import stats
```

```python id="Mt9k-G01w5Wy" outputId="0a11b32b-603d-4365-ce1e-02d0e4866d16" executionInfo={"status": "ok", "timestamp": 1563918184766, "user_tz": -60, "elapsed": 86089, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
y_test.min()
```

```python id="6bzva7ftw5W2" outputId="86f40a8e-8020-4cc5-9e44-11ba86f93097" executionInfo={"status": "ok", "timestamp": 1563918184774, "user_tz": -60, "elapsed": 86058, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
## This calculates the RMSE confidence interval

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

## MSE
MSE_int = np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))

print("MSE Interval: ", MSE_int)

```

<!-- #region id="k-mO0SYfw5W8" -->
We could also compute the interval manually like this:
<!-- #endregion -->

```python id="rzynXm29w5W9" outputId="1d391bce-c5af-43e9-e6bf-4d97e046a294" executionInfo={"status": "ok", "timestamp": 1563918184779, "user_tz": -60, "elapsed": 86022, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
```

<!-- #region id="Bk1bJESfw5XF" -->
Alternatively, we could use a z-scores rather than t-scores:
<!-- #endregion -->

```python id="5qpnpELow5XF" outputId="360ea6ec-1817-416d-9ec8-6a9bf0468b77" executionInfo={"status": "ok", "timestamp": 1563918184782, "user_tz": -60, "elapsed": 85985, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
```

```python id="YCn2oaFgw5XJ" outputId="4b7986b7-1771-4f5e-e442-43f20efa81e7" executionInfo={"status": "ok", "timestamp": 1563918184785, "user_tz": -60, "elapsed": 85951, "user": {"displayName": "Cloud Machine", "photoUrl": "", "userId": "03543453517966682716"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
####### What about for MAE 

absolute_errors = (final_predictions - y_test).abs()
mean = absolute_errors.mean()
m = len(absolute_errors)

MAE_int = stats.t.interval(confidence, m - 1,
                         loc=np.mean(absolute_errors),
                         scale=stats.sem(absolute_errors))

print("MAE Interval: ", MAE_int)

```

<!-- #region id="LB7D70eRyaeY" -->
Credit: [Derek Snow](https://www.linkedin.com/company/18004273/admin/)


<!-- #endregion -->
