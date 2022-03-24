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

<!-- #region id="NGmFu7BJBwmm" -->
# MovieLens ML-1m EDA
<!-- #endregion -->

<!-- #region id="GeGsB0MeplsR" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HdvoHy8Av6fI" executionInfo={"status": "ok", "timestamp": 1636009231215, "user_tz": -330, "elapsed": 2124, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ef52f1c-b61b-44a5-a936-49afdcd9d61a"
import os
project_name = "chef-recsys"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="WCsPYr26wfzt" executionInfo={"status": "ok", "timestamp": 1636009427960, "user_tz": -330, "elapsed": 7243, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="35c55c65-e315-4b14-d0fd-46113fc2c2f9"
!cd /content/main && git add . && git commit -m 'commit' && git push origin main
```

```python colab={"base_uri": "https://localhost:8080/"} id="zSHWK_Tmv85D" executionInfo={"status": "ok", "timestamp": 1636009242623, "user_tz": -330, "elapsed": 607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5c64789e-9a13-45c7-fad2-1962f976b692"
%cd /content
```

```python id="DSvEg5WkjK10"
!pip install tensorflow_recommenders
!pip install folium
```

```python id="TbOkzY7Ihxxp"
import os
import pprint
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Text
from wordcloud import WordCloud
import requests
import folium
from folium.plugins import MarkerCluster

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Text
import pandas as pd
import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

plt.style.use('ggplot')
```

```python id="eG-godXJxSsp"
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="_hI8irXopjcO" -->
## Data Loading
<!-- #endregion -->

<!-- #region id="2THcxHuYjkuf" -->
Note that since the MovieLens dataset does not have predefined splits, all data are under train split.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 497, "referenced_widgets": ["718ca1ef6ea747fba167caa5c7857320", "1144249c364a4c5d906be10978513e9d", "b57310ecfb97471daf737c77b95e6405", "f45b245429534fbd9a2aeecf0949152b", "6464b278156f491aa48713e8df37df62", "0d17215c9d514b038c615da117f00293", "7fb665eafd284190a7170cf8cbd90aa5", "b306c2543a914c16929aa21464b7c493", "13acf131f5a0432da3a8d62b760a91bd", "0285ffb51dc14b3c9423d36dcdb45212", "e4085ce4bc25478483578faf576f3de9", "7755984592324664b63035577969fa11", "4cd57593e4114bbc87a19d2827a1b21e", "d7e2f50e9e16456280e6be37987b9804", "65d95dc412bd4fa3911a6704a39b5f21", "2a11274eea2b48a78c349d3667c16e0a", "d7d50a0eaf5741fba3608a70b1e70626", "04af3d99a2bc4c778f387336c77acc06", "495cdec4989446908e7d69a1fd4154df", "b80f98cbe8d74f6fbd522b933cae145f", "288b5c726d8a421b983c72ee47f84926", "938e11adc906494986376552252e0eea", "2d45ed445bca481f8291b5633e8de1de", "691cd7ee90aa438abae7400ab45c78c2", "da0148d563334758955c9645cbb3f722", "beeca3844b7d40f8b4009df2b45873df", "99f71360fcf5473eb85322dbab3afdd7", "0a5ce1ea67df421989f4f70434f799f1", "c853ad51bb174bc0936c224824dcf7aa", "44eff8ed86f44e4d81fc1cfb9a0644a5", "0d373a830de24a9995c7c06a810b7446", "c73959dd8f624a39a13c72ee62f7d2dd", "f7102d6a4a604bb4802855030ee8e837", "e8490da3a07d460a8163a97150070008", "149ac513348943f08ed04c35d5bd3ee0", "2b1395cea5cb4a0fbc5efa1ba24c57c6", "7aa8fe718d0d43ff96242c54179a4eca", "84e76fb3e9794367abdcd9f4ca0c0a98", "ff43a188c26e491db3dc1de378e2970f", "f6b8106a294840ff93acb08f92cd1e29", "d3042b5bfda548548926d9d2d91e6f5e", "ef71de669de94ec1b0f1783362861ee4", "f1c9bc8e512542e495092a834d366a64", "a2824a89e4dd495d880ab57048a0203c", "e14724e2333b425ebef21fdedc66936c", "df2407fc7c4a48199f3fcc9db812bbbc", "7a7f5db599044dfa872ab3904dae623c", "0c8c69a8e693449c8ecddef6639ff3bd", "f601d7635b9b4f93a95db7eae860b092", "5d7b9c961d3d43ffb1620d189a18e027", "0f2be25b7f2e429a824b9a29709394d5", "a3706e1bf4744aada420ea6ef5959371", "2c6807b4e2e34639b39a46386962a1df", "6e5d1c7cfa0149f0a10d98b860a25475", "d92e6d9268804a658095ae7350aad4d0", "26c4521a5ec44230b5e17a76277517d8", "cca2f9d56cc147d4b70e9f6622ce1e11", "243507bfe4794446b90daef4090c5349", "529fdd57c8844a14a2314a2615016e17", "474191659a7b452786c93143a78ff243", "21dac601e2484d1eb156103894065e6a", "e0d97e9a96374d019bfe74a7d56c3d07", "3e7a6903864e481ca8f48e0789e94737", "cb10f0ee6c8742de89236620943c8a7d", "988fd7415eb34fc5be4349a225f87440", "483502b6fe7744aa9bc1b819ef5622b8", "8b7f6ae43efc4d9b8f830832f5396e07", "d8059f7b15bb4b3caad8fba113386e07", "0d19f4b9a3bf48ebbc6445849378beb0", "d0bc823b225f4a868e5775e4e65d776a", "4998810f17594ae2a2605d92095c0089", "5c21e41cf3a34a6bab957a3a362818e0", "e816a9aefab04416a1a581e83aabe90d", "0fcc9673a6874ff8a85ac847bb13ded8", "297e79ebc1e84eb9a60cadd50041557a", "55ae38e0c02a4f069c6ae0b90b410cb6", "94d27b520f2b49fbb98c3adae9f26e79", "51e1a318b4324762bc046d7fa79d9ed1", "d37c7dc3d7fd43569eef7ae2a756408d", "58918a1db431455cb4f65464942648f0", "1b9c7da49c9d479cad6e177eec435b1f", "cbf9b1748341414892fe7761295a7554", "ffa6f998b8194796b302e050cb4b28eb", "8c1a6843518d4f21ba10511a9d742fd7", "7c1c2f84021c4c538db5d9db0202e21f", "77d8097041ff4fe584e9afd0bf720713", "b9049cc1bede4a77a430268a187fda85", "89fc12e3a2fd426cb43fcb9c67e0219c", "00638f607c2b414c84461fde67852668", "c77e2135282c42f0bdc75276123ca2dd", "0dc36c671fc1499dbd3f4aa2893d628f", "0ae023c50cf0417eb7a17cec1c029692", "f25d745ff01140f2abbe8c97fda8e13d", "4fdef37527a9421cbe993627f493c6bb", "ad08bbcf77ad4dc8a32764a4f8f522e5", "ebd77859493f4013ad72eb68645f96d0", "f64e313e3bd94e3db4d26ed4ef6d4ff7", "bd9bf991480d418b93298b4baf62a849", "f4906b1183be487483fb83d380eff766", "11d07f9b81064e4b9691f983c9ffea6b", "754480fc9f2c4334bfd5841880cd10ce", "9aa68678346f42f1b9a0507a488fb7e4", "b5512b41ed7448dbae1b0b01945a04d1", "f3966234aae14e3f98d568d44641b5cd", "07fb1820c2384c95bf28a2a8930c100b", "c15707cbae8e47f2bb39a70dd2880888", "30c2ee5cfcc4468093aa9f8e1e1889de", "07e86370720b427b9e06387d63398e9f", "8469b1f0ba7f417680b94250866d3a09", "d9e06a3895934659905da91515ada2cc"]} id="NT-fhWEvjVnF" executionInfo={"status": "ok", "timestamp": 1636006931208, "user_tz": -330, "elapsed": 892034, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="224d85ef-f04a-403a-abb4-b87b625758ce"
# Ratings data.
ratings = tfds.load("movielens/1m-ratings", split="train", shuffle_files=True)
# Features of all the available movies.
movies = tfds.load("movielens/1m-movies", split="train", shuffle_files=True)
```

<!-- #region id="E1U0QWvcjrIl" -->
The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QTrPZDEAjp-6" executionInfo={"status": "ok", "timestamp": 1636006931211, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3f9e6b29-173c-43ab-bf65-b3acb2620873"
#View the data from ratings dataset:
for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)
```

<!-- #region id="Olu8uuPzjvqW" -->
The movies dataset contains the movie id, movie title, and data on what genres it belongs to. Note that the genres are encoded with integer labels:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rOFc83vUnrkK" executionInfo={"status": "ok", "timestamp": 1636007075628, "user_tz": -330, "elapsed": 759, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fade605c-83ee-48b4-cb7b-0cff36332604"
#View the data from movies dataset:
for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)
```

<!-- #region id="QOMbutT9ntiL" -->
Now let's convert Tensforflow Dataset into a Pandas Data Frame:
<!-- #endregion -->

```python id="huKX_YHWn4TG"
# df = tfds.as_dataframe(ratings.take(10000)) # Use This line if you want to limit the conversion into small Pandas DataFrame
df = tfds.as_dataframe(ratings)
df1 = tfds.as_dataframe(movies)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 360} id="OC5ZrR0eoJId" executionInfo={"status": "ok", "timestamp": 1636007697819, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cf1a51de-9c18-4621-a373-f0589535ae75"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="jL92tirwoK0-" executionInfo={"status": "ok", "timestamp": 1636007697824, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e7d58b8e-1d85-44ee-fc82-e27c2945d0e9"
df.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="px5W4L-hoMR3" executionInfo={"status": "ok", "timestamp": 1636007698603, "user_tz": -330, "elapsed": 797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="357c4eda-8f6b-4f80-fd57-0c401ac22ace"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="7C4pu-H8oNcs" executionInfo={"status": "ok", "timestamp": 1636007698605, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="216f9743-6d7a-4917-8b60-e28655f678ef"
df.describe()
```

```python id="lyhGNO8jrVpA"
df.to_parquet('ratings.parquet.snappy', compression='snappy')
df1.to_parquet('movies.parquet.snappy', compression='snappy')
```

```python id="DmqKRAcnwFoZ"
!mkdir -p main/T590975/data
!cp ratings.parquet.snappy main/T590975/data
!cp movies.parquet.snappy main/T590975/data
```

<!-- #region id="dsHo9RleoRW7" -->
## Data Wrangling

**Data Wrangling Objective:**

 * Let's change user_gender from boolian Female or Male: True --> Male, False --> Female
 * Let's remove the symbols: (b), (') and (").
 * Let's drop columns: user_occupation_label and movie_genres.
 * Let's change "timestamp" which is in the unix epoch (units of seconds) to datetime64[ns]
 * Let's fix any wrong values in user_zip_code (Any zipcode >5 characters)
<!-- #endregion -->

```python id="Y1IiXBIerJXb"
ratings_1 = pd.read_parquet('ratings.parquet.snappy')
```

```python id="YsmORNzDrwYW"
# let's Change user_gender from boolian Female or Male: True --> Male, False --> Female:
ratings_1.loc[ratings_1['user_gender'] == True, 'user_gender'] = 'm'
ratings_1.loc[ratings_1['user_gender'] == False, 'user_gender'] = 'f'

# Now let's remove (b) and ('')
# Normal method .str.strip wont't work because it doesn't recognize it as a string column and it will raise an error. 
# So, let's force the columns to string dtype using .apply(str) as shown below:
ratings_1['user_zip_code'] = ratings_1['user_zip_code'].astype('str').str.strip("b")
ratings_1['user_zip_code'] = ratings_1['user_zip_code'].astype('str').str.strip("'")

# Ok, Now let's looks at user_zip_code (Any zipcode >5 characters):
ratings_1[ratings_1.user_zip_code.str.len() ==10]
# for all user_zip_code  >5 characters, let's only keep the first 5 characters:
ratings_1['user_zip_code'] = ratings_1['user_zip_code'].str[:5]
# Let's confirm that all user_zip_code are 5 characters or less:
ratings_1[ratings_1.user_zip_code.str.len() ==10]
# Let's change now user_zip_code data type from str to int:
ratings_1['user_zip_code'] = ratings_1['user_zip_code'].astype('int64')

# Now let's remove (b), (') and (") from 'user_occupation_text', 'user_id', 'movie_title' and 'movie_id'
ratings_1['user_occupation_text'] = ratings_1['user_occupation_text'].astype('str').str.strip("b")
ratings_1['user_occupation_text'] = ratings_1['user_occupation_text'].astype('str').str.strip("'")
ratings_1['user_id'] = ratings_1['user_id'].astype('str').str.strip("b")
ratings_1['user_id'] = ratings_1['user_id'].astype('str').str.strip("'")
ratings_1['movie_title'] = ratings_1['movie_title'].astype('str').str.strip("b")
ratings_1['movie_title'] = ratings_1['movie_title'].astype('str').str.strip("'")
ratings_1['movie_title'] = ratings_1['movie_title'].astype('str').str.strip('"')
ratings_1['movie_id'] = ratings_1['movie_id'].astype('str').str.strip("b")
ratings_1['movie_id'] = ratings_1['movie_id'].astype('str').str.strip("'")

# Let's change the datatype of 'user_id' and 'movie_id' from str to int:
ratings_1['user_id'] = ratings_1['user_id'].astype('int64')
ratings_1['movie_id'] = ratings_1['movie_id'].astype('int64')

# Let's extract the release year from title:
ratings_1['movie_release_year'] = ratings_1['movie_title'].str[-5:-1]
ratings_1['movie_release_year'] = pd.to_datetime(ratings_1['movie_release_year'])

# Finally, let's change "timestamp" which is in the unix epoch (units of seconds) to datetime64[ns]:
ratings_1['timestamp'] = pd.to_datetime(ratings_1['timestamp'], unit = 's')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="wy9h5Fexok5T" executionInfo={"status": "ok", "timestamp": 1636008713117, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d5523de-02c6-4e66-cfd5-b45f9a86bee9"
ratings_1.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="dEaSdcEFtL8t" executionInfo={"status": "ok", "timestamp": 1636008715895, "user_tz": -330, "elapsed": 478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3cd8e04c-fea0-4718-fddc-00aacda6e569"
ratings_1.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="PmiT9IlAtNrh" executionInfo={"status": "ok", "timestamp": 1636008723258, "user_tz": -330, "elapsed": 536, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="757e30f5-4c77-4126-dc05-f0e5799fe0c8"
pd.set_option('float_format', '{:.1f}'.format)
ratings_1.describe()
```

```python id="BL-z9SHwuoS-"
ratings_1.to_parquet('ratings_cleaned.parquet.snappy', compression='snappy')
```

```python id="-1wn09IQwYlj"
!cp ratings_cleaned.parquet.snappy main/T590975/data
```

<!-- #region id="3a5f54b2" -->
**Awesome, There's no missing values :)**
<!-- #endregion -->

<!-- #region id="230668ee" -->
**Alright, As you can see above, we managed to:**

 * Change 'user_gender' from boolian Female or Male: True --> Male, False --> Female.
 * Remove all special characters from all features: (b) and (').
 * Correct 'user_zip_code' to keep only 5 Characters for each.
 * Change datatype to reflect the data.
 * Change "timestamp" which is in the unix epoch (units of seconds) to datetime64[ns].
 * No missing values.
    
<!-- #endregion -->

<!-- #region id="Z8R347p4tgQe" -->
## EDA

**Alright, let's start the fun part, let's extract insights from the dataset by asking very useful questions:**

- What is the preferable month of the year to rate/watch movies?
- What is the preferable day of the week to rate/watch movies?
- Who watches/rates more movies Men/Women?
- What age group watches more movies?
- Which kind of occupant watches/rates more movies?
- How much rating people give mostly? distributed between gendors?
- What are the most rated movies?
- What are the most loved Movies?
- Which year the users were interested the most to rate/watch movies?
- What are the worst movies per rating? Using worldcloud Package
<!-- #endregion -->

```python id="KkiONmpduvb8"
df = pd.read_parquet('ratings_cleaned.parquet.snappy')
```

<!-- #region id="kXwon21FxePl" -->
**What is the preferable month of the year to rate/watch movies?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="OJ78jWEXuGs3" executionInfo={"status": "ok", "timestamp": 1636009617747, "user_tz": -330, "elapsed": 718, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d0006c62-d606-449f-97bb-4961a2a2374c"
plt.figure(figsize=(6,4))
movies_view_habit = df.groupby(df.timestamp.dt.month).size()
sns.barplot(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'july', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], movies_view_habit.values)
plt.title('Movies View/Rate per Month');
plt.xlabel('Months')
plt.ylabel('Count')
plt.show()
```

<!-- #region id="19STdmInuedf" -->
Ok, Summer & Holidays Months are the highest, which make sense!!!
<!-- #endregion -->

<!-- #region id="WzT_7Y0qxkEb" -->
**What is the preferable day of the week to rate/watch movies?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="m9hYSGDRxpBj" executionInfo={"status": "ok", "timestamp": 1636009699246, "user_tz": -330, "elapsed": 618, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5da9f25a-d609-454e-a4d5-bc2530546699"
plt.figure(figsize=(6,4))
movies_view_daily = df.groupby(df.timestamp.dt.dayofweek).size()
sns.barplot(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'], movies_view_daily.values);
plt.title('Movies View/Rate per Day');
plt.xlabel('Days');
plt.ylabel('Count')
plt.show()
```

<!-- #region id="DRM45x9xx1ws" -->
As shown above, looks like people enjoys watching/rating movies during weekdays and probably going out for a theater during the weekend (low rating/watching). Or MovieLens team asked users to rate movies in their workdays.
<!-- #endregion -->

<!-- #region id="xpZIfajGx7eb" -->
**Who watches/rates more movies Men/Women?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="avlc5Y2Cx_93" executionInfo={"status": "ok", "timestamp": 1636009788873, "user_tz": -330, "elapsed": 737, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a45b8a17-1a0c-434a-d9e4-808485c9d779"
plt.figure(figsize=(6,4))
sns.barplot(df.groupby('user_gender').size().index, df.groupby('user_gender').size().values)
plt.title('Male/Female movie rating ratio')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
```

<!-- #region id="F0wwhakByFWw" -->
Males look like are more interesting in rating movies than females!!
<!-- #endregion -->

<!-- #region id="2FtUqkWqyFTk" -->
**What age group watches more movies?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="Sv2XaSC7yFQO" executionInfo={"status": "ok", "timestamp": 1636009826947, "user_tz": -330, "elapsed": 592, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4d254cdb-bdd9-4704-c415-c59fb9e26d96"
plt.figure(figsize=(6,4))
sns.barplot(df.groupby('bucketized_user_age').size().index, df.groupby('bucketized_user_age').size().values)
plt.title('Movie Watchers per Age Group');
plt.xlabel('Age Group');
plt.ylabel('Count')
plt.show()
```

<!-- #region id="zuQZ-eAwyFNN" -->
As shown above, users aged between 18 to 34 are most who watch or rate Movies.
<!-- #endregion -->

<!-- #region id="-HBYpUs3yFJ7" -->
**Which kind of occupant watches/rates more movies?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="W7Tv8sjYyFG2" executionInfo={"status": "ok", "timestamp": 1636010035691, "user_tz": -330, "elapsed": 1882, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1988dbba-c50d-4a9c-ff5b-c1efa5de352d"
plt.figure(figsize=(6,4))
movie_watcher_occupants = df.groupby('user_occupation_text').size().sort_values(ascending=False)
sns.barplot(y=movie_watcher_occupants.index, x=movie_watcher_occupants.values)
plt.title('Movie Watchers Occupation Group')
plt.xlabel('Occupation Group')
# plt.xticks(rotation=90);
plt.ylabel('Count')
plt.show()
```

<!-- #region id="ckgL-Cs-yrD9" -->
**How much rating people give mostly? distributed between gendors?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="9aWWTJC9zVma" executionInfo={"status": "ok", "timestamp": 1636010136405, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6456cb7f-d98a-4fa6-f45e-887c82e1c53f"
temp_df = df.groupby(['user_gender', 'user_rating']).size()
plt.figure(figsize=(6, 4))
m_temp_df = temp_df.m.sort_values(ascending=False)
f_temp_df = temp_df.f.sort_values(ascending=False)

plt.bar(x=m_temp_df.index, height=m_temp_df.values, label="Male", align="edge", width=0.3, color='green')
plt.bar(x=f_temp_df.index, height=f_temp_df.values, label="Female", width=0.3, color='red')
plt.title('Ratings given by Male/Female Viewers')
plt.legend()
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()
```

<!-- #region id="F-vSwXdezY0H" -->
Ok, That's interesting both males and females have shown the same trend in ratings and both have geven 4 as the highest ratings!!!
<!-- #endregion -->

<!-- #region id="Y_Z-VhiEzceL" -->
**What are the most rated movies? In terms of: All Time/Gender Group/Age Group?**
<!-- #endregion -->

```python id="IRy90nh8zk5-"
def draw_horizontal_movie_bar(movie_titles, ratings_count, title=''):
    plt.figure(figsize=(6, 4))
    sns.barplot(y=movie_titles, x=ratings_count, orient='h')
    plt.title(title)
    plt.ylabel('Movies')
    plt.xlabel('Count')
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="nMXG5uUozo9t" executionInfo={"status": "ok", "timestamp": 1636010208930, "user_tz": -330, "elapsed": 524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f036c43a-965f-4e00-c50b-1d670626faeb"
top_ten_rated_movies = df.groupby('movie_id').size().sort_values(ascending=False)[:10]
top_ten_movie_titles = df.iloc[top_ten_rated_movies.index].movie_title

draw_horizontal_movie_bar(top_ten_movie_titles.values, top_ten_rated_movies.values, 'Top 10 watched movies - All Time')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 581} id="SLqdhetazqbl" executionInfo={"status": "ok", "timestamp": 1636010210050, "user_tz": -330, "elapsed": 1133, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fccd28a1-c80c-4b6b-a7c5-1c3e95d782c0"
top_rated_movies_gender_wise = df.groupby(['user_gender','movie_id']).size()

for index_label in top_rated_movies_gender_wise.index.get_level_values(0).unique():

    top_10_userkind_rated_movies = top_rated_movies_gender_wise[index_label].sort_values(ascending=False)[:10]
    top_10_userkind_rated_movie_titles = df.iloc[top_10_userkind_rated_movies.index].movie_title
    draw_horizontal_movie_bar(top_10_userkind_rated_movie_titles.values, top_10_userkind_rated_movies.values, f'Top 10 {index_label} watched movies')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="gWzwyrsWzqra" executionInfo={"status": "ok", "timestamp": 1636010267803, "user_tz": -330, "elapsed": 3493, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a42935e1-c78e-4aaf-e887-c0998ef7c4ca"
top_rated_movies_age_group_wise = df.groupby(['bucketized_user_age','movie_id']).size()

for index_label in top_rated_movies_age_group_wise.index.get_level_values(0).unique():
    top_10_userkind_rated_movies = top_rated_movies_age_group_wise[index_label].sort_values(ascending=False)[:10]
    top_10_userkind_rated_movie_titles = df.iloc[top_10_userkind_rated_movies.index].movie_title
    draw_horizontal_movie_bar(top_10_userkind_rated_movie_titles.values, top_10_userkind_rated_movies.values, f'Top 10 {index_label} watched movies')
```

<!-- #region id="ayl31SaJ0l9b" -->
Now, let's create a new dataFrame so we can have only the unique movies to have more insights!!?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="aS1IpgTi0mUZ" executionInfo={"status": "ok", "timestamp": 1636010462463, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5c38d740-b316-4fcc-a123-4be15042f304"
# Create a dataframe called 'films':
films = df[['movie_title', 'movie_release_year']]
films.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Iq4frOVu0of6" executionInfo={"status": "ok", "timestamp": 1636010463016, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="25de63a0-997e-4abd-f6cb-d30b0b83bcca"
# Let's check how many unique movies we have:
films.movie_title.nunique()
```

```python id="EV1MFm1Y0otV"
# Let's sort the movies per 'movie_title':
films.sort_values("movie_title", inplace = True)
```

```python id="io6kp0f_0q35"
# dropping ALL duplicate values:
films.drop_duplicates(subset ="movie_title", keep = 'first', inplace = True)
```

<!-- #region id="-CzmEXfz0v9J" -->
**Which year the users were interested the most to rate/watch movies?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 393} id="x7Rh38110xAq" executionInfo={"status": "ok", "timestamp": 1636010528113, "user_tz": -330, "elapsed": 3395, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e4d19f43-f8e4-4f86-929f-3981a837b551"
plt.figure(figsize=(20,7))
yearly_release_counts = films.groupby(films.movie_release_year.dt.year).size().sort_values(ascending=False)
sns.barplot(yearly_release_counts.index, yearly_release_counts.values);
plt.xlabel('Release Year')
plt.xticks(rotation=90);
plt.title('Movies Release Count Per Year');
```

<!-- #region id="mxP9o4DR00Jm" -->
Alright, looks like our users were mainly interested in rating/watching the 90s Movies.
<!-- #endregion -->

<!-- #region id="uf26JeZ007UE" -->
**What are the worst movies per rating?**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 575} id="WeTlrxmh0_O9" executionInfo={"status": "ok", "timestamp": 1636010573692, "user_tz": -330, "elapsed": 4327, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="78fb5d2f-c161-41ee-bd08-e70e433236b2"
movies_ratings_sum = df.groupby('movie_id').sum().user_rating.sort_values()
movies_ratings_sum.index = df.iloc[movies_ratings_sum.index].movie_title
# Will show movies with 0 < total_rating<= 10
lowest_rated_movies = movies_ratings_sum[movies_ratings_sum <= 10]


wordcloud = WordCloud(min_font_size=7, width=1200, height=800, random_state=21, max_font_size=50, relative_scaling=0.2, colormap='Dark2')
# Substracted lowest_rated_movies from 11 so that we can have greater font size of least rated movies.
wordcloud.generate_from_frequencies(frequencies=(11-lowest_rated_movies).to_dict())
plt.figure(figsize=(16,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```
