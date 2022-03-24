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

<!-- #region id="jM5F1Clckebn" -->
# RekoNet Anime Recommender
> The data crawled from the popular anime website [MyAnimeList.net](http://myanimelist.net/), and cleaned of duplicates as well as missing values and false data. Following that, autoencoders used to learn embeddings of all the anime titles present in the dataset, which were then used to cluster the same.

- toc: true
- badges: true
- comments: true
- categories: [anime, autoencoder, pytorch]
- image:
<!-- #endregion -->

<!-- #region id="aoZY5KoFj60J" -->
### Overview
1. deep autoencoders for predicting ratings and generating embeddings.
2. form clusters using embeddings of anime titles
3. find similar animes using similarity metric based on like and dislike of user
4. combine with rating prediction to create a hybrid recommender
<!-- #endregion -->

<!-- #region id="i8KumC_xkBbX" -->
### Background
<!-- #endregion -->

<!-- #region id="IZMeWF-Tjxi9" -->
Anime (a term derived from the English word animation) is a form of hand-drawn computer animation which originated in Japan and has now developed a cult following around the world. In recent years, the Anime industry has been growing at an enormous pace making billions of dollars in profit every year. Its market has gained attention from major streaming platforms like Netflix and Amazon Prime. In the pre-internet era, Anime enthusiasts discovered new titles through word of mouth. Hence personalized recommendations were not required. Moreover, the number of titles released were quite less to facilitate a data-based approach for personalized recommendations. However, in recent years, with the boom of streaming services and the amount of newly released anime titles, people can watch Anime as much as they like. This calls for a personalized recommendation system for this new generation of Anime watchers.
<!-- #endregion -->

<!-- #region id="9IRd-q6NkRci" -->
<!-- #endregion -->

<!-- #region id="JjX8CzAOlGwY" -->
The data used for training Rikonet was crawled from the popular anime website [MyAnimeList.net](http://myanimelist.net/) using the Jikan API. The collected data was cleaned of duplicates as well as missing values and false data and reduced to 6668 anime titles while retaining all the key information.

Following that, autoencoders used to learn embeddings of all the anime titles present in the dataset, which were then used to cluster the same.

The logically opposite clusters of the anime titles are estimated as well.

At run-time, when a user requests a new recommendation list, the user’s context, i.e., the anime titles rated so far is fed into the primary autoencoder, which computes the predicted ratings for the unrated titles.

These ratings are further fed to a hybrid filter, which generates 2 lists, namely - Similar Anime and Anime You May Like, the former showing anime titles similar to the ones the user rated highly and the later showing titles which the user may like based on his overall ratings.
<!-- #endregion -->

<!-- #region id="A52oi8A7dk9w" -->
### Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NW5qXlV2dQet" outputId="7033b9d1-7e49-4df6-9133-7b2b975e7658"
!pip install google_trans_new
```

```python id="QfdiNlKmKZI-"
from collections import OrderedDict
from tabulate import tabulate
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from google_trans_new import google_translator  
```

<!-- #region id="f9CoQ9_6dgNs" -->
### Download data and pre-trained model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="czltDsN7YnOF" outputId="0f95d4cc-9f97-426a-84b0-c1eff0e10cca"
!wget https://github.com/sparsh-ai/reco-data/raw/master/anime/anime_cleaned.csv
!wget https://github.com/sparsh-ai/reco-data/raw/master/anime/anime_genres.csv
!wget https://github.com/sparsh-ai/reco-data/raw/master/anime/clusters.csv
!wget https://github.com/sparsh-ai/reco-data/raw/master/anime/inputFormater.csv
!gdown --id 1LV7VHOTqU5WgBYxfRcUeY31dbhcBqyzb
!gdown --id 14x3TgzhFl-XCHjJHtX-mZrtTSkJgIIey
```

```python id="xSX9cwTnXfnn"
def top_animes(genre, ani_genre, all_anime):
    top = []
    print("\nTop", genre)
    temp = list(ani_genre[ani_genre[genre]==1]['anime_id'])
    temp = list(filter(lambda x: x in all_anime.index, temp))
    temp.sort(key=lambda x: all_anime['score'][x], reverse=True)

    for i in range(5):
        r = [i+1, temp[i], all_anime['title'][temp[i]], all_anime['title_english'][temp[i]],
             all_anime['score'][temp[i]], all_anime['genre'][temp[i]]]
        top.append(r)

    table = tabulate(top, headers=['S.No.', 'Anime ID', 'Title', 'English Title',
                                   'Anime Score', 'Anime Genre'], tablefmt='orgtbl')
    print(table)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="M5AMVrYVZt9Q" outputId="1cbaa86c-1792-4ca4-ef41-acae64d4e3dd"
results = pd.read_csv('clusters.csv')
results.head()
```

```python id="HBoyd0c-XwOt"
clusters = []

for i in range(222):
    clusters.append([])

for i in range(len(results)):
    clusters[results['alpha'][i]].append(results['anime_id'][i])

def getCluster(anime_id, opposite=False):
    if opposite == False:
        temp = results[results['anime_id'] == anime_id]['alpha'].reset_index(drop=True)
        clusterID = temp[0]
        return clusters[clusterID]
    else:
        temp = results[results['anime_id'] == anime_id]['zeta'].reset_index(drop=True)
        clusterID = temp[0]
        return clusters[clusterID]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="qNbommA0aZzp" outputId="d76b6fb9-0a71-4cff-8e8a-1697e01c46ae"
input_formater = pd.read_csv("inputFormater.csv")
input_formater.iloc[:5, :5]
```

```python id="SnR4_hbjXhru"
class UserVector(Dataset):
    def __init__(self, age, gender, uratings):
        if age<11:
            self.age=2
        elif age<16:
            self.age=3
        elif age<20:
            self.age=4
        else:
            self.age=5

        if gender.lower() == 'male':
            self.gender = 0
        else:
            self.gender = 1

        self.data = input_formater
        self.data.loc[0, 'Gender'] = self.gender
        self.data.loc[0, 'Category'+str(self.age)] = 1

        self.columns = list(self.data.columns)

        self.aniId_to_ind = pd.Series(data=range(len(self.columns)), index=self.columns)

        for aniId in uratings.keys():
            self.data.loc[0, str(aniId)] = uratings[aniId]
        self.data.fillna(0, inplace=True)

        self.data = self.data.iloc[:,:]

        self.transform =  transforms.Compose([transforms.ToTensor()])

        self.data = self.transform(np.array(self.data))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        user_vector = self.data.data[0][ind]
        return user_vector

    def get_anime_id(self, ind):
        return int(self.columns[ind])

    def anime_to_index(self):
        return self.aniId_to_ind
```

```python colab={"base_uri": "https://localhost:8080/", "height": 258} id="mKobFJclat0K" outputId="56e905d8-b45a-4849-c35a-1f78a652860f"
ani_genre = pd.read_csv("anime_genres.csv", index_col=[0])
ani_genre.head()
```

```python id="PewmXorBbCVP"
# Get similar anime
def similarAnime(uratings, all_anime):

    if len(uratings) == 0:
        anime_data = all_anime.set_index('anime_id')

        top_animes('Shounen', ani_genre, anime_data)
        top_animes('Supernatural', ani_genre, anime_data)
        top_animes('Romance', ani_genre, anime_data)
        top_animes('Slice of Life', ani_genre, anime_data)

        return []

    else:
        temp = list(reversed(uratings.items()))
        SimilarAnime = []

        i = 0
        while i < len(temp) and i < 3:
            if temp[i][1] >= 6:
                SimilarAnime += getCluster(temp[i][0], opposite=False)
            else:
                SimilarAnime += getCluster(temp[i][0], opposite=True)
            i+=1
        return SimilarAnime
```

```python id="JneRj7nRbLTa"
# Get Anime You May Like
def animeYouMayLike(age, gender, uratings, model, all_anime, aniId_to_index):
    # Get Similar Anime
    SimilarAnime = similarAnime(uratings, all_anime)
    if len(SimilarAnime) == 0:
        return [], []
    
    # User Data Column
    user_data = UserVector(age, gender, uratings)

    # User Data Loader
    user_dl = DataLoader(dataset=user_data, num_workers=1)

    # Get model Predictions
    preds = model.getPredictedRatings(user_data, user_dl)
    preds = preds.reshape(-1)

    # Get top predicted anime
    animes = list(preds.argsort()[-1000:][::-1])
    animes = list(map(user_data.get_anime_id, animes))

    # Generate 'Similar Anime' and 'Anime You May Like'
    FinalList1 = []
    FinalList2 = []

    for aniID in animes:
        index = int(aniId_to_index.at[aniID])
        r = [aniID, all_anime['title'][index], all_anime['title_english'][index], all_anime['genre'][index]]

        if aniID in SimilarAnime and len(FinalList1) <10 and aniID not in uratings:
            FinalList1.append(r)
        elif aniID not in SimilarAnime and len(FinalList2) <10 and aniID not in uratings:
            FinalList2.append(r)
        elif len(FinalList1) == 10 and len(FinalList2) == 10:
            break

    return FinalList1, FinalList2
```

```python id="TqXP5dMfXR58"
def showRecommendations(age, gender, uratings, model, all_anime, aniId_to_index):
    # Get both the lists
    List1, List2 = animeYouMayLike(age, gender, uratings, model, all_anime, aniId_to_index)
    if len(List1) == 0 and len(List2) == 0:
        return
    
    # Tabulate the Results
    print("similar Anime")
    table = tabulate(List1, headers=['Anime ID', 'JP Title', 'EN Title', 'Genre'], tablefmt='orgtbl')
    print(table)

    print("Anime You May Like")
    table = tabulate(List2, headers=['Anime ID', 'JP Title', 'EN Title', 'Genre'], tablefmt='orgtbl')
    print(table)
```

<!-- #region id="CoO-mxETdWF6" -->
### Autoencoder model
<!-- #endregion -->

```python id="ZxBeNZVCbUuD"
def activation(input, type):
  
    if type.lower()=='selu':
        return F.selu(input)
    elif type.lower()=='elu':
        return F.elu(input)
    elif type.lower()=='relu':
        return F.relu(input)
    elif type.lower()=='relu6':
        return F.relu6(input)
    elif type.lower()=='lrelu':
        return F.leaky_relu(input)
    elif type.lower()=='tanh':
        return F.tanh(input)
    elif type.lower()=='sigmoid':
        return F.sigmoid(input)
    elif type.lower()=='swish':
        return F.sigmoid(input)*input
    elif type.lower()=='identity':
        return input
    else:
        raise ValueError("Unknown non-Linearity Type")
```

```python id="dI30w2ZuYa8Z"
class AutoEncoder(nn.Module):

  def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations=True):

    super(AutoEncoder, self).__init__()

    self.layer_sizes = layer_sizes
    self.nl_type = nl_type
    self.is_constrained = is_constrained
    self.dp_drop_prob = dp_drop_prob
    self.last_layer_activations = last_layer_activations

    if dp_drop_prob>0:
      self.drop = nn.Dropout(dp_drop_prob)

    self._last = len(layer_sizes) - 2

    # Initaialize Weights
    self.encoder_weights = nn.ParameterList( [nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)  ] )

    # "Xavier Initialization" ( Understanding the Difficulty in training deep feed forward neural networks - by Glorot, X. & Bengio, Y. )
    # ( Values are sampled from uniform distribution )
    for weights in self.encoder_weights:
      init.xavier_uniform_(weights)

    # Encoder Bias
    self.encoder_bias = nn.ParameterList( [nn.Parameter(torch.zeros(layer_sizes[i+1])) for i in range(len(layer_sizes) - 1) ] )

    reverse_layer_sizes = list(reversed(layer_sizes)) 
    # reversed returns iterator

    # Decoder Weights
    if is_constrained == False:
      self.decoder_weights = nn.ParameterList( [nn.Parameter(torch.rand(reverse_layer_sizes[i+1], reverse_layer_sizes[i])) for i in range(len(reverse_layer_sizes) - 1) ] )

      for weights in self.decoder_weights:
        init.xavier_uniform_(weights)

    self.decoder_bias = nn.ParameterList( [nn.Parameter(torch.zeros(reverse_layer_sizes[i+1])) for i in range(len(reverse_layer_sizes) - 1) ] )



  def encode(self,x):
    for i,w in enumerate(self.encoder_weights):
      x = F.linear(input=x, weight = w, bias = self.encoder_bias[i] )
      x = activation(input=x, type=self.nl_type)

    # Apply Dropout on the last layer
    if self.dp_drop_prob > 0:
      x = self.drop(x)

    return x


  def decode(self,x):
    if self.is_constrained == True:
      # Weights are tied
      for i,w in zip(range(len(self.encoder_weights)),list(reversed(self.encoder_weights))):
        x = F.linear(input=x, weight=w.t(), bias = self.decoder_bias[i] )
        x = activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'relu')

    else:

      for i,w in enumerate(self.decoder_weights):
        x = F.linear(input=x, weight = w, bias = self.decoder_weights[i])
        x = activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'relu')

    return x

  def forward(self,x):
    # Forward Pass
    return self.decode(self.encode(x))
```

```python id="r23EDHkzX_zu"
class PredictionEngine:
  def __init__(self):
    self.layer_sizes = [6673, 8192, 2048, 512, 256]
    self.model = AutoEncoder(layer_sizes=self.layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations=False)
    self.model.load_state_dict(torch.load('autoEncoder.pth'))
    try:
        self.model = self.model.cuda()
    except:
        pass

  def getPredictedRatings(self, user_dat, user_dl):
    for data in user_dl:
      inputs = data
      try:
          inputs = inputs.cuda()
      except:
          pass
      inputs = inputs.float()

      outputs = self.model(inputs)
      break

    return outputs.cpu().detach().numpy()
```

```python id="MhcplucvYiOS"
# Search anime in database
def find_anime(input_anime, name_to_id):
    print('Anime Id', '\t', 'Title')
    flag = 0
    for n in name_to_id.index:
        if input_anime in n.lower():
            flag = 1
            print(name_to_id[n], '\t', n)
    return flag
```

<!-- #region id="BvjZhG8ydSYa" -->
### Load cleaned dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 819} id="H8YpRXF-cCKq" outputId="ef60e4b7-ccab-474e-9d82-424de672dd5d"
# Load all the datasets
all_anime = pd.read_csv("anime_cleaned.csv")
display(all_anime.head())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="C-x5SG7Wb31_" outputId="3bea2389-1771-4b4f-a75f-ab70598935f9"
all_anime.image_url[0]
```

```python id="7czeb-D4btrZ"
from PIL import Image
import requests
from io import BytesIO

def id_to_image(anime_id):
  url = all_anime.loc[all_anime['anime_id']==anime_id, 'image_url'].values[0]
  url = url.split('/')
  url = ['https://cdn.myanimelist.net'] + url[3:]
  url = '/'.join(url)
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  return img
```

```python colab={"base_uri": "https://localhost:8080/", "height": 336} id="x6XBTAhee-M_" outputId="d6e8ff92-ff8a-4e2d-e3ed-57a43a7ca6e0"
id_to_image(12365)
```

<!-- #region id="EWJKynzydJZG" -->
### Japanese to english translated titles
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jShYjAqMlo7Y" outputId="1b556ab9-fa23-4b25-fa74-a9b208ee2beb"
all_anime_small = all_anime.sample(200, random_state=42)
translator = google_translator()
all_anime_small['title_english_new'] = all_anime_small.apply(lambda row: translator.translate(row.title_japanese), axis=1)
all_anime_small.title_english_new.unique()
```

```python id="dr3RlFDVdmRD"
name_to_id = pd.Series(list(all_anime['anime_id']), index=all_anime['title'])
aniId_to_index = pd.Series(all_anime.index, index=all_anime['anime_id'])
```

<!-- #region id="Gj_XFnA9dEi2" -->
### Run time process
<!-- #endregion -->

<!-- #region id="1RRyghn2cvID" -->
<!-- #endregion -->

<!-- #region id="zJSWVjmLc9jE" -->
### Hybrid filter
<!-- #endregion -->

<!-- #region id="wwR4kcekc_de" -->
<!-- #endregion -->

<!-- #region id="zqHJNvtodBPQ" -->
### Run time
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xFOd2lgEX_xA" outputId="dae2ffb0-3f63-48b3-e546-3ad9fe4568a5"
print("Starting...\n")

# Load the AutoEncoder
model = PredictionEngine()

# Get basic information from the user
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
input_ratings = OrderedDict()

# Let the user rate some animes
print("\nIt is recommended to rate atleast 5 animes in the beginning.")
print("Note:- Currently search mechanism searches for anime using the Japanese Title only.")

# List for storing the user ratings of recommended table
user_score = []
c = 1

# Start the recommendation process
k1 = input("\nStart the process? [y/n]: ")

while k1 == 'y' or k1 == 'Y':

    # If user want to search and rate
    k2 = input("\nSearch and rate? [y/n]: ")
    while k2 == 'y' or k2 == 'Y':
        p = 'n'
        while p == 'n' or p == 'N':
            input_anime = input("Enter Anime title: ")
            flag = find_anime(input_anime.lower(), name_to_id)
            if flag==0:
                print("\nAnime not found in dataset. Please try searching only a part of the title or another anime!!")
                continue
            p = input("Anime found? [y/n]: ")

        aniId = int(input("Enter anime id: "))
        rate = int(input("Your rating (1 - 10): "))
        if not type(rate) is int:
              raise TypeError("Only integers are allowed")
              
        input_ratings[aniId] = rate

        k2 = input("Search and rate more? [y/n]: ")

    # Main Game
    showRecommendations(age, gender, input_ratings, model, all_anime, aniId_to_index)

    # If user want to rate anime from above list
    k2 = input("\nRate anime from above list? [y/n]:")
    while k2 == 'y' or k2 == 'Y':
        aniId = int(input("Enter anime id: "))
        rate = int(input("Your rating (1 - 10): "))
        input_ratings[aniId] = rate

        k2 = input("Rate again from above list? [y/n]: ")

    k2 = int(input("Your score for the table of recommended anime (1 - 10):"))
    user_score.append([c, k2])
    c += 1
    
    k1 = input("\nKeep going? [y/n]: ")

# Displaying the user score over iterations
print('\n\nTable of user scores')
table = tabulate(user_score, headers=['Iterations', 'User Score'], tablefmt='grid')
print(table)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 259} id="hHPDj5EKiTAm" outputId="c6b91101-4efa-414d-b93c-c62c22489b51"
rated_anime = [10690, 3768, 31699, 2056, 37858]
ratings = [7, 6, 8, 7, 9]
rated_anime_images = [id_to_image(id) for id in rated_anime]
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(rated_anime_images):
    plt.subplot(len(rated_anime_images) / columns + 1, columns, i + 1)
    plt.gca().set_title(ratings[i])
    plt.imshow(image)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 251} id="sgKXLPNLhZpp" outputId="d5edb248-69f8-4c4e-f19a-283421fc688c"
similar_anime = [1724, 1771, 1155, 2723, 8353]
similar_anime_images = [id_to_image(id) for id in similar_anime]
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(similar_anime_images):
    plt.subplot(len(similar_anime_images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 249} id="PS9_B9f6hpGT" outputId="e33d2541-6cfc-4162-89b3-fd9caaead880"
personalized_anime = [30851, 33473, 5267, 7308, 35805]
personalized_anime_images = [id_to_image(id) for id in personalized_anime]
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(personalized_anime_images):
    plt.subplot(len(personalized_anime_images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

<!-- #region id="dnFp8lsNkFdv" -->
### References
1. Paper: [https://arxiv.org/pdf/2106.12970.pdf](https://arxiv.org/pdf/2106.12970.pdf)
2. Code: [https://github.com/NilutpalNath/RikoNet](https://github.com/NilutpalNath/RikoNet)
<!-- #endregion -->
