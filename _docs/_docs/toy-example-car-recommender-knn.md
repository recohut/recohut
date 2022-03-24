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

<!-- #region id="s_V0wG8RzCrT" -->
# Toy example - Car Recommender using KNN method
> Finding top-k cars using different distance metrics like euclidean, cosine, minkowski

- toc: false
- badges: true
- comments: true
- categories: [KNN, Toy]
- image:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="jK5CPdichhR8" outputId="7fc9c81a-5085-4fe7-8c35-a363e11c5a0d"
%%writefile cardata.csv
CarName,Size,Convenience,Economical,Speed,Price
Toyota Agya,4,4,9,6,1.0
Daihatsu Alya,4,3,9,6,1.1
Toyota Avanza,6,5,6,6,2.0
Daihatsu Xenia,6,4,6,6,1.75
Xpander,7,7,6,7,2.25
Livina,7,7,6,7,2.1
Karimun,3,4,10,5,1.2
Toyota Innova,8,8,5,7,4.0
Alphard,9,10,4,8,10.0
Toyota Vios,5,7,9,8,2.5
Honda City,5,8,7,8,2.7
Toyota Hiace,10,5,8,6,5.0
Toyota Fortuner,9,8,5,8,5.0
Toyota Foxy,9,9,5,7,5.5
Toyota Corolla Altis,5,9,7,9,6.0
Suzuki Ertiga,7,7,7,7,2.3
Suzuki Carry,7,3,9,5,0.8
```

```python id="ClzLMBrGhluw"
import numpy as np
import pandas as pd 

from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
```

```python colab={"base_uri": "https://localhost:8080/", "height": 576} id="OAH4h2mYhrJY" outputId="8505f3f7-9da0-420c-843c-fbb6f4124996"
df = pd.read_csv('cardata.csv')
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="ZEWnVakKiQZV" outputId="05503f9f-7c04-42c7-d7a2-fd754ec3719f"
df.describe().round(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="yz_kDfGfiTVe" outputId="74f99ea1-0099-4b5c-ae76-1948a23a11f9"
scaler = MinMaxScaler()
data = scaler.fit_transform(df[['Size','Convenience', 'Economical','Speed','Price']].values)
pd.DataFrame(data).describe().round(2)
```

```python id="ic-fFuabnipQ"
def calc_distance(a,b, method='euclidean'):
  if method=='euclidean':
    return distance.euclidean(a,b)
  elif method=='minkowski':
    return distance.minkowski(a,b)
  elif method=='cityblock':
    return distance.cityblock(a,b)
  elif method=='cosine':
    return distance.cosine(a,b)
  elif method=='hamming':
    return distance.hamming(a,b)
```

```python id="ZAHYQ8wZjcOV"
def _recommend(user_vector, method='euclidean', topk=3):
  # convert to array
  uvec = np.array(user_vector).reshape(1,-1)
  # normalize
  uvec = scaler.transform(uvec)
  # distance
  distances = [calc_distance(uvec, dfvec, method=method) for dfvec in data]
  distances = np.array(distances).flatten()
  # tok-k items
  idx = np.argsort(distances)[:topk]
  recs = df.iloc[idx,:].set_index('CarName')
  # return the results
  return recs
```

```python id="vwSEc2zXoMR6"
def recommend_car():
  uvec = []
  uvec.append(int(input("Car size preference (3-10, default=5): ") or "5"))
  uvec.append(int(input("Convenience level (3-10, default=6): ") or "6"))
  uvec.append(int(input("Economical (4-10, default=7): ") or "7"))
  uvec.append(int(input("Speed(5-9, default=7): ") or "7"))
  uvec.append(int(input("Price (1-10, default=3): ") or "3"))
  topk = int(input("How many recommendations you would like to get? (default=3): ") or "3")
  method = input("Which distance algorithm you would like to use? (euclidean/ minkowski/ cityblock/ cosine/ hamming, default=euclidean): ") or "euclidean"
  print(f"\n\n Your Top {topk} recommendations are:\n\n")
  return _recommend(uvec, method=method, topk=topk)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 428} id="ejPTflDDoYZk" outputId="56e6e0ca-1cb5-43c8-fbe4-930c879c7835"
recommend_car()
```
