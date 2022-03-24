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

<!-- #region id="8jvk9njivAgx" -->
# Simple Data analysis in Jupyter note book
<!-- #endregion -->

```python id="gdMjKErInABW"
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="iH-Db4R9v1MP"
# !unzip data.zip
```

```python id="ur_PuZnTsMVA"
# import glob
files = glob.glob('/content/*.xlsx')
files.sort()
years = {x:int(Path(x).stem.split('_')[1]) for x in files}
```

<!-- #region id="O26Bv8ViLOX0" -->
## Section 1
<!-- #endregion -->

```python id="kh0I7NHJt1ui" colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"status": "ok", "timestamp": 1598513016587, "user_tz": -330, "elapsed": 857, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="be7b4715-9041-4dd9-9c9e-95ad486e50ca"
def temp(filename):
  df = pd.read_excel(filename)
  df.columns = ['x1','industry','x3','x4','x5','x6','wales']
  df = df[~pd.notnull(df.x3)]
  df = df[['industry','wales']]
  df = df.iloc[2:].reset_index(drop=True)
  industry_labels = {
      'Agriculture, forestry and fishing ': 'Agriculture',
      'Production ': 'Production',
      'Construction ': 'Construction', 
      'Wholesale, retail, transport, hotels and food ': 'Retail',
      'Information and communication ': 'ICT',
      'Finance and insurance activities ': 'Finance', 
      'Real estate activities ': 'Real_Estate',
      'Professional, scientific and technical activities; administrative and support service activities ': 'Professional_Service',
      'Public administration, defence, education and health ': 'Public_Adminstration',
      'Other service activities ': 'Other_Service'}
  df = df.replace({'industry': industry_labels})
  df['year'] = years[filename]
  return df

df = pd.DataFrame(columns=['industry','wales','year'])
for fpath in files:
  df = df.append(temp(fpath))#.append(temp(files[1]))
df['wales'] = df['wales'].astype('int')
df['year'] = df['year'].astype('int')
df = pd.pivot_table(df, values = 'wales', index=['industry'], columns = 'year').reset_index()
df = df.set_index('industry').reindex(list(industry_labels.values())).iloc[:, ::-1].reset_index()
df
```

<!-- #region id="v_uW-tphet7D" -->
Comment 1: 
<!-- #endregion -->

<!-- #region id="x7dYY1tNLSkF" -->
## Section 2
<!-- #endregion -->

<!-- #region id="5Ag3RDitLVNa" -->
2.1
<!-- #endregion -->

```python id="_pOCqVN8yWi3" colab={"base_uri": "https://localhost:8080/", "height": 645} executionInfo={"status": "ok", "timestamp": 1598513582011, "user_tz": -330, "elapsed": 2658, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="27403704-c6ac-4f63-858a-cbc0514ec54b"
# fig=plt.figure(figsize=(10,30))
df.set_index('industry').T.plot(kind='line', figsize=(17, 10))
plt.show()
```

```python id="mMNbLo6m0b4-" colab={"base_uri": "https://localhost:8080/", "height": 757} executionInfo={"status": "ok", "timestamp": 1598513757238, "user_tz": -330, "elapsed": 5257, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2a6dad03-9675-4746-8ec9-ab3efdd144ed"
figure, axes = plt.subplots(figsize=(17, 10))
sns.heatmap(df.set_index('industry').T, annot=True, fmt="d", linewidths=.5, ax=axes, cmap='Blues')
```

<!-- #region id="oyjQAkkJex3E" -->
Comment 2.1: 
<!-- #endregion -->

<!-- #region id="Ls3M1YXBLXY8" -->
2.2
<!-- #endregion -->

```python id="8xYun3xH1hAE" colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"status": "ok", "timestamp": 1598516332006, "user_tz": -330, "elapsed": 1946, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0281c3b9-efec-4bce-da11-8b719e5031d6"
dfx = df.set_index('industry').T
dfx
```

```python id="oGI89khn3LnQ" colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"status": "ok", "timestamp": 1598514402085, "user_tz": -330, "elapsed": 1277, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b8a5e5ac-b4e1-4c4f-e912-f0f85c75a330"
dft = df.set_index('industry').copy()
dft['growth'] = (dft[2011] - dft[2002])*100/dft[2002]
dft
```

```python id="HT3KDGlK3pwB" colab={"base_uri": "https://localhost:8080/", "height": 400} executionInfo={"status": "ok", "timestamp": 1598514746413, "user_tz": -330, "elapsed": 1787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b701d5ad-92ab-4dcd-d279-d9341aaeed6b"
dft.growth.plot()
```

<!-- #region id="FVVd0yRfe0lF" -->
Comment 2.2: 
<!-- #endregion -->

<!-- #region id="TVBNcCYlLbn2" -->
2.3
<!-- #endregion -->

```python id="7x_pPxff6F_2" colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"status": "ok", "timestamp": 1598516337594, "user_tz": -330, "elapsed": 1268, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c90b7eaa-982a-4be3-fd1a-110347542aa1"
dfx['Total'] = dfx.sum(axis=1)
dfx
```

```python id="aW3SWaBj5CQs" colab={"base_uri": "https://localhost:8080/", "height": 662} executionInfo={"status": "ok", "timestamp": 1598514942958, "user_tz": -330, "elapsed": 5470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d152e7d4-3444-4628-fe1d-be3c81aa469f"
dfx.pct_change().plot(kind='line', figsize=(17, 10))
```

```python id="AKjnHuVP5Skn" colab={"base_uri": "https://localhost:8080/", "height": 370} executionInfo={"status": "ok", "timestamp": 1598514958152, "user_tz": -330, "elapsed": 1954, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a63405f-c15a-4449-b8c9-3c5132117676"
dfx.Total.pct_change().plot(kind='line', figsize=(17, 5))
```

<!-- #region id="Oq35dYMblFwO" -->
Comment 2.3

<!-- #endregion -->

<!-- #region id="2W3A3o3wLeCd" -->
## Section 4
<!-- #endregion -->

<!-- #region id="LuGzg4C2Lhsp" -->
4.1
<!-- #endregion -->

```python id="jLc1edHK_gPr" colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"status": "ok", "timestamp": 1598516342301, "user_tz": -330, "elapsed": 1319, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b0a4eebc-f221-424f-c989-7476456967a6"
dfx.loc['Average'] = dfx.mean(axis=0)
dfx
```

```python id="x-HFnCqwAJVV" colab={"base_uri": "https://localhost:8080/", "height": 499} executionInfo={"status": "ok", "timestamp": 1598516484944, "user_tz": -330, "elapsed": 3359, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9c4ac2f0-7694-4ecf-d01a-95d4f8b0c576"
sns.heatmap(dfx.corr())
```

<!-- #region id="iOeH9WcwlNPG" -->
Comment 4.1: 
<!-- #endregion -->

<!-- #region id="DH094CPdLjKj" -->
4.2
<!-- #endregion -->

```python id="e4nhmcFQAqCw" colab={"base_uri": "https://localhost:8080/", "height": 403} executionInfo={"status": "ok", "timestamp": 1598516542595, "user_tz": -330, "elapsed": 2669, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8927c05-7ce6-49c8-a498-66bdcac6e049"
sns.heatmap(dfx.T.corr())
```

<!-- #region id="0X5lTtA4lQbs" -->
Comment 4.2: 
<!-- #endregion -->

<!-- #region id="SD2D8_5DLlmJ" -->
## Section 3
<!-- #endregion -->

```python id="YrjUlm4L9B4T" colab={"base_uri": "https://localhost:8080/", "height": 194} executionInfo={"status": "ok", "timestamp": 1598516054801, "user_tz": -330, "elapsed": 1180, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f18c419-134b-4262-e3f8-68f9436c9b6d"
xx = dfxpct.copy().reset_index(); xx.columns
xx = pd.melt(xx, id_vars=['year'], var_name='Name').fillna(0).round(2)
xx.head()
```

```python id="9FVhocij6nD0" colab={"base_uri": "https://localhost:8080/", "height": 617} executionInfo={"status": "ok", "timestamp": 1598516148155, "user_tz": -330, "elapsed": 1212, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="163a36c3-b91b-42aa-f7dc-a1972ff1ebd5"
# !pip install -q plotly==4.2.1
# import plotly.express as px
# dfxpct = dfx.pct_change() 
# fig = px.scatter(x=dfxpct.index, y=dfxpct['Production'].values)
fig = px.scatter(xx, x='year', y='value', color='Name')
fig.show()
```

<!-- #region id="s1EfRqVllS6E" -->
Comment 3:
<!-- #endregion -->

<!-- #region id="R34VKozCLony" -->
## Section 5
<!-- #endregion -->

```python id="7D_MTakb8UFO"
dfx = df.set_index('industry').T
dfx['Total'] = dfx.sum(axis=1)
xx = dfx.pct_change()
best_performing_year = xx.index[xx.Total.argmax()]
worst_performing_year = xx.index[xx.Total.argmin()]
```

```python id="nCsfd8msBr3a"
best_performing_year_data = dfx.T[best_performing_year]
worst_performing_year_data = dfx.T[worst_performing_year]
```

```python id="dLsuD61eEhzS"
xz = worst_performing_year_data.reset_index().merge(best_performing_year_data.reset_index()).set_index('industry')
```

<!-- #region id="GP8Lu7DWLsr5" -->
5.1
<!-- #endregion -->

```python id="4ZNQ9P_gCG1R"
from sklearn.cluster import KMeans
kmeans_2n = KMeans(n_clusters=2).fit(xz)
kmeans_3n = KMeans(n_clusters=3).fit(xz)
xz['cluster_kmeans_2n'] = kmeans_2n.labels_
xz['cluster_kmeans_3n'] = kmeans_3n.labels_
```

<!-- #region id="ZonLDDpJla8U" -->
Comment 5.1: 
<!-- #endregion -->

<!-- #region id="tgLK26RoLt31" -->
5.2
<!-- #endregion -->

```python id="p6-J1jh4CIVv"
from sklearn.cluster import AgglomerativeClustering
hierar_2n = AgglomerativeClustering(n_clusters=2).fit(xz)
hierar_3n = AgglomerativeClustering(n_clusters=3).fit(xz)
xz['cluster_hierar_2n'] = hierar_2n.labels_
xz['cluster_hierar_3n'] = hierar_3n.labels_
```

```python id="8vJbcgtjDyWu" colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"status": "ok", "timestamp": 1598517987001, "user_tz": -330, "elapsed": 1480, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0bca37b-7f4d-4cb1-8783-32cb28afbe7f"
xz
```

<!-- #region id="B3rTrRaLl5Mi" -->
Comment 5.2:
<!-- #endregion -->
