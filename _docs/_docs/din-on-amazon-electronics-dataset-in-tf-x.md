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

```python id="N-C8WVmWyt1c"
!pip install tensorflow==2.5.0
```

```python colab={"base_uri": "https://localhost:8080/"} id="R3QiS0AHzPNr" executionInfo={"status": "ok", "timestamp": 1637067177691, "user_tz": -330, "elapsed": 5576, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1591f011-5266-4b41-e607-004bb4fc9136"
!wget -q --show-progress https://github.com/RecoHut-Datasets/amazon_electronics/raw/main/rating_electronics.pbz2
```

```python id="tOgRfWKuzGx3"
import bz2
import pandas as pd
import pickle
import _pickle as cPickle
```

```python id="rdWjLyFCzKSF"
def load_pickle(path):
    data = bz2.BZ2File(path+'.pbz2', 'rb')
    data = cPickle.load(data)
    return data
```

```python id="u_rAHb45zuJY"
data = load_pickle('rating_electronics')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="SPm7h9KPz3FK" executionInfo={"status": "ok", "timestamp": 1637067243225, "user_tz": -330, "elapsed": 508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="315d4e9b-bf5d-4bf8-d658-2d4634c583d2"
data.head()
```

```python id="8cAmR0J4zSa-"
!pip install pyBaiduPan
```

```python id="fQ_dw2BU3728"
!BdPan download https://pan.baidu.com/s/1sYsY88APFTNldcZ2n3sKlA .
```

```python id="qTaga32Y39Kl"

```
