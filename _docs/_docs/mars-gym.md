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

```python id="FpReF3yqK3xu"
!pip install mars-gym
```

```python id="iI4OTphXPEXH" executionInfo={"status": "ok", "timestamp": 1634624896140, "user_tz": -330, "elapsed": 750, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="OEyFT-x_NlQR" executionInfo={"status": "ok", "timestamp": 1634624897693, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="54cb4388-0d68-4763-e423-b024dc9cde06"
from mars_gym.data import utils
utils.datasets()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 323} id="XoGGEmWON0X6" executionInfo={"status": "ok", "timestamp": 1634624899568, "user_tz": -330, "elapsed": 1233, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a849de50-3c5b-400f-8b3b-32f3b205bb16"
df, df_meta = utils.load_dataset('processed_trivago_rio')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="aZjteiFFN3wr" executionInfo={"status": "ok", "timestamp": 1634624909288, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0348d05-26ea-4197-933d-fb39e87a7f5e"
df_meta[['list_metadata']].head()
```
