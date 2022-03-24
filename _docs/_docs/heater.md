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

<!-- #region id="YfQ_kuP3Vh0a" -->
# Heater

> Recommendation for New Users and New Items via Randomized Training and Mixture-of-Experts Transformation.
<!-- #endregion -->

<!-- #region id="q7vWdlIXd4DS" -->
## API run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JQ1OSkzQVkv6" executionInfo={"status": "ok", "timestamp": 1635432029672, "user_tz": -330, "elapsed": 12953, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e7fea472-66cd-4ef2-94c1-d1921de7e4c5"
!git clone https://github.com/Zziwei/Heater--Cold-Start-Recommendation.git
%cd Heater--Cold-Start-Recommendation
```

```python colab={"base_uri": "https://localhost:8080/"} id="2ujtgjmDd6hw" executionInfo={"status": "ok", "timestamp": 1635432029673, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="aec56663-06ad-4533-fbc8-08900f94733e"
%tensorflow_version 1.x
```

```python colab={"base_uri": "https://localhost:8080/"} id="72a8ZeqDT12w" executionInfo={"status": "ok", "timestamp": 1635433073237, "user_tz": -330, "elapsed": 1043569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7eb8a9cd-4967-4697-84ce-454a55168349"
!python main_CiteULike.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="nf2IQk0YUAOt" executionInfo={"status": "ok", "timestamp": 1635434113539, "user_tz": -330, "elapsed": 1038764, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="75f4c091-92d7-4fc4-848b-fbe73b89104e"
!python main_LastFM.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="lyz-WB6RWTLC" executionInfo={"status": "ok", "timestamp": 1635434647602, "user_tz": -330, "elapsed": 534074, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9655bf7a-2646-418b-f06c-7af2929a23ce"
!python main_XING.py
```
