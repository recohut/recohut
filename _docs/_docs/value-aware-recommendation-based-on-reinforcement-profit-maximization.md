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

```python colab={"base_uri": "https://localhost:8080/"} id="1si5ADNBuJ7g" executionInfo={"status": "ok", "timestamp": 1634817650739, "user_tz": -330, "elapsed": 53570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d36809ed-e203-4961-8888-2b6449bc28e4"
!gdown --id 14OtIC8eiDkzoWCTtaUZHcb7eB-bUmtTT
```

```python colab={"base_uri": "https://localhost:8080/"} id="SxtCZzKCuodn" executionInfo={"status": "ok", "timestamp": 1634817847470, "user_tz": -330, "elapsed": 127979, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a8b940c-9bfe-461b-90b1-e97ea9c1fab4"
!tar -xvf /content/wwwdata.tar
```

```python colab={"base_uri": "https://localhost:8080/"} id="gejzcSyOv4PT" executionInfo={"status": "ok", "timestamp": 1634818039760, "user_tz": -330, "elapsed": 770, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f1d7a3c-29ed-47c1-cfa1-ff55949165d0"
!head www.data
```

<!-- #region id="3-nL6GUvuNDN" -->
In our recommendation platform, items are shown in cascade on a mobile App one by one. Each time the user initiates a request, 50 items are recommended to him/her. As user scrolls down the list and have seen all 50 items, a new request is triggered. �is process is repeated until the user leaves the App or return the top of the cascade, labelled as ”exit” in Figure 2. We use a metric called ”pageid” to distinguish diferent requests in this interaction, similar to the concept of ”page” to a search engine. As the user and the system interact with each other, the system learns how to respond to the state to obtain an optimized accumulative reward.
<!-- #endregion -->

<!-- #region id="xx0mBKZEudYE" -->
Each line contains 15 columns. The meaning of each column is as follows:

- column 1: The id of returned page for the current request, which ranges from 0 to 11.  Note that for each page, we return 50 items to the user.
- column 2: The hour when the request is launched by the user.
- column 3-5: The features used to profile the user which includes age-level, gender and the level of purchase power.
- column 6-14: The item-sepcific features/labels. We concat the values of 50 returned items belonging to a request toghether to form a list and separat them by comma.  

More specifically, 
- column 6: The concatenated list of **position** for each item in the returned list, which ranges from 0 to 600 (12pages * 50items/page).
- column 7-9: The concatenated list of **predicted ctr/cvr/price** for each item in the returned list.
- column 10-12: The concatednated list of **isclcik/iscart/isfav** for each item in the returned list to indicate whether the item is cliked/added to cart/added to whishlist by the user.
- column 13: The concatednated list of **purchase amount** for each item in the returned list. For example, 0.0 means that the user does not purchase this item. 12.0 means that the user spends 12 Yuan on this item.
- column 14: The concatednated list of **an optinal powerful feature** of the item which can be used as one dimension of the "state" vector in RL.
- column 15: To indicate whether the current page is the last page browsed by the user. 


So column 1-9,14 can be used to generate **state** in the paper. Column 10-13 can be used to calclulate the **reward** in the paper. Column 15 represents the **terminal** indicator of RL in the paper.
<!-- #endregion -->

```python id="LROy8MpFukMY" executionInfo={"status": "ok", "timestamp": 1634818215854, "user_tz": -330, "elapsed": 743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
%tensorflow_version 1.x
```

```python colab={"base_uri": "https://localhost:8080/"} id="Y3Gva4xavSG5" executionInfo={"status": "ok", "timestamp": 1634817882579, "user_tz": -330, "elapsed": 1389, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="face7b6e-0c09-47a4-b130-450ccf41f0e7"
!git clone https://github.com/rec-agent/rec-rl.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="-jYPz965vS8x" executionInfo={"status": "ok", "timestamp": 1634818221810, "user_tz": -330, "elapsed": 4456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="49d3125d-42a7-4901-dfae-00e67421799d"
!cd rec-rl && python examples/rec_es/rec_run_es_local.py -i examples/rec_es/rec_config_local.json
```

```python id="XpT90mFgvU9Z"
!pip install numpy==1.19
```

```python id="5ezH4xuxvpCT"

```
