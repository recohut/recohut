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

<!-- #region id="bt279XzUARiB" -->
# Recommender System Evaluations - Part 2
> Understanding evaluation metrics and pricing factors

- toc: true
- badges: true
- comments: true
- categories: [Evaluation]
- image:
<!-- #endregion -->

```python id="_pWvCiiDdjr_" executionInfo={"status": "ok", "timestamp": 1625636107973, "user_tz": -330, "elapsed": 898, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import  numpy  as  np 
import  pandas  as  pd 
import  math
```

<!-- #region id="XXtU4GFqdn3G" -->
## HR@K
<!-- #endregion -->

```python id="MwcAl3w3dm-w" executionInfo={"status": "ok", "timestamp": 1625636300047, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def hit_rate_at_k(recommended_list, bought_list, k=5): 
  bought_list = np.array(bought_list) 
  recommended_list = np.array(recommended_list)[:k]
  flags = np.isin(bought_list, recommended_list) 
  return (flags.sum() > 0) * 1
```

```python id="xEck0b3EeWCz" executionInfo={"status": "ok", "timestamp": 1625636373208, "user_tz": -330, "elapsed": 450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recommended_list = [156, 1134, 27, 1543, 3345, 143, 32, 533, 11, 43]  #items ids
bought_list = [521, 32, 143, 991]
```

```python colab={"base_uri": "https://localhost:8080/"} id="Mzw-ZKndeqrW" executionInfo={"status": "ok", "timestamp": 1625636498805, "user_tz": -330, "elapsed": 491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="94a28725-61fb-4167-ebe2-54fd33919d4e"
hit_rate_at_k(recommended_list, bought_list, 5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="a5M85f-ffGhO" executionInfo={"status": "ok", "timestamp": 1625636570984, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79b59344-0f3b-4000-960f-8480a029fe8d"
hit_rate_at_k(recommended_list, bought_list, 10)
```

<!-- #region id="h_d4yXPCgS67" -->
## Precision@K
<!-- #endregion -->

<!-- #region id="A3QxPFvUgcHC" -->
- Precision = (# of recommended items that are relevant) / (# of recommended items)
- Precision @ k = (# of recommended items @k that are relevant) / (# of recommended items @k)
- Money Precision @ k = (revenue of recommended items @k that are relevant) / (revenue of recommended items @k)
<!-- #endregion -->

```python id="gtweDIeDgZWw" executionInfo={"status": "ok", "timestamp": 1625637886933, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def precision_at_k(recommended_list, bought_list, k=5):
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list)[:k]
  
  flags = np.isin(bought_list, recommended_list)
  return flags.sum() / len(recommended_list)
```

```python id="2KWbIvmsglul" executionInfo={"status": "ok", "timestamp": 1625637884416, "user_tz": -330, "elapsed": 454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
  recommend_list = np.array(recommended_list)[:k] 
  prices_recommended = np.array(prices_recommended)[:k]
  flags = np.isin(recommend_list, bought_list)
  precision = np.dot(flags, prices_recommended) / prices_recommended.sum()
  return precision
```

```python id="w1IHDZ8dkZqk" executionInfo={"status": "ok", "timestamp": 1625637951529, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recommended_list = [156, 1134, 27, 1543, 3345, 143, 32, 533, 11, 43]  #items ids
bought_list = [521, 32, 143, 991]
prices_recommendede_list = [400, 60, 40, 90, 60, 340, 70, 190,110, 240]
```

```python colab={"base_uri": "https://localhost:8080/"} id="ISqhoHgwksM6" executionInfo={"status": "ok", "timestamp": 1625638081449, "user_tz": -330, "elapsed": 412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2a6b2232-a4d9-4436-f038-b026c7bdd9d0"
precision_at_k(recommended_list, bought_list, 5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6ukYqJRHkxGO" executionInfo={"status": "ok", "timestamp": 1625638083877, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="14f26a85-6e9e-4269-f3ee-5f154ffdb34c"
precision_at_k(recommended_list, bought_list, 10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gPi_uXyElLqD" executionInfo={"status": "ok", "timestamp": 1625638131340, "user_tz": -330, "elapsed": 763, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5cbeaee4-522e-4c34-e6f4-17b775348678"
money_precision_at_k(recommended_list, bought_list, prices_recommendede_list, 5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ee_aWCJhlLmz" executionInfo={"status": "ok", "timestamp": 1625638131341, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b2e73f90-577e-48ae-8d19-b22dfcd39bd3"
money_precision_at_k(recommended_list, bought_list, prices_recommendede_list, 10)
```

<!-- #region id="34bVpwgslYcp" -->
## Recall@K
<!-- #endregion -->

<!-- #region id="7MStCbMXlbiI" -->
- Recall = (# of recommended items that are relevant) / (# of relevant items)
- Recall @ k = (# of recommended items @k that are relevant) / (# of relevant items)
- Money Recall @ k = (revenue of recommended items @k that are relevant) / (revenue of relevant items)
<!-- #endregion -->

```python id="a8KPrGjllaWA" executionInfo={"status": "ok", "timestamp": 1625638356422, "user_tz": -330, "elapsed": 462, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recommended_list=[143,156,1134,991,27,1543,3345,533,11,43] #itemsid
prices_recommended_list=[400,60,40,90,60,340,70,190,110,240]

bought_list=[521,32,143,991]
prices_bought=[150,30,400,90]
```

```python id="Ni6hjugKlkkM" executionInfo={"status": "ok", "timestamp": 1625638211143, "user_tz": -330, "elapsed": 436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def recall_at_k(recommended_list, bought_list, k=5):
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list)[:k]
  
  flags = np.isin(bought_list, recommended_list)
  return flags.sum() / len(bought_list)
```

```python id="ZlrS7hZPloF9" executionInfo={"status": "ok", "timestamp": 1625638330271, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
  bought_list = np.array(bought_list)
  prices_bought = np.array(prices_bought)
  recommended_list = np.array(recommended_list)[:k]
  prices_recommended = np.array(prices_recommended)[:k]

  flags = np.isin(recommended_list, bought_list)
  return np.dot(flags, prices_recommended)/prices_bought.sum()
```

```python colab={"base_uri": "https://localhost:8080/"} id="f5fepPoqmP2l" executionInfo={"status": "ok", "timestamp": 1625638389315, "user_tz": -330, "elapsed": 415, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="65d4ae56-af27-4e86-9bc6-935ddccec26a"
recall_at_k(recommended_list, bought_list, 5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eiVBQBvfmP2m" executionInfo={"status": "ok", "timestamp": 1625638389739, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="72056673-b92f-44c3-e8ee-d275cec7d247"
recall_at_k(recommended_list, bought_list, 10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="8eYgatL-mP2n" executionInfo={"status": "ok", "timestamp": 1625638392207, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f02d8b2c-aa74-4c91-f7f7-67a8797ce183"
money_recall_at_k(recommended_list, bought_list, prices_recommendede_list, 5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="y-x5T3HOmP2o" executionInfo={"status": "ok", "timestamp": 1625638392208, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d294c3be-32ae-408e-c997-ae4ff6f22dfc"
money_recall_at_k(recommended_list, bought_list, prices_recommendede_list, 10)
```

<!-- #region id="FsR6hHRembdu" -->
## MAP@K
- MAP @ k (Mean Average Precision @ k )
- Average AP @ k for all users
<!-- #endregion -->

```python id="g1C1deyvmhba" executionInfo={"status": "ok", "timestamp": 1625638454859, "user_tz": -330, "elapsed": 473, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    
    amount_relevant = len(relevant_indexes)
            
    sum_ = sum([precision_at_k(recommended_list, bought_list, k=index_relevant+1) for index_relevant in relevant_indexes])
    return sum_/amount_relevant
```

```python id="x6m00Jc3mjWF" executionInfo={"status": "ok", "timestamp": 1625638455609, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def map_k(recommended_list, bought_list, k=5):

    amount_user = len(bought_list)
    list_ap_k = [ap_k(recommended_list[i], bought_list[i], k) for i in np.arange(amount_user)]
    
    sum_ap_k = sum(list_ap_k)  
    return sum_ap_k/amount_user
```

```python id="x8GehNJDmpKA" executionInfo={"status": "ok", "timestamp": 1625638518079, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
#list of 3 users
recommended_list_3_users = [[143,156,1134,991,27,1543,3345,533,11,43],
                            [1134,533,14,4,15,1543,1,99,27,3345],
                            [991,3345,27,533,43,143,1543,156,1134,11]]

bought_list_3_users= [[521,32,143], #user1
                      [143,156,991,43,11], #user2
                      [1,2]] #user3
```

```python colab={"base_uri": "https://localhost:8080/"} id="UnHTv6kvm0LX" executionInfo={"status": "ok", "timestamp": 1625638554271, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0dd9f665-5a8e-4bbb-9580-8b04619df0fd"
map_k(recommended_list_3_users, bought_list_3_users, 5)
```

<!-- #region id="OFt1EYJjm-6T" -->
## MRR@K
<!-- #endregion -->

```python id="JoOHG0rvnAUJ" executionInfo={"status": "ok", "timestamp": 1625638626316, "user_tz": -330, "elapsed": 413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def reciprocal_rank(recommended_list, bought_list, k=1):
    recommended_list = np.array(recommended_list)
    bought_list = np.array(bought_list)
    
    amount_user = len(bought_list)
    rr = []
    for i in np.arange(amount_user):    
        relevant_indexes = np.nonzero(np.isin(recommended_list[i][:k], bought_list[i]))[0]
        if len(relevant_indexes) != 0:
            rr.append(1/(relevant_indexes[0]+1))
    
    if len(rr) == 0:
        return 0
    
    return sum(rr)/amount_user
```

```python colab={"base_uri": "https://localhost:8080/"} id="hVhEiJ3NnGX_" executionInfo={"status": "ok", "timestamp": 1625638647501, "user_tz": -330, "elapsed": 437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6901d23-e342-47e5-96d1-db0c5b57a3af"
reciprocal_rank(recommended_list_3_users, bought_list_3_users, 5)
```

<!-- #region id="tlb3LmFunVEK" -->
## NDCG@K
<!-- #endregion -->

```python id="5blGg05wfSB8" executionInfo={"status": "ok", "timestamp": 1625638666358, "user_tz": -330, "elapsed": 559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def ndcg_at_k(recommended_list, bought_list, k=5):
    rec = recommended_list
    b = bought_list
    
    recommended_list = np.array(recommended_list)[:k]
    bought_list = np.array(bought_list)
    
    flags = np.isin(recommended_list, bought_list)
    rank_list = []
    for i in np.arange(len(recommended_list)):
        if i < 2:
            rank_list.append(i+1)
        else:
            rank_list.append(math.log2(i+1))
    if len(recommended_list) == 0:
        return 0
    dcg = sum(np.divide(flags, rank_list)) / len(recommended_list)

    i_dcg = sum(np.divide(1, rank_list)) / len(recommended_list)
#     print(i_dcg)
    return dcg/i_dcg
```

```python id="8XO4G7VvnXtd" executionInfo={"status": "ok", "timestamp": 1625638697567, "user_tz": -330, "elapsed": 635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
recommended_list = [143,156,1134,991,27,1543,3345,533,11,43] #iditems
prices_recommended_list = [400,60,40,90,60,340,70,190,110,240]

bought_list = [521,32,143,991]
prices_bought = [150,30,400,90]
```

```python colab={"base_uri": "https://localhost:8080/"} id="6fqcvbvMnfUC" executionInfo={"status": "ok", "timestamp": 1625638783457, "user_tz": -330, "elapsed": 412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a135ea43-30d6-48a4-af03-f4d68565f2dc"
ndcg_at_k(recommended_list, bought_list, 5)
```
