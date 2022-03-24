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

```python colab={"base_uri": "https://localhost:8080/"} id="CBeab3ljhCr5" executionInfo={"status": "ok", "timestamp": 1628086504340, "user_tz": -330, "elapsed": 4541, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0421af33-f4a7-4857-c9e2-440636d13e0b"
import os
project_name = "reco-tut-mba"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="1jVaxI2Mh5cQ" executionInfo={"status": "ok", "timestamp": 1628086813929, "user_tz": -330, "elapsed": 577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
sys.path.insert(0,'./code')
```

```python id="YgCiBOCeaza_" executionInfo={"status": "ok", "timestamp": 1628087342258, "user_tz": -330, "elapsed": 588, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd

# local modules
from apriori import apriori
```

```python colab={"base_uri": "https://localhost:8080/", "height": 165} id="_wVjp5VebLMO" executionInfo={"status": "ok", "timestamp": 1628086707013, "user_tz": -330, "elapsed": 741, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="467bf347-6d22-4c2b-b621-245384557c15"
df = pd.read_csv('./data/grocery.csv', header=None)
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="QlnV6oBYjHHG" executionInfo={"status": "ok", "timestamp": 1628087034667, "user_tz": -330, "elapsed": 560, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2738b8a0-f9c0-4e6e-f5be-4529e26ef3a5"
df.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="QnpEBki3bXU4" executionInfo={"status": "ok", "timestamp": 1628085002595, "user_tz": -330, "elapsed": 617, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5b8d6717-acd7-410b-ee03-c8b35391f448"
!head -20 ./data/grocery.csv
```

```python id="_jARS3rAjDBg" executionInfo={"status": "ok", "timestamp": 1628087108242, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
transactions = []
list_of_products=[]
basket=[]
totalcol = df.shape[1]
```

```python id="Mv6i07h0jbGI" executionInfo={"status": "ok", "timestamp": 1628087112099, "user_tz": -330, "elapsed": 1686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
for i in range(0, len(df)):
    cart = []
    for j in range(0,totalcol):
        if str(df.values[i,j] ) != "nan":
            cart.append(str(df.values[i,j]))            
        if str(df.values[i,j]) not in list_of_products:
            list_of_products.append(str(df.values[i,j]))  
    transactions.append(cart)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 222} id="KusFTQZ_jcmK" executionInfo={"status": "ok", "timestamp": 1628087466549, "user_tz": -330, "elapsed": 772, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1b339986-7e3f-4106-ff31-adc43eaa8452"
', '.join(list_of_products)
```

```python colab={"base_uri": "https://localhost:8080/"} id="YpO0v6RxjlcK" executionInfo={"status": "ok", "timestamp": 1628087200639, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="53ae20ce-8fe4-4997-9ced-7c0044a9faaa"
[', '.join(x) for x in transactions[:10]]
```

```python id="uI_MkItPj5AY" executionInfo={"status": "ok", "timestamp": 1628087233366, "user_tz": -330, "elapsed": 1010, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.04, min_lift = 3)
results = list(rules)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xXTlGJJej6aD" executionInfo={"status": "ok", "timestamp": 1628087236416, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6675bb32-cbd9-4c83-9ff1-b64785a1a3e6"
results
```

```python id="U2drRtXXkC2-" executionInfo={"status": "ok", "timestamp": 1628088190863, "user_tz": -330, "elapsed": 703, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def recommendation(basket):    
    recommendations=[]
    prints = []
    for item in results:
        pair = item[0] 
        items = [x for x in pair]
        for product in basket:
            if items[0]==product:
                prints.append('Rule: {} -> {}'.format(items[0],items[1]))
                prints.append('Support: {}'.format(item[1]))
                prints.append('Confidence: {}'.format(str(item[2][0][2])))
                prints.append('{}'.format('-'*50))
                if items[1] not in recommendations:
                    recommendations.append(items[1])
    return recommendations, prints
```

```python id="vxtSMaflqo9Q" executionInfo={"status": "ok", "timestamp": 1628089078022, "user_tz": -330, "elapsed": 569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def recommend_randomly(nrec=2):
    count = 0
    while True:
        basket = np.random.choice(list_of_products,5)
        recs, prints = recommendation(basket)
        if recs:
            count+=1
            print('\n{}\n'.format('='*100))
            print('Basket:\n\t{}'.format('\n\t'.join(list(basket))))
            print('\nRecommendation:\n\t{}'.format('\n\t'.join(list(recs))))
            print('\n{}\n'.format('='*100))
            print('\n'.join(prints))
        if count>=nrec:
                break
```

```python colab={"base_uri": "https://localhost:8080/"} id="Te4n4H-on7G0" executionInfo={"status": "ok", "timestamp": 1628089086046, "user_tz": -330, "elapsed": 632, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="94cdec97-ce4a-439f-a56a-12b7aeb53e16"
recommend_randomly()
```

```python colab={"base_uri": "https://localhost:8080/"} id="lQOBTYU7rQOi" executionInfo={"status": "ok", "timestamp": 1628089172227, "user_tz": -330, "elapsed": 623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3bc7dcca-f910-4333-b028-9f3f60c8dee3"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="MxCapHPXrSNX" executionInfo={"status": "ok", "timestamp": 1628089416781, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1b2059f1-c93c-4425-b562-1f33d81e31bb"
!git add . && git commit -m 'commit' && git push origin main
```

```python colab={"base_uri": "https://localhost:8080/"} id="sd3mfTzYsNJ8" executionInfo={"status": "ok", "timestamp": 1628089432474, "user_tz": -330, "elapsed": 2313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e20ffa0a-383d-47d5-d3e1-6ca41effb7db"
!git push origin main
```

```python colab={"base_uri": "https://localhost:8080/"} id="hMRTVVnXr-xo" executionInfo={"status": "ok", "timestamp": 1628089387389, "user_tz": -330, "elapsed": 866, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a59ff8b6-0c51-4131-f419-0a3d88a61827"
%%writefile README.md
# reco-tut-mba

This repository contains tutorials related to Market Basket Analysis for Recommenders.
```
