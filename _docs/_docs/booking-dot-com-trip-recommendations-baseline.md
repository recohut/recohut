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
    language: python
    name: python3
---

# Booking.com trip recommendation part 1 - baseline model
> Booking.com challenge on trip recommendation part 1

- toc: true
- badges: true
- comments: true
- categories: [travel]
- image: 

```python id="YrHhkJNbghNP"
import pandas as pd
```

```python id="LR03pKu4hTyH"
!wget https://github.com/sparsh-ai/reco-data/raw/master/BookingChallenge.zip
!unzip BookingChallenge.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="L-wi74ZtgvVH" outputId="f6fcd4c5-ada2-4b88-9dba-5f449d7c9226"
train_set = pd.read_csv('train_set.csv').sort_values(by=['utrip_id','checkin'])

print(train_set.shape)
train_set.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="z1m-r3qLhH0x" outputId="91c73e93-2946-4448-c8f9-b7c7841be235"
test_set = pd.read_csv('test_set.csv').sort_values(by=['utrip_id','checkin'])

print(test_set.shape)
test_set.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="9QKI_nxuhtVP" outputId="7eaf4633-813b-41c7-fd36-46f31b344110"
# what are the top 4 most visited cities?
topcities = train_set.city_id.value_counts().index[:4]
topcities
```

```python colab={"base_uri": "https://localhost:8080/"} id="xz0---cgiG7X" outputId="6672ddf9-f1f6-42ad-fff1-a701056ecc9b"
# how many trips are there in the test set?
test_trips = (test_set[['utrip_id']].drop_duplicates()).reset_index().drop('index', axis=1)
len(test_trips)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ptKQJIE-iMnL" outputId="3635433b-a717-453d-cbad-6eef6d8892ca"
# baseline - a simple logical rule - recommend top 4 most visitied cities to everyone
cities_prediction = pd.DataFrame([topcities]*test_trips.shape[0],
                                 columns= ['city_id_1','city_id_2','city_id_3','city_id_4'])
cities_prediction[:5]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="I4kLpNQVirxr" outputId="35a862c1-8126-46d2-d043-ebf2ac43d84b"
predictions = pd.concat([test_trips, cities_prediction], axis=1)

print(predictions.shape)
predictions.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 252} id="OeOWrUdujVer" outputId="131db80f-2718-4c65-e01a-fdd7cb040001"
ground_truth = pd.read_csv('ground_truth.csv', index_col=[0])

print(ground_truth.shape)
ground_truth.head()
```

```python id="4YhzkrLnjgeo"
def evaluate_accuracy_at_4(predictions, ground_truth):
    '''checks if the true city is within the four recommended cities'''
    data = predictions.join(ground_truth, on='utrip_id')

    hits = ((data['city_id']==data['city_id_1'])|(data['city_id']==data['city_id_2'])|
        (data['city_id']==data['city_id_3'])|(data['city_id']==data['city_id_4']))*1
    return hits.mean()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8t0sGGLNjr6J" outputId="10bce096-8673-4343-adc0-05c4c27f39b1"
evaluate_accuracy_at_4(predictions, ground_truth)
```

```python id="8Oorct55jydh"

```
