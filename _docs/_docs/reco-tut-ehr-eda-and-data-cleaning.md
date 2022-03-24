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

```python id="3iExonjl1fDJ" executionInfo={"status": "ok", "timestamp": 1628596757706, "user_tz": -330, "elapsed": 455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-ehr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9PDMOfig1tw6" executionInfo={"status": "ok", "timestamp": 1628595231141, "user_tz": -330, "elapsed": 1132, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd93f016-e6c9-4369-a706-437bba4a48af"
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

```python id="z8LWxLdN1txA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628596754314, "user_tz": -330, "elapsed": 534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="69b7ccee-3012-400d-bfcf-ec023577533a"
!git status
```

```python id="9-Dgqtof1txB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628596765521, "user_tz": -330, "elapsed": 4012, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bc2dba59-d6c7-4a15-e839-64b1b7a70b02"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="zIQrefs61zkm" -->
---


<!-- #endregion -->

```python id="o69ZBeGP15gX"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!cd /content && kaggle competitions download -c expedia-hotel-recommendations
```

```python colab={"base_uri": "https://localhost:8080/"} id="GMeAfXsC1_8p" executionInfo={"status": "ok", "timestamp": 1628595406914, "user_tz": -330, "elapsed": 86044, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d7ba654e-988c-4e7c-96b6-9de68c50abf3"
!cd /content && unzip /content/expedia-hotel-recommendations.zip
```

```python id="yh0QUVj04cjK"
!pip install ml_metrics
```

```python id="TfSsoX822HU5"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_absolute_error
import ml_metrics as metrics

from datetime import timedelta
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="NrG6oHtI2Kjt" executionInfo={"status": "ok", "timestamp": 1628595899687, "user_tz": -330, "elapsed": 1787, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="40c4d037-2734-4ffa-ea83-717fb87734c2"
train = pd.read_csv('/content/train.csv', nrows=100000)
train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ghHO5PAt2PA4" executionInfo={"status": "ok", "timestamp": 1628595899689, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="41edaa3f-1545-4f7e-927a-a07ee4cfa78e"
train.info()
```

<!-- #region id="4Snvedo62P5K" -->
Train.csv have 2 more columns 'is_booking' and 'hotel_cluster' compared to Test.csv. Since we are predicting which hotel cluster a customer will book, only those data that make the booking would be relevant and hotel_cluster would be our target variable. The rest of the data where the user did not make a booking would be just noise.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="H2d0zEgi5EYS" executionInfo={"status": "ok", "timestamp": 1628596115135, "user_tz": -330, "elapsed": 455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b274048-a187-4a86-9367-db6341d5ac86"
bookings = train[train['is_booking'] == 1].drop('is_booking', axis = 1)
bookings.info()
```

<!-- #region id="rpy5kufh40Gt" -->
Notice that date_time, srch_ci, srch_co are objects, we will change this to datetime type and create 2 features - length of stay and dates to checkin
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="1uiVSZnM5ld_" executionInfo={"status": "ok", "timestamp": 1628596238166, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="49b57a58-157d-4225-cd70-bf571c7df98d"
# changing type to datetime
bookings['date_time'] = pd.to_datetime(bookings['date_time'], format = '%Y-%m-%d').dt.normalize()
bookings['srch_ci'] = pd.to_datetime(bookings['srch_ci'], format = '%Y-%m-%d')
bookings['srch_co'] = pd.to_datetime(bookings['srch_co'], format = '%Y-%m-%d')

# check for bookings later than the check in dates
bookings[bookings['srch_ci'] < bookings['date_time']][['date_time', 'srch_ci', 'srch_co']]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="xGJbsYKg5oPw" executionInfo={"status": "ok", "timestamp": 1628596244131, "user_tz": -330, "elapsed": 451, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9c6ffac3-2cf5-401d-82db-6337d78e52f7"
# check for check in dates later than the check out dates
bookings[bookings['srch_co'] < bookings['srch_ci']][['date_time', 'srch_ci', 'srch_co']]
```

<!-- #region id="41DrOtzF5qBG" -->
We can notice that there are a couple of dates that 'srch_ci' that are earlier than the 'date_time' but only by 1 day. It is probably because the user book the hotel just after midnight of because of time difference. We will adjust this by setting the 'date_time' to be on the same date as the 'srch_ci'

We also notice that there are 'srch_co' that are earlier than the 'srch_ci'. This could be possible because the dates are inversed. We will switch back the dates.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="WL3MiFun5KZc" executionInfo={"status": "ok", "timestamp": 1628596292884, "user_tz": -330, "elapsed": 637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5fe0c206-978f-45c2-9085-b0d9db088def"
# for those bookings that the 'date_time' is later than the 'srch_ci' timing, 
# we change the 'date_time' to 'srch_ci' timing as explained in the text cell above

bookings['date_time'] = np.where(bookings['srch_ci'] < bookings['date_time'],
                                 bookings['srch_ci'],
                                 bookings['date_time'])

#verifing the change take place
bookings[bookings['srch_ci'] < bookings['date_time']][['date_time', 'srch_ci', 'srch_co']]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49} id="ORXsQBih5xNE" executionInfo={"status": "ok", "timestamp": 1628596317853, "user_tz": -330, "elapsed": 519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9dfde19-71d5-49f8-f2cb-71952c3e1de6"
# for those bookings that the 'srch_ci' is later than the 'srch_out' timing, 
# we reverse them as explained in the text cell above

bookings['srch_ci'], bookings['srch_co'] = np.where(bookings['srch_co'] < bookings['srch_ci'],
                                                    (bookings['srch_co'], bookings['srch_ci']),
                                                    (bookings['srch_ci'], bookings['srch_co'])
                                                   )

bookings[bookings['srch_co'] < bookings['srch_ci']][['date_time', 'srch_ci', 'srch_co']]
```

```python colab={"base_uri": "https://localhost:8080/"} id="ragS6wV256jA" executionInfo={"status": "ok", "timestamp": 1628596331943, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a9d978d-9135-4874-bc2c-6b5d70e299b4"
for x in bookings.columns:
    print(f'{x}: {bookings[x].nunique()}')
```

<!-- #region id="mluiVRD96F2u" -->
Also notice that hotel_market has much more categories than hotel_country which in turn have more categories than hotel_continent. It is highly likely that hotel_market is a subset of hotel_country which is in turn a subset of hotel_continent. As such, only hotel_market is needed.

Similary for user_location_city, user_location_region and user_location_country, only user_location_city is needed
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 470} id="f0jN8YDS6GOl" executionInfo={"status": "ok", "timestamp": 1628596384872, "user_tz": -330, "elapsed": 483, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="598dc40d-a759-42e2-a1dc-a0eb4e08f2ed"
groupby_cluster = bookings.groupby('hotel_cluster').nunique()
groupby_cluster
```

```python id="fvkpMlpD6K-5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628596558732, "user_tz": -330, "elapsed": 1892, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0154c1bb-80b1-4eb0-b13a-b5adcaecf827"
!cd /content && git clone https://github.com/namkungchew/Recommendation-System-Expedia.git
```

```python id="DI6YnL4o61E0" executionInfo={"status": "ok", "timestamp": 1628596747431, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}

```

```python id="VhBfDwTq7ar5"

```
