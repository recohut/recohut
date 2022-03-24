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

```python colab={"base_uri": "https://localhost:8080/"} id="XAZAGKscAtUb" executionInfo={"status": "ok", "timestamp": 1609254018430, "user_tz": -330, "elapsed": 1914, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ee82b1d-ff07-4f0a-f75b-b471899ffdbc"
import sys
sys.path.append("/content/drive/MyDrive")
import mykeys

project_name = "simple-docker-flask-app"
path = "/content/" + project_name
!mkdir "{path}"
%cd "{path}"

import sys
sys.path.append(path)

!git config --global user.email "<email>"
!git config --global user.name  "sparsh-ai"

!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/sparsh-ai/"{project_name}".git

!git pull origin master
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="IipnKEw6Gy71" executionInfo={"status": "ok", "timestamp": 1609255583540, "user_tz": -330, "elapsed": 1717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5105b95e-cf0b-4af8-f1ae-5198294fa006"
mykeys.git_token
```

```python colab={"base_uri": "https://localhost:8080/"} id="wXRrXKCYAq30" executionInfo={"status": "ok", "timestamp": 1609257124421, "user_tz": -330, "elapsed": 5068, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a1ee8f6b-b99a-41d7-e71c-ba3e99a1312d"
!git add .
!git commit -m 'commit'
!git push origin master
```

```python id="Wy-EHkSkC9KB" executionInfo={"status": "ok", "timestamp": 1609254561645, "user_tz": -330, "elapsed": 2144, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```

```python id="ewCv3FrPC96f" executionInfo={"status": "ok", "timestamp": 1609254566235, "user_tz": -330, "elapsed": 1301, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
```

```python id="mgssc2T3C_PQ" executionInfo={"status": "ok", "timestamp": 1609254576546, "user_tz": -330, "elapsed": 1294, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
model = LogisticRegression(solver='liblinear', random_state=0)
```

```python colab={"base_uri": "https://localhost:8080/"} id="D0d7ddT2DBwe" executionInfo={"status": "ok", "timestamp": 1609254587437, "user_tz": -330, "elapsed": 1383, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6a11bc3f-e94d-46e2-ba79-787bfe6577c8"
model.fit(x, y)
```

```python id="4gjmRCOjDEIX" executionInfo={"status": "ok", "timestamp": 1609254723466, "user_tz": -330, "elapsed": 1137, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pickle

with open("logreg.pkl", "wb") as pickle_out:
  pickle.dump(model, pickle_out)
```

```python id="xMw4RTskDuwA"
!pip install -q flasgger
```

```python colab={"base_uri": "https://localhost:8080/"} id="ibozyhckDGDM" executionInfo={"status": "ok", "timestamp": 1609255054429, "user_tz": -330, "elapsed": 1309, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5835fc59-38e5-45d2-a67c-56bfe55443e1"
%%writefile flask_api.py

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("logreg.pkl","rb")
model=pickle.load(pickle_in)

@app.route('/predict',methods=["Get"])
def predict_class():
  """Predict if Customer would buy the product or not.
  ---
  parameters:
    - name: input_x
      in: query
      type: number
      required: true
  responses:
      500:
          description: Prediction
  """
  x = int(request.args.get("input_x"))
  prediction=model.predict([[x]])
  print(prediction[0])
  return "Model prediction is"+str(prediction)

  if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
```

```python colab={"base_uri": "https://localhost:8080/"} id="2MQZoxrVFeTv" executionInfo={"status": "ok", "timestamp": 1609255411494, "user_tz": -330, "elapsed": 1515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="386af283-1e34-4da1-bd44-fa3d9f3333c1"
%%writefile requirements.txt

flasgger==0.9.5
Flask==1.1.2
numpy==1.19.4
pandas==1.1.5
```

```python colab={"base_uri": "https://localhost:8080/"} id="q2p9c6K8DsLy" executionInfo={"status": "ok", "timestamp": 1609256413988, "user_tz": -330, "elapsed": 1393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64618302-e603-482f-89ff-dd84a02c3ce5"
%%writefile Dockerfile

FROM continuumio/miniconda
COPY . /usr/ML/app
EXPOSE 5000
WORKDIR /usr/ML/app
RUN pip install -r requirements.txt
CMD python flask_api.py
```

```python id="p4NBAhZEGYgh" executionInfo={"status": "ok", "timestamp": 1609257107285, "user_tz": -330, "elapsed": 2859, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!cp "/content/drive/MyDrive/Colab Notebooks/simple-docker-flask-app.ipynb" .
```

<!-- #region id="tb7qsQjQH7G-" -->
---
<!-- #endregion -->

```python id="ja99l6WHGsA-"
# docker build -t ml_app_docker .
```

```python id="h3ApPzgMMhtW"
# docker container run -p 5000:5000 ml_app_docker
```
