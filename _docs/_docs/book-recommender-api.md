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

<!-- #region id="gVBiHPtWsxMP" -->
# Book Recommender API
> Converting book short description into vectors using Universal Sentence Encoder model and wrapping in an interactive Flask API with Front end HTML page

- toc: true
- badges: true
- comments: true
- categories: [Book, Flask, API, FrontEnd, NLP, TFHub, KNN]
- author: "<a href='https://github.com/staniher/kcaDeepRecommenderSystem'>staniher</a>"
- image:
<!-- #endregion -->

<!-- #region id="PnVizeduss0V" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Q6ukaNKTfraS" executionInfo={"status": "ok", "timestamp": 1625569553026, "user_tz": -330, "elapsed": 4951, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c76aa79d-3410-413d-df24-38d84cd22e09"
!pip install -q tensorflow_text
```

```python id="b2gBnmftfEuB" executionInfo={"status": "ok", "timestamp": 1625570199462, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import nltk
import json
import re
import csv
import pickle

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_hub as hub
import tensorflow_text
```

<!-- #region id="TTKCRg9bsq7O" -->
## Data loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 615} id="fxAe0x-FfK0j" executionInfo={"status": "ok", "timestamp": 1625569446683, "user_tz": -330, "elapsed": 736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a5b64d3-9865-46c7-87ac-04b77f156952"
data = pd.read_json('https://raw.githubusercontent.com/sparsh-ai/reco-data/master/books.json', lines=True)
data.head()
```

```python id="pGKqotkffQrQ" executionInfo={"status": "ok", "timestamp": 1625569568694, "user_tz": -330, "elapsed": 542, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = data[['title', 'authors', 'isbn','shortDescription','thumbnailUrl']].copy()
df['authors'] = df['authors'].str[0]
df.dropna(subset = ["shortDescription"], inplace=True)
```

<!-- #region id="8B5A8oBGf0T9" -->
## Encoding book description into vector using pre-trained USE model
<!-- #endregion -->

<!-- #region id="iXiP7DEMf9At" -->
<!-- #endregion -->

```python id="A3dnrVL3fyD6" executionInfo={"status": "ok", "timestamp": 1625569635744, "user_tz": -330, "elapsed": 7306, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
```

<!-- #region id="Ekz0yP8egDXd" -->
We convert our dataframe to a dictionnary where each row becomes a dictionary where key is column name and value is the data in the cell
<!-- #endregion -->

```python id="AvvghcdigAsE" executionInfo={"status": "ok", "timestamp": 1625569944955, "user_tz": -330, "elapsed": 4425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
full_data = df.to_dict('records')

# add vectors to the data
for element in full_data:
  element['vector'] = embed(element['shortDescription'])[0]
```

```python id="q7WcJ3WShMt_" executionInfo={"status": "ok", "timestamp": 1625569977332, "user_tz": -330, "elapsed": 487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
vectors = [item['vector'] for item in full_data]
X = np.array(vectors)
```

<!-- #region id="scCpEq16hNHN" -->
## Make Nearest Neighbor Model
<!-- #endregion -->

```python id="Fm_hy_3KhWQb" executionInfo={"status": "ok", "timestamp": 1625570004591, "user_tz": -330, "elapsed": 530, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# calculate similarity based on Euclidean distance
sim = euclidean_distances(X)
indices = np.vstack([np.argsort(-arr) for arr in sim])
```

```python id="kuRjJKGMhb8h" executionInfo={"status": "ok", "timestamp": 1625570039439, "user_tz": -330, "elapsed": 519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# calculate similarity based on cosine distance
cos_sim = cosine_similarity(X)
cos_indices = np.vstack([np.argsort(-arr) for arr in cos_sim])
```

```python id="YRpHEk9AhcsI" executionInfo={"status": "ok", "timestamp": 1625570107632, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# find top-k most similar books for each case
topk = 20
for i, book in enumerate(full_data):
  book['euclidean'] = indices[i][1:topk+1]
  book['cosine'] = cos_indices[i][1:topk+1]
```

<!-- #region id="o8zS-buBslYv" -->
## Model Serialization
<!-- #endregion -->

```python id="OGX7vmBlh1t6" executionInfo={"status": "ok", "timestamp": 1625570125141, "user_tz": -330, "elapsed": 791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# remove vectors from dict
for book in full_data:
  book.pop('vector')
```

```python colab={"base_uri": "https://localhost:8080/"} id="KanE9WKsh5tr" executionInfo={"status": "ok", "timestamp": 1625570130706, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="66e84dae-fc27-4576-fc57-531a5b926155"
full_data[0]
```

```python id="XIbk6D_nh7Xg" executionInfo={"status": "ok", "timestamp": 1625570203684, "user_tz": -330, "elapsed": 747, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# save the data
with open('model_01.pkl', 'wb') as f:
  pickle.dump(full_data, f)
```

<!-- #region id="xS0dIky8sZLP" -->
## Front-end Design
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7U8rNjaHiKYQ" executionInfo={"status": "ok", "timestamp": 1625570257400, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b7b9c919-8185-401d-f25d-b045a0ffea3b"
%%writefile Procfile
web: gunicorn app:app
```

```python colab={"base_uri": "https://localhost:8080/"} id="Nkh-Cq6niyYA" executionInfo={"status": "ok", "timestamp": 1625572017355, "user_tz": -330, "elapsed": 766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6741d98d-cc82-4f96-8919-a51d276f5725"
!mkdir templates static
```

```python colab={"base_uri": "https://localhost:8080/"} id="Kz3qrKO2mFr8" executionInfo={"status": "ok", "timestamp": 1625572017356, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e4722562-0379-4187-ae70-9e20b91bf594"
!wget -O ./static/logo.png https://images-platform.99static.com//JY78phRJ6tT1yo1QGGfhZOVlrAw=/68x2062:932x2926/fit-in/500x500/99designs-contests-attachments/87/87917/attachment_87917977
```

```python colab={"base_uri": "https://localhost:8080/"} id="mXa6q8sRioRx" executionInfo={"status": "ok", "timestamp": 1625572080381, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="caa3df9e-4566-4729-d40a-a1220a0c8ef0"
%%writefile ./templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Book recommendation</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</head>
<body class="bg-dark">
<style>
body {
	background-color:coral;
}
   #content {
        position: relative;
    }
    #content img {
        position: absolute;
        top: 0px;
        right: 25px;
    }
	 #content p {
        position: absolute;
        top: 150px;
        right: 0px;}
</style>
    <div class="p-3 text-white">
	<div id="content">
		<img src="{{ url_for('static', filename='logo.png') }}" width=150 class="ribbon"/>
	</div>
        <div class="row">
            <div class="col-6">
                <form method="post">
                    <div class="form-group">
                        <label for="sel2" style="font-size:25pt;color:yellow;font-style:bold;">CHOOSE A METRIC:</label>
                        <select class="form-control" id="sel2" name="selected_metric">
                            <option>cosine</option>
                            <option>euclidean</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="sel1" style="font-size:25pt;color:yellow;font-style:bold;">CHOOSE A BOOK:</label>
                        <select class="form-control" id="sel1" name="selected_title">
                            {% for title in list_books %}
                            <option>{{ title }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <input type="submit" class="btn btn-light" value="Recommender"/>
                </form>
            </div>
            <div class="col-6">
                {% if book_selected is defined %}
                <h2 style="font-size:11pt;color:red;font-style:bold;">SELECTED BOOK</h2>
                &nbsp;&nbsp;<img src="{{ book_selected.thumbnailUrl }}">
                {% endif %}
            </div>
        </div>
        {% if similar_books is defined %}
		<br/><br/>
        <h2>Here are your other reading suggestions:</h2>
        <div class="row">
            {% for book in similar_books %}
            <div class="col-2 p-3 d-flex justify-content-center">
                <img src="{{ book.thumbnailUrl }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
```

<!-- #region id="i2z9Uk2isU4d" -->
## Flask API
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oFp63ToyjY0x" executionInfo={"status": "ok", "timestamp": 1625570536060, "user_tz": -330, "elapsed": 4307, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e655404-086a-4bd9-9975-f2b71aef7379"
!pip install flask_ngrok
from flask_ngrok import run_with_ngrok
```

```python colab={"base_uri": "https://localhost:8080/"} id="9zq_6TtqiaOU" executionInfo={"status": "ok", "timestamp": 1625572031957, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e34f12e-1cd3-4047-f2c7-849e04157660"
%%writefile app.py
import pickle

from flask import Flask, request, render_template, jsonify
from flask_ngrok import run_with_ngrok
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)

# load data and extract all the vectors
with open('model_01.pkl', 'rb') as f:
  book_data = pickle.load(f)

list_books = sorted([book['title'] for book in book_data])
isbn_list = [item['isbn'] for item in book_data]


@app.route("/", methods=['GET', 'POST'])
def template_test():
    if request.method == 'POST':
        selected_title = request.form.get('selected_title')
        selected_metric = request.form.get('selected_metric')
        selected_book = next(item for item in book_data if item['title'] == selected_title)
        similar_books = [book_data[i] for i in selected_book[selected_metric]]
        return render_template('index.html',
                               list_books=list_books,
                               book_selected=selected_book,
                               similar_books=similar_books[:6])
    else:
        return render_template('index.html', list_books=list_books)


@app.route("/recommendations", methods=['GET'])
def get_recommendations():
    isbn = request.args.get('isbn', default=None, type=str)
    num_reco = request.args.get("number", default=5, type=int)
    distance = request.args.get("distance", default="cosine", type=str)
    field = request.args.get("field", default="isbn", type=str)
    if not isbn:
        return jsonify("Missing ISBN for the book"), 400
    elif distance not in ["cosine", "euclidean"]:
        return jsonify("Distance can only be cosine or euclidean"), 400
    elif num_reco not in range(1, 21):
        return jsonify("Can only request between 1 and 20 books"), 400
    elif isbn not in isbn_list:
        return jsonify("ISBN not in supported books"), 400
    elif field not in book_data[0].keys():
        return jsonify("Field not available in the data"), 400
    else:
        try:
            selected_book = next(item for item in book_data if item['isbn'] == isbn)
            similar_books = [book_data[i][field] for i in selected_book[distance]]
            return jsonify(similar_books[:num_reco]), 200
        except Exception as e:
            return jsonify(str(e)), 500


if __name__ == '__main__':
  app.run()
```

```python colab={"base_uri": "https://localhost:8080/"} id="U_4H90QLniO_" executionInfo={"status": "ok", "timestamp": 1625572498447, "user_tz": -330, "elapsed": 415828, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cb108b8b-99c7-4983-a1fa-4a95dcc7bae4"
!python app.py
```

<!-- #region id="3l0_ijxyq-O5" -->
## Evaluation
<!-- #endregion -->

<!-- #region id="9PBbEZ2mrQH1" -->
> Tip: Ignore the selection dropdowns in below images. They gets reset after clicking on Recommender button. 
<!-- #endregion -->

<!-- #region id="wBPj1scgq_-s" -->
### Cosine vs Euclidean for *Android in Practice* book
<!-- #endregion -->

<!-- #region id="MjXFiJSSreIm" -->
**Cosine**
<!-- #endregion -->

<!-- #region id="0lv6EN-kqPuz" -->
<!-- #endregion -->

<!-- #region id="B7WoByDmrilt" -->
**Euclidean**
<!-- #endregion -->

<!-- #region id="L7TEIe_ZqhPb" -->
<!-- #endregion -->

<!-- #region id="dM_lMIN4rknS" -->
Cosine performed much better.
<!-- #endregion -->

<!-- #region id="i0bgQ9y0rsZK" -->
### Cosine vs Euclidean for *Scala in Depth* book
<!-- #endregion -->

<!-- #region id="JF-Nf6Vsrv64" -->
**Cosine**
<!-- #endregion -->

<!-- #region id="xzOQZ9Y4qr4E" -->
<!-- #endregion -->

<!-- #region id="ft7GXA5MrxTH" -->
**Euclidean**
<!-- #endregion -->

<!-- #region id="H73m2_2vq2ha" -->
<!-- #endregion -->

<!-- #region id="xtzkS5JTr0rC" -->
There are not much Scala related books. So Cosine recommender suggest Groovy, Java books. But Euclidean recommender suggesting Coffeehouse, Client server books. In this case also, cosine clearly outperformed euclidean.
<!-- #endregion -->
