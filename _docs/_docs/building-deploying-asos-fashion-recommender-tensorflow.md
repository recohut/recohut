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

<!-- #region id="sR6J3gYpqhoe" -->
# Building and deploying ASOS fashion recommender
> Building a fashion recommender using tensorflow and deploying using tensorflow serving

- toc: true
- badges: true
- comments: true
- categories: [Fashion, Tensorflow, Tensorflow Serving, Workshop, Vidoe Tutorial]
- image:
<!-- #endregion -->

<!-- #region id="8c-8imDMqFE0" -->
## Loading data
<!-- #endregion -->

```python id="kLzP5JelqtTf"
import numpy as np
import pandas as pd
import tensorflow as tf
import os
```

```python id="DVdPnbV-qtTk"
train = pd.read_parquet("https://raw.githubusercontent.com/ASOS/dsf2020/main/dsf_asos_train_with_alphanumeric_dummy_ids.parquet")
valid = pd.read_parquet("https://raw.githubusercontent.com/ASOS/dsf2020/main/dsf_asos_valid_with_alphanumeric_dummy_ids.parquet")
dummy_users = pd.read_csv("https://raw.githubusercontent.com/ASOS/dsf2020/main/dsf_asos_dummy_users_with_alphanumeric_dummy_ids.csv", header=None).values.flatten().astype(str)
products = pd.read_csv("https://raw.githubusercontent.com/ASOS/dsf2020/main/dsf_asos_productIds.csv", header=None).values.flatten().astype(int)
```

<!-- #region id="Jq5BLW6eQzl6" -->
## Beginners guide
<!-- #endregion -->

<!-- #region id="DmVZ0dgYqtUD" -->
The embedding layer gives a list of random numbers for each user and each product.
<!-- #endregion -->

```python id="yzIol1E3qtUE"
# we can think of this like we are representing 5 users with 8 features
embed1 = tf.keras.layers.Embedding(5, 8)

# these features values are initialized randomly
embed1.get_weights()

# what is the embedding for user 2
embed1(1)
```

<!-- #region id="u3tKDBNIqtUE" -->
Scores can be found using the dot product.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WRGiMOvbtZDY" outputId="c08a68f8-31bc-4a2d-a4f8-75b88327bcda"
dummy_users
```

```python id="PUxUdpC2qtUE"
dummy_user_embedding = tf.keras.layers.Embedding(len(dummy_users), 6)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5vFrOcrFtzF-" outputId="21c28637-11dc-433e-ce97-f13b1a436ad7"
# embedding for user 11
dummy_user_embedding(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="vxGI26ZSttt8" outputId="51198230-ba41-4a0c-d135-b3c5e2a54dda"
products
```

```python id="SM549-O8ttpi"
product_embedding = tf.keras.layers.Embedding(len(products), 6)
```

```python colab={"base_uri": "https://localhost:8080/"} id="SxgzRSxTtxAI" outputId="795cea52-dead-4ad8-d176-850fbbea8369"
# embedding for item 100
product_embedding(99)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5nxryt5PuJ_M" outputId="8927e985-55b3-4a2a-cb19-d826b633e989"
# what is the dot product of user 11 and item 100
tf.tensordot(dummy_user_embedding(10), product_embedding(99), axes=[[0],[0]])
```

<!-- #region id="fQkUd3iAqtUF" -->
We can score multiple products at the same time, which is what we need to create a ranking.
<!-- #endregion -->

```python id="8s-U6fZyqtUI"
# let's select any 4 products which we want to rank for a given user
example_product = tf.constant([1, 99, 150, 1893])
```

```python colab={"base_uri": "https://localhost:8080/"} id="MOZE8RPWvL3l" outputId="bdb46e6f-e38b-4d41-a70e-c38ba0e73f5d"
# we can now lookup the embeddings for these products
product_embedding(example_product)
```

```python colab={"base_uri": "https://localhost:8080/"} id="K83qARaVvWoL" outputId="f3d47fca-a16c-416a-8f29-9e346c4224bc"
# we can now rank the products for user 11
tf.tensordot(dummy_user_embedding(10), product_embedding(example_product), axes=[[0],[1]])
```

<!-- #region id="HGbJC1c1qtUJ" -->
And we can score multiple users for multiple products which we will need to do if we are to train quickly.
<!-- #endregion -->

```python id="s7CTor-_qtUL"
# let's select any 5 users
example_dummy_users = tf.constant([1, 15, 64, 143, 845])
```

```python colab={"base_uri": "https://localhost:8080/"} id="gst5AxIr0kWb" outputId="06e858d8-5c93-466a-f088-00b01fd0e5cf"
# we can now lookup the embeddings for these users
dummy_user_embedding(example_dummy_users)
```

```python colab={"base_uri": "https://localhost:8080/"} id="z_ebWIn90REd" outputId="725a8773-a9f2-41cd-c7cc-1f7efd83ffd1"
# we can now rank the products for all 5 users in one go
tf.tensordot(dummy_user_embedding(example_dummy_users), product_embedding(example_product), axes=[[1],[1]])
```

<!-- #region id="AhqTVjCQqtUO" -->
But we need to map product ids to embedding ids.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pFlvOHxNzdY2" outputId="f2ecc396-ac15-4596-b8f7-cf5a8705c1c2"
products
```

```python id="ASw6k_0HqtUQ"
product_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(products, dtype=tf.int32), 
                                        range(len(products))), -1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="o8B4KBxMy9Cx" outputId="fbd934fa-37e5-459b-e72f-8f274f288fd3"
# We can now ask the product table for a id of a product
product_table.lookup(tf.constant([9961521]))
```

<!-- #region id="FWWC2GrgQ7Z3" -->
## Model
<!-- #endregion -->

<!-- #region id="GyUJDr_KqtUV" -->
Let's put those two things together
<!-- #endregion -->

```python id="TA-deNCgqtUZ"
class SimpleRecommender(tf.keras.Model):
    def __init__(self, dummy_users, products, len_embed):
        super(SimpleRecommender, self).__init__()
        self.products = tf.constant(products, dtype=tf.int32)
        self.dummy_users = tf.constant(dummy_users, dtype=tf.string)
        self.dummy_user_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.dummy_users, range(len(dummy_users))), -1)
        self.product_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.products, range(len(products))), -1)
        
        self.user_embedding = tf.keras.layers.Embedding(len(dummy_users), len_embed)
        self.product_embedding = tf.keras.layers.Embedding(len(products), len_embed)

        self.dot = tf.keras.layers.Dot(axes=-1)
        
    # task idiom: "personalized ranking"
    # business use case: "recommended for you, mainly served on the home page"
    def call(self, inputs):
        users = inputs[0]
        products = inputs[1]

        user_embedding_index = self.dummy_user_table.lookup(users)
        product_embedding_index = self.product_table.lookup(products)

        user_embedding_values = self.user_embedding(user_embedding_index)
        product_embedding_values = self.product_embedding(product_embedding_index)

        return tf.squeeze(self.dot([user_embedding_values, product_embedding_values]))

    # task idiom: "item to item similarity"
    # business use case: "similar items you might like, mainly served on the product page"
    @tf.function
    def call_item_item(self, product):
        product_x = self.product_table.lookup(product)
        pe = tf.expand_dims(self.product_embedding(product_x), 0)
        
        all_pe = tf.expand_dims(self.product_embedding.embeddings, 0)#note this only works if the layer has been built!
        scores = tf.reshape(self.dot([pe, all_pe]), [-1])
        
        top_scores, top_indices = tf.math.top_k(scores, k=100)
        top_ids = tf.gather(self.products, top_indices)
        return top_ids, top_scores
```

```python colab={"base_uri": "https://localhost:8080/"} id="YQY-tX9j4OYb" outputId="1c7297cb-0ff6-4908-d1b5-fb8b5f7d91c6"
dummy_users
```

```python colab={"base_uri": "https://localhost:8080/"} id="qlgf9oXd4PuV" outputId="f4814af7-d7a8-4ea9-ce0f-a863eab2e15c"
products
```

```python colab={"base_uri": "https://localhost:8080/"} id="BQ2EK03F3R4t" outputId="e5a493b6-b381-4fbc-a63c-8fb11a628b53"
# let's sanity check the model
srl = SimpleRecommender(dummy_users, products, 15)

# let's check for 2 users and 3 products
srl([tf.constant([['pmfkU4BNZhmtLgJQwJ7x'], ['UDRRwOlzlWVbu7H8YCCi']]),
     tf.constant([[8650774,  9306139,  9961521], [12058614, 12058615, 11927550]])
     ])
```

<!-- #region id="lSatcB2FQqT_" -->
## Dataset
<!-- #endregion -->

<!-- #region id="LB5H5MGsqtUd" -->
First create a tf.data.Dataset from the user purchase pairs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="aySISv9v-Tuk" outputId="270535b9-f9a0-4a55-b6d8-bf6ec1838239"
train.head()
```

```python id="3svJogL0qtUf" colab={"base_uri": "https://localhost:8080/"} outputId="40a6b1ba-0d9c-46d4-c6ea-9e74803e6694"
dummy_user_tensor = tf.constant(train[["dummyUserId"]].values, dtype=tf.string)
product_tensor = tf.constant(train[["productId"]].values, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((dummy_user_tensor, product_tensor))
for x, y in dataset:
    print(x)
    print(y)
    break
```

<!-- #region id="rx2kdWPxqtUg" -->
For each purchase let's sample a number of products that the user did not purchase. Then the model can score each of the products and we will know we are doing a good job if the product with the highest score is the product that the user actually purchased.

We can do this using dataset.map
<!-- #endregion -->

```python id="H1hQWJfcqtUh"
class Mapper():
    
    def __init__(self, possible_products, num_negative_products):
        self.num_possible_products = len(possible_products)
        self.possible_products_tensor = tf.constant(possible_products, dtype=tf.int32)
        
        self.num_negative_products = num_negative_products
        self.y = tf.one_hot(0, num_negative_products+1)
    
    def __call__(self, user, product):
        random_negative_indices = tf.random.uniform((self.num_negative_products, ), minval=0, maxval=self.num_possible_products, dtype=tf.int32)
        negatives = tf.gather(self.possible_products_tensor, random_negative_indices)
        candidates = tf.concat([product, negatives], axis=0)
        return (user, candidates), self.y
```

```python colab={"base_uri": "https://localhost:8080/"} id="zSEb61inClMI" outputId="ec264c58-c4c8-4069-f410-25142324823d"
# let's sanity check the mapper fucntion
dataset = tf.data.Dataset.from_tensor_slices((dummy_user_tensor, product_tensor)).map(Mapper(products, 10))
for (u,c), y in dataset:
  print(u)
  print(c)
  print(y)
  break
```

<!-- #region id="jPEFhg2OEg-s" -->
Note: we are selecting negative samples from all products, which might include products that user in fact purchased, and this is going to add a factor of error but we are intentionally ignoring for now because this error factor is insignificant for the time being. Before productionizing, we will indeed take care of this.
<!-- #endregion -->

<!-- #region id="WET5T4LPqtUk" -->
Let's bring the steps together to define a function which creates a dataset 
<!-- #endregion -->

```python id="uZrHFLGxqtUl"
# let's wrap the dataset operations and check for a single user
def get_dataset(df, products, num_negative_products):
    dummy_user_tensor = tf.constant(df[["dummyUserId"]].values, dtype=tf.string)
    product_tensor = tf.constant(df[["productId"]].values, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_user_tensor, product_tensor))
    dataset = dataset.map(Mapper(products, num_negative_products))
    return dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="i6hU1x7MDTp6" outputId="2d886b41-7f7c-4cee-ffe9-bedb5cf10aa1"
for (u, c), y in get_dataset(train, products, 3):
  print(u)
  print(c)
  print(y)
  break
```

```python id="4wu5S1CPD-xv"
# let's now make it for a whole batch of users at a time that we will pass to the model
def get_dataset(df, products, num_negative_products):
    dummy_user_tensor = tf.constant(df[["dummyUserId"]].values, dtype=tf.string)
    product_tensor = tf.constant(df[["productId"]].values, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_user_tensor, product_tensor))
    dataset = dataset.map(Mapper(products, num_negative_products))
    dataset = dataset.batch(1024)
    return dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="ePx7iVQbD-sE" outputId="a260bfbe-91b1-4c9c-bfbb-b9bf1a364dc7"
for (u, c), y in get_dataset(train, products, 3):
  print(u)
  print(c)
  print(y)
  break
```

<!-- #region id="mGNhZ6BHQla_" -->
## Training
<!-- #endregion -->

<!-- #region id="UrAHAb-QqtUo" -->
We need to compile a model, set the loss and create an evaluation metric. Then we need to train the model.
<!-- #endregion -->

```python id="Rwh4r08fGBW0"
# we are using categorical cross entropy, which means we are formulating our task as a classification problem now
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 100 works well for this use case but we can make this as a hyperparameter to find a more optimal lr
optimizer = tf.keras.optimizers.SGD(learning_rate=100.)

metrics = [tf.keras.metrics.CategoricalAccuracy()]
```

```python id="_0F-ZL7QqtUp" colab={"base_uri": "https://localhost:8080/"} outputId="ea226a62-67f9-43f0-956b-56f91f81b501"
#hide-output
model = SimpleRecommender(dummy_users, products, 15)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(get_dataset(train, products, num_negative_products=100),
          validation_data=get_dataset(valid, products, 100),
          epochs=5)
```

<!-- #region id="3cmXxzSaqtUr" -->
Let's do a manual check on whether the model is any good.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="asBSeMzqK751" outputId="80a37490-a6b5-451b-a37c-68ddfcf15398"
products
```

```python id="BPJUxLO5qtUs" colab={"base_uri": "https://localhost:8080/"} outputId="51d84f48-72f5-4ca9-e402-67fdd61fd52b"
test_product = np.random.choice(products)
test_product
```

```python colab={"base_uri": "https://localhost:8080/"} id="IV0S3zvcKiMG" outputId="bc4e7d09-2ee5-42e4-9228-a73cfaf28399"
print("Go to https://www.asos.com/prd/{} to see the product description.".format(test_product))
```

<!-- #region id="NxrNZndpLTN4" -->
<!-- #endregion -->

```python id="H3YHeoswqtUt" colab={"base_uri": "https://localhost:8080/"} outputId="2fcb9e80-3cc7-4cc5-997a-4accaa70d7ec"
similar_recs = model.call_item_item(tf.constant(test_product, dtype=tf.int32))
print("Recs for item {}: {}".format(test_product, similar_recs))
```

```python colab={"base_uri": "https://localhost:8080/"} id="KFf1XB-VLu0Q" outputId="33989841-d513-4e5c-d5b8-809dd22451af"
print("The user also likes to purchase {}, and {}. Go to https://www.asos.com/prd/{}, https://www.asos.com/prd/{} to see the recommended product description."\
      .format(similar_recs[0][0].numpy(), similar_recs[0][1].numpy(),
              similar_recs[0][0].numpy(), similar_recs[0][1].numpy()))
```

<!-- #region id="f58XZPIuN1X9" -->
It seems people like to buy this t-shirt and sunglasses along with the shoes:


<!-- #endregion -->

<!-- #region id="ea2FIEx8PQmO" -->
Insight: Instead of learning user embeddings, we can say - "user embedding is the product sum of product embeddings the user purchased in the past". Then we can take the dot product of this user embedding with the product embeddings (which we can enhance by adding image and text features) to calculate the similarity.
<!-- #endregion -->

<!-- #region id="Mpqlu-9wRIv2" -->
## Save the model
<!-- #endregion -->

```python id="TL2KdkTUUAjx"
model_path = "models/recommender/1"
```

```python id="0UNgAKavUAdZ"
inpute_signature = tf.TensorSpec(shape=(), dtype=tf.int32)
```

```python id="mKR4GPBVUbox"
signatures = { 'call_item_item': model.call_item_item.get_concrete_function(inpute_signature)}
```

```python id="7p8BuTLQUqEn"
tf.saved_model.save(model, model_path, signatures=signatures)
```

```python id="YQpMPDkHZQ_F"
from zipfile import ZipFile
import os
# create a ZipFile object
with ZipFile('models.zip', 'w') as zipObj:
   # Iterate over all the files in directory
    for folderName, subfolders, filenames in os.walk("models"):
        for filename in filenames:
           #create complete filepath of file in directory
           filePath = os.path.join(folderName, filename)
           # Add file to zip
           zipObj.write(filePath)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5ASpnrMVfiA5" outputId="a19cfab5-7372-43fa-f2c2-230ddf2b50fc"
#hide-output
# let's examine the saved model by using the command line utility saved_model_cli
# to look at the MetaGraphDefs (the models) and SignatureDefs (the methods you
# can call) in our SavedModel

!saved_model_cli show --dir {model_path} --all
```

<!-- #region id="PnQ1ShpmVIv0" -->
## Load model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="g6W1QAeVUJBz" outputId="5a057e96-858f-4842-944a-cea0bf8768af"
imported_model = tf.saved_model.load(model_path)
list(imported_model.signatures.keys())
```

```python colab={"base_uri": "https://localhost:8080/"} id="8dpVf6xXZ0M0" outputId="f3973e57-2b4d-44c2-9890-21aa47e3e544"
products
```

```python id="o_8yTOFnUI8k"
result_tensor = imported_model.signatures['call_item_item'](tf.constant([8650774]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 521} id="VYYCqF78UI36" outputId="150c23f3-2652-44f4-b498-27ccd93e9ead"
#hide-output
from IPython.core.display import HTML

def path_to_image_html(path):
  return '<img src="https://images.asos-media.com/products/ugg-classic-mini-boots-in-black-suede/'+str(path)+'-2" width="60" >'

result_df = pd.DataFrame(result_tensor['output_0'].numpy(), columns=['ProductUrl']).head()
HTML(result_df.to_html(escape=False, formatters=dict(ProductUrl=path_to_image_html)))
```

<!-- #region id="_cEecY3jeSMK" -->
## Serve
<!-- #endregion -->

<!-- #region id="_S4WT-Hcg1PX" -->
### On colab
<!-- #endregion -->

```python id="BRaf64-hg4MI"
import sys
# We need sudo prefix if not on a Google Colab.
if 'google.colab' not in sys.modules:
  SUDO_IF_NEEDED = 'sudo'
else:
  SUDO_IF_NEEDED = ''
```

```python colab={"base_uri": "https://localhost:8080/"} id="--CU_mFQg4Ed" outputId="8a8f6ae3-e048-4057-d131-53dcdc6eff96"
# This is the same as you would do from your command line, but without the [arch=amd64], and no sudo
# You would instead do:
# echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

!echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | {SUDO_IF_NEEDED} tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | {SUDO_IF_NEEDED} apt-key add -
!{SUDO_IF_NEEDED} apt update
```

```python colab={"base_uri": "https://localhost:8080/"} id="pI7ajuvOg38i" outputId="df71883a-42ca-4d2f-eb38-8d52d0b8b279"
!{SUDO_IF_NEEDED} apt-get install tensorflow-model-server
```

<!-- #region id="KalOYKyehLH_" -->
This is where we start running TensorFlow Serving and load our model.  After it loads we can start making inference requests using REST.  There are some important parameters:

* `rest_api_port`: The port that you'll use for REST requests.
* `model_name`: You'll use this in the URL of REST requests.  It can be anything.
* `model_base_path`: This is the path to the directory where you've saved your model.

<!-- #endregion -->

```python id="dbYha4L5hQue"
os.environ["MODEL_PATH"] = "/content/models/recommender"
```

```bash colab={"base_uri": "https://localhost:8080/"} id="JvnN5y_QhY2c" outputId="c1faaf19-01c0-4fad-df49-0266fc8baca2" magic_args="--bg "
nohup tensorflow_model_server \
  --rest_api_port=8508 \
  --model_name=recommender \
  --model_base_path="${MODEL_PATH}" >server.log 2>&1
```

```python colab={"base_uri": "https://localhost:8080/"} id="mdyqkbvQhYvj" outputId="77dd57f0-bbaa-4be5-e8ce-3aaaac902950"
!tail server.log
```

<!-- #region id="d4IUY9sGgMie" -->
### Running tensorflow serving in local Docker container
<!-- #endregion -->

```python id="ONM89Y7NoIRk"
# # The recommended way of running Tensorflow serving is with Docker image.

# # Environment setup
# - docker engine installed and running to run a serve
#     General installation instructions are on the [Docker site](https://docs.docker.com/get-docker/), but some quick links here:
#     [Docker for macOS](https://docs.docker.com/docker-for-mac/install/)
#     [Docker for Windows](https://docs.docker.com/docker-for-windows/install/)
# - http client installed to run a client 
#     [Curl for mac](https://curl.haxx.se/dlwiz/?type=source&os=Mac+OS+X)  
  

# cd {recommender-model-folder}
# docker pull tensorflow/serving

# # Windows
# docker run -d -p 8501:8501 -v "$PWD/:/models/recommender" -e MODEL_NAME=recommender tensorflow/serving

# # Mac
# docker run -d -p 8501:8501 --mount type=bind,source=${PWD}/,target='/models/recommender' -e MODEL_NAME=recommender tensorflow/serving

# # Windows
# $rec_request = @"
# {
# "signature_name" : "call_item_item",
# "inputs" : {
# "item": [123123]
# }
# }
# "@
# $rec_response = Invoke-RestMethod -Uri "http://localhost:8501/v1/models/recommender:predict" -Method Post -Body $rec_request -ContentType "application/json"
# $rec_response | convertto-json

# # Mac
# curl --header "Content-Type: application/json" --request POST --data '{"signature_name":"call_item_item","inputs": {"item": [123123] } }' http://localhost:8501/v1/models/recommender:predict

# # Windows
# $output = Invoke-RestMethod http://localhost:8501/v1/models/recommender/metadata
# $output | convertto-json

# # Mac
# curl http://localhost:8501/v1/models/recommender/metadata
```

<!-- #region id="h6DCa8ePiOVw" -->
## Inference
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="SU0l72WheTAy" outputId="c570a769-2c90-40a5-c803-9c9b49241356"
import json
test_sample = json.dumps({"signature_name": "call_item_item", "inputs": {"product":[8650774]}})
test_sample
```

```python id="aNVarthgixIB"
import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8508/v1/models/recommender:predict', data=test_sample, headers=headers)
# json_response = requests.post('http://localhost:8508/v1/models/recommender/versions/1:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['outputs']
```

```python colab={"base_uri": "https://localhost:8080/"} id="k5VoWT_Gnnfs" outputId="128daa80-0075-4785-c675-a99c2339b385"
predictions['output_0'][0:10]
```
