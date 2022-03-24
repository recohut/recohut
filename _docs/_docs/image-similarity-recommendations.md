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

<!-- #region id="mdcFFjIG8dot" -->
# Similar Product Recommendations
> A tutorial on building a recommender system that will find similar looking products

- toc: true
- badges: true
- comments: true
- categories: [similarity, visual]
- image: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MVHAW10n1F74" outputId="1bf9840e-03c4-4760-f45d-ef067a666ee9"
#hide
from google.colab import drive
drive.mount('/content/drive')
```

```python colab={"base_uri": "https://localhost:8080/"} id="KqMPSQ8xsTbu" outputId="09d0e696-5c5a-45bb-a177-ac3c47ec9dfe"
#hide
%%writefile kaggle.json
{"username":"<your kaggle username>","key":"<your kaggle api key>"}
```

```python colab={"base_uri": "https://localhost:8080/"} id="j6tw-MsbsYpt" outputId="33c41e38-802e-4f05-eb84-ff6ac3eee4ba"
#hide
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

<!-- #region id="nyI8Bl1d4t1Y" -->
<!-- #endregion -->

<!-- #region id="8GlqfNWm6ciB" -->
### Choice of variables

- Image Encoder: Any pre-trained image classification model can be selected. These models are commonly known as encoders because their job is to encode an image into a feature vector. In our case, we analyzed four encoders named 1) MobileNet, 2) EfficientNet, 3) ResNet and 4) [BiT](https://tfhub.dev/google/bit/m-r152x4/1). After some basic research, we decided to select BiT model because of its performance. I selected the BiT-M-50x3 variant of model which is of size 748 MB. More details about this architecture can be found on the official page [here](https://tfhub.dev/google/bit/m-r50x3/1).
- Vector Similarity System: Images are represented in a fixed-length feature vector format. For the given input vector, we need to find the TopK most similar vectors, keeping the memory efficiency and real-time retrival objective in mind. We explored the most popular techniques and listed down five of them: Annoy, Cosine distance, L1 distance, Locally Sensitive Hashing (LSH) and Image Deep Ranking. We selected Annoy because of its fast and efficient nature. More details about Annoy can be found on the official page [here](https://github.com/spotify/annoy).
- Dataset: This system is able to handle all kind of image dataset. Only the basic preprocessing (in step 1) would need some modifications depending on the dataset. We chose [Fashion Product Images (Small)](https://www.kaggle.com/bhaskar2443053/fashion-small?). Other examples can be [Food-11 image dataset](https://www.kaggle.com/trolukovich/food11-image-dataset?), and [Caltech 256 Image Dataset](https://www.kaggle.com/jessicali9530/caltech256?).
<!-- #endregion -->

<!-- #region id="ushnAKJ5VyfI" -->
### Step 1: Data Acquisition
<!-- #endregion -->

<!-- #region id="SdAaHU-x5R7U" -->
Download the raw image dataset into a directory. Categorize these images into their respective category directories. Make sure that images are of the same type, JPEG recommended. We will also process the metadata and store it in a serialized file, CSV recommended. 
<!-- #endregion -->

```python id="Q72U45CZtzAZ"
#hide-output
# downloading raw images from kaggle
!kaggle datasets download -d paramaggarwal/fashion-product-images-small
!unzip fashion-product-images-small.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="xS6CNOmrUfmi" outputId="ce79c260-8082-42f7-b81f-d5685654ba2b"
import pandas as pd
from shutil import move
import os
from tqdm import tqdm

os.mkdir('/content/Fashion_data')
os.chdir('/content/Fashion_data')

df = pd.read_csv('/content/styles.csv', usecols=['id','masterCategory']).reset_index()
df['id'] = df['id'].astype('str')

all_images = os.listdir('/content/images/')
co = 0
os.mkdir('/content/Fashion_data/categories')
for image in tqdm(all_images):
    category = df[df['id'] == image.split('.')[0]]['masterCategory']
    category = str(list(category)[0])
    if not os.path.exists(os.path.join('/content/Fashion_data/categories', category)):
        os.mkdir(os.path.join('/content/Fashion_data/categories', category))
    path_from = os.path.join('/content/images', image)
    path_to = os.path.join('/content/Fashion_data/categories', category, image)
    move(path_from, path_to)
    co += 1
print('Moved {} images.'.format(co))
```

<!-- #region id="dx-LREyDWEYK" -->
### Step 2: Encoder Fine-tuning [optional]
<!-- #endregion -->

<!-- #region id="TEYVHxkN5U_K" -->
Download the pre-trained image model and add two additional layers on top of that: the first layer is a feature vector layer and the second layer is the classification layer. We will only train these 2 layers on our data and after training, we will select the feature vector layer as the output of our fine-tuned encoder. After fine-tuning the model, we will save the feature extractor for later use.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gdpzJdFpUfiu" outputId="d97f458e-5054-45cf-c890-bfd59bcbb93a"
import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
```

```python colab={"base_uri": "https://localhost:8080/"} id="F5D6EK18Ufg_" outputId="9961fdde-8972-46bc-f07b-64ce2c3ffa8e"
MODULE_HANDLE = 'https://tfhub.dev/google/bit/m-r50x3/1'
IMAGE_SIZE = (224, 224)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
BATCH_SIZE = 32 
N_FEATURES = 256
```

```python id="6W9wx6VvUfff"
#hide
data_dir = '/content/Fashion_data/categories'
```

```python colab={"base_uri": "https://localhost:8080/"} id="LGyqBujZUfd0" outputId="b918eebd-e801-4b4e-ff24-4d0c4afed25f"
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = False 
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="acJCBUeiUfcC" outputId="6304df64-b293-47ff-a2b4-594c8f687410"
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(N_FEATURES,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()
```

```python id="g2CMlRLFUfZC"
# Define optimiser and loss
lr = 0.003 * BATCH_SIZE / 512 
SCHEDULE_LENGTH = 500
SCHEDULE_BOUNDARIES = [200, 300, 400]

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES, 
                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="i8lBtDaOUfWD" outputId="4214c3dd-0e5c-415a-d9ff-6ef83f4ccadc"
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history
```

```python colab={"base_uri": "https://localhost:8080/", "height": 566} id="NJKzJ1q1W6BO" outputId="cafa44c3-d4ad-4245-ce65-86859c113fc8"
#hide
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
```

```python colab={"base_uri": "https://localhost:8080/"} id="NY6gNBhyW5-5" outputId="b6236a5b-0694-44a9-bdf8-4b2204e6699e"
if not os.path.exists('/content/drive/MyDrive/ImgSim/'):
    os.mkdir('/content/drive/MyDrive/ImgSim/')

feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)
feature_extractor.save('/content/drive/MyDrive/ImgSim/bit_feature_extractor', save_format='tf')

saved_model_path = '/content/drive/MyDrive/ImgSim/bit_model'
tf.saved_model.save(model, saved_model_path)
```

<!-- #region id="QoymCsSMXOeu" -->
### Step 3: Image Vectorization
Now, we will use the encoder (prepared in step 2) to encode the images (prepared in step 1). We will save feature vector of each image as an array in a directory. After processing, we will save these embeddings for later use.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VbWu23iuW56S" outputId="f74fb73f-5929-4d27-e890-df0c88e6ee8d"
#hide
import tensorflow as tf
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
tqdm.pandas()
```

```python id="_USmBUpiW58t"
img_paths = []
for path in Path('/content/Fashion_data/categories').rglob('*.jpg'):
  img_paths.append(path)
np.random.shuffle(img_paths)
```

```python id="992vULeyW5zD"
def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img
```

```python colab={"base_uri": "https://localhost:8080/"} id="2B8cJuQgaT1r" outputId="8a6baf17-1403-4ac2-9004-6387d6e33910"
#hide-output
TRANSFER_LEARNING_FLAG = 1
if TRANSFER_LEARNING_FLAG:
  module = tf.keras.models.load_model('/content/drive/MyDrive/ImgSim/bit_feature_extractor')
else:
  module_handle = "https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1" 
  module = hub.load(module_handle)
```

```python id="WtfPQPWIW5wD"
imgvec_path = '/content/img_vectors/'
Path(imgvec_path).mkdir(parents=True, exist_ok=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1kLn4sOyW5sx" outputId="08e36c9e-406e-4e47-8871-2b54740694fd"
for filename in tqdm(img_paths[:5000]):
    img = load_img(str(filename))
    features = module(img)
    feature_set = np.squeeze(features)
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    out_path_file = os.path.join(imgvec_path, outfile_name)
    np.savetxt(out_path_file, feature_set, delimiter=',')
```

<!-- #region id="UsSVlCOhf5b0" -->
### Step 4: Metadata and Indexing
<!-- #endregion -->

<!-- #region id="iUMsk_kd5bPU" -->
We will assign a unique id to each image and create dictionaries to locate information of this image: 1) Image id to Image name dictionary, 2) Image id to image feature vector dictionary, and 3) (optional) Image id to metadata product id dictionary. We will also create an image id to image feature vector indexing. Then we will save these dictionaries and index object for later use.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="21NzEoTegKxZ" outputId="31b52417-aa6c-4363-e568-1be0c2059b1a"
#hide
import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm
tqdm.pandas()
!pip install -q annoy
import json
from annoy import AnnoyIndex
from scipy import spatial
import pickle
from IPython.display import Image as dispImage
```

```python colab={"base_uri": "https://localhost:8080/", "height": 97} id="9jv-lnwzHvlC" outputId="5f72ac85-c3fa-4e69-9305-bc3f4058b32b"
test_img = '/content/Fashion_data/categories/Accessories/1941.jpg'
dispImage(test_img)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1niXWpbXW5m3" outputId="98c52ad2-e0d8-4252-9f38-cc0df278b667"
#hide-output
styles = pd.read_csv('/content/styles.csv', error_bad_lines=False)
styles['id'] = styles['id'].astype('str')
styles.to_csv(root_path+'/styles.csv', index=False)
```

```python id="W-nFnff8W5j7"
def match_id(fname):
  return styles.index[styles.id==fname].values[0]
```

```python id="Fq594UchW5hz"
# Defining data structures as empty dict
file_index_to_file_name = {}
file_index_to_file_vector = {}
file_index_to_product_id = {}

# Configuring annoy parameters
dims = 256
n_nearest_neighbors = 20
trees = 10000

# Reads all file names which stores feature vectors 
allfiles = glob.glob('/content/img_vectors/*.npz')

t = AnnoyIndex(dims, metric='angular')
```

```python colab={"base_uri": "https://localhost:8080/"} id="OvKGnMU-W5fh" outputId="0787700d-abec-44af-82c6-742da632ead3"
for findex, fname in tqdm(enumerate(allfiles)):
  file_vector = np.loadtxt(fname)
  file_name = os.path.basename(fname).split('.')[0]
  file_index_to_file_name[findex] = file_name
  file_index_to_file_vector[findex] = file_vector
  try:
    file_index_to_product_id[findex] = match_id(file_name)
  except IndexError:
    pass 
  t.add_item(findex, file_vector)
```

```python colab={"base_uri": "https://localhost:8080/"} id="PHB-JNObW5cs" outputId="7cbe28ed-e6cf-4392-c6a5-750983234c0f"
#hide-output
t.build(trees)
t.save('t.ann')
```

```python id="c-6vhz4Dhjz3"
#hide
file_path = '/content/drive/MyDrive/ImgSim/'
```

```python id="WKzVD4PdW5ZD"
t.save(file_path+'indexer.ann')
pickle.dump(file_index_to_file_name, open(file_path+"file_index_to_file_name.p", "wb"))
pickle.dump(file_index_to_file_vector, open(file_path+"file_index_to_file_vector.p", "wb"))
pickle.dump(file_index_to_product_id, open(file_path+"file_index_to_product_id.p", "wb"))
```

<!-- #region id="WQ9BN_MsiwSy" -->
## Step 5: Local Testing
<!-- #endregion -->

<!-- #region id="6k1SHCwd5fO8" -->
We will load a random image and find top-K most similar images.
<!-- #endregion -->

```python id="QK_ChEiYf1n0"
#hide
from PIL import Image
import matplotlib.image as mpimg
```

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="V32QozZfHtDU" outputId="da444127-620d-471e-a316-c6801362c2fd"
img_addr = 'https://images-na.ssl-images-amazon.com/images/I/81%2Bd6eSA0eL._UL1500_.jpg'

!wget -q -O img.jpg $img_addr
test_img = 'img.jpg'
topK = 4

test_vec = np.squeeze(module(load_img(test_img)))

basewidth = 224
img = Image.open(test_img)
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img
```

```python colab={"base_uri": "https://localhost:8080/", "height": 381} id="HFs56RynGrjO" outputId="08c14364-79de-4d4e-8dfc-b17e4c4d4b0d"
path_dict = {}
for path in Path('/content/Fashion_data/categories').rglob('*.jpg'):
  path_dict[path.name] = path

nns = t.get_nns_by_vector(test_vec, n=topK)
plt.figure(figsize=(20, 10))
for i in range(topK):
  x = file_index_to_file_name[nns[i]]
  x = path_dict[x+'.jpg']
  y = file_index_to_product_id[nns[i]]
  title = '\n'.join([str(j) for j in list(styles.loc[y].values[-5:])])
  plt.subplot(1, topK, i+1)
  plt.title(title)
  plt.imshow(mpimg.imread(x))
  plt.axis('off')
plt.tight_layout()
```

<!-- #region id="7-pbfJyqkS9T" -->
## Step 6: API Call
<!-- #endregion -->

<!-- #region id="lATZdqAI5iQU" -->
We will build two kinds of API - 1) UI based API using Streamlit, and 2) REST API using Flask. In both the cases, we will receive an image from user, encode it with our image encoder, find TopK similar vectors using Indexing object, and retrieve the image (and metadata) using dictionaries. We send these images (and metadata) back to the user.
<!-- #endregion -->

```python id="bEuchptAlDIK"
#hide
import os
import time
```

```python id="4xqniAiWk-da"
#hide
root_path = '/content/drive/MyDrive/ImgSim'
```

```python colab={"base_uri": "https://localhost:8080/"} id="kUXYz5QZBiDG" outputId="a6a4fa70-7924-4ad4-9c4b-5042836ee97c"
%%writefile utils.py
### ----utils.py---- ###

import os
import tensorflow as tf
from annoy import AnnoyIndex

root_path = '/content/drive/MyDrive/ImgSim'

class Encoder:
  encoder = tf.keras.models.load_model(os.path.join(root_path, 'bit_feature_extractor'))

class Indexer:
  dims = 256
  topK = 6
  indexer = AnnoyIndex(dims, 'angular')
  indexer.load(os.path.join(root_path, 'indexer.ann'))
  
encoder = Encoder()
indexer = Indexer()
```

<!-- #region id="OSO01jdL5rHK" -->
### Streamlit App
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d2xGrUSvyAHt" outputId="c6865549-15e6-4342-de56-202c021cc913"
%%writefile app.py
### ----app.py---- ###

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from annoy import AnnoyIndex
import glob
import os
import tensorflow as tf
import tarfile
import pickle
from pathlib import Path
import time
from utils import encoder, indexer

root_path = '/content/drive/MyDrive/ImgSim'

start_time = time.time()
encoder = encoder.encoder
print("---Encoder--- %s seconds ---" % (time.time() - start_time))

topK = 6

start_time = time.time()
t = indexer.indexer
print("---Indexer--- %s seconds ---" % (time.time() - start_time))

# load the meta data
meta_data = pd.read_csv(os.path.join(root_path, 'styles.csv'))

# load the mappings
file_index_to_file_name = pickle.load(open(os.path.join(root_path ,'file_index_to_file_name.p'), 'rb'))
file_index_to_file_vector = pickle.load(open(os.path.join(root_path ,'file_index_to_file_vector.p'), 'rb'))
file_index_to_product_id = pickle.load(open(os.path.join(root_path ,'file_index_to_product_id.p'), 'rb'))

# load image path mapping
path_dict = {}
for path in Path('/content/Fashion_data/categories').rglob('*.jpg'):
  path_dict[path.name] = path

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

query_path = '/content/user_query.jpg'

st.title("Image Similarity App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save(query_path)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Top similar images...")
    
    start_time = time.time()
    test_vec = np.squeeze(encoder(load_img(query_path)))
    print("---Encoding--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    nns = t.get_nns_by_vector(test_vec, n=topK)
    print("---SimilarityIndex--- %s seconds ---" % (time.time() - start_time))

    img_files = []
    img_captions = []

    start_time = time.time()
    for i in nns:
      #image files
      img_path = str(path_dict[file_index_to_file_name[i]+'.jpg'])
      img_file = Image.open(img_path)
      img_files.append(img_file)
      #image captions
      item_id = file_index_to_product_id[i]
      img_caption = '\n'.join([str(j) for j in list(meta_data.loc[item_id].values[-5:])])
      img_captions.append(img_caption)
    print("---Output--- %s seconds ---" % (time.time() - start_time))

    st.image(img_files, caption=img_captions, width=200)
```

```python id="HTGShJ4HvhOr"
! pip install -q pyngrok
! pip install -q streamlit
! pip install -q colab-everything

from colab_everything import ColabStreamlit
ColabStreamlit('app.py')
```

<!-- #region id="GUAHHR5ZjWi1" -->
### Flask API
<!-- #endregion -->

<!-- #region id="IlPaLLHq4ENq" -->
#### server-side
<!-- #endregion -->

```python id="5fahnFX0lWUp"
%%writefile flask_app.py
### ----flask_app.py---- ###

import pandas as pd
import numpy as np
from PIL import Image
from annoy import AnnoyIndex
import glob
import os
import tensorflow as tf
import tarfile
import pickle
from pathlib import Path
import time
from utils import encoder, indexer
import io
import base64

from flask import Flask, request, jsonify, send_file

_PPATH = '/content/drive/MyDrive/ImgSim/'

start_time = time.time()
encoder = encoder.encoder
print("---Encoder--- %s seconds ---" % (time.time() - start_time))

topK = 6

start_time = time.time()
t = indexer.indexer
print("---Indexer--- %s seconds ---" % (time.time() - start_time))

# load the meta data
meta_data = pd.read_csv(_PPATH+'styles.csv')

# load the mappings
file_index_to_file_name = pickle.load(open(_PPATH+'file_index_to_file_name.p', 'rb'))
file_index_to_file_vector = pickle.load(open(_PPATH+'file_index_to_file_vector.p', 'rb'))
file_index_to_product_id = pickle.load(open(_PPATH+'file_index_to_product_id.p', 'rb'))

# load image path mapping
path_dict = {}
for path in Path('/content/Fashion_data/categories').rglob('*.jpg'):
  path_dict[path.name] = path

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

query_path = '/content/user_query.jpg'

def get_encoded_img(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img


app = Flask(__name__)

@app.route("/fashion", methods=["POST"])
def home():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    img.save(query_path)

    start_time = time.time()
    test_vec = np.squeeze(encoder(load_img(query_path)))
    print("---Encoding--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    nns = t.get_nns_by_vector(test_vec, n=topK)
    print("---SimilarityIndex--- %s seconds ---" % (time.time() - start_time))

    img_files = {}
    img_captions = {}

    start_time = time.time()
    for count, i in enumerate(nns):
      #image files
      img_path = str(path_dict[file_index_to_file_name[i]+'.jpg'])
      img_file = Image.open(img_path)
      img_files[count] = get_encoded_img(img_file)
      #image captions
      item_id = file_index_to_product_id[i]
      img_caption = '\n'.join([str(j) for j in list(meta_data.loc[item_id].values[-5:])])
      img_captions[count] = img_caption
    print("---Output--- %s seconds ---" % (time.time() - start_time))

    return jsonify(img_files)

app.run(debug=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="X0ktfvwxlWQt" outputId="3faa21d0-362e-4936-cc9c-a8fb33d44a0b"
!nohup python3 -u flask_app.py &
```

<!-- #region id="k63_TOdX3_jU" -->
#### client-side
<!-- #endregion -->

```python id="DX7NLTjUIx-4"
from PIL import Image
from io import BytesIO
import base64
import requests

!wget -O 'img.jpg' -q 'https://images-na.ssl-images-amazon.com/images/I/61utX8kBDlL._UL1100_.jpg'

url = 'http://localhost:5000/fashion'
my_img = {'image': open('img.jpg', 'rb')}
r = requests.post(url, files=my_img)

imgs = []
for i in range(6):
  img = base64.decodebytes(r.json()[str(i)].encode('ascii'))
  img = Image.open(BytesIO(img)).convert('RGBA')
  imgs.append(img)

imgs[1]
```

<!-- #region id="dasr882iulPo" -->
## Step 7: Deployment on AWS Elastic BeanStalk
<!-- #endregion -->

<!-- #region id="mlGzPLof6AQ5" -->
The API was deployed on AWS cloud infrastructure using AWS Elastic Beanstalk service.
<!-- #endregion -->

<!-- #region id="8MJlMv2C6EFi" -->
<!-- #endregion -->

<!-- #region id="K-flZYOU3UHC" -->
### application.py
<!-- #endregion -->

```python id="AQFwCf4Du_w6"
#collapse
%%writefile ./ebsapp/application.py
import os
import zipfile
import requests
from tqdm import tqdm
from shutil import move
from pandas import read_csv
from pathlib import Path
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import glob
import json
from tqdm import tqdm
from annoy import AnnoyIndex
from scipy import spatial
import pickle
import time
from PIL import Image
import tarfile
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask import redirect, url_for, flash, render_template

# path = Path(__file__)
# _PPATH = str(path.parents[1])+'/'
_PPATH = os.path.join(os.getcwd(), 'mytemp')

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

module_handle = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
module = hub.load(module_handle)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

topK = 5
threshold = 0.3
UPLOAD_FOLDER = _PPATH
ALLOWED_EXTENSIONS = set(['zip'])

application = Flask(__name__)
           
@application.route('/')
def hello_world():
    return "Hello world!"

@application.route('/original_images', methods=['POST'])
def upload_zip1():
  os.makedirs(_PPATH, exist_ok=True)
  shutil.rmtree(_PPATH)
  os.makedirs(_PPATH, exist_ok=True)
  os.chdir(_PPATH)
  file = request.files['file']
  if file and allowed_file(file.filename):
      file.save(os.path.join(UPLOAD_FOLDER, 'zip1.zip'))

  zip1_path = os.path.join(_PPATH, 'zip1.zip')
  zip1_ipath = os.path.join(_PPATH, 'zip1')
  os.makedirs(zip1_ipath, exist_ok=True)
  with zipfile.ZipFile(zip1_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(_PPATH, 'zip1'))

  img_paths = []
  for path in Path(zip1_ipath).rglob('*.jpg'):
    img_paths.append(path)

  imgvec_path = os.path.join(_PPATH, 'vecs1')
  Path(imgvec_path).mkdir(parents=True, exist_ok=True)

  for filename in tqdm(img_paths):
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    out_path_file = os.path.join(imgvec_path, outfile_name)
    if not os.path.exists(out_path_file):
      img = load_img(str(filename))
      features = module(img)
      feature_set = np.squeeze(features)
      print(features.shape)
      np.savetxt(out_path_file, feature_set, delimiter=',')

  # Defining data structures as empty dict
  file_index_to_file_name = {}
  file_index_to_file_vector = {}

  # Configuring annoy parameters
  dims = 2048
  n_nearest_neighbors = 20
  trees = 10000

  # Reads all file names which stores feature vectors 
  allfiles = glob.glob(os.path.join(_PPATH, 'vecs1', '*.npz'))

  t = AnnoyIndex(dims, metric='angular')

  for findex, fname in tqdm(enumerate(allfiles)):
    file_vector = np.loadtxt(fname)
    file_name = os.path.basename(fname).split('.')[0]
    file_index_to_file_name[findex] = file_name
    file_index_to_file_vector[findex] = file_vector
    t.add_item(findex, file_vector)

  t.build(trees)

  file_path = os.path.join(_PPATH,'models/indices/')
  Path(file_path).mkdir(parents=True, exist_ok=True)

  t.save(file_path+'indexer.ann')
  pickle.dump(file_index_to_file_name, open(file_path+"file_index_to_file_name.p", "wb"))
  pickle.dump(file_index_to_file_vector, open(file_path+"file_index_to_file_vector.p", "wb"))

  return 'File processed on the server status OK!'

@application.route('/test_images', methods=['POST'])
def upload_zip2():
  os.chdir(_PPATH)
  file = request.files['file']
  if file and allowed_file(file.filename):
      file.save(os.path.join(UPLOAD_FOLDER, 'zip2.zip'))

  zip2_path = os.path.join(_PPATH, 'zip2.zip')
  zip2_ipath = os.path.join(_PPATH, 'zip2')
  os.makedirs(zip2_ipath, exist_ok=True)
  with zipfile.ZipFile(zip2_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(_PPATH, 'zip2'))

  query_files = []
  for path in Path(zip2_ipath).rglob('*.jpg'):
    query_files.append(path)
  
  dims = 2048
  indexer = AnnoyIndex(dims, 'angular')
  indexer.load(os.path.join(_PPATH,'models/indices/indexer.ann'))
  file_index_to_file_name = pickle.load(open(os.path.join(_PPATH,'models/indices/file_index_to_file_name.p'), 'rb'))

  results = pd.DataFrame(columns=['qid','fname','dist'])

  for q in query_files:
    temp_vec = np.squeeze(module(load_img(str(q))))
    nns = indexer.get_nns_by_vector(temp_vec, n=topK, include_distances=True)
    col1 = [q.stem]*topK
    col2 = [file_index_to_file_name[x] for x in nns[0]]
    col3 = nns[1]
    results = results.append(pd.DataFrame({'qid':col1,'fname':col2,'dist':col3}))
    results = results[results.dist<=threshold]
  
  results = results.reset_index(drop=True).T.to_json()
  return results
  
 # run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
```

<!-- #region id="ckTHq6dl3bp0" -->
### requirements.txt
<!-- #endregion -->

```python id="UHjFGVqDu_ut"
#collapse
%%writefile ./ebsapp/requirements.txt
annoy==1.16.3
Pillow==2.2.2
click==7.1.2
Flask==1.1.2
itsdangerous==1.1.0
Jinja2==2.11.2
MarkupSafe==1.1.1
Werkzeug==1.0.1
numpy==1.18.5
pandas==1.0.5
pathlib==1.0.1
pip-tools==4.5.1
requests==2.23.0
scipy==1.4.1
tensorflow==2.0.0b1
tensorflow-hub==0.8.0
tqdm==4.41.1
urllib3==1.24.3
zipfile36==0.1.3
```

<!-- #region id="OQ2y7D2L3d6h" -->
### packages.config
<!-- #endregion -->

```python id="kc9FLbpVvj4S"
#collapse
%%writefile ./ebsapp/.ebextensions/01_packages.config
packages:
  yum:
    gcc-c++: []
    unixODBC-devel: []
files:
  "/etc/httpd/conf.d/wsgi_custom.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      WSGIApplicationGroup %{GLOBAL}
```

<!-- #region id="w0y_hOmR3pPq" -->
### extras
some handy EBS commands
<!-- #endregion -->

```python id="vJRxjqlwtIu_"
#collapse
# !pip install aws.sam.cli
# !sam init

# %cd my-app

# !git config --global user.email "<email>"
# !git config --global user.name  "sparsh-ai"
# !git init
# !git status
# !git add .
# !git commit -m 'commit'

# !pip install awscli
# !aws configure

# !sam build && sam deploy
# !pip install flask-lambda-python36

# !aws ec2 create-key-pair --key-name MyKeyPair --query 'KeyMaterial' --output text > MyKeyPair.pem
# !chmod 400 MyKeyPair.pem
# !aws ec2 describe-instances --filters "Name=instance-type,Values=t2.micro" --query "Reservations[].Instances[].InstanceId"
# !ssh -i MyKeyPair.pem ec2-user@ec2-50-xx-xx-xxx.compute-1.amazonaws.com

# !git clone https://github.com/aws/aws-elastic-beanstalk-cli-setup.git
# build-essential zlib1g-dev libssl-dev libncurses-dev libffi-dev libsqlite3-dev libreadline-dev libbz2-dev

# !pip install awscli
# !pip install awsebcli
# !aws configure

# !mkdir eb-flask
# %cd eb-flask

# python --version
# pip install awsebcli --upgrade --user
# eb --version
# mkdir eb-flask
# cd eb-flask
# pip install virtualenv
# virtualenv virt
# source virt/bin/activate
# pip install flask
# pip freeze
# pip freeze > requirements.txt
# python application.py
# eb init -p python-3.6 my-app --region us-east-1
# pip install zip-files
```

<!-- #region id="Ps5JnIhv6JHJ" -->
<!-- #endregion -->

```python id="wy8NfNTA6JwS"

```
