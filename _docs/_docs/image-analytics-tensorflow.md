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

<!-- #region id="iRRufUm7BBfX" -->
# Image Analysis with Tensorflow
> Applying image analytics and computer vision methods using Tensorflow

- toc: true
- badges: true
- comments: true
- categories: [tensorflow, vision]
- image:
<!-- #endregion -->

```python id="Irr_-WvMKP-j"
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
```

```python id="KBtrPnwMLPNI" colab={"base_uri": "https://localhost:8080/"} outputId="aedf8838-9d51-4ed5-d55a-f6f81e421332"
!curl https://raw.githubusercontent.com/PracticalDL/Practical-Deep-Learning-Book/master/sample-images/cat.jpg --output cat.jpg
IMG_PATH = 'cat.jpg'
```

```python id="3Z0_4rEHLeqC" colab={"base_uri": "https://localhost:8080/", "height": 269} outputId="6db2e69a-cbbb-4f8b-c1a5-514bf0908809"
# data loading
img = image.load_img(IMG_PATH, target_size=(224, 224))
plt.imshow(img)
plt.show()
```

```python id="KleqXOmaLhfT"
# model loading
model = tf.keras.applications.resnet50.ResNet50()
```

```python id="9-WQyt1GLmDT"
# inference pipe
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    model = tf.keras.applications.resnet50.ResNet50()
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    print(decode_predictions(prediction, top=3)[0])
```

```python id="X_iBHZlmLpXq" colab={"base_uri": "https://localhost:8080/"} outputId="01d48117-c659-45f4-b18c-ace4204d54f1"
# inference
predict(IMG_PATH)
```

```python id="RvmMblYFMSW8" colab={"base_uri": "https://localhost:8080/"} outputId="ca360366-f3f3-4002-9ab1-7d7a1a8339fb"
!pip install tf-explain
```

```python id="EIHcGt0sLr8P"
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.activations import ExtractActivations

import matplotlib.image as mpimg
from matplotlib import rcParams

import requests

%matplotlib inline
%reload_ext tensorboard
```

```python id="9IjBejEIMRGa"
def download_sample_image(filename):
  url = f'https://raw.githubusercontent.com/PracticalDL/Practical-Deep-Learning-Book/master/sample-images/{filename}'
  open(filename, 'wb').write(requests.get(url).content)

IMAGE_PATHS = ['dog.jpg', 'cat.jpg']
for each_filename in IMAGE_PATHS:
    download_sample_image(each_filename)
```

```python id="2rUFZfqlM1x9"
def display_images(paths):
  # figure size in inches optional
  rcParams['figure.figsize'] = 11 ,8

  # read images
  img_A = mpimg.imread(paths[0])
  img_B = mpimg.imread(paths[-1])
  
  # display images
  fig, ax = plt.subplots(1,2)
  ax[0].imshow(img_A);
  ax[1].imshow(img_B);
```

```python colab={"base_uri": "https://localhost:8080/"} id="lCpiFD-0F2ze" outputId="fd0474d2-bba1-476e-c750-07e5351192c5"
model.summary()
```

```python id="TEMb3sQSNYSY"
indices = [263, 281]

layers_name = ['activation_6']

from IPython.display import Image

for i in range(len(IMAGE_PATHS)):
    each_path = IMAGE_PATHS[i]
    index = indices[i]

    img = tf.keras.preprocessing.image.load_img(each_path,
                                                target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    data = ([img], None)
    # Define name with which to save the result as
    name = each_path.split("/")[-1].split(".jpg")[0]

    #Save the Grad Cam visualization
    explainer = GradCAM()
    # model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
    #                                           include_top=True)
    # grid = explainer.explain(data, model, index, 'conv5_block3_add')
    # explainer.save(grid, '.', name + 'grad_cam.png')
    # display_images([each_path, name + 'grad_cam.png'])
```

<!-- #region id="QuBCyyLEXWR_" -->
---

### Building a Custom Classifier in Keras with Transfer Learning

- **Organize the data**: Download labeled images of cats and dogs from Kaggle. Then divide the images into training and validation folders.
- **Set up the configuration**: Define a pipeline for reading data, including preprocessing the images (e.g. resizing) and batching multiple images together.
- **Load and augment the data**: In the absence of a ton of training images, make small changes (augmentation) like rotation, zooming, etc to increase variation in training data.
- **Define the model**: Take a pre-trained model, remove the last few layers, and append a new classifier layer. Freeze the weights of original layers (i.e. make them unmodifiable). Select an optimizer algorithm and a metric to track (like accuracy).
- **Train and test**: Start training for a few iterations. Save the model to eventually load inside any application for predictions.
<!-- #endregion -->

```python id="WPLBHS1MOSoC"
!wget -x 'https://storage.googleapis.com/kagglesdsdata/competitions/3362/31148/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1608793667&Signature=IUAf0shLM4frn2DhvD8F2%2BD2Uk6hTZYV%2FMF3XkK7DFzYTZ5yGQS%2B4wf5eVe8DnZGjuVl0Gc30TpPoO%2B7uOL9DkUdKG8aUvcgfBVLS6nMadrUqawPyW1ODxz16tKIbKCmT8gLhff0ORDeN1H9Y0JjPu3pepAGZ8Nr0fktZOyI8ONQjW2h0c%2B%2FjnW9ayVtLQy4fZdaTlbU4rpTWTlahg2lI0eX57giPswH%2B%2F7lSJtfaCvDOVrOQFTer%2FqR%2F%2BFf73ynHH6zrJae9%2BrUd4lZ9XINqhfAZ%2FYfnC7HR%2F5%2FJ2TOnGr1%2FD4L6jckSm0RKEbizk%2BiWm%2FnnkTgLFkmpKvmYAotpA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.zip' -O train.zip
!unzip train.zip
%mv train data
%cd data
%mkdir train val
%mkdir train/cat train/dog
%mkdir val/cat val/dog

%ls | grep cat | sort -R | head -250 | xargs -I {} mv {} train/cat/
%ls | grep dog | sort -R | head -250 | xargs -I {} mv {} train/dog/
%ls | grep cat | sort -R | head -250 | xargs -I {} mv {} val/cat/
%ls | grep dog | sort -R | head -250 | xargs -I {} mv {} val/dog/
```

```python id="jDenWMi5ZcQs"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import math
```

```python id="NF_hNHn1a5YL"
TRAIN_DATA_DIR = 'train/'
VALIDATION_DATA_DIR = 'val/'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64
```

```python id="khznlEdha-yS" colab={"base_uri": "https://localhost:8080/"} outputId="010e00ae-0954-4a1a-ee4b-8dd5b93e5100"
# load and augment
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(VALIDATION_DATA_DIR,
                                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=False,
                                                       class_mode='categorical')
```

```python id="nA02rqmMbsXl"
# define the model
def model_maker():
    base_model = MobileNet(include_top=False,
                           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)
```

```python id="Z4NCVKlub-21" colab={"base_uri": "https://localhost:8080/"} outputId="de9d5c6d-dc3d-4b8e-b1f7-02b040afefa0"
model = model_maker()
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['acc'])
model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE))
```

```python id="_zOcyig3cPye"
model.save('model.h5')
```

```python id="XB7yUtpqItrx"
!ls /content/data
```

```python id="_t5OCprWcPwF"
# inference
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = load_model('model.h5')

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255.  # Preprocess the image
prediction = model.predict(preprocessed_img)
print(prediction)
print(validation_generator.class_indices)
```

<!-- #region id="xvmownFQdJ_o" -->
Result Analysis
<!-- #endregion -->

```python id="D96Xf4VocPtZ"
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
```

```python id="ogUB7yaEdQDS"
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

with CustomObjectScope(
    {'GlorotUniform': glorot_uniform()}):
    model = load_model('model.h5')
```

```python id="cZIeVOKbeIEp"
# Let's view the names of the files
filenames = validation_generator.filenames
print(len(filenames))
print(filenames[:10])

# Let's check what the ground truth looks like
ground_truth = validation_generator.classes
print(ground_truth[:10])
print(len(ground_truth))

# Let's confirm the which category names corresponds to which category id
label_to_index = validation_generator.class_indices
print(label_to_index)

# Now, let's develop a reverse mapping
index_to_label = dict((v, k) for k, v in label_to_index.items())
print(index_to_label)
```

```python id="Wx2-P6qTexOL"
predictions = model.predict_generator(validation_generator, steps=None)
print(predictions[:10])
prediction_index = []
for prediction in predictions:
    prediction_index.append(np.argmax(prediction))

def accuracy(predictions, ground_truth):
    total = 0
    for i, j in zip(predictions, ground_truth):
        if i == j:
            total += 1
    return total * 1.0 / len(predictions)

print(accuracy(prediction_index, ground_truth))

# To make our analysis easier, we make a dictionary storing the image index to 
# the prediction and ground truth (the expected prediction) for each image
prediction_table = {}
for index, val in enumerate(predictions):
    index_of_highest_probability = np.argmax(val)
    value_of_highest_probability = val[index_of_highest_probability]
    prediction_table[index] = [
        value_of_highest_probability, index_of_highest_probability,
        ground_truth[index]
    ]
assert len(predictions) == len(ground_truth) == len(prediction_table)
```

<!-- #region id="ScUKE3uvf2sJ" -->
`get_images_with_sorted_probabilities` finds the images with the highest/lowest probability value for a given category. These are the input arguments:
- `prediction_table`: dictionary from the image index to the prediction and ground truth for that image
- `get_highest_probability`: boolean flag to indicate if the results need to be highest (True) or lowest (False) probabilities
- `label`: intgeger id of category
- `number_of_items`: num of results to return
- `only_false_predictions`: boolean flag to indicate if results should only contain incorrect predictions

<!-- #endregion -->

```python id="R5l7IN1Lflj2"
def get_images_with_sorted_probabilities(prediction_table,
                                         get_highest_probability,
                                         label,
                                         number_of_items,
                                         only_false_predictions=False):
    sorted_prediction_table = [(k, prediction_table[k])
                               for k in sorted(prediction_table,
                                               key=prediction_table.get,
                                               reverse=get_highest_probability)
                               ]
    result = []
    for index, key in enumerate(sorted_prediction_table):
        image_index, [probability, predicted_index, gt] = key
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != gt:
                    result.append(
                        [image_index, [probability, predicted_index, gt]])
            else:
                result.append(
                    [image_index, [probability, predicted_index, gt]])
    return result[:number_of_items]
```

```python id="RP_8FpZIgC34"
def plot_images(filenames, distances, message):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 15))
    columns = 5
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        ax.set_title("\n\n" + filenames[i].split("/")[-1] + "\n" +
                     "\nProbability: " +
                     str(float("{0:.2f}".format(distances[i]))))
        plt.suptitle(message, fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.imshow(image)
        
def display(sorted_indices, message):
    similar_image_paths = []
    distances = []
    for name, value in sorted_indices:
        [probability, predicted_index, gt] = value
        similar_image_paths.append(VALIDATION_DATA_DIR + filenames[name])
        distances.append(probability)
    plot_images(similar_image_paths, distances, message)
```

<!-- #region id="oJBbPXJEgPKk" -->
Which images are we most confident contain dogs? Let's find images with the highest prediction probability (i.e. closest to 1.0) with the predicted class dog (i.e. 1)
<!-- #endregion -->

```python id="xumK7fj5gDPZ"
most_confident_dog_images = get_images_with_sorted_probabilities(prediction_table, True, 1, 10, False)
message = 'Images with highest probability of containing dogs'
display(most_confident_dog_images, message)
```

<!-- #region id="tYRZePszgfju" -->
What about the images that are least confident of containing dogs?
<!-- #endregion -->

```python id="8Rs12mcAgXjR"
least_confident_dog_images = get_images_with_sorted_probabilities(prediction_table, False, 1, 10, False)
message = 'Images with lowest probability of containing dogs'
display(least_confident_dog_images, message)
```

<!-- #region id="cTuQeLmqhPqT" -->
Incorrect predictions of dog
<!-- #endregion -->

```python id="CLLrtR48g4Z3" colab={"base_uri": "https://localhost:8080/", "height": 678} outputId="ada5cfe3-dee1-4a6e-9641-55056b5d1ca7"
incorrect_dog_images = get_images_with_sorted_probabilities(prediction_table, True, 1, 10, True)
message = 'Images of cats with highest probability of containing dogs'
display(incorrect_dog_images, message)
```

<!-- #region id="Qv64k7iihXXW" -->
Most confident predictions of cat
<!-- #endregion -->

```python id="vu9VpKwRhRK0" colab={"base_uri": "https://localhost:8080/", "height": 654} outputId="c7d84e1f-d020-4ff9-f59f-dc638da62499"
most_confident_cat_images = get_images_with_sorted_probabilities(prediction_table, True, 0, 10, False)
message = 'Images with highest probability of containing cats'
display(most_confident_cat_images, message)
```

<!-- #region id="RLD0ZOVripyO" -->
---
### Feature Extraction
- Extract features from pretrained models like VGG-16, VGG-19, ResNet-50, InceptionV3 and MobileNet and benchmark them using the Caltech101 dataset
- Write an indexer to index features and search for most similar features using various nearest neighbor algorithms, and explore various methods of visualizing plots
- Benchmark the algorithms based on the time it takes to index images and locate the most similar image based on its features using the Caltech-101 dataset. Also experiment with t-SNE and PCA
- Calculate the accuracies of the features obtained from the pretrained and finetuned models
- Experiment with PCA and figure out what is the optimum length of the features to improve the speed of feature extraction and similarity search
- Improve the accuracy with Fine-Tuning
<!-- #endregion -->

```python id="vramKdlghZoZ"
!mkdir -p /content/datasets
!pip install gdown
!gdown https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp --output /content/datasets/caltech101.tar.gz
!tar -xvzf /content/datasets/caltech101.tar.gz --directory /content/datasets
!mv /content/datasets/101_ObjectCategories /content/datasets/caltech101
!rm -rf /content/datasets/caltech101/BACKGROUND_Google
```

```python id="DlAK3xVnkjsY"
import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import random
import time
import math
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
```

```python id="rjS8XAcnksw1"
def model_picker(name):
    if (name == 'vgg16'):
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'vgg19'):
        model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'mobilenet'):
        model = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='max',
                          depth_multiplier=1,
                          alpha=1)
    elif (name == 'inception'):
        model = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max')
    elif (name == 'resnet'):
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')
    elif (name == 'xception'):
        model = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                         pooling='max')
    else:
        print("Specified model not available")
    return model
```

```python id="zXxBNsnfkyvu" colab={"base_uri": "https://localhost:8080/"} outputId="d6a1f010-37bc-46b5-8e98-85633717a9fc"
model_architecture = 'resnet'
model = model_picker(model_architecture)
```

```python id="0ry4ZiqjlRTE" colab={"base_uri": "https://localhost:8080/"} outputId="42772726-d39d-4837-d252-2b6908dd5b0e"
!curl https://raw.githubusercontent.com/PracticalDL/Practical-Deep-Learning-Book/master/sample-images/cat.jpg --output /content/sample_cat.jpg
```

```python id="h22n67m6k1E8" colab={"base_uri": "https://localhost:8080/"} outputId="9688fe5f-eb5b-46e0-e8e0-e521ea03a207"
def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path,
                         target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

# Let's see the feature length the model generates
features = extract_features('/content/sample_cat.jpg', model)
print(len(features))

# Now, we will see how much time it takes to extract features of one image
%timeit features = extract_features('/content/sample_cat.jpg', model)
```

```python id="AJUql-lElaTA" colab={"base_uri": "https://localhost:8080/", "height": 100, "referenced_widgets": ["4ab1b7fbbbef42b380db6f8eab2695e5", "37873755634c49daa016ab10f715c48f", "f3382b38d88a493e8d1972c817af8dd3", "d14b132aeb484caabe8a6f84d2bca52b", "63c02b82cd964c1db18331ad19cd6b71", "ada61ae1547849c091776057f6834ad7", "086ebeb238404465b8a09cdeaac101da", "6027d9c5c37b408bbfabfb5040145bd5"]} outputId="d89a2513-181a-4f3c-e3c6-82c15fa0340c"
# Let's make a handy function to recursively get all the image files under a root directory
extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    return file_list

# Now, let's run the extraction over the entire dataset and time it
root_dir = '/content/datasets/caltech101'
filenames = sorted(get_file_list(root_dir))

feature_list = []
for i in tqdm_notebook(range(len(filenames))):
    feature_list.append(extract_features(filenames[i], model))
```

```python id="vL90i5bKlwAb" colab={"base_uri": "https://localhost:8080/"} outputId="ed88e96f-d750-40f0-d0e4-8c2d0fbdaddb"
# Now let's try the same with the Keras Image Generator functions
batch_size = 64
datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(root_dir,
                                        target_size=(224, 224),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

num_images = len(generator.filenames)
num_epochs = int(math.ceil(num_images / batch_size))

start_time = time.time()
feature_list = []
feature_list = model.predict_generator(generator, num_epochs)
end_time = time.time()

for i, features in enumerate(feature_list):
    feature_list[i] = features / norm(features)

feature_list = feature_list.reshape(num_images, -1)

print("Num images   = ", len(generator.classes))
print("Shape of feature_list = ", feature_list.shape)
print("Time taken in sec = ", end_time - start_time)
```

```python id="LCSB9lXonV6k"
# Let's save the features as intermediate files to use later
filenames = [root_dir + '/' + s for s in generator.filenames]
pickle.dump(generator.classes, open('./class_ids-caltech101.pickle', 'wb'))
pickle.dump(filenames, open('./filenames-caltech101.pickle', 'wb'))
pickle.dump(feature_list, open('./features-caltech101-' + model_architecture + '.pickle', 'wb'))
```

```python id="PiUsEcuJn4RD" colab={"base_uri": "https://localhost:8080/"} outputId="227f4e5a-9561-4165-8631-51c179314a6b"
# Let's train a finetuned model as well and save the features for that as well
TRAIN_SAMPLES = 8677
NUM_CLASSES = 101
IMG_WIDTH, IMG_HEIGHT = 224, 224

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(root_dir,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')

def model_maker():
    base_model = ResNet50(include_top=False,
                           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)

model_finetuned = model_maker()
model_finetuned.compile(loss='categorical_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(0.001),
              metrics=['acc'])
model_finetuned.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / batch_size),
    epochs=10)

model_finetuned.save('./model-finetuned.h5')

start_time = time.time()
feature_list_finetuned = []
feature_list_finetuned = model_finetuned.predict_generator(generator, num_epochs)
end_time = time.time()

for i, features_finetuned in enumerate(feature_list_finetuned):
    feature_list_finetuned[i] = features_finetuned / norm(features_finetuned)

feature_list = feature_list_finetuned.reshape(num_images, -1)

print("Num images   = ", len(generator.classes))
print("Shape of feature_list = ", feature_list.shape)
print("Time taken in sec = ", end_time - start_time)

pickle.dump(feature_list, open('./features-caltech101-resnet-finetuned.pickle', 'wb'))                                         
```

```python id="UozeHSvtolAF"
import numpy as np
import pickle
from tqdm import tqdm, tqdm_notebook
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import PIL
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
```

```python id="GcW5huz5qJC8" colab={"base_uri": "https://localhost:8080/"} outputId="00cb62f5-b159-4907-98c9-5006947d04f4"
filenames = pickle.load(open('./filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('./features-caltech101-resnet.pickle', 'rb'))
class_ids = pickle.load(open('./class_ids-caltech101.pickle', 'rb'))

num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)
```

```python id="lMJ6lmJOqJWk"
# Helper function to get the classname
def classname(str):
    return str.split('/')[-2]


# Helper function to get the classname and filename
def classname_filename(str):
    return str.split('/')[-2] + '/' + str.split('/')[-1]


# Helper functions to plot the nearest images given a query image
def plot_images(filenames, distances):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 10))
    columns = 4
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + classname_filename(filenames[i]))
        else:
            ax.set_title("Similar Image\n" + classname_filename(filenames[i]) +
                         "\nDistance: " +
                         str(float("{0:.2f}".format(distances[i]))))
        plt.imshow(image)
        # To save the plot in a high definition format i.e. PDF, uncomment the following line:
        #plt.savefig('results/' + str(random.randint(0,10000))+'.pdf', format='pdf', dpi=1000)
        # We will use this line repeatedly in our code.
```

```python id="r0EspAcCtLUE"
neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)
```

```python id="-zmn39kNqJji" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="1eeca467-17bf-422c-b58c-134621c1f1d5"
for i in range(6):
    random_image_index = random.randint(0, num_images)
    distances, indices = neighbors.kneighbors(
        [feature_list[random_image_index]])
    # Don't take the first closest image as it will be the same image
    similar_image_paths = [filenames[random_image_index]] + \
        [filenames[indices[0][i]] for i in range(1, 4)]
    plot_images(similar_image_paths, distances[0])
```

```python id="FeIM37OZqJOM" colab={"base_uri": "https://localhost:8080/"} outputId="54301e13-fccd-41e1-e8cd-92841834609f"
# Let us get a sense of the similarity values by looking at distance stats over the dataset
neighbors = NearestNeighbors(n_neighbors=len(feature_list),
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)
distances, indices = neighbors.kneighbors(feature_list)

print("Median distance between all photos: ", np.median(distances))
print("Max distance between all photos: ", np.max(distances))
print("Median distance among most similar photos: ", np.median(distances[:, 2]))
```

```python id="gjixxj6AqI25"
# Select the amount of data you want to run the experiments on
start = 7000; end = 8000
selected_features = feature_list[start:end]
selected_class_ids = class_ids[start:end]
selected_filenames = filenames[start:end]
```

```python id="KDokBWc7r0tf" colab={"base_uri": "https://localhost:8080/", "height": 405} outputId="a85f59a8-96d9-4d7e-e341-4b7c4697ee10"
# The t-SNE algorithm is useful for visualizing high dimensional data

from sklearn.manifold import TSNE

# You can play with these values and see how the results change
n_components = 2
verbose = 1
perplexity = 30
n_iter = 1000
metric = 'euclidean'

time_start = time.time()
tsne_results = TSNE(n_components=n_components,
                    verbose=verbose,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    metric=metric).fit_transform(selected_features)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

# Plot a scatter plot from the generated t-SNE results
color_map = plt.cm.get_cmap('coolwarm')
scatter_plot = plt.scatter(tsne_results[:, 0],
                           tsne_results[:, 1],
                           c=selected_class_ids,
                           cmap=color_map)
plt.colorbar(scatter_plot)
plt.show()
# To save the plot in a high definition format i.e. PDF, uncomment the following line:
#plt.savefig('results/' + str(ADD_NAME_HERE)+'.pdf', format='pdf', dpi=1000)
```

```python id="hf1j8xenr00l" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="310955fb-a2cd-42e6-dde0-6d9be9dbefde"
# Visualize the patterns in the images using t-SNE

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


def plot_images_in_2d(x, y, image_paths, axis=None, zoom=1):
    if axis is None:
        axis = plt.gca()
    x, y = np.atleast_1d(x, y)
    for x0, y0, image_path in zip(x, y, image_paths):
        image = Image.open(image_path)
        image.thumbnail((100, 100), Image.ANTIALIAS)
        img = OffsetImage(image, zoom=zoom)
        anno_box = AnnotationBbox(img, (x0, y0),
                                  xycoords='data',
                                  frameon=False)
        axis.add_artist(anno_box)
    axis.update_datalim(np.column_stack([x, y]))
    axis.autoscale()

def show_tsne(x, y, selected_filenames):
    fig, axis = plt.subplots()
    fig.set_size_inches(22, 22, forward=True)
    plot_images_in_2d(x, y, selected_filenames, zoom=0.3, axis=axis)
    plt.show()

show_tsne(tsne_results[:, 0], tsne_results[:, 1], selected_filenames)
```

<!-- #region id="lfIfazIos67W" -->
The show_tsne function piles images one on top of each other, making it harder to discern the patterns as the density of images is high. To help visualize the patterns better, we write another helper function tsne_to_grid_plotter_manual that spaces the images evenly
<!-- #endregion -->

```python id="dU0JBAaPr0ja"
def tsne_to_grid_plotter_manual(x, y, selected_filenames):
    S = 2000
    s = 100
    x = (x - min(x)) / (max(x) - min(x))
    y = (y - min(y)) / (max(y) - min(y))
    x_values = []
    y_values = []
    filename_plot = []
    x_y_dict = {}
    for i, image_path in enumerate(selected_filenames):
        a = np.ceil(x[i] * (S - s))
        b = np.ceil(y[i] * (S - s))
        a = int(a - np.mod(a, s))
        b = int(b - np.mod(b, s))
        if str(a) + "|" + str(b) in x_y_dict:
            continue
        x_y_dict[str(a) + "|" + str(b)] = 1
        x_values.append(a)
        y_values.append(b)
        filename_plot.append(image_path)
    fig, axis = plt.subplots()
    fig.set_size_inches(22, 22, forward=True)
    plot_images_in_2d(x_values, y_values, filename_plot, zoom=.58, axis=axis)
    plt.show()
```

```python id="msE3fSQ1r0dU" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="3080ede5-a2d1-4206-fa5b-5be858d5f834"
tsne_to_grid_plotter_manual(tsne_results[:, 0], tsne_results[:, 1], selected_filenames)
```

<!-- #region id="aq6_rAXit0Y2" -->
### PCA
<!-- #endregion -->

```python id="AsPLHIaDthD5" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="d90c5985-3891-4ad4-d90e-ce794489c230"
num_feature_dimensions = 100
pca = PCA(n_components=num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list)

neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list_compressed)
distances, indices = neighbors.kneighbors([feature_list_compressed[0]])

for i in range(6):
    random_image_index = random.randint(0, num_images)
    distances, indices = neighbors.kneighbors(
        [feature_list_compressed[random_image_index]])
    # Don't take the first closest image as it will be the same image
    similar_image_paths = [filenames[random_image_index]] + \
        [filenames[indices[0][i]] for i in range(1, 4)]
    plot_images(similar_image_paths, distances[0])

selected_features = feature_list_compressed[:4000]
selected_class_ids = class_ids[:4000]
selected_filenames = filenames[:4000]

time_start = time.time()
tsne_results = TSNE(n_components=2, verbose=1,
                    metric='euclidean').fit_transform(selected_features)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

color_map = plt.cm.get_cmap('coolwarm')
scatter_plot = plt.scatter(tsne_results[:, 0],
                           tsne_results[:, 1],
                           c=selected_class_ids,
                           cmap=color_map)
plt.colorbar(scatter_plot)
plt.show()

tsne_to_grid_plotter_manual(tsne_results[:, 0], tsne_results[:, 1], selected_filenames)
```

<!-- #region id="2_lWr_NTvYHg" -->
Calculate the accuracies of the features obtained from the pretrained and finetuned models
<!-- #endregion -->

```python id="tibECDFvuD1H"
import numpy as np
import pickle
from tqdm import tqdm, tqdm_notebook
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import PIL
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
```

```python id="p16mnJLPvank" colab={"base_uri": "https://localhost:8080/"} outputId="d0bb6d13-f6f2-44b0-879d-66548eb65762"
filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('features-caltech101-resnet.pickle', 'rb'))
class_ids = pickle.load(open('class_ids-caltech101.pickle', 'rb'))

num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)
```

```python id="um0eQu0nvhUK"
# Helper function to get the classname
def classname(str):
    return str.split('/')[-2]


# Helper function to get the classname and filename
def classname_filename(str):
    return str.split('/')[-2] + '/' + str.split('/')[-1]


def calculate_accuracy(feature_list):
    num_nearest_neighbors = 5
    correct_predictions = 0
    incorrect_predictions = 0
    neighbors = NearestNeighbors(n_neighbors=num_nearest_neighbors,
                                 algorithm='brute',
                                 metric='euclidean').fit(feature_list)
    for i in tqdm_notebook(range(len(feature_list))):
        distances, indices = neighbors.kneighbors([feature_list[i]])
        for j in range(1, num_nearest_neighbors):
            if (classname(filenames[i]) == classname(
                    filenames[indices[0][j]])):
                correct_predictions += 1
            else:
                incorrect_predictions += 1
    print(
        "Accuracy is ",
        round(
            100.0 * correct_predictions /
            (1.0 * correct_predictions + incorrect_predictions), 2))
```

```python id="Z5Zun2w7vkF3" colab={"base_uri": "https://localhost:8080/", "height": 217, "referenced_widgets": ["f25a6c864a3042d4b6bf5d533200cb2a", "e6004582afb14031965f6b1ada823e33", "7f8afaa03bb1435487f9c9be8ed7dcab", "d81f886f3f574bccb90f7ee4a399571b", "34b1da920def4b3d96b36ca2154ee5a1", "91b94268fd7f421ab55f64bb5f7cc059", "49eda2c54803406ea3cc087bd4c0ada5", "c4fe152a3170411fb92d9495dc17ba54", "e8c9d54665304965bceb6d409183c382", "bc6981087d544a4683d316f8949019cf", "89101646d762417d9a7d4a04996c00e6", "15b9635b45914b808bd09b1a81a3e00e", "b9adf2c643584d709d5037a6a8e489cb", "ebf637f97e3f4563ada3311a3a8fae17", "bca07a4f0a8546feb5296eae0480ae21", "1e3dabe0da6f4876ac67ffff5f15ac23"]} outputId="2ae0f0b5-a218-4ff7-ea3d-a2a9116502ce"
# Accuracy of Brute Force over Caltech101 features
calculate_accuracy(feature_list[:])

# Accuracy of Brute Force over the PCA compressed Caltech101 features
num_feature_dimensions = 100
pca = PCA(n_components=num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list[:])
calculate_accuracy(feature_list_compressed[:])
```

```python id="L9Tfi85Yv7sL"
# Use the features from the finetuned model
filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(
    open('features-caltech101-resnet-finetuned.pickle', 'rb'))
class_ids = pickle.load(open('class_ids-caltech101.pickle', 'rb'))

num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)
```

```python id="06yKshHvv98t"
# Accuracy of Brute Force over the finetuned Caltech101 features
calculate_accuracy(feature_list[:])

# Accuracy of Brute Force over the PCA compressed finetuned Caltech101 features
num_feature_dimensions = 100
pca = PCA(n_components=num_feature_dimensions)
pca.fit(feature_list)
feature_list_compressed = pca.transform(feature_list[:])
calculate_accuracy(feature_list_compressed[:])
```

<!-- #region id="iU_JSOk8wb7a" -->
### Accuracy 

These results lead to the accuracy on Caltech101. Repeating Level 3 on the Caltech256 features we get its corresponding accuracy. 

Accuracy on Caltech101.

| Algorithm | Accuracy using Pretrained features| Accuracy using Finetuned features | 
|-------------|----------------------------|------------------------|
| Brute Force | 87.06 | 89.48 | 
| PCA + Brute Force | 87.65  |  89.39 |


Accuracy on Caltech256.

| Algorithm | Accuracy using Pretrained features| Accuracy using Finetuned features | 
|-------------|----------------------------|------------------------|
| Brute Force | 58.38 | 96.01 | 
| PCA + Brute Force | 56.64  | 95.34|
<!-- #endregion -->

```python id="khkkyiy6wgRN"

```
