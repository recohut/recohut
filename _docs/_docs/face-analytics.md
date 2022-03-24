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

# Face Analytics

```python colab={} colab_type="code" id="-Pl2yOByRse2"
MODEL_PATH = '/content/drive/My Drive/pickles/facenet_keras.h5'
DATA_PATH = '/content/drive/My Drive/sample_images/mathematicians/'
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 7705, "status": "ok", "timestamp": 1588941753458, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="HLdPfMJqRcZv" outputId="365cc0d3-d4f0-4283-d12e-e691e6f9d216"
# load pre-trained facenet model
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)
```

```python colab={} colab_type="code" id="HDgyZmEFUhpy"
# load packages
# !pip install mtcnn
import mtcnn
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np
import matplotlib.pylab as plt
import os
```

```python colab={} colab_type="code" id="zaFf_Qw1Uhmv"
# extract face from image
def extract_face(image_file, required_size=(160, 160)):
 image = imread(image_file)
 image = gray2rgb(image) if len(image.shape) < 3 else \
 image[...,:3]
 detector = MTCNN()
 results = detector.detect_faces(image)
 x1, y1, width, height = results[0]['box']
 x2, y2 = abs(x1) + width, abs(y1) + height
 # extract the face
 face = image[y1:y2, x1:x2]
 return resize(face, required_size)

def load_faces(folder):
  faces = []
  for filename in os.listdir(folder):
    face = extract_face(folder + filename)
    faces.append(face)
  return faces
```

```python colab={"base_uri": "https://localhost:8080/", "height": 255} colab_type="code" executionInfo={"elapsed": 22673, "status": "ok", "timestamp": 1588942629605, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="GWFlMyCpVd05" outputId="22c81af4-2a7a-4541-fdb3-6e6415f9b668"
# load dataset
def load_dataset(folder):
 X, y = [], []
 for sub_folder in os.listdir(folder):
    path = folder + sub_folder + '/'
    if not os.path.isdir(path): continue
    faces = load_faces(path)
    labels = [sub_folder for _ in range(len(faces))]
    print('>loaded %d examples for class: %s' % (len(faces), sub_folder))
    X.extend(faces)
    y.extend(labels)
 return np.array(X), np.array(y)

X_train, y_train = load_dataset(DATA_PATH+'train/')
print(X_train.shape, y_train.shape)
X_test, y_test = load_dataset(DATA_PATH+'test/')
print(X_test.shape, y_test.shape)

np.savez_compressed('6-mathematicians.npz', X_train, y_train, X_test, y_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 380} colab_type="code" executionInfo={"elapsed": 3206, "status": "ok", "timestamp": 1588942767140, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="tlufHf9mV0bH" outputId="e75d4a97-5a38-4783-fb12-47ed5f9d2352"
# Uncompress the training and test datasets
data = np.load('6-mathematicians.npz')
X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print(X_train.shape)
indices = np.random.choice(X_train.shape[0], 12, replace=False)
plt.figure(figsize=(15,15))
plt.subplots_adjust(0,0,1,0.9,0.05,0.1)
for i in range(len(indices)):
    plt.subplot(6,6,i+1), plt.imshow(X_train[indices[i]]), plt.axis('off'), plt.title(y_train[indices[i]], size=15)
plt.show()
```

```python colab={} colab_type="code" id="dIK5iTLPUhh7"
  
```

```python colab={} colab_type="code" id="nayxrDE5UhgG"

```

<!-- #region colab_type="text" id="IU4yPhxxXUXG" -->
### Face Recognition with FaceNet (Keras)
<!-- #endregion -->

```python colab={} colab_type="code" id="Vty7Sbe-XUXH" outputId="8b0fa784-3db7-4128-ce99-11902a2381cb"
#https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
# googleimagesdownload --keywords "Euler" --limit 20
%matplotlib inline
#from keras.models import load_model
from tensorflow.keras.models import load_model
#!pip install mtcnn
import mtcnn
# print version
print(mtcnn.__version__)
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np
import matplotlib.pylab as plt
import os
```

```python colab={} colab_type="code" id="-0IexTJnXUXk" outputId="f0d30bce-ec73-4543-a34f-2f5a7415fefa"
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# example of loading the keras facenet model
# https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_
# load the model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model('models/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)
```

```python colab={} colab_type="code" id="Pr4KUPkPXUXr" outputId="edf9d5a1-fbf6-4c08-e87b-7c0e279e8d92"
# extract a single face from a given photograph
def extract_face(image_file, required_size=(160, 160)):
    # load image from file
    image = imread(image_file)
    #print(image_file, image.shape)
    if len(image.shape) < 3: image = gray2rgb(image) 
    else: image = image[...,:3]
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1:y2, x1:x2]
    # resize pixels to the model size
    return resize(face, required_size)

face = extract_face('images/beatles.png')
plt.figure(figsize=(5,5))
plt.imshow(face), plt.axis('off')
plt.show()
```

```python colab={} colab_type="code" id="hBUlV6j1XUXy" outputId="e458dd52-c4a1-4152-dee8-7aa5e7751c99"
# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    #i = 1
    for filename in os.listdir(directory):
        # path
        path = directory + filename
        #new_path = directory + str(i) + filename[-4:]
        #os.rename(path, new_path)
        # get face
        face = extract_face(path)
        print(path, face.shape)
        # store
        faces.append(face)
        #i += 1
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = [], []
    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
        #print('extended')  
    #X, y = np.concatenate(X, axis=0), np.array(y)
    return np.array(X), np.array(y)

# load train dataset
X_train, y_train = load_dataset('images/mathematicians/train/')
print(X_train.shape, y_train.shape)
# (72, 160, 160, 3) (72,)
# load test dataset
X_test, y_test = load_dataset('images/mathematicians/test/')
print(X_test.shape, y_test.shape)
# (36, 160, 160, 3) (36,)
# save arrays to one file in compressed format
#np.savez_compressed('images/5-celebrity-faces-dataset.npz', X_train, y_train, X_test, y_test)
np.savez_compressed('images/6-mathematicians.npz', X_train, y_train, X_test, y_test)
```

```python colab={} colab_type="code" id="LihnQ3nIXUX4" outputId="9a5dbb0f-e140-4f47-845f-e227d9284fd5"
data = np.load('images/6-mathematicians.npz')
X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print(X_train.shape)
indices = np.random.choice(X_train.shape[0], 36, replace=False)
plt.figure(figsize=(15,15))
plt.subplots_adjust(0,0,1,0.9,0.05,0.1)
for i in range(len(indices)):
    plt.subplot(6,6,i+1), plt.imshow(X_train[indices[i]]), plt.axis('off'), plt.title(y_train[indices[i]], size=15)
plt.show()
```

```python colab={} colab_type="code" id="XT_HPuiSXUX-" outputId="04bf9f51-4c01-4593-8ef5-0e414238f882"
# get the face embedding for one face
def get_embedding(model, face):
    # scale pixel values
    #face = face.astype('float32')
    # standardize pixel values across channels (global)
    #mean, std = face.mean(), face.std()
    #face = (face - mean) / std
    # transform face into one sample
    # make prediction to get embedding
    yhat = model.predict(np.expand_dims(face, axis=0))
    return yhat[0]

# load the face dataset
#data = np.load('images/5-celebrity-faces-dataset.npz')
data = np.load('images/6-mathematicians.npz')
X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# convert each face in the train set to an embedding
X_train_em = []
for face in X_train:
    embedding = get_embedding(model, face)
    X_train_em.append(embedding)
X_train_em = np.asarray(X_train_em)
print(X_train_em.shape)
# convert each face in the test set to an embedding
X_test_em = []
for face in X_test:
    embedding = get_embedding(model, face)
    X_test_em.append(embedding)
X_test_em = np.asarray(X_test_em)
print(X_test_em.shape)
# save arrays to one file in compressed format
#np.savez_compressed('models/5-celebrity-faces-embeddings.npz', X_train_em, y_train, X_test_em, y_test)
np.savez_compressed('models/6-mathematicians-embeddings.npz', X_train_em, y_train, X_test_em, y_test)
```

```python colab={} colab_type="code" id="48-Mip2-XUYE" outputId="8b36a39d-476c-4d2f-8e7a-27eb0e3f154e"
#data = np.load('models/5-celebrity-faces-embeddings.npz')
data = np.load('models/6-mathematicians-embeddings.npz')
X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (X_train.shape[0], X_test.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
X_train = in_encoder.transform(X_train)
X_test = in_encoder.transform(X_test)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)
y_test = out_encoder.transform(y_test)
# fit model
model_svc = SVC(kernel='linear', probability=True)
model_svc.fit(X_train, y_train)
# predict
yhat_train = model_svc.predict(X_train)
yhat_test = model_svc.predict(X_test)
# score
score_train = accuracy_score(y_train, yhat_train)
score_test = accuracy_score(y_test, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
```

```python colab={} colab_type="code" id="IWvWwrjgXUYK" outputId="7690fe4d-a662-40a0-bea1-d04fbea47b8a"
data = np.load('images/6-mathematicians.npz')
X_test_faces = data['arr_2']
# test model on a random example from the test dataset
selection = np.random.choice(X_test.shape[0], 1)
random_face_pixels = X_test_faces[selection]
random_face_emb = X_test[selection]
random_face_class = y_test[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
yhat_class = model_svc.predict(random_face_emb)
yhat_prob = model_svc.predict_proba(random_face_emb)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
plt.imshow(random_face_pixels[0]), plt.title('%s (%.3f)' % (predict_names[0], class_probability)), plt.axis('off')
plt.show()
```

```python colab={} colab_type="code" id="DxCkLKrUXUYQ"
#from sklearn.metrics import classification_report
#import keras
#y_hat_classes = keras.utils.np_utils.to_categorical(yhat_test)
#print(classification_report(y_test, y_hat_classes))#, target_names=target_names))
```

<!-- #region colab_type="text" id="hPlzvkzrXUYW" -->
### Age/Gender Recognition with Deep Learning Model
<!-- #endregion -->

```python colab={} colab_type="code" id="8k2fc2q2XUYX" outputId="a8416ae4-d00f-469b-e076-b5e60c1799ff"
# https://github.com/yu4u/age-gender-estimation
# https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.29-3.76_utk.hdf5
# https://github.com/thegopieffect/computer_vision
# https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
# https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE
# https://susanqq.github.io/UTKFace/
# https://stackoverflow.com/questions/53859419/dlib-get-frontal-face-detector-gets-faces-in-full-image-but-does-not-get-in-c

import sys
import numpy as np
from keras.models import Model
from keras import backend as K
import cv2
import dlib
from keras.models import model_from_json
from glob import glob
import matplotlib.pylab as plt

#sys.setrecursionlimit(2 ** 20)
#np.random.seed(2 ** 10)

depth = 16
k = width = 8
margin = 0.4
img_size = 64
#conf_threshold = 0.5

#! pip install dlib
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2, thickness=3):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('models/weights.29-3.76_utk.hdf5') #weights.28-3.73.hdf5')
# for face detection
detector = dlib.get_frontal_face_detector()

plt.figure(figsize=(15,7))
plt.subplots_adjust(0,0,1,1,0.05,0.05)
j = 1
for img_file in glob('images/musicians/*.jpg'): #['images/all.png']: 
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(img)
    r = 640 / max(img_h, img_w)
    img = cv2.resize(img, (int(img_w * r), int(img_h * r)))

    # detect faces using dlib detector
    detected = detector(img, 0) #0)

    faces = np.empty((len(detected), img_size, img_size, 3))
    
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = loaded_model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        print(predicted_ages)

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]), "F" if predicted_genders[i][0] > 0.5 else "M")
            draw_label(img, (d.left(), d.top()), label)

    plt.subplot(1,3,j), plt.imshow(img), plt.title(img_file.split('\\')[-1].split('.')[0], size=20), plt.axis('off')
    j += 1
plt.show()
```

<!-- #region colab_type="text" id="GLPSpjHbXUYc" -->
### Emotion Recognition with Deep Learning Model
<!-- #endregion -->

```python colab={} colab_type="code" id="sxY2acu2XUYe" outputId="68de2616-8931-451c-ff98-fad94ae7f845"
# https://github.com/oarriaga/face_classification/tree/master/trained_models
# http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
# https://github.com/serengil/tensorflow-101
# https://github.com/jalajthanaki/Facial_emotion_recognition_using_Keras
# https://drive.google.com/file/d/0B6yZu81NrMhSV2ozYWZrenJXd1E
# https://www.istockphoto.com/in/photo/useful-faces-gm108686357-5595555
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pylab as plt 

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
model = load_model('models/model_5-49-0.62.hdf5')

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX

#image = cv2.imread('images/facial_expressions.png')
image = cv2.imread('images/emotions.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
    face_crop = image[y:y + h, x:x + w]
    face_crop = cv2.resize(face_crop, (48, 48))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_crop = face_crop.astype('float32') / 255
    face_crop = np.asarray(face_crop)
    face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
    result = target[np.argmax(model.predict(face_crop))]
    cv2.putText(image, result, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

plt.figure(figsize=(15,15))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.show()
```
