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

<!-- #region id="3byH-9FCMMVO" -->
# TF Serving in action
> Chapter 19 of the hands-on-ml book

- toc: true
- badges: true
- comments: true
- categories: [TensorflowServing, TFServing, Tensorflow, Deployment]
- author: "<a href='https://github.com/ageron/handson-ml2'>Aurélien Geron</a>"
- image:
<!-- #endregion -->

<!-- #region id="tBwZ6FDlMMVW" -->
## Setup
First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20 and TensorFlow ≥2.0.

<!-- #endregion -->

```python id="k5vBPPzqMMVc"
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab or Kaggle?
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

if IS_COLAB or IS_KAGGLE:
    !echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" > /etc/apt/sources.list.d/tensorflow-serving.list
    !curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
    !apt update && apt-get install -y tensorflow-model-server
    !pip install -q -U tensorflow-serving-api

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
    if IS_KAGGLE:
        print("Go to Settings > Accelerator and select GPU.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deploy"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

<!-- #region id="cSLg2BXJMMVe" -->
## Deploying TensorFlow models to TensorFlow Serving (TFS)
We will use the REST API or the gRPC API.
<!-- #endregion -->

<!-- #region id="791wqyHTMMVf" -->
### Save/Load a `SavedModel`
<!-- #endregion -->

```python id="Uh2eHPqxMMVg"
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_new = X_test[:3]
```

```python id="DkTw1RHZMMVi" outputId="6497fedd-7398-494e-864c-0b202fb92814"
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
```

```python id="yUs4iqmBMMVk" outputId="92fc3271-489a-47e8-d318-cc0609e9d26b"
np.round(model.predict(X_new), 2)
```

```python id="2D6sIKU1MMVl" outputId="4654c873-b6dc-4b08-fcdc-66cefa6f14a2"
model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
model_path
```

```python id="GQFzkjwjMMVl"
!rm -rf {model_name}
```

```python id="3dfp2WpaMMVm" outputId="1c61a756-b3ea-4aaa-9434-79c846192d6e"
tf.saved_model.save(model, model_path)
```

```python id="ARRn-CkSMMVm" outputId="91afcfb7-7519-448b-82d6-23a36a528446"
for root, dirs, files in os.walk(model_name):
    indent = '    ' * root.count(os.sep)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + '    ', filename))
```

```python id="JIpxEeTuMMVn" outputId="425c5af4-de9b-446d-f8aa-f0e3b5a628ba"
!saved_model_cli show --dir {model_path}
```

```python id="--S30tcSMMVo" outputId="730e8fb6-7641-4370-ff65-6256c071120e"
!saved_model_cli show --dir {model_path} --tag_set serve
```

```python id="hC8XyNfrMMVo" outputId="20f320ed-c831-430e-eda4-9618b2256eaa"
!saved_model_cli show --dir {model_path} --tag_set serve \
                      --signature_def serving_default
```

```python id="qmQqljA-MMVp" outputId="53b9f23c-6b99-43ee-89e3-d0fb9f52ed17"
!saved_model_cli show --dir {model_path} --all
```

<!-- #region id="uoI6eag7MMVq" -->
Let's write the new instances to a `npy` file so we can pass them easily to our model:
<!-- #endregion -->

```python id="2QQqjmRiMMVq"
np.save("my_mnist_tests.npy", X_new)
```

```python id="viLWHfgDMMVq" outputId="f098914f-04b5-47db-9196-a30003657c23"
input_name = model.input_names[0]
input_name
```

<!-- #region id="ygWzF7XMMMVr" -->
And now let's use `saved_model_cli` to make predictions for the instances we just saved:
<!-- #endregion -->

```python id="VP2NQ-TqMMVr" outputId="943634cc-bba2-42a7-b911-5fdbb7b34d2a"
!saved_model_cli run --dir {model_path} --tag_set serve \
                     --signature_def serving_default    \
                     --inputs {input_name}=my_mnist_tests.npy
```

```python id="WVzOsT0hMMVr" outputId="843f6152-a0d9-4b1e-d425-a69db3d03a43"
np.round([[1.1347984e-04, 1.5187356e-07, 9.7032893e-04, 2.7640699e-03, 3.7826971e-06,
           7.6876910e-05, 3.9140293e-08, 9.9559116e-01, 5.3502394e-05, 4.2665208e-04],
          [8.2443521e-04, 3.5493889e-05, 9.8826385e-01, 7.0466995e-03, 1.2957400e-07,
           2.3389691e-04, 2.5639210e-03, 9.5886099e-10, 1.0314899e-03, 8.7952529e-08],
          [4.4693781e-05, 9.7028232e-01, 9.0526715e-03, 2.2641101e-03, 4.8766597e-04,
           2.8800720e-03, 2.2714981e-03, 8.3753867e-03, 4.0439744e-03, 2.9759688e-04]], 2)
```

<!-- #region id="njT2JlsGMMVs" -->
### TensorFlow Serving
<!-- #endregion -->

<!-- #region id="HDnC-waSMMVs" -->
Install [Docker](https://docs.docker.com/install/) if you don't have it already. Then run:

```bash
docker pull tensorflow/serving

export ML_PATH=$HOME/ml # or wherever this project is
docker run -it --rm -p 8500:8500 -p 8501:8501 \
   -v "$ML_PATH/my_mnist_model:/models/my_mnist_model" \
   -e MODEL_NAME=my_mnist_model \
   tensorflow/serving
```
Once you are finished using it, press Ctrl-C to shut down the server.
<!-- #endregion -->

<!-- #region id="KTosiaqjMMVt" -->
Alternatively, if `tensorflow_model_server` is installed (e.g., if you are running this notebook in Colab), then the following 3 cells will start the server:
<!-- #endregion -->

```python id="njjJMJmdMMVt"
os.environ["MODEL_DIR"] = os.path.split(os.path.abspath(model_path))[0]
```

```bash id="OeBBlWwoMMVt" magic_args="--bg"
nohup tensorflow_model_server \
     --rest_api_port=8501 \
     --model_name=my_mnist_model \
     --model_base_path="${MODEL_DIR}" >server.log 2>&1
```

```python id="q3SaRXB2MMVt" outputId="e3421a3c-5cf9-4baa-f344-c014aa0ae3d2"
!tail server.log
```

```python id="fWywelDmMMVu"
import json

input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})
```

```python id="JHPPRKrHMMVu" outputId="88a6116c-dbb3-4243-d5a7-e360bf98dbed"
repr(input_data_json)[:1500] + "..."
```

<!-- #region id="kCaIoVHSMMVu" -->
Now let's use TensorFlow Serving's REST API to make predictions:
<!-- #endregion -->

```python id="jtzCndhkMMVu"
import requests

SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()
```

```python id="q1IdWf55MMVv" outputId="ed0227de-ba3a-4b79-ae01-c060a4b9db4c"
response.keys()
```

```python id="pInmOnplMMVv" outputId="ea30adf4-fa75-4c25-cd45-37c39181727c"
y_proba = np.array(response["predictions"])
y_proba.round(2)
```

<!-- #region id="puNixLrZMMVv" -->
Using the gRPC API
<!-- #endregion -->

```python id="JmE5VuqOMMVw"
from tensorflow_serving.apis.predict_pb2 import PredictRequest

request = PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))
```

```python id="ECKQgkT2MMVw"
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)
```

```python id="5mQKip-1MMVw" outputId="5ed6d69a-9fea-44b4-8d21-92d2a11f3ad8"
response
```

<!-- #region id="XwTAZmpZMMVx" -->
Convert the response to a tensor:
<!-- #endregion -->

```python id="s_P7-32gMMVx" outputId="8bca34d4-bdf2-47a9-d057-7dfee90faf04"
output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)
y_proba.round(2)
```

<!-- #region id="K613UqqqMMVx" -->
Or to a NumPy array if your client does not include the TensorFlow library:
<!-- #endregion -->

```python id="T22bo8RKMMVy" outputId="e67e9b0b-2cb5-4406-b149-621717d6f634"
output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
shape = [dim.size for dim in outputs_proto.tensor_shape.dim]
y_proba = np.array(outputs_proto.float_val).reshape(shape)
y_proba.round(2)
```

<!-- #region id="RRnUMBJ_MMVy" -->
### Deploying a new model version
<!-- #endregion -->

```python id="DSpx5xAqMMVy" outputId="833d2c6c-99f2-45a5-934c-172aed290568"
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
```

```python id="E_aKHS4RMMVz" outputId="5c9971e3-5b6f-44d7-f0d3-08d361edb6c0"
model_version = "0002"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
model_path
```

```python id="Ayq7CcYFMMVz" outputId="bfa168cd-4e8f-473d-8135-15261fce3b14"
tf.saved_model.save(model, model_path)
```

```python id="Y3qaorHgMMV0" outputId="dd47e59d-bdb9-4c77-ade0-5cbf2c090a02"
for root, dirs, files in os.walk(model_name):
    indent = '    ' * root.count(os.sep)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + '    ', filename))
```

<!-- #region id="Nj7OhNTdMMV0" -->
**Warning**: You may need to wait a minute before the new model is loaded by TensorFlow Serving.
<!-- #endregion -->

```python id="SeXgjlaGMMV0"
import requests

SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
            
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status()
response = response.json()
```

```python id="vQqGEobqMMV1" outputId="19d0ea03-1e71-4d3e-c2e8-42265ed8a71e"
response.keys()
```

```python id="CnO_rUE2MMV1" outputId="c215d9b6-4a7e-4cdf-cfb2-be836fd7328c"
y_proba = np.array(response["predictions"])
y_proba.round(2)
```

<!-- #region id="n4GNO0HsMMV2" -->
## Deploy the model to Google Cloud AI Platform
<!-- #endregion -->

<!-- #region id="KgxDdJXOMMV2" -->
Follow the instructions in the book to deploy the model to Google Cloud AI Platform, download the service account's private key and save it to the `my_service_account_private_key.json` in the project directory. Also, update the `project_id`:
<!-- #endregion -->

```python id="OgC-wMkwMMV2"
project_id = "onyx-smoke-242003"
```

```python id="ZoionJWBMMV2"
import googleapiclient.discovery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_service_account_private_key.json"
model_id = "my_mnist_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
model_path += "/versions/v0001/" # if you want to run a specific version
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()
```

```python id="1ebFPWl5MMV3"
def predict(X):
    input_data_json = {"signature_name": "serving_default",
                       "instances": X.tolist()}
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred[output_name] for pred in response["predictions"]])
```

```python id="pUu65wdEMMV4" outputId="c6c1036b-afca-4fb7-8387-f1a86bed25ce"
Y_probas = predict(X_new)
np.round(Y_probas, 2)
```

<!-- #region id="WXWPZiJxMMV4" -->
## Using GPUs
<!-- #endregion -->

<!-- #region id="-AjRLX-gMMV5" -->
**Note**: `tf.test.is_gpu_available()` is deprecated. Instead, please use `tf.config.list_physical_devices('GPU')`.
<!-- #endregion -->

```python id="nvwjeCTWMMV5" outputId="74f1e10d-f92e-4933-8d5d-1bdc108a9de0"
#tf.test.is_gpu_available() # deprecated
tf.config.list_physical_devices('GPU')
```

```python id="-gt7ipT9MMV6" outputId="21dc3080-da77-47f4-b1fb-f3876b4487ef"
tf.test.gpu_device_name()
```

```python id="kvFOBE04MMV6" outputId="edd2f878-0f27-4bb0-c4d9-497b049ace2a"
tf.test.is_built_with_cuda()
```

```python id="Cf6qgfIDMMV7" outputId="e3a5a30a-9402-4ffe-e03e-6a54a9c23cd8"
from tensorflow.python.client.device_lib import list_local_devices

devices = list_local_devices()
devices
```

<!-- #region id="lq8ArF2iMMV7" -->
## Distributed Training
<!-- #endregion -->

```python id="yiI353HlMMV8"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
```

```python id="lu9J_PgLMMV8"
def create_model():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                            padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"), 
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])
```

```python id="U_KLdaD-MMV9" outputId="ee2fcaaf-c8b8-4eda-f904-33bc5811f9ba"
batch_size = 100
model = create_model()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)
```

```python id="aWUhzMFgMMV9" outputId="db2c4683-618f-4065-8785-04d69ea2271e"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

distribution = tf.distribute.MirroredStrategy()

# Change the default all-reduce algorithm:
#distribution = tf.distribute.MirroredStrategy(
#    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# Specify the list of GPUs to use:
#distribution = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

# Use the central storage strategy instead:
#distribution = tf.distribute.experimental.CentralStorageStrategy()

#if IS_COLAB and "COLAB_TPU_ADDR" in os.environ:
#  tpu_address = "grpc://" + os.environ["COLAB_TPU_ADDR"]
#else:
#  tpu_address = ""
#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
#tf.config.experimental_connect_to_cluster(resolver)
#tf.tpu.experimental.initialize_tpu_system(resolver)
#distribution = tf.distribute.experimental.TPUStrategy(resolver)

with distribution.scope():
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-2),
                  metrics=["accuracy"])
```

```python id="h0tWEDY0MMV-" outputId="1444bc65-ee34-4e5e-b8c9-4cd1284de074"
batch_size = 100 # must be divisible by the number of workers
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)
```

```python id="S3YhoFrUMMWA" outputId="de203616-8097-4b11-8ced-b91e37512a24"
model.predict(X_new)
```

<!-- #region id="-HvACqDPMMWA" -->
Custom training loop:
<!-- #endregion -->

```python id="dlg--ccjMMWA" outputId="43413b97-17c3-40f9-af60-4a20ee25d8b8"
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

K = keras.backend

distribution = tf.distribute.MirroredStrategy()

with distribution.scope():
    model = create_model()
    optimizer = keras.optimizers.SGD()

with distribution.scope():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
    input_iterator = distribution.make_dataset_iterator(dataset)
    
@tf.function
def train_step():
    def step_fn(inputs):
        X, y = inputs
        with tf.GradientTape() as tape:
            Y_proba = model(X)
            loss = K.sum(keras.losses.sparse_categorical_crossentropy(y, Y_proba)) / batch_size

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    per_replica_losses = distribution.experimental_run(step_fn, input_iterator)
    mean_loss = distribution.reduce(tf.distribute.ReduceOp.SUM,
                                    per_replica_losses, axis=None)
    return mean_loss

n_epochs = 10
with distribution.scope():
    input_iterator.initialize()
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for iteration in range(len(X_train) // batch_size):
            print("\rLoss: {:.3f}".format(train_step().numpy()), end="")
        print()
```

<!-- #region id="cvznPMh9MMWB" -->
## Training across multiple servers
<!-- #endregion -->

<!-- #region id="DQwlTLLiMMWC" -->
A TensorFlow cluster is a group of TensorFlow processes running in parallel, usually on different machines, and talking to each other to complete some work, for example training or executing a neural network. Each TF process in the cluster is called a "task" (or a "TF server"). It has an IP address, a port, and a type (also called its role or its job). The type can be `"worker"`, `"chief"`, `"ps"` (parameter server) or `"evaluator"`:
* Each **worker** performs computations, usually on a machine with one or more GPUs.
* The **chief** performs computations as well, but it also handles extra work such as writing TensorBoard logs or saving checkpoints. There is a single chief in a cluster, typically the first worker (i.e., worker #0).
* A **parameter server** (ps) only keeps track of variable values, it is usually on a CPU-only machine.
* The **evaluator** obviously takes care of evaluation. There is usually a single evaluator in a cluster.

The set of tasks that share the same type is often called a "job". For example, the "worker" job is the set of all workers.

To start a TensorFlow cluster, you must first define it. This means specifying all the tasks (IP address, TCP port, and type). For example, the following cluster specification defines a cluster with 3 tasks (2 workers and 1 parameter server). It's a dictionary with one key per job, and the values are lists of task addresses:
<!-- #endregion -->

```python id="lEeH0O4lMMWC"
cluster_spec = {
    "worker": [
        "machine-a.example.com:2222",  # /job:worker/task:0
        "machine-b.example.com:2222"   # /job:worker/task:1
    ],
    "ps": ["machine-c.example.com:2222"] # /job:ps/task:0
}
```

<!-- #region id="9cEfITEZMMWD" -->
Every task in the cluster may communicate with every other task in the server, so make sure to configure your firewall to authorize all communications between these machines on these ports (it's usually simpler if you use the same port on every machine).

When a task is started, it needs to be told which one it is: its type and index (the task index is also called the task id). A common way to specify everything at once (both the cluster spec and the current task's type and id) is to set the `TF_CONFIG` environment variable before starting the program. It must be a JSON-encoded dictionary containing a cluster specification (under the `"cluster"` key), and the type and index of the task to start (under the `"task"` key). For example, the following `TF_CONFIG` environment variable defines the same cluster as above, with 2 workers and 1 parameter server, and specifies that the task to start is worker #1:
<!-- #endregion -->

```python id="L89Vg3adMMWE" outputId="7eea3f4c-f12f-439f-b188-c07ece5d3dd0"
import os
import json

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec,
    "task": {"type": "worker", "index": 1}
})
os.environ["TF_CONFIG"]
```

<!-- #region id="fUrUw8QtMMWE" -->
Some platforms (e.g., Google Cloud ML Engine) automatically set this environment variable for you.
<!-- #endregion -->

<!-- #region id="9CWqlv-oMMWE" -->
TensorFlow's `TFConfigClusterResolver` class reads the cluster configuration from this environment variable:
<!-- #endregion -->

```python id="i52cpy9kMMWF" outputId="684415ed-126f-458b-f254-24104dc1e682"
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
resolver.cluster_spec()
```

```python id="ocjW6jGfMMWF" outputId="c72ac882-ec89-42b3-d0a9-1d28ab4cb46f"
resolver.task_type
```

```python id="tZwQx3YgMMWF" outputId="c90a5e83-ba55-4e6f-fdee-83a30ef2d37c"
resolver.task_id
```

<!-- #region id="3EgkwFHUMMWG" -->
Now let's run a simpler cluster with just two worker tasks, both running on the local machine. We will use the `MultiWorkerMirroredStrategy` to train a model across these two tasks.

The first step is to write the training code. As this code will be used to run both workers, each in its own process, we write this code to a separate Python file, `my_mnist_multiworker_task.py`. The code is relatively straightforward, but there are a couple important things to note:
* We create the `MultiWorkerMirroredStrategy` before doing anything else with TensorFlow.
* Only one of the workers will take care of logging to TensorBoard and saving checkpoints. As mentioned earlier, this worker is called the *chief*, and by convention it is usually worker #0.
<!-- #endregion -->

```python id="0tCjjbT7MMWG" outputId="e1282ba2-3628-4032-9ae4-728fa7e8ed68"
%%writefile my_mnist_multiworker_task.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

# At the beginning of the program
distribution = tf.distribute.MultiWorkerMirroredStrategy()

resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print("Starting task {}{}".format(resolver.task_type, resolver.task_id))

# Only worker #0 will write checkpoints and log to TensorBoard
if resolver.task_id == 0:
    root_logdir = os.path.join(os.curdir, "my_mnist_multiworker_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_dir = os.path.join(root_logdir, run_id)
    callbacks = [
        keras.callbacks.TensorBoard(run_dir),
        keras.callbacks.ModelCheckpoint("my_mnist_multiworker_model.h5",
                                        save_best_only=True),
    ]
else:
    callbacks = []

# Load and prepare the MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis] / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

with distribution.scope():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                            padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"), 
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-2),
                  metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=10, callbacks=callbacks)
```

<!-- #region id="fhBUcT2QMMWH" -->
In a real world application, there would typically be a single worker per machine, but in this example we're running both workers on the same machine, so they will both try to use all the available GPU RAM (if this machine has a GPU), and this will likely lead to an Out-Of-Memory (OOM) error. To avoid this, we could use the `CUDA_VISIBLE_DEVICES` environment variable to assign a different GPU to each worker. Alternatively, we can simply disable GPU support, like this:
<!-- #endregion -->

```python id="zswoXr5uMMWI"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

<!-- #region id="hbh3ypMtMMWI" -->
We are now ready to start both workers, each in its own process, using Python's `subprocess` module. Before we start each process, we need to set the `TF_CONFIG` environment variable appropriately, changing only the task index:
<!-- #endregion -->

```python id="HIKVyu6dMMWI"
import subprocess

cluster_spec = {"worker": ["127.0.0.1:9901", "127.0.0.1:9902"]}

for index, worker_address in enumerate(cluster_spec["worker"]):
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "worker", "index": index}
    })
    subprocess.Popen("python my_mnist_multiworker_task.py", shell=True)
```

<!-- #region id="GRBBkWjSMMWJ" -->
That's it! Our TensorFlow cluster is now running, but we can't see it in this notebook because it's running in separate processes (but if you are running this notebook in Jupyter, you can see the worker logs in Jupyter's server logs).

Since the chief (worker #0) is writing to TensorBoard, we use TensorBoard to view the training progress. Run the following cell, then click on the settings button (i.e., the gear icon) in the TensorBoard interface and check the "Reload data" box to make TensorBoard automatically refresh every 30s. Once the first epoch of training is finished (which may take a few minutes), and once TensorBoard refreshes, the SCALARS tab will appear. Click on this tab to view the progress of the model's training and validation accuracy.
<!-- #endregion -->

```python id="J34Bvd1gMMWK" outputId="35d7e823-cfa5-4d4a-a4b9-e9f6cc7ae0ed"
%load_ext tensorboard
%tensorboard --logdir=./my_mnist_multiworker_logs --port=6006
```

<!-- #region id="GGQu5pphMMWL" -->
That's it! Once training is over, the best checkpoint of the model will be available in the `my_mnist_multiworker_model.h5` file. You can load it using `keras.models.load_model()` and use it for predictions, as usual:
<!-- #endregion -->

```python id="T2Tyo-yaMMWM" outputId="42281594-c650-477c-b2db-d641092bdfd6"
from tensorflow import keras

model = keras.models.load_model("my_mnist_multiworker_model.h5")
Y_pred = model.predict(X_new)
np.argmax(Y_pred, axis=-1)
```

<!-- #region id="1QwgqdJ6MMWM" -->
And that's all for today! Hope you found this useful.
<!-- #endregion -->
