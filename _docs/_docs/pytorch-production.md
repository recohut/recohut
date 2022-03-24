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

<!-- #region id="_k0sGYBl_wwQ" -->
# Deploying PyTorch to Production - torchscript, torchserve, Flask, ONNX
<!-- #endregion -->

<!-- #region id="AIH22088bZ3x" -->
In the past, going from research to production was a challenging task that required a team of software engineers to move PyTorch models to a framework and integrate them nto a (often non-Python) production environment. Today, PyTorch includes built-in tools and external libraries to support rapid deployment to a variety of production environments.
<!-- #endregion -->

<!-- #region id="f-J0XMNtbaIc" -->
We will focus on deploying your model for inference, not training, and we’ll explore how to deploy your trained PyTorch models into a variety of applications. First, I’ll describe the various built-in capabilities and tools within PyTorch that you can use for deployment. Tools like TorchServe and TorchScript allow you to easily deploy your PyTorch models to the cloud and to mobile or edge devices.

Depending on the application and environment, you may have several options for deployment, each with its own trade-offs. I’ll show you examples of how you can deploy your PyTorch models in multiple cloud and edge environments. You’ll learn how to deploy to web servers for development and production at scale, to iOS and Android mobile devices, and to Internet of Things (IoT) devices based on ARM processors, GPUs, and field-programmable gate array (FPGA) hardware.
<!-- #endregion -->

<!-- #region id="octSI6UkbkWI" -->
Table below summarizes the various resources available for deployment and indicates how to appropriately use each one.
<!-- #endregion -->

<!-- #region id="fy9AjJSbbylc" -->
| Resource | Use |
| -------- | --- |
| Python API | Perform fast prototyping, training, and experimentation; program Python runtimes. |
| TorchScript | Improve performance and portability (e.g., load and run a model in C++); </br>program non-Python runtimes or strict latency and performance requirements. |
| TorchServe | A fast production environment tool with model store, A/B testing, monitoring, and RESTful API. |
| ONNX | Deploy to systems with ONNX runtimes or FPGA devices. |
| Mobile libraries | Deploy to iOS and Android devices. |
<!-- #endregion -->

<!-- #region id="AhZUOMzHcHYN" -->
 For our examples, we’ll deploy an image classifier using a VGG16 model pretrained with ImageNet data. That way, each section can focus on the deployment approach used and not the model itself. For each approach, you can replace the VGG16 model with one of your own and follow the same workflow to achieve results with your own designs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 103, "referenced_widgets": ["2a1988f6a0a34a0a99bbd99f24258f45", "b35989fd782d418ba4b2f048f749018d", "5b5eb7f6259142e492bced287506b910", "81e04d276642438681832cb7b6fdd634", "f45cd15a931e4aa18b92922e077cd9cf", "a74b85e57f11464f838fcbd3f4aae999", "76fe117af6e9495b884dde2290399e9b", "edb608353ddb455ea000c5148ceacc24", "8baca337637541b9b129ac8126ee7149", "4dfc58f9d43a4ec6acfe8f98cab88c82", "ad029e16c5a84880bb57e939e1b939cb"]} id="6rFvzHR9eT5I" executionInfo={"status": "ok", "timestamp": 1631176815691, "user_tz": -330, "elapsed": 7903, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2437d5bc-6b14-493d-a79e-3f85d5cc25c9"
import numpy as np
from torchvision.models import vgg16

model = vgg16(pretrained=True)

model_parameters = filter(lambda p: 
      p.requires_grad, model.parameters())

params = sum([np.prod(p.size()) for 
      p in model_parameters])
print(params)
```

```python id="iHjAwVv0jIdX"
import torch
torch.save(model.state_dict(), "./vgg16_model.pt")
```

<!-- #region id="jQ56otMTe7ni" -->
The VGG16 model has 138,357,544 trainable parameters. As we go through each approach, keep in mind the performance at this level of complexity. You can use this as a rough benchmark when comparing the complexity of your models.

After we instantiate the VGG16 model, it requires minimal effort to deploy it in a Python application.
<!-- #endregion -->

<!-- #region id="Sn3EMcuZe--U" -->
> Tip: Python is not always used in production environments due to its slower performance and lack of true multithreading. If your production environment uses another language (e.g., C++, Java, Rust, or Go), you can convert your models to TorchScript code.
<!-- #endregion -->

<!-- #region id="jdTEXC9xfJXk" -->
TorchScript is a way to serialize and optimize your PyTorch model code so that your PyTorch models can be saved and executed in non-Python runtime environments with no dependency on Python. TorchScript is commonly used to run PyTorch models in C++ and with any language that supports C++ bindings.

TorchScript represents a PyTorch model in a format that can be understood, compiled, and serialized by the TorchScript compiler. The TorchScript compiler creates a serialized, optimized version of your model that can be used in C++ applications. To load your TorchScript model in C++, you would use the PyTorch C++ API library called LibTorch.

There are two ways to convert your PyTorch models to TorchScript. The first one is called tracing, which is a process in which you pass in an example input and perform the conversion with one line of code. It’s used in most cases. The second is called scripting, and it’s used when your model has more complex control code. For example, if your model has conditional if statements that depend on the input itself, you’ll want to use scripting. Let’s take a look at some reference code for each case.

Since our VGG16 example model does not have any control flow, we can use tracing to convert our model to TorchScript, as shown in the following code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hZXq7ugAfxen" executionInfo={"status": "ok", "timestamp": 1631178358620, "user_tz": -330, "elapsed": 21769, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0ea3961b-afe4-4dee-df48-f97e0e2c238e"
import torch

model = vgg16(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
torchscript_model = torch.jit.trace(model,
                            example_input)
torchscript_model.save("traced_vgg16_model.pt")
```

<!-- #region id="ejtVptEegAsH" -->
If our model used control flow, we would need to use the annotation method to convert it to TorchScript. Let’s consider the following model:
<!-- #endregion -->

```python id="Lii7cb_4gQif"
import torch.nn as nn

class ControlFlowModel(nn.Module):
  def __init__(self, N):
    super(ControlFlowModel, self).__init__()
    self.fc = nn.Linear(N,100)

  def forward(self, input):
    if input.sum() > 0:
      output = input
    else:
      output = -input
    return output

model = ControlFlowModel(10)
torchcript_model = torch.jit.script(model)
torchscript_model.save("scripted_vgg16_model.pt")
```

<!-- #region id="FqrTTQnsgTtf" -->
Now we can use our model in a C++ application, as shown in the following C++ code:
<!-- #endregion -->

<!-- #region id="GOyE5f-QgqQG" -->
```cpp
include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app" >> \
      "<path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  model = torch::jit::load(argv[1]);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( \
      torch::ones({1, 3, 224, 224}));

  at::Tensor output = model.forward(inputs).toTensor();
  std::cout \
    << output.slice(/*dim=*/1, \
        /*start=*/0, /*end=*/5) \
    << '\N';
  }

}
```
<!-- #endregion -->

<!-- #region id="Z1ElKSnKhBYS" -->
In this section, we used TorchScript to increase the performance of our model when it’s used in a C++ application or in a language that binds to C++. However, deploying PyTorch models at scale requires additional capabilities, like packaging models, configuring runtime environments, exposing API endpoints, logging and monitoring, and managing multiple model versions. Fortunately, PyTorch provides a tool called TorchServe to facilitate these tasks and rapidly deploy your models for inference at scale.
<!-- #endregion -->

<!-- #region id="FPLXvE9JhQud" -->
TorchServe is an open-source model-serving framework that makes it easy to deploy trained PyTorch models. It was developed by AWS engineers and jointly released with Facebook in April 2020, and it is actively maintained by AWS. TorchServe supports all the features needed to deploy models to production at scale, including multimodel serving, model versioning for A/B testing, logging and metrics for monitoring, and a RESTful API for integration with other systems.
<!-- #endregion -->

<!-- #region id="BQRLNxkjhnB4" -->
<!-- #endregion -->

<!-- #region id="jv0S9C7GhfV1" -->
The client application interfaces with TorchServe through multiple APIs. The Inference API provides the main inference requests and predictions. The client application sends input data through the RESTful API request and receives the prediction results. The Management API allows you to register and manage your deployed models. You can register, unregister, set default models, configure A/B testing, check status, and specify the number of workers for a model. The Metrics API allows you to monitor each model’s performance.

TorchServe runs all model instances and captures server logs. It processes the frontend APIs and manages the model storage to disk. TorchServe also provides a number of default handlers for common applications like object detection and text classification. The handlers take care of converting data from the API into a format that your model will process. This helps speed up deployment since you don’t have to write custom code for these common applications.
<!-- #endregion -->

<!-- #region id="jzKrDmpTh_CC" -->
To deploy your models via TorchServe, you will need to follow a few steps. First you need to install TorchServe’s tools. Then you’ll package your model using the model archiver tool. Once your models are archived, you’ll then run the TorchServe web server. Once the web server is running, you can use its APIs to request predictions, manage your models, perform monitoring, or access server logs.
<!-- #endregion -->

```python id="vjp7DZDsiHWJ"
!pip install torchserve torch-model-archiver
!pip install image_classifier captum
```

<!-- #region id="TDErgaNni0hY" -->
TorchServe has the ability to package all model artifacts into a single-model archive file. To do so, we will use the torch-model-archiver command-line tool that we installed in the previous step. It packages model checkpoints as well as the state_dict into a .mar file that the TorchServe server uses to serve the model.

You can use the torch-model-archiver to archive your TorchScript models as well as the standard “eager-mode” implementations, as shown in the following code.
<!-- #endregion -->

```python id="CQ2cD5c_jAGG"
!torch-model-archiver --model-name vgg16 \
  --version 1.0 --serialized-file traced_vgg16_model.pt --handler \
  image_classifier
```

```python id="-pzHqEg_o-r3"
!mkdir -p /content/models && mv ./*.mar /content/models
```

<!-- #region id="7mbTM7uXjyzA" -->
TorchServe includes a built-in web server that is run from the command line. It wraps one or more PyTorch models in a set of REST APIs and provides controls for configuring the port, host, and logging. The following command starts the web server with all models in the model store located in the /models folder:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="B9m6cczWrnSY" executionInfo={"status": "ok", "timestamp": 1631176891683, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1fc789b9-adb1-4888-b271-7fed9d49e496"
%%writefile config.properties
inference_address=https://0.0.0.0:8091
management_address=https://0.0.0.0:8092
metrics_address=https://0.0.0.0:8093
```

```python colab={"base_uri": "https://localhost:8080/"} id="E1z42ss7n06P" executionInfo={"status": "ok", "timestamp": 1631178418135, "user_tz": -330, "elapsed": 450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="97f22199-ddc3-4201-c8ef-1cdaecb405e2"
!nohup torchserve --model-store ./models --start --models all --ts-config ./config.properties --ncs --foreground &
```

```python colab={"base_uri": "https://localhost:8080/"} id="LYMkyNgyuemQ" executionInfo={"status": "ok", "timestamp": 1631178421316, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b690ea2d-27c0-4b2e-a21e-4c917d59b808"
!tail nohup.out
```

```python colab={"base_uri": "https://localhost:8080/"} id="IyFjEuEwvRxX" executionInfo={"status": "ok", "timestamp": 1631178454865, "user_tz": -330, "elapsed": 707, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03064fa4-231e-4738-f65a-24333c5907d9"
!wget -O hotdog.jpg -q --show-progress https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTk5onR1hxG2h_yGFkgZvLVu7b7IY2PIuekKaagBG0nYFsqktcIwjYu6a7LT6OjTfEHWAU&usqp=CAU
```

```python colab={"base_uri": "https://localhost:8080/"} id="fN9mCqqzwn2r" executionInfo={"status": "ok", "timestamp": 1631178460734, "user_tz": -330, "elapsed": 1052, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="10b34086-e242-42e5-a62b-056c2f670bc3"
!curl --insecure https://localhost:8091/ping
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZNW6rFTZuzcA" executionInfo={"status": "ok", "timestamp": 1631178477023, "user_tz": -330, "elapsed": 1211, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="216ecda0-3ba1-4a79-de79-73bbfaec87dc"
!curl --insecure https://localhost:8091/predictions/vgg16 -T hotdog.jpg
```

```python colab={"base_uri": "https://localhost:8080/"} id="cM4HAOmJn4_m" executionInfo={"status": "ok", "timestamp": 1631178320718, "user_tz": -330, "elapsed": 435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9bde44ef-22f4-4983-8b57-41fe29f60320"
!torchserve --stop
```

<!-- #region id="peYYCsZ0yQn2" -->
You can configure metrics using the Metrics API and monitor and log your models’ performance when deployed.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ysiXL3KT0QOl" executionInfo={"status": "ok", "timestamp": 1631178627264, "user_tz": -330, "elapsed": 381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6f40f54-71d6-4166-8cef-14ed10affff9"
!curl --insecure https://127.0.0.1:8093/metrics
```

<!-- #region id="O5fAtWKT0UwD" -->
The default metrics endpoint returns Prometheus-formatted metrics. Prometheus is a free software application used for event monitoring and alerting that records real-time metrics in a time series database built using an HTTP pull model. You can query metrics using curl requests or point a Prometheus Server to the endpoint and use Grafana for dashboards.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oLK9D-Bs060e" executionInfo={"status": "ok", "timestamp": 1631178705649, "user_tz": -330, "elapsed": 387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="13f073e0-ab7e-4bf3-d3c3-f427ff7b36eb"
!ls -al ./logs
```

<!-- #region id="DvMWsigQ00Jg" -->
Metrics are logged to a file. TorchServe also supports other types of server logging, including access logs and TorchServe logs. Access logs record the inference requests and the time it takes to complete the requests. As defined in the properties file, the access logs are collected in the <log_location>/access_log.log file. TorchServe logs collect all the logs from TorchServe and its backend workers.

TorchServe supports capabilities beyond the default settings for metrics and logging. Metrics and logging can be configured in many different ways. In addition, you can create custom logs. For more information on metric and logging customization and other advanced features of TorchServe, refer to the TorchServe documentation.
<!-- #endregion -->

<!-- #region id="b4VuOcQI08YV" -->
The NVIDIA Triton Inference Server is becoming more popular and is also used to deploy AI models at scale in production. Although not part of the PyTorch project, you may want to consider the Triton Inference Server as an alternative to TorchServe, especially when deploying to NVIDIA GPUs.

The Triton Inference Server is open source software and can load models from local storage, GCP, or AWS S3. Triton supports running multiple models on single or multiple GPUs, low latency and shared memory, and model ensembles. Some possible advantages of Triton over TorchServe include:

- Triton is out of beta.
- It is the fastest way to infer on NVIDIA hardware (common).
- It can use int4 quantization.
- You can port directly from PyTorch without ONNX.

Available as a Docker container, Triton Inference Server also integrates with Kubernetes for orchestration, metrics, and auto-scaling. For more information, visit the NVIDIA Triton Inference Server documentation.
<!-- #endregion -->

<!-- #region id="vsyxzX1m1Mvf" -->
If your platform doesn’t support PyTorch and you cannot use TorchScript/C++ or TorchServe for your deployment, it may be possible that your deployment platform supports the Open Neural Network Exchange (ONNX) format. The ONNX format defines a common set of operators and a common file format so that deep learning engineers can use models across a variety of frameworks, tools, runtimes, and compilers.

ONNX was developed by Facebook and Microsoft to allow model interoperability between PyTorch and other frameworks, such as Caffe2 and Microsoft Cognitive Toolkit (CTK). ONNX is currently supported by inference runtimes from a number of providers, including Cadence Systems, Habana, Intel AI, NVIDIA, Qualcomm, Tencent, Windows, and Xilinx.
<!-- #endregion -->

<!-- #region id="gdSZDxYV3VM1" -->
An example use case is edge deployment on a Xilinx FPGA device. FPGA devices are custom chips that can be programmed with specific logic. They are used by edge devices for low-latency or high-performance applications, like video. If you want to deploy your new innovative model to an FPGA device, you would first convert it to ONNX format and then use the Xilinx FPGA development tools to generate an FPGA image with your model’s implementation.

Let’s take a look at an example of how to export a model to ONNX, again using our VGG16 model. The ONNX exporter can use tracing or scripting. We learned about tracing and scripting, described in the earlier section on TorchScript. We can use tracing by simply providing the model and an example input. The following code shows how we’d export our VGG16 model to ONNX using tracing:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cXNIAeV53gCP" executionInfo={"status": "ok", "timestamp": 1631179384484, "user_tz": -330, "elapsed": 6555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c8d84a16-39b5-4514-8074-af4becee541e"
!pip install onnx
```

```python id="zqG53qe83WnT"
import onnx

model = vgg16(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
onnx_model = torch.onnx.export(model, 
                               example_input, 
                               "vgg16.onnx")
```

<!-- #region id="liFkPfnE3sEM" -->
We define an example input and call torch.onnx.export(). The resulting file, vgg16.onnx, is a binary protobuf file that contains both the network structure and the parameters of the VGG16 model we exported.

If we want to verify that our model was converted to ONNX properly, we can use the ONNX checker, as shown in the following code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 630} id="gwZPQCjo3sV9" executionInfo={"status": "ok", "timestamp": 1631179449888, "user_tz": -330, "elapsed": 680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6adcdb37-3e35-482b-9bf4-21123defc164"
model = onnx.load("vgg16.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)
```

<!-- #region id="WOp5RRno3twm" -->
This code uses the Python ONNX library to load the model, run the checker, and print out a human-readable version of the model.
<!-- #endregion -->

<!-- #region id="s9mdnqB13y4K" -->
Before deploying to full-scale production, you may want to deploy your models to a development web server. This enables you to integrate your deep learning algorithms with other systems and quickly build prototypes to demonstrate your new models. One of the easiest ways to build a development server is with Python using Flask.

Flask is a simple micro web framework written in Python. It is called a “micro” framework because it does not include a database abstraction layer, form validation, upload handling, various authentication technologies, or anything else that might be provided with other libraries. We won’t cover Flask in depth in this book, but I’ll show you how to use Flask to deploy your models in Python.

We’ll also expose a REST API so that other applications can pass in data and receive predictions. In the following examples, we’ll deploy our pretrained VGG16 model and classify images. First we’ll define our API endpoints, request types, and response types. Our API endpoint will be at /predict, which takes in POST requests (including the image file). The response will be in JSON format and contain a class_id and class_name from the ImageNet dataset.
<!-- #endregion -->

<!-- #region id="1xT7m0L04V5S" -->
Since our model will return a number indicating the class, we’ll need a lookup table to convert this number to a class name. We create a dictionary called imagenet_class_index by reading in the JSON conversion file. We then instantiate our VGG16 model and define our image transforms to preprocess a PIL image by resizing it, center-cropping it, converting it to a tensor, and normalizing it. These steps are required prior to sending the image into our model.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DvXu2jFJ4PEo" executionInfo={"status": "ok", "timestamp": 1631179606140, "user_tz": -330, "elapsed": 660, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4ce7dc4c-bf15-4d78-864a-203669fa55f5"
!wget -q --show-progress "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
```

```python colab={"base_uri": "https://localhost:8080/"} id="yEt2UCvW82ge" executionInfo={"status": "ok", "timestamp": 1631181726902, "user_tz": -330, "elapsed": 2425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fb929622-c25c-4e8c-d4af-52cf138f5605"
import socket
print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))

import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

imagenet_class_index = json.load(
    open("./imagenet_class_index.json"))

model = models.vgg16(pretrained=True)

image_transforms = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
          [0.485, 0.456, 0.406],
          [0.229, 0.224, 0.225])])

def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    tensor = image_transforms(image)
    outputs = model(tensor.unsqueeze(0))
    _, y = outputs.max(1)
    predicted_idx = str(y.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
  img_bytes = file.read()
  class_id, class_name = \
    get_prediction(image_bytes=img_bytes)
  return jsonify({'class_id': class_id,
                 'class_name': class_name})

import threading
threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':5062}).start() 
```

```python colab={"base_uri": "https://localhost:8080/"} id="U-SB5eog5AXy" executionInfo={"status": "ok", "timestamp": 1631181733705, "user_tz": -330, "elapsed": 1115, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9ca0e503-c513-42f1-eaa9-3c0f8561f93d"
import requests

resp = requests.post(
    "http://localhost:5062/predict",
    files={"file": open('hotdog.jpg','rb')})

print(resp.json())
```

```python id="Fs1SEQ1WAfgQ"

```
