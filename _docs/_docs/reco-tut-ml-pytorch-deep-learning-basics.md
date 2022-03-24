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

<!-- #region id="xCdsm69B-zyJ" -->
You’ll build a deep learning model and train the model using a common training loop structure. Then, you’ll test your model’s performance and tweak hyperparameters to improve your results and training speed. Finally, we’ll explore ways to deploy your model to prototype systems or production.
<!-- #endregion -->

<!-- #region id="X_NpMaWa-0It" -->
First, we load this data and convert it to numeric values in the form of tensors. The tensors will act as inputs during the model training stage; however, before they are passed in, the tensors are usually preprocessed via transforms and grouped into batches for better training performance. Thus, the data preparation stage takes generic data and converts it to batches of tensors that can be passed into your NN model.
<!-- #endregion -->

<!-- #region id="w3DtG7yY_FSm" -->
Next, in the model experimentation and development stage, we will design an NN model, train the model with our training data, test its performance, and optimize our hyperparameters to improve performance to a desired level. To do so, we will separate our dataset into three parts: one for training, one for validation, and one for testing. We’ll design an NN model and train its parameters with our training data. PyTorch provides elegantly designed modules and classes in the torch.nn module to help you create and train your NNs. We will define a loss function and optimizer from a selection of the many built-in PyTorch functions. Then we’ll perform backpropagation and update the model parameters in our training loop.
<!-- #endregion -->

<!-- #region id="MDCzltgR_Wf7" -->
Within each epoch, we’ll also validate our model by passing in validation data, measuring performance, and potentially tuning hyperparameters. Finally, we’ll test our model by passing in test data and measuring the model’s performance against unseen data. In practice, validation and test loops may be optional, but we show them here for completeness.
<!-- #endregion -->

<!-- #region id="JJltU6wr_h7o" -->
The last stage of deep learning model development is the model deployment stage. In this stage, we have a fully trained model—so what do we do with it? If you are a deep learning research scientist conducting experiments, you may want to simply save the model to a file and load it for further research and experimentation, or you may want to provide access to it via a repository like PyTorch Hub. You may also want to deploy it to an edge device or local server to demonstrate a prototype or a proof of concept.

On the other hand, if you are a software developer or systems engineer, you may want to deploy your model to a product or service. In this case, you can deploy your model to a production environment on a cloud server or deploy it to an edge device or mobile phone. When deploying trained models, the model often requires additional postprocessing. For example, you may classify a batch of images, but you only want to report the most confident result. The model deployment stage also handles any postprocessing that is needed to go from your model’s output values to the final solution.
<!-- #endregion -->

<!-- #region id="Ml-b4gkD_xYS" -->
PyTorch provides powerful built-in classes and utilities, such as the Dataset, DataLoader, and Sampler classes, for loading various types of data. The Dataset class defines how to access and preprocess data from a file or data sources. The Sampler class defines how to sample data from a dataset in order to create batches, while the DataLoader class combines a dataset with a sampler and allows you to iterate over a set of batches.
<!-- #endregion -->

```python id="pIC4ZPIOATAb" executionInfo={"status": "ok", "timestamp": 1631165196298, "user_tz": -330, "elapsed": 4686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch
import torchvision

from torchvision.datasets import CIFAR10
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["3b1bbde53e234e2ea985f0f07877c792", "0a799c17626b43b8b38b79dacde55151", "dfee4821887947a08e77015f08dbb098", "d21e616effd549e88a3b6ae5a9111dd3", "b6b4b2c8fb56426391335802bb4a308b", "e1ea823e09fc45dc8ae837c9f125f3f3", "ba713e8d5e2a4fbd985879ace34a8ab2", "f40e9824762146c8b7da2a0fa7c40899", "85df6f0c4928469dbed5c8eb554e612e", "5e10a87722e747c3861e535c1bbcb6ee", "0a9fb274ebdb43d5901e59bd3c9caacb"]} id="d-kAioq6BZMk" executionInfo={"status": "ok", "timestamp": 1631165207095, "user_tz": -330, "elapsed": 7346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70f85ff8-da99-413f-fc9a-570e6f7f73d3"
train_data = CIFAR10(root="./train/",
                    train=True, 
                    download=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qcw-ruQ6BbLr" executionInfo={"status": "ok", "timestamp": 1631165341115, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4e36831-db04-43cd-fc06-045b1a60256b"
print(train_data)
print(len(train_data))
print(train_data.data.shape)
print(len(train_data.targets))
print(train_data.classes)
print(train_data.class_to_idx)
print(type(train_data[0]))
print(len(train_data[0]))

data, label = train_data[0]
print(type(data))
print(data)
print(type(label))
print(label)
print(train_data.classes[label])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["81e9d38f7d484dd790e7b6927de2b447", "be1fa3882b9646aa84f518b34b4382a8", "4986bf68729f42a283f2ff29fb07e293", "ce6f449b2b94421c9d71c9c01d9c0026", "6d82a6732a28409aadba1a17802e0a58", "c239f2f37cee48e68ec47b676dd5be3b", "aaf297df8b214dabb0b7cfc4153b25e3", "566005b140a843ec95618ad33f3ee20b", "295277ab3eb84e4180f04e3cd3c49f1e", "433630fb825a4f518a328b59d93a70ad", "1063afb7a51f4e7bb8c48189cac7a4b5"]} id="X8PkwxC_Bbar" executionInfo={"status": "ok", "timestamp": 1631165383490, "user_tz": -330, "elapsed": 6130, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3ddb8d77-9593-4b61-e987-fa18fb6dd55c"
test_data = CIFAR10(root="./test/", 
                    train=False, 
                    download=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="l2PA41w2B2zb" executionInfo={"status": "ok", "timestamp": 1631165383493, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6b390aaa-e9e8-42f2-c080-8dfd465ae69b"
print(test_data)
print(len(test_data))
print(test_data.data.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="JtXwIBBTCG08" executionInfo={"status": "ok", "timestamp": 1631165511329, "user_tz": -330, "elapsed": 1533, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3cfbcc61-a412-46a9-94d6-e65a195a815e"
from torchvision import transforms

train_transforms = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(
      (0.4914, 0.4822, 0.4465),
      (0.2023, 0.1994, 0.2010))])

train_data = CIFAR10(root="./train/",
                    train=True, 
                    download=True,
                    transform=train_transforms)

print(train_data)
print(train_data.transforms)

data, label = train_data[0]
print(type(data))
print(data.size())
print(data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="vbfud9n2Cdk3" executionInfo={"status": "ok", "timestamp": 1631165571307, "user_tz": -330, "elapsed": 1019, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="324530ee-f55b-42ae-95c1-773f4b8712d2"
test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
      (0.4914, 0.4822, 0.4465),
      (0.2023, 0.1994, 0.2010))])

test_data = torchvision.datasets.CIFAR10(
      root="./test/", 
      train=False, 
      transform=test_transforms)

print(test_data)
```

<!-- #region id="-5xEDXNeC1o-" -->
Now that we have defined the transforms and created the datasets, we can access data samples one at a time. However, when you train your model, you will want to pass in small batches of data at each iteration. Sending data in batches not only allows more efficient training but also takes advantage of the parallel nature of GPUs to accelerate training.

Batch processing can easily be implemented using the torch.utils.data.DataLoader class. Let’s start with an example of how Torchvision uses this class, and then we’ll cover it in more detail.
<!-- #endregion -->

```python id="SS_RrZScDFvw" executionInfo={"status": "ok", "timestamp": 1631165686729, "user_tz": -330, "elapsed": 471, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
trainloader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=16,
                    shuffle=True)
```

```python id="x-IEueVMDs-w" executionInfo={"status": "ok", "timestamp": 1631165798590, "user_tz": -330, "elapsed": 423, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
testloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=16,
                    shuffle=False)
```

<!-- #region id="JW_yheMmDR9m" -->
The dataloader object combines a dataset and a sampler, and provides an iterable over the given dataset. In other words, your training loop can use this object to sample your dataset and apply transforms one batch at a time instead of applying them for the complete dataset at once. This considerably improves efficiency and speed when training and testing models.
<!-- #endregion -->

<!-- #region id="7QQ6ss7YDSPi" -->
The following code shows how to retrieve a batch of samples from the trainloader:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yFxW7zDFDaCU" executionInfo={"status": "ok", "timestamp": 1631165738759, "user_tz": -330, "elapsed": 376, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="328cc60b-8537-4e7c-f2e5-918e7e81c0c5"
data_batch, labels_batch = next(iter(trainloader))

print(data_batch.size())
print(labels_batch.size())
```

<!-- #region id="DJcGmSRLDerp" -->
We need to use iter() to cast the trainloader to an iterator and then use next() to iterate over the data one more time. This is only necessary when accessing one batch. As we’ll see later, our training loops will access the dataloader directly without the need for iter() and next(). After checking the sizes of the data and labels, we see they return batches of size 16.
<!-- #endregion -->

<!-- #region id="Te-zeXO7DpsL" -->
So far, I’ve shown you how to load, transform, and batch image data using Torchvision. However, you can use PyTorch to prepare other types of data as well. PyTorch libraries such as Torchtext and Torchaudio provide dataset and dataloader classes for text and audio data, and new external libraries are being developed all the time.

PyTorch also provides a submodule called torch.utils.data that you can use to create your own dataset and dataloader classes like the ones you saw in Torchvision. It consists of Dataset, Sampler, and DataLoader classes.
<!-- #endregion -->

<!-- #region id="semtaM19D4jV" -->
PyTorch supports map- and iterable-style dataset classes. A map-style dataset is derived from the abstract class torch.utils.data.Dataset. It implements the getitem() and len() functions, and represents a map from (possibly nonintegral) indices/keys to data samples. For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk. Map-style datasets are more commonly used than iterable-style datasets, and all datasets that represent a map made from keys or data samples should use this subclass.
<!-- #endregion -->

<!-- #region id="YtARpb62O95_" -->
All subclasses should overwrite getitem(), which fetches a data sample for a given key. Subclasses can also optionally overwrite len(), which returns the size of the dataset by many Sampler implementations and the default options of DataLoader.
<!-- #endregion -->

<!-- #region id="DDbVqGrpPLFv" -->
An iterable-style dataset, on the other hand, is derived from the torch.utils.data.IterableDataset abstract class. It implements the iter() protocol and represents an iterable over data samples. This type of dataset is typically used when reading data from a database or a remote server, as well as data generated in real time. Iterable datasets are useful when random reads are expensive or uncertain, and when the batch size depends on fetched data.
<!-- #endregion -->

<!-- #region id="jKbI_kYyPeEV" -->
In addition to dataset classes PyTorch also provides sampler classes, which offer a way to iterate over indices of dataset samples. Sampler are derived from the torch.utils.data.Sampler base class.

Every Sampler subclass needs to implement an iter() method to provide a way to iterate over indices of dataset elements and a len() method that returns the length of the returned iterators.
<!-- #endregion -->

<!-- #region id="P4-YSshXQCyK" -->
The dataset and sampler objects are not iterables, meaning you cannot run a for loop on them. The dataloader object solves this problem. The Dataset class returns a dataset object that includes data and information about the data. The Sampler class returns the actual data itself in a specified or random fashion. The DataLoader class combines a dataset with a sampler and returns an iterable.
<!-- #endregion -->

<!-- #region id="XlIm6jP-QKRZ" -->
One of the most powerful features of PyTorch is its Python module torch.nn, which makes it easy to design and experiment with new models. The following code illustrates how you can create a simple model with torch.nn. In this example, we will create a fully connected model called SimpleNet. It consists of an input layer, a hidden layer, and an output layer that takes in 2,048 input values and returns 2 output values for classification:
<!-- #endregion -->

```python id="l4Td7yLBQ8Lt" executionInfo={"status": "ok", "timestamp": 1631169361686, "user_tz": -330, "elapsed": 448, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,2)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ub45qjGyRTLT" executionInfo={"status": "ok", "timestamp": 1631169362176, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4265488e-5330-400f-c02b-ab43a6a66723"
simplenet = SimpleNet()
print(simplenet)
```

<!-- #region id="zaOuo05hRTXq" -->
This simple model demonstrates the following decisions you need to make during model design:
1. **Module definition**: How will you define the layers of your NN? How will you combine these layers into building blocks? In the example, we chose three linear or fully connected layers.
2. **Activation functions**: Which activation functions will you use at the end of each layer or module? In the example, we chose to use relu activation for the input and hidden layers and softmax for the output layer.
3. **Module connections**: How will your modules be connected to each other? In the example, we chose to simply connect each linear layer in sequence.
4. **Output selection**: What output values and formats will be returned? In this example, we return two values from the softmax() function.
<!-- #endregion -->

<!-- #region id="ZiE7TMF-R_z1" -->
The next step in model development is to train your model with your training data. Training a model involves nothing more than estimating the model’s parameters, passing in data, and adjusting the parameters to achieve a more accurate representation of how the data is generally modeled.

In other words, you set the parameters to some values, pass through data, and then compare the model’s outputs with true outputs to measure the error. The goal is to change the parameters and repeat the process until the error is minimized and the model’s outputs are the same as the true outputs.
<!-- #endregion -->

<!-- #region id="EGhjUdCvTJ5W" -->
In this example, we will train the LeNet5 model with the CIFAR-10 dataset that we used earlier in this chapter. The LeNet5 model is a simple convolutional NN developed by Yann LeCun and his team at Bell Labs in the 1990s to classify hand-written digits. (Unbeknownst to me at the time, I actually worked for Bell Labs in the same building in Holmdel, NJ, while this work was being performed.)
<!-- #endregion -->

```python id="qyM0Bi4iTgUu" executionInfo={"status": "ok", "timestamp": 1631169991026, "user_tz": -330, "elapsed": 419, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # <1>
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet5().to(device=device)
```

<!-- #region id="JV3k8lOLTszU" -->
Next, we need to define the loss function (which is also called the criterion) and the optimizer algorithm. The loss function determines how we measure the performance of our model and computes the loss or error between predictions and truth. We’ll attempt to minimize the loss by adjusting the model parameters during training. The optimizer defines how we update our model’s parameters during training.

To define the loss function and the optimizer, we use the torch.optim and torch.nn packages as shown in the following code:
<!-- #endregion -->

```python id="xR3rFFNiTtDZ" executionInfo={"status": "ok", "timestamp": 1631170042928, "user_tz": -330, "elapsed": 403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from torch import optim
from torch import nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.001, 
                      momentum=0.9)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xArCva7zT5fM" executionInfo={"status": "ok", "timestamp": 1631170522614, "user_tz": -330, "elapsed": 341743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afecda60-2966-473d-c9c7-72fe74dd7936"
N_EPOCHS = 10 
for epoch in range(N_EPOCHS): # <1>

    epoch_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(device) # <2>
        labels = labels.to(device)

        optimizer.zero_grad() # <3>

        outputs = model(inputs) # <4>
        loss = criterion(outputs, labels) # <5>
        loss.backward() # <6>
        optimizer.step() # <7>

        epoch_loss += loss.item() # <8>
    print("Epoch: {} Loss: {}".format(epoch, 
                  epoch_loss/len(trainloader)))
```

<!-- #region id="nzNfnkUgUbQF" -->
1. Outer training loop; loop over 10 epochs.
2. Move inputs and labels to GPU if available.
3. Zero out gradients before each backpropagation pass, or they’ll accumulate.
4. Perform forward pass.
5. Compute loss.
6. Perform backpropagation; compute gradients.
7. Adjust parameters based on gradients.
8. Accumulate batch loss so we can average over the epoch.
<!-- #endregion -->

<!-- #region id="7izhHuA_Uo70" -->
The training loop consists of two loops. In the outer loop, we will process the entire set of training data during every iteration or epoch. However, instead of waiting to process the entire dataset before updating the model’s parameters, we process smaller batches of data, one batch at a time. The inner loop loops over each batch.
<!-- #endregion -->

<!-- #region id="1i5HdMfOU7Uv" -->
> Warning: By default, PyTorch accumulates the gradients during each call to loss.backward() (i.e., the backward pass). This is convenient while training some types of NNs, such as RNNs; however, it is not desired for convolutional neural networks (CNNs). In most cases, you will need to call optimizer.zero_grad() to zero the gradients before doing backpropagation so the optimizer updates the model parameters correctly.
<!-- #endregion -->

<!-- #region id="tNCw6Ci5Wv9u" -->
Now that we have trained our model and attempted to minimize the loss, how can we evaluate its performance? How do we know that our model will generalize and work with data it has never seen before?

Model development often includes validation and testing loops to ensure that overfitting does not occur and that the model will perform well against unseen data. Let’s address validation first. Here, I’ll provide you with a quick reference for how you can add validation to your training loops with PyTorch.

Typically, we will reserve a portion of the training data for validation. The validation data will not be used to train the NN; instead, we’ll use it to test the performance of the model at the end of each epoch.

Validation is good practice when training your models. It’s commonly performed when adjusting hyperparameters. For example, maybe we want to slow down the learning rate after five epochs.
<!-- #endregion -->

<!-- #region id="g00vfkZEW80J" -->
Before we perform validation, we need to split our training dataset into a training dataset and a validation dataset. We use the random_split() function from torch.utils.data to reserve 10,000 of our 50,000 training images for validation. Once we create our train_set and val_set, we create our dataloaders for each one.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="viaCRVKdXJer" executionInfo={"status": "ok", "timestamp": 1631170911674, "user_tz": -330, "elapsed": 497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ae8b391-d7bd-4caa-918d-bfb817416e10"
from torch.utils.data import random_split

train_set, val_set = random_split(
                      train_data,
                      [40000, 10000])

trainloader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=16,
                    shuffle=True)

valloader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=16,
                    shuffle=True)

print(len(trainloader))
print(len(valloader))
```

<!-- #region id="-wY5C8ycX4hI" -->
If the loss decreases for validation data, then the model is doing well. However, if the training loss decreases but the validation loss does not, then there’s a good chance the model is overfitting.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="q3JcA2nLXjFO" executionInfo={"status": "ok", "timestamp": 1631171348512, "user_tz": -330, "elapsed": 333293, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c69de3b0-c8c2-436e-fa85-13b50510abcb"
from torch import optim
from torch import nn

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                      lr=0.001, 
                      momentum=0.9)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):

    # Training 
    train_loss = 0.0
    model.train() # <1>
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    val_loss = 0.0
    model.eval() # <2>
    for inputs, labels in valloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

    print("Epoch: {} Train Loss: {} Val Loss: {}".format(
                  epoch, 
                  train_loss/len(trainloader), 
                  val_loss/len(valloader)))
```

<!-- #region id="eFNMCQiyXm62" -->
> Note: Running the .train() or .eval() method on your model object puts the model in training or testing mode, respectively. Calling these methods is only necessary if your model operates differently for training and evaluation. For example, dropout and batch normalization are used in training but not in validation or testing. It’s good practice to call .train() and .eval() in your loops.
<!-- #endregion -->

<!-- #region id="M2AsMS_YX03g" -->
As you can see, our model is training well and does not seem to be overfitting, since both the training loss and the validation loss are decreasing. If we train the model for more epochs, we may get even better results.

We’re not quite finished, though. Our model may still be overfitting. We might have just gotten lucky with our choice of hyperparameters, leading to good validation results. As a further test against overfitting, we will run some test data through our model.

The model has never seen the test data during training, nor has the test data had any influence on the hyperparameters. Let’s see how we perform against the test dataset.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GXWxEZP8YBFl" executionInfo={"status": "ok", "timestamp": 1631171352290, "user_tz": -330, "elapsed": 3808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="139bdffb-e3b6-48ff-90e9-a3dd72dd011a"
num_correct = 0.0

for x_test_batch, y_test_batch in testloader:
  model.eval()
  y_test_batch = y_test_batch.to(device)
  x_test_batch = x_test_batch.to(device)
  y_pred_batch = model(x_test_batch)
  _, predicted = torch.max(y_pred_batch, 1)
  num_correct += (predicted == y_test_batch).float().sum()
  
accuracy = num_correct/(len(testloader)*testloader.batch_size) 

print(len(testloader), testloader.batch_size)

print("Test Accuracy: {}".format(accuracy))
```

<!-- #region id="cosznu2cYl3A" -->
> Tip: You now know how to create training, validation, and test loops using PyTorch. Feel free to use this code as a reference when creating your own loops.
<!-- #endregion -->

<!-- #region id="xu8H1j2mYdUl" -->
Now that you have a fully trained model, let’s explore what you can do with it in the model deployment stage. One of the simplest things you can do is save your trained model for future use. When you want to run your model against new inputs, you can simply load it and call the model with the new values.

The following code illustrates the recommended way to save and load a trained model. It uses the state_dict() method, which creates a dictionary object that maps each layer to its parameter tensor. In other words, we only need to save the model’s learned parameters. We already have the model’s design defined in our model class, so we don’t need to save the architecture. When we load the model, we use the constructor to create a “blank model,” and then we use load_state_dict() to set the parameters for each layer:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iiJyV2pvYpiv" executionInfo={"status": "ok", "timestamp": 1631171428335, "user_tz": -330, "elapsed": 408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="86a16ec6-0b0d-48e0-d2b1-0a8029fe91d5"
torch.save(model.state_dict(), "./lenet5_model.pt")

model = LeNet5().to(device)
model.load_state_dict(torch.load("./lenet5_model.pt"))
```

<!-- #region id="PF7IUdY_ZT0B" -->
> Note: A common PyTorch convention is to save models using either a .pt or .pth file extension.
<!-- #endregion -->
