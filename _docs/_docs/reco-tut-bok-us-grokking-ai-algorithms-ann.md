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

```python id="4sv1j8lRtwKQ" executionInfo={"status": "ok", "timestamp": 1630133974731, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-bok"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="n8RiXVLstuoO" executionInfo={"status": "ok", "timestamp": 1630133976706, "user_tz": -330, "elapsed": 1989, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d6232f6-8f85-47d3-ba85-6b5a2750b236"
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

```python id="drkDUesKtuoT"
!git status
```

```python id="pWkwGc0ltuoT"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="RfQnUMWgjmMA" -->
---
<!-- #endregion -->

```python id="1zfK4Z7ajrSl" executionInfo={"status": "ok", "timestamp": 1630133993665, "user_tz": -330, "elapsed": 724, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
```

<!-- #region id="TjkvQNRvltND" -->
## The Perceptron
<!-- #endregion -->

<!-- #region id="hxf8n8l2lwai" -->
The neuron is the fundamental concept that makes up the brain and nervous systems.
It accepts many inputs from other neurons, processes those inputs, and transfers the result to other “connected”
neurons. Artificial neural networks are based on the fundamental concept of the Perceptron. The Perceptron is a
logical representation of a single biological neuron.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ljLprWZhlqj1" executionInfo={"status": "ok", "timestamp": 1630134563963, "user_tz": -330, "elapsed": 2120, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="503cfe5b-a951-4eb2-e67a-996c641613d6"
# Features
# Smoking, Obesity, Exercise
dataset = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 1, 1]])
dataset_labels = np.array([[1, 0, 0, 1, 1, 0, 0, 1]])
dataset_labels = dataset_labels.reshape(8, 1)

np.random.seed(42)
weights = np.random.rand(3, 1)
bias = 1  # np.random.rand(1)
learning_rate = 0.05


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


for epoch in range(10000):
    # Multiply every input with it's respective weight and sum the outputs
    weight_and_sum_results = np.dot(dataset, weights) + bias
    # Apply the sigmoid activation function to all the input sums
    activation_results = sigmoid(weight_and_sum_results)
    # Determine error for each data row
    error = activation_results - dataset_labels
    # Find slope of the predicated results using derivatives
    predicted_results_derivative = sigmoid_derivative(activation_results)
    # Find amount to adjust weights by
    z_delta = error * predicted_results_derivative
    # Transpose array to work with consistent shaped matrices
    inputs = dataset.transpose()
    # Update weights using gradient descent
    weights -= learning_rate * np.dot(inputs, z_delta)
    # Update bias
    for num in z_delta:
        bias -= learning_rate * num


# Smoker, obese, no exercise
single_point = np.array([1, 0, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Smoker, not obese, no exercise')
print(result)

# Non smoker, obese, no exercise
single_point = np.array([0, 1, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Non smoker, obese, no exercise')
print(result)

# Non smoker, not obese, exercise
single_point = np.array([0, 0, 1])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Non smoker, not obese, does exercise')
print(result)
```

<!-- #region id="uHkjE1zcjwyO" -->
### Artificial Neural Network
Let’s walk through the phases and different operations involved in the back propagation algorithm.

Phase A: Setup

1.	Define ANN architecture: This involves defining the input nodes, the output nodes, the number of hidden layers,
the number of neurons in each hidden layer, the activation functions used, and more. We will dive into some of these
details in the next section. For now, we will stick to the same ANN architecture that we have already used in the
previous section.

2.	Initialize ANN weights: The weights in the ANN must be initialized to some value. There are various approaches
to this, however, the key principle is that the weights will be constantly adjusted as the ANN learns from training
examples but we need to start somewhere.

Phase B: Forward propagation: This is the same process that we covered in the previous section. The same calculations
are carried out; however, the predicted output will be compared with the actual class for each example in the training
set to train the network.

Phase C: Training
1.	Calculate cost: Following from forward propagation, the cost is the difference between the predicted output and
the actual class for the examples in the training set. The cost is effectively determining how bad the ANN is at
predicting the class of examples.

2.	Update weights in the ANN: The weights of the ANN are the only thing that can be adjusted by the network itself.
The architecture and configurations that we defined in phase A doesn’t change during training the network. The weights
are essentially encoding the “intelligence” of the network. Weights are adjusted to be larger or smaller which impacts
the strength of the inputs.

3.	Stopping condition: Training cannot happen indefinitely. Similarly to many of the algorithms explored in this
book, a sensible stopping condition needs to be determined. If we have a large dataset, we might decide that we will
use 500 examples in our training dataset over 1000 iterations to train the ANN. This means that the 500 examples will
be passed through the network 1000 times and adjust the weights in every iteration.
<!-- #endregion -->

```python id="s3Xn-RuZkCfn" executionInfo={"status": "ok", "timestamp": 1630134088382, "user_tz": -330, "elapsed": 722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Scale a dataset to values between 0 and 1 using min-max scaling
def scale_dataset(dataset, feature_count, feature_min, feature_max):
    scaled_data = []
    for data in dataset:
        example = []
        for i in range(0, feature_count):
            example.append(scale_data_feature(data[i], feature_min[i], feature_max[i]))
        scaled_data.append(example)
    return np.array(scaled_data)


# Scale features of a dataset to values between 0 and 1 using min-max scaling
def scale_data_feature(data, feature_min, feature_max):
    return (data - feature_min) / (feature_max - feature_min)


# The sigmoid function will be used as the activation function
# np.exp is a mathematical constant called Euler's number, approximately 2.71828
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

```python id="QscMC2Z2kSIN" executionInfo={"status": "ok", "timestamp": 1630134162426, "user_tz": -330, "elapsed": 540, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# A class to encapsulate the functionality of the artificial neural network
class NeuralNetwork:
    def __init__(self, features, labels, hidden_node_count):
        # Map the features as inputs to the neural network
        self.input = features
        # Initialize weights between the input layer and hidden layer to random values
        self.weights_input = np.random.rand(self.input.shape[1], hidden_node_count)
        print(self.weights_input)
        # Initialize hidden node outputs to nothing
        self.hidden = None
        # Initialize weights between the hidden layer and output node
        self.weights_hidden = np.random.rand(hidden_node_count, 1)
        print(self.weights_hidden)
        # Map the actual expected outputs to the labels
        self.expected_output = labels
        # Initialize the output results to zeros
        self.output = np.zeros(self.expected_output.shape)

    def add_example(self, features, label):
        np.append(self.input, features)
        np.append(self.expected_output, label)

    # Process forward propagation by calculating the weighted sum and activation values
    def forward_propagation(self):
        hidden_weighted_sum = np.dot(self.input, self.weights_input)
        self.hidden = sigmoid(hidden_weighted_sum)
        output_weighted_sum = np.dot(self.hidden, self.weights_hidden)
        self.output = sigmoid(output_weighted_sum)

    # Process back propagation by calculating the cost and updating the weights
    def back_propagation(self):
        cost = self.expected_output - self.output
        print('ACTUAL: ')
        print(self.expected_output)
        print('PREDICTED: ')
        print(self.output)
        print('COSTS: ')
        print(cost)
        print('HIDDEN: ')
        print(self.hidden)
        weights_hidden_update = np.dot(self.hidden.T, (2 * cost * sigmoid_derivative(self.output)))
        print('WEIGHTS HIDDEN UPDATE:')
        print(weights_hidden_update)
        weights_input_update = np.dot(self.input.T, (np.dot(2 * cost * sigmoid_derivative(self.output), self.weights_hidden.T) * sigmoid_derivative(self.hidden)))
        print('WEIGHTS INPUT UPDATE:')
        print(weights_hidden_update)

        # update the weights with the derivative (slope) of the loss function
        self.weights_hidden += weights_hidden_update
        print('WEIGHTS HIDDEN:')
        print(weights_hidden_update)

        self.weights_input += weights_input_update
        print('WEIGHTS INPUT:')
        print(weights_hidden_update)
```

```python id="DBt9tPG6kXSi" executionInfo={"status": "ok", "timestamp": 1630134188850, "user_tz": -330, "elapsed": 823, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def run_neural_network(feature_data, label_data, feature_count, features_min, features_max, hidden_node_count, epochs):
    # Apply min-max scaling to the dataset
    scaled_feature_data = scale_dataset(feature_data, feature_count, features_min, features_max)
    # Initialize a neural network with the scaled data and hidden nodes
    nn = NeuralNetwork(scaled_feature_data, label_data, hidden_node_count)
    # Train the artificial neural network over many iterations with the same training data
    for epoch in range(epochs):
        nn.forward_propagation()
        nn.back_propagation()

    print('OUTPUTS: ')
    for r in nn.output:
        print(r)

    print('INPUT WEIGHTS: ')
    print(nn.weights_input)

    print('HIDDEN WEIGHTS: ')
    print(nn.weights_hidden)
```

<!-- #region id="tnrwJfajlbkH" -->
## Car Collision Prediction
<!-- #endregion -->

<!-- #region id="0Snx9kVrlAfL" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uG1qEwPHjmeJ" executionInfo={"status": "ok", "timestamp": 1630134227277, "user_tz": -330, "elapsed": 13892, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11e8cacc-d333-43e7-ca88-c106482baaec"
# Number of features in the dataset
FEATURE_COUNT = 4
# Minimum possible values for features (Speed, Terrain Quality, Degree of Vision, Driving Experience)
FEATURE_MIN = [0, 0, 0, 0]
# Maximum possible values for features (Speed, Terrain Quality, Degree of Vision, Driving Experience)
FEATURE_MAX = [120, 10, 360, 400000]
# Number of hidden nodes
HIDDEN_NODE_COUNT = 5
# Number of iterations to train the neural network
EPOCHS = 1500

# Training dataset (Speed, Terrain Quality, Degree of Vision, Driving Experience)
car_collision_data = np.array([
    [65, 5,	180, 80000],
    [120, 1, 72, 110000],
    [8,	6,	288, 50000],
    [50, 2,	324, 1600],
    [25, 9,	36, 160000],
    [80, 3,	120, 6000],
    [40, 3,	360, 400000]
])

# Labels for training dataset (0 = No collision occurred, 1 = Collision occurred)
car_collision_data_labels = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0]])

# Run neural network
run_neural_network(car_collision_data,
                    car_collision_data_labels,
                    FEATURE_COUNT,
                    FEATURE_MIN,
                    FEATURE_MAX,
                    HIDDEN_NODE_COUNT,
                    EPOCHS)
```

<!-- #region id="dCoTkR0clLvZ" -->
<!-- #endregion -->

```python id="YZ_H7V31khgF"

```
