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

<!-- #region id="wMJvhqEB5iC-" -->
# Predicting Student Admissions with Neural Networks
In this notebook, we predict student admissions to graduate school at UCLA based on three pieces of data:
- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)

The dataset originally came from here: http://www.ats.ucla.edu/
<!-- #endregion -->

<!-- #region id="qQHxydHZ5qaV" -->
## Loading the data
<!-- #endregion -->

```python id="X2hOA87D5iDB" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1631196738357, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6239a19-beb5-488d-a8e3-09927caaec92"
# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-neural-networks/student-admissions/student_data.csv')

# Printing out the first 10 rows of our data
data.head()
```

<!-- #region id="nHyMLmcM5iDE" -->
## Plotting the data

First let's make a plot of our data to see how it looks. In order to have a 2D plot, let's ingore the rank.
<!-- #endregion -->

```python id="lOVx1Waa5iDF" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1631196744289, "user_tz": -330, "elapsed": 674, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e83bb84c-0446-4354-aa89-9a3c7aa8288e"
# %matplotlib inline
import matplotlib.pyplot as plt

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
# Plotting the points
plot_points(data)
plt.show()
```

<!-- #region id="WsL43x5v5iDG" -->
Roughly, it looks like the students with high scores in the grades and test passed, while the ones with low scores didn't, but the data is not as nicely separable as we hoped it would. Maybe it would help to take the rank into account? Let's make 4 plots, each one for each rank.
<!-- #endregion -->

```python id="XNtmwfBS5iDG" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1631196748158, "user_tz": -330, "elapsed": 1230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b52a77a1-58c1-4764-de1d-f644c9367d43"
# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()
```

<!-- #region id="OtZrzIZJ5iDH" -->
This looks more promising, as it seems that the lower the rank, the higher the acceptance rate. Let's use the rank as one of our inputs. In order to do this, we should one-hot encode it.

## One-hot encoding the rank
Use the `get_dummies` function in pandas in order to one-hot encode the data.

Hint: To drop a column, it's suggested that you use `one_hot_data`[.drop( )](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html).
<!-- #endregion -->

```python id="JH_R-AIm5iDI" colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"status": "ok", "timestamp": 1631196791193, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03a91117-96b5-4ae0-f241-af27b22ad1d5"
# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Print the first 10 rows of our data
one_hot_data[:10]
```

<!-- #region id="j-AfPHnK5iDI" -->
## Scaling the data
The next step is to scale the data. We notice that the range for grades is 1.0-4.0, whereas the range for test scores is roughly 200-800, which is much larger. This means our data is skewed, and that makes it hard for a neural network to handle. Let's fit our two features into a range of 0-1, by dividing the grades by 4.0, and the test score by 800.
<!-- #endregion -->

```python id="X68T9T2f5iDJ" colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"status": "ok", "timestamp": 1631196829494, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bac0abcf-e771-4ed3-c2e7-ae2bf9b6038c"
# Copying our data
processed_data = one_hot_data[:]

# Scaling the columns
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
processed_data[:10]
```

<!-- #region id="mQB4RAJh5iDK" -->
## Splitting the data into Training and Testing
<!-- #endregion -->

<!-- #region id="2nm16-HL5iDK" -->
In order to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.
<!-- #endregion -->

```python id="m2MzzBT65iDK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196836661, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98650c0d-8e50-4003-b1b2-de9d1653e363"
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])
```

<!-- #region id="CxTuMi0B5iDL" -->
## Splitting the data into features and targets (labels)
Now, as a final step before the training, we'll split the data into features (X) and targets (y).
<!-- #endregion -->

```python id="fHTwjUce5iDL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196840524, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4dd5991a-a39b-4a27-d890-c004409ba32b"
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])
```

<!-- #region id="YeIFAlZj5iDM" -->
## Training the 1-layer Neural Network
The following function trains the 1-layer neural network.  
First, we'll write some helper functions.
<!-- #endregion -->

```python id="obL2qqRx5iDM" executionInfo={"status": "ok", "timestamp": 1631196844303, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
    
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)
```

<!-- #region id="X1rQSGrW5iDM" -->
## Backpropagate the error
Now it's your turn to shine. Write the error term. Remember that this is given by the equation $$ (y-\hat{y})x $$ for binary cross entropy loss function and 
$$ (y-\hat{y})\sigma'(x)x $$ for mean square error. 
<!-- #endregion -->

```python id="6l8htF5M5iDN" executionInfo={"status": "ok", "timestamp": 1631196881823, "user_tz": -330, "elapsed": 412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def error_term_formula(x, y, output):
#    for binary cross entropy loss
    return (y - output)*x
#    for mean square error
#    return (y - output)*sigmoid_prime(x)*x
```

```python id="xkPopj_E5iDN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196887826, "user_tz": -330, "elapsed": 3917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9eb95e1e-bd44-4592-815d-8c00e5863c42"
# Neural Network hyperparameters
epochs = 1000
learnrate = 0.0001

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term

        # Update the weights here. The learning rate times the 
        # change in weights
        # don't have to divide by n_records since it is compensated by the learning rate
        weights += learnrate * del_w #/ n_records  

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean(error_formula(targets, out))
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)
```

<!-- #region id="UpSbWJFB5iDO" -->
## Calculating the Accuracy on the Test Data
<!-- #endregion -->

```python id="AReeZ67y5iDO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196888241, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2339ead3-1b6e-4fee-98a7-3f7920ceb00e"
# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```
