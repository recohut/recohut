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

<!-- #region id="uFnpg9coKEpG" -->
# From Batch to Online/Stream (Concept)
> Learning and validating the concept of incremental (online) learning and comparing to its counterpart batch learning method

- toc: true
- badges: true
- comments: true
- categories: [Concept, OnlineLearning]
- image:
<!-- #endregion -->

<!-- #region id="PasR1CuXJxNv" -->
## Setup
<!-- #endregion -->

```python id="It-vcW2kIjMr"
!pip install river
!pip install -U numpy
```

```python id="sw24c-Z0JpdE"
import numpy as np

from sklearn import datasets, linear_model, metrics
from sklearn import model_selection, pipeline, preprocessing

from river import preprocessing, linear_model
from river import optim, compat, compose, stream
```

```python colab={"base_uri": "https://localhost:8080/"} id="anFBvNsfJ3_d" outputId="d20d9671-5613-4455-d4ce-f4bb6456d045"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

<!-- #region id="gIOe7-dQNXAO" -->
## A quick overview of batch learning

If you've already delved into machine learning, then you shouldn't have any difficulty in getting to use incremental learning. If you are somewhat new to machine learning, then do not worry! The point of this notebook in particular is to introduce simple notions. We'll also start to show how `river` fits in and explain how to use it.

The whole point of machine learning is to *learn from data*. In *supervised learning* you want to learn how to predict a target $y$ given a set of features $X$. Meanwhile in an unsupervised learning there is no target, and the goal is rather to identify patterns and trends in the features $X$. At this point most people tend to imagine $X$ as a somewhat big table where each row is an observation and each column is a feature, and they would be quite right. Learning from tabular data is part of what's called *batch learning*, which basically that all of the data is available to our learning algorithm at once. Multiple libraries have been created to handle the batch learning regime, with one of the most prominent being Python's [scikit-learn](https://scikit-learn.org/stable/).

As a simple example of batch learning let's say we want to learn to predict if a women has breast cancer or not. We'll use the [breast cancer dataset available with scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). We'll learn to map a set of features to a binary decision using a [logistic regression](https://www.wikiwand.com/en/Logistic_regression). Like many other models based on numerical weights, logistic regression is sensitive to the scale of the features. Rescaling the data so that each feature has mean 0 and variance 1 is generally considered good practice. We can apply the rescaling and fit the logistic regression sequentially in an elegant manner using a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). To measure the performance of the model we'll evaluate the average [ROC AUC score](https://www.wikiwand.com/en/Receiver_operating_characteristic) using a 5 fold [cross-validation](https://www.wikiwand.com/en/Cross-validation_(statistics)). 
<!-- #endregion -->

```python id="aC6f3NOsNXAR" colab={"base_uri": "https://localhost:8080/"} outputId="8e9f9eed-3062-4516-bffc-6e863aa84403"
# Load the data
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

# Define the steps of the model
model = pipeline.Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LogisticRegression(solver='lbfgs'))
])

# Define a determistic cross-validation procedure
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# Compute the MSE values
scorer = metrics.make_scorer(metrics.roc_auc_score)
scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# Display the average score and it's standard deviation
print(f'ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')
```

<!-- #region id="1vCeknVRNXAX" -->
This might be a lot to take in if you're not accustomed to scikit-learn, but it probably isn't if you are. Batch learning basically boils down to:

1. Loading (and preprocessing) the data
2. Fitting a model to the data
3. Computing the performance of the model on unseen data

This is pretty standard and is maybe how most people imagine a machine learning pipeline. However, this way of proceeding has certain downsides. First of all your laptop would crash if the `load_boston` function returned a dataset who's size exceeds your available amount of RAM. Sometimes you can use some tricks to get around this. For example by optimizing the data types and by using sparse representations when applicable you can potentially save precious gigabytes of RAM. However, like many tricks this only goes so far. If your dataset weighs hundreds of gigabytes then you won't go far without some special hardware. One solution is to do out-of-core learning; that is, algorithms that can learn by being presented the data in chunks or mini-batches. If you want to go down this road then take a look at [Dask](https://examples.dask.org/machine-learning.html) and [Spark's MLlib](https://spark.apache.org/mllib/).

Another issue with the batch learning regime is that it can't elegantly learn from new data. Indeed if new data is made available, then the model has to learn from scratch with a new dataset composed of the old data and the new data. This is particularly annoying in a real situation where you might have new incoming data every week, day, hour, minute, or even setting. For example if you're building a recommendation engine for an e-commerce app, then you're probably training your model from 0 every week or so. As your app grows in popularity, so does the dataset you're training on. This will lead to longer and longer training times and might require a hardware upgrade.

A final downside that isn't very easy to grasp concerns the manner in which features are extracted. Every time you want to train your model you first have to extract features. The trick is that some features might not be accessible at the particular point in time you are at. For example maybe that some attributes in your data warehouse get overwritten with time. In other words maybe that all the features pertaining to a particular observations are not available, whereas they were a week ago. This happens more often than not in real scenarios, and apart if you have a sophisticated data engineering pipeline then you will encounter these issues at some point. 
<!-- #endregion -->

<!-- #region id="jVsH2HaqNXAd" -->
## A hands-on introduction to incremental learning

Incremental learning is also often called *online learning* or *stream learning*, but if you [google online learning](https://www.google.com/search?q=online+learning) a lot of the results will point to educational websites. Hence, the terms "incremental learning" and "stream learning" (from which `river` derives it's name) are prefered. The point of incremental learning is to fit a model to a stream of data. In other words, the data isn't available in it's entirety, but rather the observations are provided one by one. As an example let's stream through the dataset used previously.
<!-- #endregion -->

```python id="8LyFqjlMNXAe"
for xi, yi in zip(X, y):
    # This is where the model learns
    pass
```

<!-- #region id="z3BvgUFuNXAf" -->
In this case we're iterating over a dataset that is already in memory, but we could just as well stream from a CSV file, a Kafka stream, an SQL query, etc. If we look at `xi` we can notice that it is a `numpy.ndarray`.
<!-- #endregion -->

```python id="ZTpjG4dRNXAg" colab={"base_uri": "https://localhost:8080/"} outputId="8a30e71e-a1a2-4776-f56b-e5e87f1f759e"
xi
```

<!-- #region id="jTHGTFTzNXAi" -->
`river` by design works with `dict`s. We believe that `dict`s are more enjoyable to program with than `numpy.ndarray`s, at least for when single observations are concerned. `dict`'s bring the added benefit that each feature can be accessed by name rather than by position.
<!-- #endregion -->

```python id="faTWoh1nNXAk" colab={"base_uri": "https://localhost:8080/"} outputId="950e22b7-29e6-4b64-971e-ba7ff9069269"
for xi, yi in zip(X, y):
    xi = dict(zip(dataset.feature_names, xi))
    pass

xi
```

<!-- #region id="ZiCA406mNXAn" -->
Conveniently, `river`'s `stream` module has an `iter_sklearn_dataset` method that we can use instead.
<!-- #endregion -->

```python id="w3WbsuX0NXAo"
for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    pass
```

<!-- #region id="77_fWU-UNXAo" -->
The simple fact that we are getting the data as a stream means that we can't do a lot of things the same way as in a batch setting. For example let's say we want to scale the data so that it has mean 0 and variance 1, as we did earlier. To do so we simply have to subtract the mean of each feature to each value and then divide the result by the standard deviation of the feature. The problem is that we can't possible known the values of the mean and the standard deviation before actually going through all the data! One way to proceed would be to do a first pass over the data to compute the necessary values and then scale the values during a second pass. The problem is that this defeats our purpose, which is to learn by only looking at the data once. Although this might seem rather restrictive, it reaps sizable benefits down the road.

The way we do feature scaling in `river` involves computing *running statistics* (also know as *moving statistics*). The idea is that we use a data structure that estimates the mean and updates itself when it is provided with a value. The same goes for the variance (and thus the standard deviation). For example, if we denote $\mu_t$ the mean and $n_t$ the count at any moment $t$, then updating the mean can be done as so:

$$
\begin{cases}
n_{t+1} = n_t + 1 \\
\mu_{t+1} = \mu_t + \frac{x - \mu_t}{n_{t+1}}
\end{cases}
$$

Likewise, the running variance can be computed as so:

$$
\begin{cases}
n_{t+1} = n_t + 1 \\
\mu_{t+1} = \mu_t + \frac{x - \mu_t}{n_{t+1}} \\
s_{t+1} = s_t + (x - \mu_t) \times (x - \mu_{t+1}) \\
\sigma_{t+1} = \frac{s_{t+1}}{n_{t+1}}
\end{cases}
$$

where $s_t$ is a running sum of squares and $\sigma_t$ is the running variance at time $t$. This might seem a tad more involved than the batch algorithms you learn in school, but it is rather elegant. Implementing this in Python is not too difficult. For example let's compute the running mean and variance of the `'mean area'` variable.
<!-- #endregion -->

```python id="-yUEM0MtNXAq" colab={"base_uri": "https://localhost:8080/"} outputId="572bd9cd-0c23-4eda-b392-c8d128508bd3"
n, mean, sum_of_squares, variance = 0, 0, 0, 0

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    n += 1
    old_mean = mean
    mean += (xi['mean area'] - mean) / n
    sum_of_squares += (xi['mean area'] - old_mean) * (xi['mean area'] - mean)
    variance = sum_of_squares / n
    
print(f'Running mean: {mean:.3f}')
print(f'Running variance: {variance:.3f}')
```

<!-- #region id="F8vbyBe4NXAs" -->
Let's compare this with `numpy`. But remember, `numpy` requires access to "all" the data.
<!-- #endregion -->

```python id="uLrCoJG5NXAt" colab={"base_uri": "https://localhost:8080/"} outputId="b8d62011-06cc-437b-f200-7e2acbc05e31"
i = list(dataset.feature_names).index('mean area')
print(f'True mean: {np.mean(X[:, i]):.3f}')
print(f'True variance: {np.var(X[:, i]):.3f}')
```

<!-- #region id="9BrrseaBNXAt" -->
The results seem to be exactly the same! The twist is that the running statistics won't be very accurate for the first few observations. In general though this doesn't matter too much. Some would even go as far as to say that this descrepancy is beneficial and acts as some sort of regularization...

Now the idea is that we can compute the running statistics of each feature and scale them as they come along. The way to do this with `river` is to use the `StandardScaler` class from the `preprocessing` module, as so:
<!-- #endregion -->

```python id="n6P3W4JVNXAu"
scaler = preprocessing.StandardScaler()

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    scaler = scaler.learn_one(xi)
```

<!-- #region id="ZVTLgC5LNXAv" -->
Now that we are scaling the data, we can start doing some actual machine learning. We're going to implement an online linear regression task. Because all the data isn't available at once, we are obliged to do what is called *stochastic gradient descent*, which is a popular research topic and has a lot of variants. SGD is commonly used to train neural networks. The idea is that at each step we compute the loss between the target prediction and the truth. We then calculate the gradient, which is simply a set of derivatives with respect to each weight from the linear regression. Once we have obtained the gradient, we can update the weights by moving them in the opposite direction of the gradient. The amount by which the weights are moved typically depends on a *learning rate*, which is typically set by the user. Different optimizers have different ways of managing the weight update, and some handle the learning rate implicitly. Online linear regression can be done in `river` with the `LinearRegression` class from the `linear_model` module. We'll be using plain and simple SGD using the `SGD` optimizer from the `optim` module. During training we'll measure the squared error between the truth and the predictions.
<!-- #endregion -->

```python id="__4AHNo4NXAw" colab={"base_uri": "https://localhost:8080/"} outputId="f9cb90ed-11eb-4e70-974a-43fa0c6d84af"
scaler = preprocessing.StandardScaler()
optimizer = optim.SGD(lr=0.01)
log_reg = linear_model.LogisticRegression(optimizer)

y_true = []
y_pred = []

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer(), shuffle=True, seed=42):
    
    # Scale the features
    xi_scaled = scaler.learn_one(xi).transform_one(xi)
    
    # Test the current model on the new "unobserved" sample
    yi_pred = log_reg.predict_proba_one(xi_scaled)
    # Train the model with the new sample
    log_reg.learn_one(xi_scaled, yi)
    
    # Store the truth and the prediction
    y_true.append(yi)
    y_pred.append(yi_pred[True])
    
print(f'ROC AUC: {metrics.roc_auc_score(y_true, y_pred):.3f}')
```

<!-- #region id="WS4_NQO_NXAx" -->
The ROC AUC is significantly better than the one obtained from the cross-validation of scikit-learn's logisitic regression. However to make things really comparable it would be nice to compare with the same cross-validation procedure. `river` has a `compat` module that contains utilities for making `river` compatible with other Python libraries. Because we're doing regression we'll be using the `SKLRegressorWrapper`. We'll also be using `Pipeline` to encapsulate the logic of the `StandardScaler` and the `LogisticRegression` in one single object.
<!-- #endregion -->

```python id="HK1UEAP5NXAy" colab={"base_uri": "https://localhost:8080/"} outputId="d726c81b-49cc-46c1-c241-10331b24c135"
# We define a Pipeline, exactly like we did earlier for sklearn 
model = compose.Pipeline(
    ('scale', preprocessing.StandardScaler()),
    ('log_reg', linear_model.LogisticRegression())
)

# We make the Pipeline compatible with sklearn
model = compat.convert_river_to_sklearn(model)

# We compute the CV scores using the same CV scheme and the same scoring
scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# Display the average score and it's standard deviation
print(f'ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')
```

<!-- #region id="S97lPgTxNXA0" -->
This time the ROC AUC score is lower, which is what we would expect. Indeed online learning isn't as accurate as batch learning. However it all depends in what you're interested in. If you're only interested in predicting the next observation then the online learning regime would be better. That's why it's a bit hard to compare both approaches: they're both suited to different scenarios.
<!-- #endregion -->

<!-- #region id="CPzjeR21NXA1" -->
## Going further
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} id="HGK4NVpvNXA3" -->
Here a few resources if you want to do some reading:

- [Online learning -- Wikipedia](https://www.wikiwand.com/en/Online_machine_learning)
- [What is online machine learning? -- Max Pagels](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5)
- [Introduction to Online Learning -- USC course](http://www-bcf.usc.edu/~haipengl/courses/CSCI699/)
- [Online Methods in Machine Learning -- MIT course](http://www.mit.edu/~rakhlin/6.883/)
- [Online Learning: A Comprehensive Survey](https://arxiv.org/pdf/1802.02871.pdf)
- [Streaming 101: The world beyond batch](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101)
- [Machine learning for data streams](https://www.cms.waikato.ac.nz/~abifet/book/contents.html)
- [Data Stream Mining: A Practical Approach](https://www.cs.waikato.ac.nz/~abifet/MOA/StreamMining.pdf)
<!-- #endregion -->
