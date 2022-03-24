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

<!-- #region id="R-6K5rt9zMlp" -->
# Warm-start Model Retraining
<!-- #endregion -->

<!-- #region id="DozX2fBrPP3Z" -->
Importantly, you have a lot of flexibility when it comes to responding to a change to the problem or the availability of new data.

For example, you can take the trained neural network model and update the model weights using the new data. Or we might want to leave the existing model untouched and combine its predictions with a new model fit on the newly available data.

These approaches might represent two general themes in updating neural network models in response to new data, they are:

- Retrain Update Strategies.
- Ensemble Update Strategies.

Let’s take a closer look at each in turn.
<!-- #endregion -->

<!-- #region id="C0GHlhYQb2Wt" -->
## Retraining Update Strategies
A benefit of neural network models is that their weights can be updated at any time with continued training.

When responding to changes in the underlying data or the availability of new data, there are a few different strategies to choose from when updating a neural network model, such as:

- Continue training the model on the new data only.
- Continue training the model on the old and new data.

We might also imagine variations on the above strategies, such as using a sample of the new data or a sample of new and old data instead of all available data, as well as possible instance-based weightings on sampled data.

We might also consider extensions of the model that freeze the layers of the existing model (e.g. so model weights cannot change during training), then add new layers with model weights that can change, grafting on extensions to the model to handle any change in the data. Perhaps this is a variation of the retraining and the ensemble approach in the next section, and we’ll leave it for now.

Nevertheless, these are the two main strategies to consider.

Let’s make these approaches concrete with a worked example.
<!-- #endregion -->

<!-- #region id="XGT0i9Ujb9-M" -->
### Update Model on New Data Only
We can update the model on the new data only.

One extreme version of this approach is to not use any new data and simply re-train the model on the old data. This might be the same as “do nothing” in response to the new data. At the other extreme, a model could be fit on the new data only, discarding the old data and old model.

- Ignore new data, do nothing.
- Update existing model on new data.
- Fit new model on new data, discard old model and data.

We will focus on the middle ground in this example, but it might be interesting to test all three approaches on your problem and see what works best.

First, we can define a synthetic binary classification dataset and split it into half, then use one portion as “old data” and another portion as “new data.”
<!-- #endregion -->

<!-- #region id="DB6c8QzFb___" -->
```python
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
```
<!-- #endregion -->

<!-- #region id="ibIKA6locERr" -->
We can then define a Multilayer Perceptron model (MLP) and fit it on the old data only.
<!-- #endregion -->

<!-- #region id="IuoV7JtScEiF" -->
```python
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)
```
<!-- #endregion -->

<!-- #region id="-RtGEKdNcNz5" -->
We can then imagine saving the model and using it for some time.

Time passes, and we wish to update it on new data that has become available.

This would involve using a much smaller learning rate than normal so that we do not wash away the weights learned on the old data.

Note: you will need to discover a learning rate that is appropriate for your model and dataset that achieves better performance than simply fitting a new model from scratch.
<!-- #endregion -->

<!-- #region id="Ogor1FfXcPwg" -->
```python
# update model on new data only with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
```
<!-- #endregion -->

<!-- #region id="cOeYSD5JcSWZ" -->
We can then fit the model on the new data only with this smaller learning rate..
<!-- #endregion -->

<!-- #region id="e-V2RalxcSyv" -->
```python
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on new data
model.fit(X_new, y_new, epochs=100, batch_size=32, verbose=0)
```
<!-- #endregion -->

<!-- #region id="sA-cjlpPcVXt" -->
Tying this together, the complete example of updating a neural network model on new data only is listed below.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zjefjTA3cVla" executionInfo={"status": "ok", "timestamp": 1635182106200, "user_tz": -330, "elapsed": 8445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="da3082a6-4529-4d3c-e32b-3080acb735d8"
# update neural network with new data only
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# update model on new data only with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on new data
model.fit(X_new, y_new, epochs=100, batch_size=32, verbose=0)
```

<!-- #region id="T7vIf_dVcrJR" -->
Next, let’s look at updating the model on new and old data.
<!-- #endregion -->

<!-- #region id="XR_oQz0Tc5Uu" -->
### Update Model on Old and New Data

We can update the model on a combination of both old and new data. An extreme version of this approach is to discard the model and simply fit a new model on all available data, new and old. A less extreme version would be to use the existing model as a starting point and update it based on the combined dataset. Again, it is a good idea to test both strategies and see what works well for your dataset. We will focus on the less extreme update strategy in this case.

- The synthetic dataset and model can be fit on the old dataset as before.
- New data comes available and we wish to update the model on a combination of both old and new data.
- First, we must use a much smaller learning rate in an attempt to use the current weights as a starting point for the search.
- We can then create a composite dataset composed of old and new data.
- Finally, we can update the model on this composite dataset.

> Note: you will need to discover a learning rate that is appropriate for your model and dataset that achieves better performance than simply fitting a new model from scratch.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lbAxQ1gRdHWT" executionInfo={"status": "ok", "timestamp": 1635182229320, "user_tz": -330, "elapsed": 7285, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d4ef181a-7652-4ec5-c8a3-fd08692958ab"
# update neural network with both old and new data
from numpy import vstack
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the model
model = Sequential()
model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# update model with a smaller learning rate
opt = SGD(learning_rate=0.001, momentum=0.9)
# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy')
# create a composite dataset of old and new data
X_both, y_both = vstack((X_old, X_new)), hstack((y_old, y_new))
# fit the model on new data
model.fit(X_both, y_both, epochs=100, batch_size=32, verbose=0)
```

<!-- #region id="8LlulZdSdJPm" -->
Next, let’s look at how to use ensemble models to respond to new data.
<!-- #endregion -->

<!-- #region id="cgLiD6skdOWx" -->
## Ensemble Update Strategies

An ensemble is a predictive model that is composed of multiple other models.

There are many different types of ensemble models, although perhaps the simplest approach is to average the predictions from multiple different models.

We can use an ensemble model as a strategy when responding to changes in the underlying data or availability of new data.

Mirroring the approaches in the previous section, we might consider two approaches to ensemble learning algorithms as strategies for responding to new data; they are:

- Ensemble of existing model and new model fit on new data only.
- Ensemble of existing model and new model fit on old and new data.

Again, we might consider variations on these approaches, such as samples of old and new data, and more than one existing or additional models included in the ensemble.

Nevertheless, these are the two main strategies to consider.

Let’s make these approaches concrete with a worked example.
<!-- #endregion -->

<!-- #region id="n-e4iu0rdVBQ" -->
### Ensemble Model With Model on New Data Only

We can create an ensemble of the existing model and a new model fit on only the new data.

The expectation is that the ensemble predictions perform better or are more stable (lower variance) than using either the old model or the new model alone. This should be checked on your dataset before adopting the ensemble.

- First, we can prepare the dataset and fit the old model, as we did in the previous sections.
- Some time passes and new data becomes available.
- We can then fit a new model on the new data, naturally discovering a model and configuration that works well or best on the new dataset only. In this case, we’ll simply use the same model architecture and configuration as the old model.
- We can then fit this new model on the new data only.
- Now that we have the two models, we can make predictions with each model, and calculate the average of the predictions as the “ensemble prediction.”
<!-- #endregion -->

```python id="mdZCzA28dkiy"
# ensemble old neural network with new model fit on new data only
from numpy import hstack
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the old model
old_model = Sequential()
old_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
old_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
old_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
old_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
old_model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# define the new model
new_model = Sequential()
new_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
new_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
new_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
new_model.fit(X_new, y_new, epochs=150, batch_size=32, verbose=0)

# make predictions with both models
yhat1 = old_model.predict(X_new)
yhat2 = new_model.predict(X_new)
# combine predictions into single array
combined = hstack((yhat1, yhat2))
# calculate outcome as mean of predictions
yhat = mean(combined, axis=-1)
```

<!-- #region id="PKK_SAwVdlbV" -->
### Ensemble Model With Model on Old and New Data

We can create an ensemble of the existing model and a new model fit on both the old and the new data.

The expectation is that the ensemble predictions perform better or are more stable (lower variance) than using either the old model or the new model alone. This should be checked on your dataset before adopting the ensemble.

- First, we can prepare the dataset and fit the old model, as we did in the previous sections.
- Some time passes and new data becomes available.
- We can then fit a new model on a composite of the old and new data, naturally discovering a model and configuration that works well or best on the new dataset only. In this case, we’ll simply use the same model architecture and configuration as the old model.
- We can create a composite dataset from the old and new data, then fit the new model on this dataset.
- Finally, we can use both models together to make ensemble predictions.
<!-- #endregion -->

```python id="flwjKe5ddvWK"
# ensemble old neural network with new model fit on old and new data
from numpy import hstack
from numpy import vstack
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# record the number of input features in the data
n_features = X.shape[1]
# split into old and new data
X_old, X_new, y_old, y_new = train_test_split(X, y, test_size=0.50, random_state=1)
# define the old model
old_model = Sequential()
old_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
old_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
old_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
old_model.compile(optimizer=opt, loss='binary_crossentropy')
# fit the model on old data
old_model.fit(X_old, y_old, epochs=150, batch_size=32, verbose=0)

# save model...

# load model...

# define the new model
new_model = Sequential()
new_model.add(Dense(20, kernel_initializer='he_normal', activation='relu', input_dim=n_features))
new_model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))
# define the optimization algorithm
opt = SGD(learning_rate=0.01, momentum=0.9)
# compile the model
new_model.compile(optimizer=opt, loss='binary_crossentropy')
# create a composite dataset of old and new data
X_both, y_both = vstack((X_old, X_new)), hstack((y_old, y_new))
# fit the model on old data
new_model.fit(X_both, y_both, epochs=150, batch_size=32, verbose=0)

# make predictions with both models
yhat1 = old_model.predict(X_new)
yhat2 = new_model.predict(X_new)
# combine predictions into single array
combined = hstack((yhat1, yhat2))
# calculate outcome as mean of predictions
yhat = mean(combined, axis=-1)
```
