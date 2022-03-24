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

# A simple recommender with tensorflow
> A tutorial on how to build a simple deep learning based movie recommender using tensorflow library.

- toc: true
- badges: true
- comments: true
- categories: [movie, tensorflow]
- image: 

```python id="hLtJPt_5idKN"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

tf.random.set_seed(343)
```

```python id="DNLlAwKUihC1"
# Clean up the logdir if it exists
import shutil
shutil.rmtree('logs', ignore_errors=True)

# Load TensorBoard extension for notebooks
%load_ext tensorboard
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8IRTF0EVjQuX" outputId="932eaa43-725c-4fb8-e9d4-dca92ced4cf0"
movielens_ratings_file = 'https://github.com/sparsh-ai/reco-data/blob/master/MovieLens_100K_ratings.csv?raw=true'
df_raw = pd.read_csv(movielens_ratings_file)
df_raw.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="El1C8OwWjhxk" outputId="f8ed06e7-8554-45f2-982d-987f153a5cc7"
df = df_raw.copy()
df.columns = ['userId', 'movieId', 'rating', 'timestamp']
user_ids = df['userId'].unique()
user_encoding = {x: i for i, x in enumerate(user_ids)}   # {user_id: index}
movie_ids = df['movieId'].unique()
movie_encoding = {x: i for i, x in enumerate(movie_ids)} # {movie_id: index}

df['user'] = df['userId'].map(user_encoding)    # Map from IDs to indices
df['movie'] = df['movieId'].map(movie_encoding)

n_users = len(user_ids)
n_movies = len(movie_ids)

min_rating = min(df['rating'])
max_rating = max(df['rating'])

print(f'Number of users: {n_users}\nNumber of movies: {n_movies}\nMin rating: {min_rating}\nMax rating: {max_rating}')

# Shuffle the data
df = df.sample(frac=1, random_state=42)
```

<!-- #region id="1W5V8T-C8Gpv" -->
### Scheme of the model

<!-- #endregion -->

```python id="G-iv9rijkaBf"
class MatrixFactorization(models.Model):
    def __init__(self, n_users, n_movies, n_factors, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        
        # We specify the size of the matrix,
        # the initializer (truncated normal distribution)
        # and the regularization type and strength (L2 with lambda = 1e-6)
        self.user_emb = layers.Embedding(n_users, 
                                         n_factors, 
                                         embeddings_initializer='he_normal',
                                         embeddings_regularizer=keras.regularizers.l2(1e-6),
                                         name='user_embedding')
        self.movie_emb = layers.Embedding(n_movies, 
                                          n_factors, 
                                          embeddings_initializer='he_normal',
                                          embeddings_regularizer=keras.regularizers.l2(1e-6),
                                          name='movie_embedding')
        
        # Embedding returns a 3D tensor with one dimension = 1, so we reshape it to a 2D tensor
        self.reshape = layers.Reshape((self.n_factors,))
        
        # Dot product of the latent vectors
        self.dot = layers.Dot(axes=1)

    def call(self, inputs):
        # Two inputs
        user, movie = inputs
        u = self.user_emb(user)
        u = self.reshape(u)
    
        m = self.movie_emb(movie)
        m = self.reshape(m)
        
        return self.dot([u, m])

n_factors = 50
model = MatrixFactorization(n_users, n_movies, n_factors)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError()
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Bac1w7u49Ddx" outputId="bb733033-9aba-446b-a56d-f971897221d0"
try:
    model.summary()
except ValueError as e:
    print(e, type(e))
```

<!-- #region id="o-JSFnJA-1dz" -->
This is why building models via subclassing is a bit annoying - you can run into errors such as this. We'll fix it by calling the model with some fake data so it knows the shapes of the inputs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7wkIhqmO92Ca" outputId="6825be87-3d5e-4d25-e276-5043ff3a3bb9"
_ = model([np.array([1, 2, 3]), np.array([2, 88, 5])])
model.summary()
```

<!-- #region id="9Nxdrz7b_HOq" -->
We're going to expand our toolbox by introducing callbacks. Callbacks can be used to monitor our training progress, decay the learning rate, periodically save the weights or even stop early in case of detected overfitting. In Keras, they are really easy to use: you just create a list of desired callbacks and pass it to the model.fit method. It's also really easy to define your own by subclassing the Callback class. You can also specify when they will be triggered - the default is at the end of every epoch.

We'll use two: an early stopping callback which will monitor our loss and stop the training early if needed and TensorBoard, a utility for visualizing models, monitoring the training progress and much more.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6N_Y7u5o-QpY" outputId="b4f299bc-25e2-4e38-b349-dc07184fb488"
callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(log_dir='logs')
]

history = model.fit(
    x=(df['user'].values, df['movie'].values),  # The model has two inputs!
    y=df['rating'],
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_split=0.1,
    callbacks=callbacks
)
```

<!-- #region id="5QUGLmtw_eWA" -->
We see that we stopped early because the validation loss was not improving. Now, we'll open TensorBoard (it's a separate program called via command-line) to read the written logs and visualize the loss over all epochs. We will also look at how to visualize the model as a computational graph.
<!-- #endregion -->

```python id="_J-v9Hua_SV8"
# Run TensorBoard and specify the log dir
%tensorboard --logdir logs
```

<!-- #region id="Ldq0DwgI_lWC" -->
We've seen how easy it is to implement a recommender system with Keras and use a few utilities to make it easier to experiment. Note that this model is still quite basic and we could easily improve it: we could try adding a bias for each user and movie or adding non-linearity by using a sigmoid function and then rescaling the output. It could also be extended to use other features of a user or movie.
<!-- #endregion -->

<!-- #region id="-dpCn5hm_nUM" -->
Next, we'll try a bigger, more state-of-the-art model: a deep autoencoder.
<!-- #endregion -->

<!-- #region id="zTGdZ0b4_4rl" -->
We'll apply a more advanced algorithm to the same dataset as before, taking a different approach. We'll use a deep autoencoder network, which attempts to reconstruct its input and with that gives us ratings for unseen user / movie pairs.
<!-- #endregion -->

<!-- #region id="4zZ5N6K0_7sN" -->
<!-- #endregion -->

<!-- #region id="rSdY5NKQAYfI" -->
Preprocessing will be a bit different due to the difference in our model. Our autoencoder will take a vector of all ratings for a movie and attempt to reconstruct it. However, our input vector will have a lot of zeroes due to the sparsity of our data. We'll modify our loss so our model won't predict zeroes for those combinations - it will actually predict unseen ratings.

To facilitate this, we'll use the sparse tensor that TF supports. Note: to make training easier, we'll transform it to dense form, which would not work in larger datasets - we would have to preprocess the data in a different way or stream it into the model.
<!-- #endregion -->

<!-- #region id="dVNWDq4aAbLE" -->
### Sparse representation and autoencoder reconstruction

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="OWybs9LyE8bB" outputId="1e108c11-a007-4942-bf0d-50409e4bbb1d"
df_raw.head()
```

```python id="M9jASOsh_gvU"
# Create a sparse tensor: at each user, movie location, we have a value, the rest is 0
sparse_x = tf.sparse.SparseTensor(indices=df[['movie', 'user']].values, values=df['rating'], dense_shape=(n_movies, n_users))

# Transform it to dense form and to float32 (good enough precision)
dense_x = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(sparse_x)), tf.float32)

# Shuffle the data
x = tf.random.shuffle(dense_x, seed=42)
```

<!-- #region id="1j2-lFANEp8t" -->
Now, let's create the model. We'll have to specify the input shape. Because we have 9724 movies and only 610 users, we'll prefer to predict ratings for movies instead of users - this way, our dataset is larger.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9s4qXdbuEpuX" outputId="45ac7925-b8f3-44dd-8e35-53fa7192b538"
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense1 = layers.Dense(28, activation='selu', kernel_initializer='glorot_uniform')
        self.dense2 = layers.Dense(56, activation='selu', kernel_initializer='glorot_uniform')
        self.dense3 = layers.Dense(56, activation='selu', kernel_initializer='glorot_uniform')
        self.dropout = layers.Dropout(0.3)
        
    def call(self, x):
        d1 = self.dense1(x)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        return self.dropout(d3)
        
        
class Decoder(layers.Layer):
    def __init__(self, n, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense1 = layers.Dense(56, activation='selu', kernel_initializer='glorot_uniform')
        self.dense2 = layers.Dense(28, activation='selu', kernel_initializer='glorot_uniform')
        self.dense3 = layers.Dense(n, activation='selu', kernel_initializer='glorot_uniform')

    def call(self, x):
        d1 = self.dense1(x)
        d2 = self.dense2(d1)
        return self.dense3(d2)

n = n_users
inputs = layers.Input(shape=(n,))

encoder = Encoder()
decoder = Decoder(n)

enc1 = encoder(inputs)
dec1 = decoder(enc1)
enc2 = encoder(dec1)
dec2 = decoder(enc2)

model = models.Model(inputs=inputs, outputs=dec2, name='DeepAutoencoder')
model.summary()
```

<!-- #region id="aqXWA_TQGMDa" -->
Because our inputs are sparse, we'll need to create a modified mean squared error function. We have to look at which ratings are zero in the ground truth and remove them from our loss calculation (if we didn't, our model would quickly learn to predict zeros almost everywhere). We'll use masking - first get a boolean mask of non-zero values and then extract them from the result.
<!-- #endregion -->

```python id="G7AyGH8IFXAj"
def masked_mse(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)
    se = tf.boolean_mask(tf.square(y_true - y_pred), mask)
    return tf.reduce_mean(se)

model.compile(
    loss=masked_mse,
    optimizer=keras.optimizers.Adam()
)
```

<!-- #region id="6O-Hqm_FGTmz" -->
The model training will be similar as before - we'll use early stopping and TensorBoard. Our batch size will be smaller due to the lower number of examples. Note that we are passing the same array for both x and y, because the autoencoder reconstructs its input.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OHoZ3IuJGSrL" outputId="7923e0b0-7bc6-42ba-b3e6-c2bfe025291d"
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-2,
        patience=5,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(log_dir='logs')
]

model.fit(
    x, 
    x, 
    batch_size=16, 
    epochs=100, 
    validation_split=0.1,
    callbacks=callbacks
)
```

<!-- #region id="kkkhIjHhGhP5" -->
Let's visualize our loss and the model itself with TensorBoard.
<!-- #endregion -->

```python id="MMVp_HbwGdGQ"
%tensorboard --logdir logs
```

<!-- #region id="eSBFppW8Gkih" -->
That's it! We've seen how to use TensorFlow to implement recommender systems in a few different ways. I hope this short introduction has been informative and has prepared you to use TF on new problems. Thank you for your attention!
<!-- #endregion -->
