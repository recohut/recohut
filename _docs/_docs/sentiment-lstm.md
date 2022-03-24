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

# IMDB Reviews Sentiment Analysis with LSTM RNNs

<!-- #region colab_type="text" id="WMucfLUS1yhH" -->
This Jupyter Notebook contains Python code for building a LSTM Recurrent Neural Network that gives 87-88% accuracy on the IMDB Movie Review Sentiment Analysis Dataset. 

More information is given on [this blogpost](https://www.bouvet.no/bouvet-deler/explaining-recurrent-neural-networks).

This code is supplied without license, warranty or support. Feel free to do with it what you will.
<!-- #endregion -->

<!-- #region colab_type="text" id="wFUKGe4x3ala" -->
## Built for Google Collaboratory

Train your network more quickly in Google Collaboratory. From the **Runtime** menu select **Change Runtime** Type and choose "GPU"!

Don't forget to select **Runtime** -> **Restart runtime** to put your changes into effect.
<!-- #endregion -->

<!-- #region colab_type="text" id="WP1VrbVp3sVu" -->
## Setting up

When running this for the first time you may get a warning telling you to restart the Runtime. You can ignore this, but feel free to select "Runtime->Restart Runtime" from the overhead menu if you encounter problems.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 56} colab_type="code" id="2e3txwbh3q76" outputId="097e3f2c-6084-4ddb-baad-b918f445f7c3"
# keras.datasets.imdb is broken in TensorFlow 1.13 and 1.14 due to numpy 1.16.3
!pip install numpy==1.16.2

# All the imports!
import tensorflow as tf 
import numpy as np
from tensorflow.keras.preprocessing import sequence
from numpy import array

# Supress deprecation warnings
import logging
logging.getLogger('tensorflow').disabled = True

# Fetch "IMDB Movie Review" data, constraining our reviews to 
# the 10000 most commonly used words
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# Map for readable classnames
class_names = ["Negative", "Positive"]
```

<!-- #region colab_type="text" id="hdyHL8FF0JJy" -->
## Create map for converting IMDB dataset to readable reviews

Reviews in the IMDB dataset have been encoded as a sequence of integers. Luckily the dataset also 
contains an index for converting the reviews back into human readable form.
<!-- #endregion -->

```python colab={} colab_type="code" id="E05AweFu0Imt"
# Get the word index from the dataset
word_index = tf.keras.datasets.imdb.get_word_index()

# Ensure that "special" words are mapped into human readable terms 
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2
word_index["<UNUSED>"] = 3

# Perform reverse word lookup and make it callable
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
```

<!-- #region colab_type="text" id="fFXK-g6G81sC" -->
## Data Insight

Here we take a closer look at our data. How many words do our reviews contain?

And what do our review look like in machine and human readable form?

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 218} colab_type="code" id="yD1qHVBn81Y_" outputId="65f35e69-8cf5-4ada-f817-40ee96ed8df9"
# Concatonate test and training datasets
allreviews = np.concatenate((x_train, x_test), axis=0)

# Review lengths across test and training whole datasets
print("Maximum review length: {}".format(len(max((allreviews), key=len))))
print("Minimum review length: {}".format(len(min((allreviews), key=len))))
result = [len(x) for x in allreviews]
print("Mean review length: {}".format(np.mean(result)))

# Print a review and it's class as stored in the dataset. Replace the number
# to select a different review.
print("")
print("Machine readable Review")
print("  Review Text: " + str(x_train[60]))
print("  Review Sentiment: " + str(y_train[60]))

# Print a review and it's class in human readable format. Replace the number
# to select a different review.
print("")
print("Human Readable Review")
print("  Review Text: " + decode_review(x_train[60]))
print("  Review Sentiment: " + class_names[y_train[60]])
```

<!-- #region colab_type="text" id="mF-Votm66zD5" -->
## Pre-processing Data

We need to make sure that our reviews are of a uniform length. This is for the LSTM's parameters.

Some reviews will need to be truncated, while others need to be padded.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 138} colab_type="code" id="uNtJTLJA6gaT" outputId="dc3a6cce-561b-493f-d306-ad2df4dd48a4"
# The length of reviews
review_length = 500

# Padding / truncated our reviews
x_train = sequence.pad_sequences(x_train, maxlen = review_length)
x_test = sequence.pad_sequences(x_test, maxlen = review_length)

# Check the size of our datasets. Review data for both test and training should 
# contain 25000 reviews of 500 integers. Class data should contain 25000 values, 
# one for each review. Class values are 0 or 1, indicating a negative 
# or positive review.
print("Shape Training Review Data: " + str(x_train.shape))
print("Shape Training Class Data: " + str(y_train.shape))
print("Shape Test Review Data: " + str(x_test.shape))
print("Shape Test Class Data: " + str(y_test.shape))

# Note padding is added to start of review, not the end
print("")
print("Human Readable Review Text (post padding): " + decode_review(x_train[60]))
```

<!-- #region colab_type="text" id="BfOdV_VCCFee" -->
## Create and build LSTM Recurrent Neural Network
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 312} colab_type="code" id="8nmO8M4aCKwT" outputId="58b748ab-6b90-41e2-c5b0-0a7a5b9c91ca"
# We begin by defining the a empty stack. We'll use this for building our 
# network, later by layer.
model = tf.keras.models.Sequential()

# The Embedding Layer provides a spatial mapping (or Word Embedding) of all the 
# individual words in our training set. Words close to one another share context 
# and or meaning. This spatial mapping is learning during the training process.
model.add(
    tf.keras.layers.Embedding(
        input_dim = vocab_size, # The size of our vocabulary 
        output_dim = 32, # Dimensions to which each words shall be mapped
        input_length = review_length # Length of input sequences
    )
)

# Dropout layers fight overfitting and forces the model to learn multiple 
# representations of the same data by randomly disabling neurons in the 
# learning phase.
model.add(
    tf.keras.layers.Dropout(
        rate=0.25 # Randomly disable 25% of neurons
    )
)

# We are using a fast version of LSTM whih is optimised for GPUs. This layer 
# looks at the sequence of words in the review, along with their word embeddings
# and uses both of these to determine to sentiment of a given review.
model.add(
    tf.keras.layers.CuDNNLSTM(
        units=32 # 32 LSTM units in this layer
    )
)

# Add a second dropout layer with the same aim as the first.
model.add(
    tf.keras.layers.Dropout(
        rate=0.25 # Randomly disable 25% of neurons
    )
)

# All LSTM units are connected to a single node in the dense layer. A sigmoid 
# activation function determines the output from this node - a value 
# between 0 and 1. Closer to 0 indicates a negative review. Closer to 1 
# indicates a positive review.
model.add(
    tf.keras.layers.Dense(
        units=1, # Single unit
        activation='sigmoid' # Sigmoid activation function (output from 0 to 1)
    )
)

# Compile the model
model.compile(
    loss=tf.keras.losses.binary_crossentropy, # loss function
    optimizer=tf.keras.optimizers.Adam(), # optimiser function
    metrics=['accuracy']) # reporting metric

# Display a summary of the models structure
model.summary()
```

<!-- #region colab_type="text" id="8Xx1Q2I8WNI9" -->
## Visualise the Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 642} colab_type="code" id="cz0Erj2WU3Vh" outputId="0d5ef724-03c0-43e0-dfea-9c7806e1e288"
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
```

<!-- #region colab_type="text" id="5KdfAoHsGwzo" -->
## Train the LSTM
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 131} colab_type="code" id="rEN1vV4nG1V3" outputId="28f14ffe-cc25-4cea-d3ba-a00320ad5649"
# Train the LSTM on the training data
history = model.fit(

    # Training data : features (review) and classes (positive or negative)
    x_train, y_train,
                    
    # Number of samples to work through before updating the 
    # internal model parameters via back propagation. The 
    # higher the batch, the more memory you need.
    batch_size=256, 

    # An epoch is an iteration over the entire training data.
    epochs=3, 
    
    # The model will set apart his fraction of the training 
    # data, will not train on it, and will evaluate the loss
    # and any model metrics on this data at the end of 
    # each epoch.
    validation_split=0.2,
    
    verbose=1
) 
```

<!-- #region colab_type="text" id="rpCS2-jFH1KY" -->
## Evaluate model with test data and view results
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 164} colab_type="code" id="nPnfxwbnITqV" outputId="08493db5-a14a-40aa-cdf4-33b46b6b68d1"
# Get Model Predictions for test data
from sklearn.metrics import classification_report
predicted_classes = model.predict_classes(x_test)
print(classification_report(y_test, predicted_classes, target_names=class_names))
```

<!-- #region colab_type="text" id="CkfHCIVHrJni" -->
## View some incorrect predictions

Lets have a look at some of the incorrectly classified reviews. For readability we remove the padding.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="bwKLBwBbp7zg" outputId="edd92930-4ee3-4f98-e2bf-2b1684498954"
predicted_classes_reshaped = np.reshape(predicted_classes, 25000)

incorrect = np.nonzero(predicted_classes_reshaped!=y_test)[0]

# We select the first 10 incorrectly classified reviews
for j, incorrect in enumerate(incorrect[0:20]):
    
    predicted = class_names[predicted_classes_reshaped[incorrect]]
    actual = class_names[y_test[incorrect]]
    human_readable_review = decode_review(x_test[incorrect])
    
    print("Incorrectly classified Test Review ["+ str(j+1) +"]") 
    print("Test Review #" + str(incorrect)  + ": Predicted ["+ predicted + "] Actual ["+ actual + "]")
    print("Test Review Text: " + human_readable_review.replace("<PAD> ", ""))
    print("")
```

<!-- #region colab_type="text" id="OlAfxIoTrtYa" -->
## Run your own text against the trained model

This is a fun way to test out the limits of the trained model. To avoid getting errors - type in lower case only and do not use punctuation! 

You'll see the raw prediction from the model - basically a value between 0 and 1.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 65} colab_type="code" id="UEKEB0DpD_8P" outputId="3b9c092a-71a1-42b2-beb3-7e85a97c238a"
# Write your own review
review = "this was a terrible film with too much sex and violence i walked out halfway through"
#review = "this is the best film i have ever seen it is great and fantastic and i loved it"
#review = "this was an awful film that i will never see again"

# Encode review (replace word with integers)
tmp = []
for word in review.split(" "):
    tmp.append(word_index[word])

# Ensure review is 500 words long (by padding or truncating)
tmp_padded = sequence.pad_sequences([tmp], maxlen=review_length) 

# Run your processed review against the trained model
rawprediction = model.predict(array([tmp_padded][0]))[0][0]
prediction = int(round(rawprediction))

# Test the model and print the result
print("Review: " + review)
print("Raw Prediction: " + str(rawprediction))
print("Predicted Class: " + class_names[prediction])
```
