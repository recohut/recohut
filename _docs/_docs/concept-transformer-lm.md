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

<!-- #region id="BFXTlKxZYEcZ" -->
# Concept - TransformerLM Quick Start and Guide 
> In this notebook, we will Use a pre-trained TransformerLM, Train a TransformerLM model, and Looking inside the Trax TransformerLM

- toc: true
- badges: true
- comments: true
- categories: [Concept, NLP, Transformer]
- author: "<a href='https://github.com/jalammar'>Jay Alammar</a>"
- image:
<!-- #endregion -->

<!-- #region id="CE38oeYU9fgv" -->
Language models are machine learning models that power some of the most impressive applications involving text and language (e.g. machine translation, sentiment analysis, chatbots, summarization). At the time of this writing, some of the largest ML models in existence are language models. They are also based on the [transformer](https://arxiv.org/abs/1706.03762) architecture. The transformer language model (TransformerLM) is a simpler [variation](https://arxiv.org/pdf/1801.10198.pdf) of the original transformer architecture and is useful for plenty of tasks.

![](https://storage.googleapis.com/ml-intro/t/transformerLM-1.png)

The [Trax](https://trax-ml.readthedocs.io/en/latest/) implementation of TransformerLM focuses on clear code and speed.  It runs without any changes on CPUs, GPUs and TPUs.

In this notebook, we will:

1. Use a pre-trained TransformerLM
2. Train a TransformerLM model
3. Looking inside the Trax TransformerLM

<!-- #endregion -->

```python id="nWKJgBwkeTax" colab={"base_uri": "https://localhost:8080/", "height": 462} outputId="b3292c68-8458-415e-f215-0432fd33515b"
import os
import numpy as np
! pip install -q -U trax
import trax
```

<!-- #region id="mWxTtb4snypO" -->
## Using a pre-trained TransformerLM

The following cell loads a pre-trained TransformerLM that sorts a list of four integers.
<!-- #endregion -->

```python id="sL3rakwb05cL" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="7219838b-68a7-4d2f-e28a-3be0928db948"
# Create a Transformer model.
# Have to use the same configuration of the pre-trained model we'll load next
model = trax.models.TransformerLM(  
    d_model=32, d_ff=128, n_layers=2, 
    vocab_size=32, mode='predict')

# Initialize using pre-trained weights.
model.init_from_file('gs://ml-intro/models/sort-transformer.pkl.gz',
                     weights_only=True, 
                     input_signature=trax.shapes.ShapeDtype((1,1), dtype=np.int32))

# Input sequence
# The 0s indicate the beginning and end of the input sequence
input = [0, 3, 15, 14, 9, 0]


# Run the model
output = trax.supervised.decoding.autoregressive_sample(
    model, np.array([input]), temperature=0.0, max_length=4)

# Show us the output
output
```

<!-- #region id="I5rohWvyoOaf" -->
This is a trivial example to get you started and put a toy transformer into your hands. Language models get their name from their ability to assign probabilities to sequences of words. This property makes them useful for generating text (and other types of sequences) by probabilistically choosing the next item in the sequence (often the highest probability one)  -- exactly like the next-word suggestion feature of your smartphone keyboard.

In Trax, TransformerLM is a series of [Layers]() combined using the [Serial]() combinator. A high level view of the TransformerLM we've declared above can look like this:

![](https://storage.googleapis.com/ml-intro/t/transformerLM-layers-1.png)

The model has two decoder layers because we set `n_layers` to 2. TransformerLM makes predictions by being fed one token at a time, with output tokens typically fed back as inputs (that's the `autoregressive` part of the `autoregressive_sample` method we used to generate the output from the model). 

If we're to think of a simple model trained to generate the fibonacci sequence, we can give it a number in the sequence and it would continue to generate the next items in the sequence:

![](https://storage.googleapis.com/ml-intro/t/transformerLM-input-output-fib.gif)


## Train a TransformerLM Model

Let's train a TransformerLM model. We'll train this one to reverse a list of integers. This is another toy task that we can train a small transformer to do. But using the concepts we'll go over, you'll be able to train proper language models on larger dataset.

**Example**: This model is to take a sequence like `[1, 2, 3, 4]` and return `[4, 3, 2, 1]`.

1. Create the Model
1. Prepare the Dataset
1. Train the model using `Trainer`

<!-- #endregion -->

<!-- #region id="Vra3JRlJtilo" -->

### Create the Model
<!-- #endregion -->

```python id="7Vz4xGIRur_C"
# Create a Transformer model.
def tiny_transformer_lm(mode='train'):
  return trax.models.TransformerLM(  
          d_model=32, d_ff=128, n_layers=2, 
          vocab_size=32, mode=mode)
```

<!-- #region id="vtHa8mOmb-NZ" -->
Refer to [TransferLM in the API reference](https://trax-ml.readthedocs.io/en/latest/trax.models.html#trax.models.transformer.TransformerLM) to understand each of its parameters and their default values. We have chosen to create a small model using these values for `d_model`, `d_ff`, and `n_layers` to be able to train the model more quickly on this simple task.

![](https://storage.googleapis.com/ml-intro/t/untrained-transformer.png)

### Prepare the Dataset

Trax models are trained on streams of data represented as python iterators. [`trax.data`](https://trax-ml.readthedocs.io/en/latest/trax.data.html) gives you the tools to construct your datapipeline. Trax also gives you readily available access to [TensorFlow Datasets](https://www.tensorflow.org/datasets).

For this simple task, we will create a python generator. Every time we invoke it, it returns a batch of training examples.
<!-- #endregion -->

```python id="0_JoHNp_1IUe"
def reverse_ints_task(batch_size, length=4):
  while True:
    random_ints = m = np.random.randint(1, 31, (batch_size,length))
    source = random_ints

    target = np.flip(source, 1)

    zero = np.zeros([batch_size, 1], np.int32)
    x = np.concatenate([zero, source, zero, target], axis=1)

    loss_weights = np.concatenate([np.zeros((batch_size, length+2)),
                                    np.ones((batch_size, length))], axis=1)
    yield (x, x, loss_weights)  # Here inputs and targets are the same.

reverse_ints_inputs =  reverse_ints_task(16)
```

<!-- #region id="RtzNKw0Flf-v" -->


This function prepares a dataset and returns one batch at a time. If we ask for a batch size of 8, for example, it returns the following:
<!-- #endregion -->

```python id="UgPlRp1AjJ2O" colab={"base_uri": "https://localhost:8080/", "height": 153} outputId="348b596f-7acc-42fc-fc52-32f64e3f4237"
a = reverse_ints_task(8)
sequence_batch, _ , masks = next(a)
sequence_batch
```

<!-- #region id="Uae3WdEA9etU" -->
You can see that each example starts with 0, then a list of integers, then another 0, then the reverse of the list of integers. The function will give us as many examples and batches as we request.

In addition to the example, the generator returns a mask vector. During the training process, the model is challenged to predict the tokens hidden by the mask (which have a value of 1 associated with that position. So for example, if the first element in the batch is the following vector:

<table><tr>
<td><strong>0</strong></td><td>5</td><td>6</td><td>7</td><td>8</td><td><strong>0</strong></td><td>8</td><td>7</td><td>6</td><td>5</td>
</tr></table> 

And the associated mask vector for this example is:
<table><tr>
<td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td>
</tr></table> 

Then the model will only be presented with the following prefix items, and it has to predict the rest:
<table><tr>
<td><strong>0</strong></td><td>5</td><td>6</td><td>7</td><td>8</td><td><strong>0</strong></td><td>_</td><td>_</td><td>_ </td><td>_</td>
</tr></table> 

It's important here to note that while `5, 6, 7, 8` constitute the input sequence, the **zeros** serve a different purpose. We are using them as special tokens to delimit where the source sequence begins and ends. 

With this, we now have a method that streams the dataset in addition to the method that creates the model.

![](https://storage.googleapis.com/ml-intro/t/untrained-transformer-and-dataset.png)


### Train the model

Trax's [training](https://trax-ml.readthedocs.io/en/latest/notebooks/trax_intro.html#Supervised-training) takes care of the training process. We hand it the model, define training and eval tasks, and create the training loop. We then start the training loop.
<!-- #endregion -->

```python id="itAKBkN81H1F" colab={"base_uri": "https://localhost:8080/", "height": 272} outputId="30731cc2-2cce-4bbd-fa4c-a710710cf64b"
from trax.supervised import training
from trax import layers as tl

# Training task.
train_task = training.TrainTask(
    labeled_data=reverse_ints_inputs,
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=500,
)


# Evaluaton task.
eval_task = training.EvalTask(
    labeled_data=reverse_ints_inputs,
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    n_eval_batches=20  # For less variance in eval numbers.
)

output_dir = os.path.expanduser('~/train_dir/')
!rm -f ~/train_dir/model.pkl.gz  # Remove old model.

# Train tiny model with Loop.
training_loop = training.Loop(
    tiny_transformer_lm(),
    train_task,
    eval_tasks=[eval_task],
    output_dir=output_dir)

# run 1000 steps (batches)
training_loop.run(1000)
```

<!-- #region id="umb-MvIJme65" -->
The Trainer is the third key component in this process that helps us arrive at the trained model.

![](https://storage.googleapis.com/ml-intro/t/transformerLM-training.png)

### Make predictions

Let's take our newly minted model for a ride. To do that, we load it up, and use the handy `autoregressive_sample` method to feed it our input sequence and return the output sequence. These components now look like this:

![](https://storage.googleapis.com/ml-intro/t/transformerLM-sampling-prediction.png)

And this is the code to do just that:
<!-- #endregion -->

```python id="psqZzQELeMxE" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="7a765372-e3db-4bb6-c07c-532481338863"

input = np.array([[0, 4, 6, 8, 10, 0]])

# Initialize model for inference.
predict_model = tiny_transformer_lm(mode='predict')
predict_signature = trax.shapes.ShapeDtype((1,1), dtype=np.int32)
predict_model.init_from_file(os.path.join(output_dir, "model.pkl.gz"),
                             weights_only=True, input_signature=predict_signature)

# Run the model
output = trax.supervised.decoding.autoregressive_sample(
    predict_model, input, temperature=0.0, max_length=4)

# Print the contents of output
print(output)
```

<!-- #region id="CfZsEnMqvmQh" -->
If things go correctly, the model would be able to reverse the string and output `[[10 8 6 4]]`
<!-- #endregion -->

<!-- #region id="ERq8BDIvD8GP" -->
## Transformer vs. TransformerLM
TransformerLM is a great place to start learning about Transformer architectures. The main difference between it and the original Transformer is that it's made up of a decoder stack, while Transformer is made up of an encoder stack and decoder stack (with the decoder stack being nearly identical to TransformerLM).

![](https://storage.googleapis.com/ml-intro/t/transformer-vs-transformerlm.png)


<!-- #endregion -->

<!-- #region id="crm0Xg-1r92p" -->

## Looking inside the Trax TransformerLM
In Trax, TransformerLM is implemented as a single Serial layer

![](https://storage.googleapis.com/ml-intro/t/transformerLM-serial-trax-layer.png)

This graph shows you two of the central concepts in Trax. Layers are the basic building blocks. Serial is the most common way to compose multiple layers together in sequence.

<!-- #endregion -->

<!-- #region id="mFWXztCysHZJ" -->

### Layers
Layers are best described in the [Trax Layers Intro](https://trax-ml.readthedocs.io/en/latest/notebooks/layers_intro.html).

For a Transformer to make a calculation (translate a sentence, summarize an article, or generate text), input tokens pass through many steps of transformation and
computation (e.g. embedding, positional encoding, self-attention, feed-forward neural networks...tec). Each of these steps is a layer (some with their own sublayers). 

Each layer you use or define takes a fixed number of input tensors and returns a fixed number of output tensors (n_in and n_out respectively, both of which default to 1).

![](https://storage.googleapis.com/ml-intro/t/trax-layer-inputs-outputs.png)

A simple example of a layer is the ReLU activation function:

![](https://storage.googleapis.com/ml-intro/t/relu-trax-layer.png)

Trax is a deep learning library, though. And so, a layer can also contain weights. An example of this is the Dense layer. Here is a dense layer that multiplies the input tensor with a weight matrix (`W`) and adds a bias (`b`) (both W and b are saved inside the `weights` property of the layer):

![](https://storage.googleapis.com/ml-intro/t/dense-trax-layer.png)

In practice, Dense and Relu often go hand in hand. With Dense first working on a tensor, and ReLu then processing the output of the Dense layer. This is a perfect job for Serial, which, in simple cases, chains two or more layers and hands over the output of the first layer to the following one:

![](https://storage.googleapis.com/ml-intro/t/serial-dense-relu-trax.png)

The Serial combinator is a layer itself. So we can think of it as a layer containing a number of sublayers:

![](https://storage.googleapis.com/ml-intro/t/serial-layer-dense-relu-trax.png)

With these concepts in mind, let's go back and unpack the layers inside the TransformerLM Serial.







<!-- #endregion -->

<!-- #region id="OXqF121jsc7R" -->
### Input, Decoder Blocks, and Output Layers

It's straightforward to read the delcaration of TransformerLM to understand the layers that make it up. In general, you can group these layers into a set of input layers, then Transformer decoder blocks, and a set of output blocks. The number of Transformer blocks (`n_layers`) is one of the key parameters when creating a TransformerLM model. This is a way to think of the layer groups of a TransformerLM:


<div align="center">
<img src="https://storage.googleapis.com/ml-intro/t/TransformerLM-layer-groups.png" />
</div>

* The **input layers** take each input token id and look up its proper embedding and positional encoding.
* The prediction calculations happen in the stack of **decoder blocks**.
* The **output layers** take the output of the final Decoder block and project it to the output vocabulary. The LogSoftmax layer then turns the scoring of each potential output token into a probability score.

<!-- #endregion -->

<!-- #region id="dDsYBvBJFKjh" -->

### Transformer Decoder Block
A decoder block has two major components:
* A **Causal self-attention** layer. Self-attention incorporates information from other tokens that could help make more sense of the current token being processed. Causal attention only allows the incorporation of information from previous positions. One key parameter when creating a TransformerLM model is `n_heads`, which is the number of "attention heads".
* A **FeedForward** component. This is where the primary prediction computation is calculated. The key parameter associated with this layer is `d_ff`, which specifies the dimensions of the neural network layer used in this block. 


![](https://storage.googleapis.com/ml-intro/t/transformerLM-d_self-attention-ff.png)

This figure also shows the `d_model` parameter, which specifies the dimension of tensors at most points in the model, including the embedding, and the majority of tensors handed off between the various layers in the model. 
<!-- #endregion -->

<!-- #region id="WKv7vr20sdGo" -->
### Multiple Inputs/Outputs, Branch, and Residual
There are a couple more central Trax concept to cover to gain a deeper understanding of how Trax implements TransformerLM



<!-- #endregion -->

<!-- #region id="SOSlCg2msdUk" -->

#### Multiple Inputs/Outputs
The layers we've seen so far all have one input tensor and one output tensor. A layer could have more. For example, the Concatenate layer:

![](https://storage.googleapis.com/ml-intro/t/trax-concatenate-layer.png)

<!-- #endregion -->

<!-- #region id="5pyHqDuwsLNm" -->

#### Branch
We saw the Serial combinator that combines layers serially. Branch combines layers in parallel. It supplies input copies to each of its sublayers.

For example, if we wrap two layers (each expecting one input) in a Branch layer, and we pass a tensor to Branch, it copies it as the input to both of its sublayers as shown here:


![](https://storage.googleapis.com/ml-intro/t/branch-combinator-trax-inputs.png)

Since the sublayers have two outputs (one from each), then the Branch layer would also end up outputing both of those tensors:

![](https://storage.googleapis.com/ml-intro/t/branch-combinator-trax-output.png)
<!-- #endregion -->

<!-- #region id="0IQ8idxqxLsB" -->

<!-- #endregion -->

<!-- #region id="RLlq8dbdxMAN" -->
#### Residual

Residual connections are an important component of Transformer architectures. Inside a Decoder Block, both the causal-attention layer and the
feed-forward layer have residual connections around them:

![](https://storage.googleapis.com/ml-intro/t/trax-residual-input.png)

What that means, is that a copy of the input tensor is added to the output of the Attention layer:

![](https://storage.googleapis.com/ml-intro/t/trax-residual-output.png)

In Trax, this is achieved using the Residual layer, which combines both the Serial and Branch combinators:

![](https://storage.googleapis.com/ml-intro/t/trax-residual-layers-1.png)


Similarly, the feed-forward sublayer has another residual connection around it:


![](https://storage.googleapis.com/ml-intro/t/trax-transformer-residual-layers-2.png)
<!-- #endregion -->
