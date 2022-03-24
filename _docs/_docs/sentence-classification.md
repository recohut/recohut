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

# Sentence Classification with Transformers

```python colab={"base_uri": "https://localhost:8080/", "height": 683} colab_type="code" executionInfo={"elapsed": 11420, "status": "ok", "timestamp": 1588872666208, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0NmMdkZO8R6q" outputId="e7193bd9-1eb0-4193-99db-555ac3ac912d"
!pip install transformers
```

<!-- #region colab_type="text" id="_9ZKxKc04Btk" -->
We'll use [The Corpus of Linguistic Acceptability (CoLA)](https://nyu-mll.github.io/CoLA/) dataset for single sentence classification. It's a set of sentences labeled as grammatically correct or incorrect. It was first published in May of 2018, and is one of the tests included in the "GLUE Benchmark" on which models like BERT are competing.

<!-- #endregion -->

<!-- #region colab_type="text" id="4JrUHXms16cn" -->
## 2.1. Download & Extract
<!-- #endregion -->

<!-- #region colab_type="text" id="08pO03Ff1BjI" -->
The dataset is hosted on GitHub in this repo: https://nyu-mll.github.io/CoLA/
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 306} colab_type="code" executionInfo={"elapsed": 4023, "status": "ok", "timestamp": 1588872903783, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="pMtmPMkBzrvs" outputId="102f2b68-542b-4e6f-cfe7-bc5c108acf82"
!wget cola_public_1.1.zip 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
```

<!-- #region colab_type="text" id="_mKctx-ll2FB" -->
Unzip the dataset to the file system. You can browse the file system of the Colab instance in the sidebar on the left.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 4127, "status": "ok", "timestamp": 1588873455780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0Yv-tNv20dnH" outputId="c648cab2-3c76-4be5-b284-d5b52f19a760"
# Unzip the dataset (if we haven't already)
if not os.path.exists('./cola_public/'):
    !unzip cola_public_1.1.zip
```

<!-- #region colab_type="text" id="oQUy9Tat2EF_" -->
## 2.2. Parse
<!-- #endregion -->

<!-- #region colab_type="text" id="xeyVCXT31EZQ" -->
We can see from the file names that both `tokenized` and `raw` versions of the data are available. 

We can't use the pre-tokenized version because, in order to apply the pre-trained BERT, we *must* use the tokenizer provided by the model. This is because (1) the model has a specific, fixed vocabulary and (2) the BERT tokenizer has a particular way of handling out-of-vocabulary words.
<!-- #endregion -->

<!-- #region colab_type="text" id="MYWzeGSY2xh3" -->
We'll use pandas to parse the "in-domain" training set and look at a few of its properties and data points.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 393} colab_type="code" executionInfo={"elapsed": 1632, "status": "ok", "timestamp": 1588873461580, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="_UkeC7SG2krJ" outputId="99678462-c83c-41bf-9294-f20e45bc8ea9"
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
df.sample(10)
```

<!-- #region colab_type="text" id="kfWzpPi92UAH" -->
The two properties we actually care about are the the `sentence` and its `label`, which is referred to as the "acceptibility judgment" (0=unacceptable, 1=acceptable).
<!-- #endregion -->

<!-- #region colab_type="text" id="H_LpQfzCn9_o" -->
Here are five sentences which are labeled as not grammatically acceptible. Note how much more difficult this task is than something like sentiment analysis!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 1121, "status": "ok", "timestamp": 1588873464102, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="blqIvQaQncdJ" outputId="a016fc3e-20e4-4230-ec36-9980b6397884"
df.loc[df.label == 0].sample(5)[['sentence', 'label']]
```

<!-- #region colab_type="text" id="4SMZ5T5Imhlx" -->


Let's extract the sentences and labels of our training set as numpy ndarrays.
<!-- #endregion -->

```python colab={} colab_type="code" id="GuE5BqICAne2"
# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values
```

<!-- #region colab_type="text" id="ex5O1eV-Pfct" -->
## Tokenization & Input Formatting

In this section, we'll transform our dataset into the format that BERT can be trained on.
<!-- #endregion -->

<!-- #region colab_type="text" id="-8kEDRvShcU5" -->
## 3.1. BERT Tokenizer
<!-- #endregion -->

<!-- #region colab_type="text" id="bWOPOyWghJp2" -->

To feed our text to BERT, it must be split into tokens, and then these tokens must be mapped to their index in the tokenizer vocabulary.

The tokenization must be performed by the tokenizer included with BERT--the below cell will download this for us. We'll be using the "uncased" version here.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["cc8306ba5a75487c89029f85eae3ce16", "781ac31eb2bb459ebb63703681a59bf0", "6da65d408b304fc58e3c6efb35c54de5", "594fb08b42a443ccb9177657f6dc78d7", "717beba549e244128cd797380bdd91b9", "11a9b77b19f945dc82dd9f009dae040a", "8374816f613a4381b22c315404d99fee", "b2fdbde4c8bc481fb9fc86ec21c72703"]} colab_type="code" executionInfo={"elapsed": 4468, "status": "ok", "timestamp": 1588873469887, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Z474sSC6oe7A" outputId="6261cb51-d488-4f04-9cda-35670454d41d"
from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
```

<!-- #region colab_type="text" id="dFzmtleW6KmJ" -->
Let's apply the tokenizer to one sentence just to see the output.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" executionInfo={"elapsed": 1428, "status": "ok", "timestamp": 1588875140623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="dLIbudgfh6F0" outputId="93b78705-271e-4f21-8890-899bc349b388"
# Print the original sentence.
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
```

<!-- #region colab_type="text" id="WeNIc4auFUdF" -->
When we actually convert all of our sentences, we'll use the `tokenize.encode` function to handle both steps, rather than calling `tokenize` and `convert_tokens_to_ids` separately. 

Before we can do that, though, we need to talk about some of BERT's formatting requirements.
<!-- #endregion -->

<!-- #region colab_type="text" id="viKGCCh8izww" -->
## 3.2. Required Formatting
<!-- #endregion -->

<!-- #region colab_type="text" id="yDcqNlvVhL5W" -->
The above code left out a few required formatting steps that we'll look at here.

*Side Note: The input format to BERT seems "over-specified" to me... We are required to give it a number of pieces of information which seem redundant, or like they could easily be inferred from the data without us explicity providing it. But it is what it is, and I suspect it will make more sense once I have a deeper understanding of the BERT internals.*

We are required to:
1. Add special tokens to the start and end of each sentence.
2. Pad & truncate all sentences to a single constant length.
3. Explicitly differentiate real tokens from padding tokens with the "attention mask".


<!-- #endregion -->

<!-- #region colab_type="text" id="V6mceWWOjZnw" -->
### Special Tokens

<!-- #endregion -->

<!-- #region colab_type="text" id="Ykk0P9JiKtVe" -->

**`[SEP]`**

At the end of every sentence, we need to append the special `[SEP]` token. 

This token is an artifact of two-sentence tasks, where BERT is given two separate sentences and asked to determine something (e.g., can the answer to the question in sentence A be found in sentence B?). 

I am not certain yet why the token is still required when we have only single-sentence input, but it is!

<!-- #endregion -->

<!-- #region colab_type="text" id="86C9objaKu8f" -->
**`[CLS]`**

For classification tasks, we must prepend the special `[CLS]` token to the beginning of every sentence.

This token has special significance. BERT consists of 12 Transformer layers. Each transformer takes in a list of token embeddings, and produces the same number of embeddings on the output (but with the feature values changed, of course!).

![Illustration of CLS token purpose](http://www.mccormickml.com/assets/BERT/CLS_token_500x606.png)

On the output of the final (12th) transformer, *only the first embedding (corresponding to the [CLS] token) is used by the classifier*.

>  "The first token of every sequence is always a special classification token (`[CLS]`). The final hidden state
corresponding to this token is used as the aggregate sequence representation for classification
tasks." (from the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))

You might think to try some pooling strategy over the final embeddings, but this isn't necessary. Because BERT is trained to only use this [CLS] token for classification, we know that the model has been motivated to encode everything it needs for the classification step into that single 768-value embedding vector. It's already done the pooling for us!


<!-- #endregion -->

<!-- #region colab_type="text" id="u51v0kFxeteu" -->
### Sentence Length & Attention Mask


<!-- #endregion -->

<!-- #region colab_type="text" id="qPNuwqZVK3T6" -->
The sentences in our dataset obviously have varying lengths, so how does BERT handle this?

BERT has two constraints:
1. All sentences must be padded or truncated to a single, fixed length.
2. The maximum sentence length is 512 tokens.

Padding is done with a special `[PAD]` token, which is at index 0 in the BERT vocabulary. The below illustration demonstrates padding out to a "MAX_LEN" of 8 tokens.

![](http://www.mccormickml.com/assets/BERT/padding_and_mask.png)

The "Attention Mask" is simply an array of 1s and 0s indicating which tokens are padding and which aren't (seems kind of redundant, doesn't it?!). This mask tells the "Self-Attention" mechanism in BERT not to incorporate these PAD tokens into its interpretation of the sentence.

The maximum length does impact training and evaluation speed, however. 
For example, with a Tesla K80:

`MAX_LEN = 128  -->  Training epochs take ~5:28 each`

`MAX_LEN = 64   -->  Training epochs take ~2:57 each`






<!-- #endregion -->

<!-- #region colab_type="text" id="l6w8elb-58GJ" -->
## 3.3. Tokenize Dataset
<!-- #endregion -->

<!-- #region colab_type="text" id="U28qy4P-NwQ9" -->
The transformers library provides a helpful `encode` function which will handle most of the parsing and data prep steps for us.

Before we are ready to encode our text, though, we need to decide on a **maximum sentence length** for padding / truncating to.

The below cell will perform one tokenization pass of the dataset in order to measure the maximum sentence length.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 3597, "status": "ok", "timestamp": 1588875147512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="cKsH2sU0OCQA" outputId="c9b3578e-2151-4902-c939-ac29960f2c6c"
max_len = 0

# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
```

<!-- #region colab_type="text" id="1M296yz577fV" -->
Just in case there are some longer test sentences, I'll set the maximum length to 64.

<!-- #endregion -->

<!-- #region colab_type="text" id="tIWAoWL2RK1p" -->
Now we're ready to perform the real tokenization.

The `tokenizer.encode_plus` function combines multiple steps for us:

1. Split the sentence into tokens.
2. Add the special `[CLS]` and `[SEP]` tokens.
3. Map the tokens to their IDs.
4. Pad or truncate all sentences to the same length.
5. Create the attention masks which explicitly differentiate real tokens from `[PAD]` tokens.

The first four features are in `tokenizer.encode`, but I'm using `tokenizer.encode_plus` to get the fifth item (attention masks). Documentation is [here](https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus).

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 153} colab_type="code" executionInfo={"elapsed": 5457, "status": "ok", "timestamp": 1588875150181, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2bBdb3pt8LuQ" outputId="375a5a8f-ce84-4fc5-d46a-711e48493169"
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])
```

<!-- #region colab_type="text" id="aRp4O7D295d_" -->
## 3.4. Training & Validation Split

<!-- #endregion -->

<!-- #region colab_type="text" id="qu0ao7p8rb06" -->
Divide up our training set to use 90% for training and 10% for validation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 5013, "status": "ok", "timestamp": 1588875150184, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="GEgLpFVlo1Z-" outputId="42864853-cf12-4402-ff6b-14ead77aae3b"
from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
```

<!-- #region colab_type="text" id="dD9i6Z2pG-sN" -->
We'll also create an iterator for our dataset using the torch DataLoader class. This helps save on memory during training because, unlike a for loop, with an iterator the entire dataset does not need to be loaded into memory.
<!-- #endregion -->

```python colab={} colab_type="code" id="XGUqOCtgqGhP"
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
```

<!-- #region colab_type="text" id="8bwa6Rts-02-" -->
## Train Our Classification Model
<!-- #endregion -->

<!-- #region colab_type="text" id="3xYQ3iLO08SX" -->
Now that our input data is properly formatted, it's time to fine tune the BERT model. 
<!-- #endregion -->

<!-- #region colab_type="text" id="D6TKgyUzPIQc" -->
## 4.1. BertForSequenceClassification
<!-- #endregion -->

<!-- #region colab_type="text" id="1sjzRT1V0zwm" -->
For this task, we first want to modify the pre-trained BERT model to give outputs for classification, and then we want to continue training the model on our dataset until that the entire model, end-to-end, is well-suited for our task. 

Thankfully, the huggingface pytorch implementation includes a set of interfaces designed for a variety of NLP tasks. Though these interfaces are all built on top of a trained BERT model, each has different top layers and output types designed to accomodate their specific NLP task.  

Here is the current list of classes provided for fine-tuning:
* BertModel
* BertForPreTraining
* BertForMaskedLM
* BertForNextSentencePrediction
* **BertForSequenceClassification** - The one we'll use.
* BertForTokenClassification
* BertForQuestionAnswering

The documentation for these can be found under [here](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html).
<!-- #endregion -->

<!-- #region colab_type="text" id="BXYitPoE-cjH" -->


We'll be using [BertForSequenceClassification](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#bertforsequenceclassification). This is the normal BERT model with an added single linear layer on top for classification that we will use as a sentence classifier. As we feed input data, the entire pre-trained BERT model and the additional untrained classification layer is trained on our specific task. 

<!-- #endregion -->

<!-- #region colab_type="text" id="WnQW9E-bBCRt" -->
OK, let's load BERT! There are a few different pre-trained BERT models available. "bert-base-uncased" means the version that has only lowercase letters ("uncased") and is the smaller version of the two ("base" vs "large").

The documentation for `from_pretrained` can be found [here](https://huggingface.co/transformers/v2.2.0/main_classes/model.html#transformers.PreTrainedModel.from_pretrained), with the additional parameters defined [here](https://huggingface.co/transformers/v2.2.0/main_classes/configuration.html#transformers.PretrainedConfig).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["f4bfd726d318479ba17f26a334131d48", "1f0dc332c051405faced0a9199615119", "1ff432472b364a10bd2f33a79a51e1f1", "3988f885fdf74f64b798b51140fec8a0", "5ef9effd20eb4ef78274776fd7cee7fe", "20aab88cae494faa880ffcab97f18d5c", "9bd5e7d2611847db84b475aad5a3a720", "f9a685737dd241778db01ddbfd8e2aa7", "a9cd931a75b74e579bd20dd2d8947ba9", "1d8d980718d14e06aef24aa8bdb78508", "e7f38ec7c1ff4167a1d1708fe8cc667d", "49e8d2b98491499597bd224fdd70b65a", "c3a5d9b5397540238c4a02b90421b735", "251b35d145254da1aa8b9cab5f5d1f4f", "321a9382437a41cdabb1705a2a63cea8", "349628b1aba841228f3f81ba425cf400"]} colab_type="code" executionInfo={"elapsed": 48642, "status": "ok", "timestamp": 1588875200802, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="gFsCTp_mporB" outputId="53f3b940-0083-436d-ddcb-6158279e12bd"
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()
```

<!-- #region colab_type="text" id="e0Jv6c7-HHDW" -->
Just for curiosity's sake, we can browse all of the model's parameters by name here.

In the below cell, I've printed out the names and dimensions of the weights for:

1. The embedding layer.
2. The first of the twelve transformers.
3. The output layer.



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 612} colab_type="code" executionInfo={"elapsed": 48131, "status": "ok", "timestamp": 1588875200804, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="8PIiVlDYCtSq" outputId="85301bdc-08df-4a51-e30a-185edfa44acd"
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
```

<!-- #region colab_type="text" id="qRWT-D4U_Pvx" -->
## 4.2. Optimizer & Learning Rate Scheduler
<!-- #endregion -->

<!-- #region colab_type="text" id="8o-VEBobKwHk" -->
Now that we have our model loaded we need to grab the training hyperparameters from within the stored model.

For the purposes of fine-tuning, the authors recommend choosing from the following values (from Appendix A.3 of the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)):

>- **Batch size:** 16, 32  
- **Learning rate (Adam):** 5e-5, 3e-5, 2e-5  
- **Number of epochs:** 2, 3, 4 

We chose:
* Batch size: 32 (set when creating our DataLoaders)
* Learning rate: 2e-5
* Epochs: 4 (we'll see that this is probably too many...)

The epsilon parameter `eps = 1e-8` is "a very small number to prevent any division by zero in the implementation" (from [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)).

You can find the creation of the AdamW optimizer in `run_glue.py` [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L109).
<!-- #endregion -->

```python colab={} colab_type="code" id="GLs72DuMODJO"
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

```

```python colab={} colab_type="code" id="-p0upAhhRiIx"
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
```

<!-- #region colab_type="text" id="RqfmWwUR_Sox" -->
## 4.3. Training Loop
<!-- #endregion -->

<!-- #region colab_type="text" id="_QXZhFb4LnV5" -->
Below is our training loop. There's a lot going on, but fundamentally for each pass in our loop we have a trianing phase and a validation phase. 

> *Thank you to [Stas Bekman](https://ca.linkedin.com/in/stasbekman) for contributing the insights and code for using validation loss to detect over-fitting!*

**Training:**
- Unpack our data inputs and labels
- Load data onto the GPU for acceleration
- Clear out the gradients calculated in the previous pass. 
    - In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out.
- Forward pass (feed input data through the network)
- Backward pass (backpropagation)
- Tell the network to update parameters with optimizer.step()
- Track variables for monitoring progress

**Evalution:**
- Unpack our data inputs and labels
- Load data onto the GPU for acceleration
- Forward pass (feed input data through the network)
- Compute loss on our validation data and track variables for monitoring progress

Pytorch hides all of the detailed calculations from us, but we've commented the code to point out which of the above steps are happening on each line. 

> *PyTorch also has some [beginner tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) which you may also find helpful.*
<!-- #endregion -->

<!-- #region colab_type="text" id="pE5B99H5H2-W" -->
Define a helper function for calculating accuracy.
<!-- #endregion -->

```python colab={} colab_type="code" id="9cQNvaZ9bnyy"
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
```

<!-- #region colab_type="text" id="KNhRtWPXH9C3" -->
Helper function for formatting elapsed times as `hh:mm:ss`

<!-- #endregion -->

```python colab={} colab_type="code" id="gpt6tR83keZD"
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

```

<!-- #region colab_type="text" id="cfNIhN19te3N" -->
We're ready to kick off the training!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" executionInfo={"elapsed": 669007, "status": "ok", "timestamp": 1588875824063, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6J-FYdx6nFE_" outputId="1bede604-c2f2-4430-bd21-e25bbfcedcb8"
import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
```

<!-- #region colab_type="text" id="VQTvJ1vRP7u4" -->
Let's view the summary of the training process.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 668834, "status": "ok", "timestamp": 1588875824065, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6O_NbXFGMukX" outputId="cbde0b14-58c9-438d-bf93-7eaf19f4e2e7"
import pandas as pd

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
df_stats
```

<!-- #region colab_type="text" id="1-G03mmwH3aI" -->
Notice that, while the the training loss is going down with each epoch, the validation loss is increasing! This suggests that we are training our model too long, and it's over-fitting on the training data. 

(For reference, we are using 7,695 training samples and 856 validation samples).

Validation Loss is a more precise measure than accuracy, because with accuracy we don't care about the exact output value, but just which side of a threshold it falls on. 

If we are predicting the correct answer, but with less confidence, then validation loss will catch this, while accuracy will not.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 481} colab_type="code" executionInfo={"elapsed": 668663, "status": "ok", "timestamp": 1588875824066, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="68xreA9JAmG5" outputId="39af4205-6d58-45ec-bc1d-bbdcd5539090"
import matplotlib.pyplot as plt
% matplotlib inline

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()
```

<!-- #region colab_type="text" id="mkyubuJSOzg3" -->
## Performance On Test Set
<!-- #endregion -->

<!-- #region colab_type="text" id="DosV94BYIYxg" -->
Now we'll load the holdout dataset and prepare inputs just as we did with the training set. Then we'll evaluate predictions using [Matthew's correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) because this is the metric used by the wider NLP community to evaluate performance on CoLA. With this metric, +1 is the best score, and -1 is the worst score. This way, we can see how well we perform against the state of the art models for this specific task.
<!-- #endregion -->

<!-- #region colab_type="text" id="Tg42jJqqM68F" -->
### 5.1. Data Preparation

<!-- #endregion -->

<!-- #region colab_type="text" id="xWe0_JW21MyV" -->

We'll need to apply all of the same steps that we did for the training data to prepare our test data set.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 668324, "status": "ok", "timestamp": 1588875824067, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mAN0LZBOOPVh" outputId="1925aec3-daa5-49e2-aea6-512a5456418b"
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
sentences = df.sentence.values
labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
```

<!-- #region colab_type="text" id="16lctEOyNFik" -->
## 5.2. Evaluate on Test Set

<!-- #endregion -->

<!-- #region colab_type="text" id="rhR99IISNMg9" -->

With the test set prepared, we can apply our fine-tuned model to generate predictions on the test set.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 6757, "status": "ok", "timestamp": 1588875995470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Hba10sXR7Xi6" outputId="22cce472-bc7f-44da-c4ab-25103d5a04aa"
# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')
```

<!-- #region colab_type="text" id="-5jscIM8R4Gv" -->
Accuracy on the CoLA benchmark is measured using the "[Matthews correlation coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)" (MCC).

We use MCC here because the classes are imbalanced:

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 6451, "status": "ok", "timestamp": 1588875995472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="hWcy0X1hirdx" outputId="8e842270-e548-46a9-8f7b-b485d84b114e"
print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" executionInfo={"elapsed": 6312, "status": "ok", "timestamp": 1588875995473, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="cRaZQ4XC7kLs" outputId="429f2690-6703-4d65-d888-2efaef1ae647"
from sklearn.metrics import matthews_corrcoef

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
  
  # The predictions for this batch are a 2-column ndarray (one column for "0" 
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  
  # Calculate and store the coef for this batch.  
  matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
  matthews_set.append(matthews)
```

<!-- #region colab_type="text" id="IUM0UA1qJaVB" -->
The final score will be based on the entire test set, but let's take a look at the scores on the individual batches to get a sense of the variability in the metric between batches. 

Each batch has 32 sentences in it, except the last batch which has only (516 % 32) = 4 test sentences in it.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 427} colab_type="code" executionInfo={"elapsed": 5971, "status": "ok", "timestamp": 1588875995474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="pyfY1tqxU0t9" outputId="07e633e4-887e-4648-90ab-3929be59cfdc"
# Create a barplot showing the MCC score for each batch of test samples.
ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

plt.title('MCC Score per Batch')
plt.ylabel('MCC Score (-1 to +1)')
plt.xlabel('Batch #')

plt.show()
```

<!-- #region colab_type="text" id="1YrjAPX2V-l4" -->
Now we'll combine the results for all of the batches and calculate our final MCC score.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 4907, "status": "ok", "timestamp": 1588875995475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="oCYZa1lQ8Jn8" outputId="bc7b1f95-3c6a-454f-ba67-f910751e5ead"
# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('Total MCC: %.3f' % mcc)
```

<!-- #region colab_type="text" id="jXx0jPc4HUfZ" -->
Cool! In about half an hour and without doing any hyperparameter tuning (adjusting the learning rate, epochs, batch size, ADAM properties, etc.) we are able to get a good score. 

> *Note: To maximize the score, we should remove the "validation set" (which we used to help determine how many epochs to train for) and train on the entire training set.*

The library documents the expected accuracy for this benchmark [here](https://huggingface.co/transformers/examples.html#glue) as `49.23`.

You can also look at the official leaderboard [here](https://gluebenchmark.com/leaderboard/submission/zlssuBTm5XRs0aSKbFYGVIVdvbj1/-LhijX9VVmvJcvzKymxy). 

Note that (due to the small dataset size?) the accuracy can vary significantly between runs.

<!-- #endregion -->

<!-- #region colab_type="text" id="GfjYoa6WmkN6" -->
## Conclusion
<!-- #endregion -->

<!-- #region colab_type="text" id="xlQG7qgkmf4n" -->
This post demonstrates that with a pre-trained BERT model you can quickly and effectively create a high quality model with minimal effort and training time using the pytorch interface, regardless of the specific NLP task you are interested in.
<!-- #endregion -->

<!-- #region colab_type="text" id="q2079Qyn8Mt8" -->
## A1. Saving & Loading Fine-Tuned Model

This first cell (taken from `run_glue.py` [here](https://github.com/huggingface/transformers/blob/35ff345fc9df9e777b27903f11fa213e4052595b/examples/run_glue.py#L495)) writes the model and tokenizer out to disk.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" executionInfo={"elapsed": 4113, "status": "ok", "timestamp": 1588876001387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6ulTWaOr8QNY" outputId="0dd2024e-1342-486b-a072-5d3e77f8db6d"
import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))

```

<!-- #region colab_type="text" id="Z-tjHkR7lc1I" -->
Let's check out the file sizes, out of curiosity.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 119} colab_type="code" executionInfo={"elapsed": 10689, "status": "ok", "timestamp": 1588876007996, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mqMzI3VTCZo5" outputId="e9a40011-9d8a-40ce-c969-4763c5466e9d"
!ls -l --block-size=K ./model_save/
```

<!-- #region colab_type="text" id="fr_bt2rFlgDn" -->
The largest file is the model weights, at around 418 megabytes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 15817, "status": "ok", "timestamp": 1588876013151, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="-WUFUIQ8Cu8D" outputId="c1a7388a-b4b6-46c6-fc39-6bc06ff37fb2"
!ls -l --block-size=M ./model_save/pytorch_model.bin
```

<!-- #region colab_type="text" id="dzGKvOFAll_e" -->
To save your model across Colab Notebook sessions, download it to your local machine, or ideally copy it to your Google Drive.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 122} colab_type="code" executionInfo={"elapsed": 29521, "status": "ok", "timestamp": 1588876055202, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Trr-A-POC18_" outputId="1d5f168e-dfb4-47b3-8d29-8d07b2db9797"
# Mount Google Drive to this Notebook instance.
from google.colab import drive
drive.mount('/content/drive')
```

```python colab={} colab_type="code" id="NxlZsafTC-V5"
# Copy the model files to a directory in your Google Drive.
!cp -r ./model_save/ "/content/drive/My Drive/pickles"
```

<!-- #region colab_type="text" id="W0vstijw85SZ" -->
The following functions will load the model back from disk.
<!-- #endregion -->

```python colab={} colab_type="code" id="nskPzUM084zL"
# Load a trained model and vocabulary that you have fine-tuned
model = model_class.from_pretrained(output_dir)
tokenizer = tokenizer_class.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)
```

<!-- #region colab_type="text" id="NIWouvDrGVAi" -->
## A.2. Weight Decay


<!-- #endregion -->

<!-- #region colab_type="text" id="f123ZAlF1OyW" -->
The huggingface example includes the following code block for enabling weight decay, but the default decay rate is "0.0", so I moved this to the appendix.

This block essentially tells the optimizer to not apply weight decay to the bias terms (e.g., $ b $ in the equation $ y = Wx + b $ ). Weight decay is a form of regularization--after calculating the gradients, we multiply them by, e.g., 0.99.
<!-- #endregion -->

```python colab={} colab_type="code" id="QxSMw0FrptiL"
# This code is taken from:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

# Don't apply weight decay to any parameters whose names include these tokens.
# (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
no_decay = ['bias', 'LayerNorm.weight']

# Separate the `weight` parameters from the `bias` parameters. 
# - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01. 
# - For the `bias` parameters, the 'weight_decay_rate' is 0.0. 
optimizer_grouped_parameters = [
    # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.1},
    
    # Filter for parameters which *do* include those.
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# Note - `optimizer_grouped_parameters` only includes the parameter values, not 
# the names.
```
