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

# NLP Transformers Search and Sampling

```python colab={} colab_type="code" id="iUrz8QzSIytG"
!pip install transformers
```

```python colab={} colab_type="code" id="SQxUJMIgEyLc"
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
```

<!-- #region colab_type="text" id="RXtlLcprJgNZ" -->
**Greedy search**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" executionInfo={"elapsed": 10215, "status": "ok", "timestamp": 1591623349458, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0LeezhJ8Iw6R" outputId="75e84c63-0117-43c7-e430-7459600678bb"
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I am an avid', return_tensors='tf')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
```

<!-- #region colab_type="text" id="p6GfsmGXJmOh" -->
**Beam Search**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" executionInfo={"elapsed": 26974, "status": "ok", "timestamp": 1591623570604, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="3QxCYMQpJMxd" outputId="f0486086-ccdf-4cb7-9819-559d4a169d3e"
# activate beam search and early_stopping
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 124} colab_type="code" executionInfo={"elapsed": 31108, "status": "ok", "timestamp": 1591623646551, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="reTaRETZJ-rI" outputId="3a4615b0-3476-477b-dbf3-0cf344e01e63"
# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 332} colab_type="code" executionInfo={"elapsed": 30365, "status": "ok", "timestamp": 1591623737000, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="8YXeNsTXKQNL" outputId="5d354537-2977-4b8f-deb9-dc1657144711"
# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
```

<!-- #region colab_type="text" id="izRRfGOALCv3" -->
**Top-K Sampling**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" executionInfo={"elapsed": 10054, "status": "ok", "timestamp": 1591623842295, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lYTEE0WeKmeY" outputId="e1f879d2-3f5b-4f93-80d3-59d7411e04c5"
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 89} colab_type="code" executionInfo={"elapsed": 10628, "status": "ok", "timestamp": 1591623904578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="KwFW94zhLFJC" outputId="e590be89-a12b-4e6e-b5e6-1dcadbe0a70a"
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0, 
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 124} colab_type="code" executionInfo={"elapsed": 10661, "status": "ok", "timestamp": 1591623998427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="axWv_SC5LUMv" outputId="6884ecd4-0b07-4e76-a077-94c4016b6920"
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k to 50
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

<!-- #region colab_type="text" id="nZY1GJJ-MDS2" -->
**Top-p Sampling**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 139} colab_type="code" executionInfo={"elapsed": 10381, "status": "ok", "timestamp": 1591624109202, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6THYEFWQLrHV" outputId="4c5663ac-fb0a-4946-ab3c-480510d93503"
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

<!-- #region colab_type="text" id="HkBHsF47MRjH" -->
**Top-K with Top-p Sampling**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 159} colab_type="code" executionInfo={"elapsed": 27165, "status": "ok", "timestamp": 1591624167232, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lzaVvvIXMGOx" outputId="0cdf3cc1-a587-4515-bb73-40e5c223f0e5"
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```
