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

<!-- #region id="rV5iuZwDZYSc" -->
# Concept - Exploring Word Embeddings
> Embeddings are important now-a-days in recommender systems. In this notebook, we'll look at trained word embeddings. We'll plot the embeddings so we can attempt to visually compare embeddings. We'll then look at analogies and word similarities. We'll use the Gensim library which makes it easy to work with embeddings.

- toc: true
- badges: true
- comments: true
- categories: [Concept, NLP, Embedding, Visualization]
- author: "<a href='https://github.com/jalammar'>Jay Alammar</a>"
- image:
<!-- #endregion -->

```python id="xB1y1EFC_6Eu"
import gensim
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="AlYeTyjz_6Ey" -->
#### Download a table of pre-trained embeddings
<!-- #endregion -->

```python id="e-_Rx6Hn_6E0" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="cc44f9c3-8309-431e-f278-2ace23963bcd"
# Download embeddings (66MB, glove, trained on wikipedia)
model = api.load("glove-wiki-gigaword-50")
```

<!-- #region id="dqK6_8vX_6E3" -->
What's the embedding of 'king'?
<!-- #endregion -->

```python id="XWeOUHjT_6E5" colab={"base_uri": "https://localhost:8080/", "height": 170} outputId="ea01d016-050b-485a-dcd6-45e798a92e2c"
model['king']
```

<!-- #region id="S2duYBkk_6E_" -->
#### How many words does this table have?
<!-- #endregion -->

```python id="_6BObQ0q_6FA" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="5665e2ca-00fb-4414-e2aa-2076f70ffd02"
model.vectors.shape
```

<!-- #region id="u9ULYBe7_6FE" -->
Which means:
* 400,000 words (vocab_size)
* Each has an embedding composed of 50 numbers (embedding_size)
<!-- #endregion -->

<!-- #region id="0vUJ6e9B_6FG" -->
### Visualizing the embedding vector
Let's plot the vector so we can have a colorful visual of values in the embedding vector
<!-- #endregion -->

```python id="RRmmYf8-_6FT"

def plot_embeddings(vectors, labels=None):
    n_vectors = len(vectors)
    fig = plt.figure(figsize=(12, n_vectors))
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax = fig.add_axes([1, 1, 1, 1])
    ax = plt.gca()
    
    sns.heatmap(vectors, cmap='RdBu', vmax=2, vmin=-2, ax=ax)
    
    if labels:
        ax.set_yticklabels(labels,rotation=0)
        ax.tick_params(axis='both', which='major', labelsize=30)
        
    plt.tick_params(axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    # From https://github.com/mwaskom/seaborn/issues/1773
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() # ta-da!
```

<!-- #region id="nW_otYufCb9b" -->
Let's plot the embedding of `king`
<!-- #endregion -->

```python id="1QJf4kfvCWOU" colab={"base_uri": "https://localhost:8080/", "height": 86} outputId="717f0aa1-43e6-491e-ba11-9dc01aadc2d6"
plot_embeddings([model['king']], ['king'])
```

<!-- #region id="xtM9lYuuCgjD" -->
We can also compare multiple embeddings:
<!-- #endregion -->

```python id="K6SADFGaCWUH" colab={"base_uri": "https://localhost:8080/", "height": 303} outputId="31280cc9-07ba-49df-f01a-9680bebc429a"
plot_embeddings([model['king'], model['man'], model['woman'], model['girl'], model['boy']],
              ['king', 'man', 'woman', 'girl', 'boy'])
```

<!-- #region id="cz1-mreNClOK" -->
Here's another example including a number of different concepts:
<!-- #endregion -->

```python id="cgBLjFK8_6FW" colab={"base_uri": "https://localhost:8080/", "height": 303} outputId="a4134d48-0252-404e-8e31-5dab9de3a9e4"
plot_embeddings([model['king'], model['water'], model['god'], model['love'], model['star']],
              ['king', 'water', 'god', 'love', 'star'])
```

<!-- #region id="jjXRJTkH_6FZ" -->
## Analogies
### king - man + woman  = ?
<!-- #endregion -->

```python id="dXs2TVam_6Fa" colab={"base_uri": "https://localhost:8080/", "height": 187} outputId="40055f86-b4cb-4144-a5ac-c324f0f4edc3"
model.most_similar(positive=["king", "woman"], negative=["man"])
```

```python id="KMLRo6DW_6Fd" colab={"base_uri": "https://localhost:8080/", "height": 303} outputId="7f7d2afa-ccf4-4a56-9c0b-b49a725691bf"
plot_embeddings([model['king'], 
                model['man'], 
                model['woman'],
                model['king'] - model['man'] + model['woman'],
                model['queen']],
                ['king', 'man', 'woman', 'king-man+woman', 'queen'])
```

<!-- #region id="Yl0kGOJn_6Fg" -->
**2019 update**: This turned out to be a misconception. The result is actually closer to "king" than it is to "queen", it's just that the code rules out the input vectors as possible outputs


[Fair is Better than Sensational:Man is to Doctor as Woman is to Doctor](https://arxiv.org/abs/1905.09866)

To verify, let's calculate cosine distance between the result of the analogy, and `queen`.
<!-- #endregion -->

```python id="246SfDTN_6Fh" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="3800a072-81a3-4b0c-ba64-98b2ebc4835d"
result = model['king'] - model['man'] + model['woman']

# Similarity between result and 'queen'
cosine_similarity(result.reshape(1, -1), model['queen'].reshape(1, -1))
```

<!-- #region id="nLouwuAzC80t" -->
Let's compare that to the distance between the result and `king`:
<!-- #endregion -->

```python id="eRL5BTC0_6Fk" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="b7d0dca9-4502-4878-b266-4eacf86fcb23"
# Similarity between result and 'king'
cosine_similarity(result.reshape(1, -1), model['king'].reshape(1, -1))
```

<!-- #region id="Vs7wB08b_6Fx" -->
So the result is more similar to king (0.8859834 similarity score) than it is to queen (0.8609581 similarity score).
<!-- #endregion -->

```python id="khmA8TJI_6Fy" colab={"base_uri": "https://localhost:8080/", "height": 243} outputId="4d3b7c65-e10e-4109-f2e2-848429b61699"
plot_embeddings( [model['king'],
                 result, 
                 model['queen']],
                 ['king', 'king-man+woman', 'queen'])
```

<!-- #region id="zwc6OsJv_6F1" -->
## Exercise: doctor - man + woman = ?
<!-- #endregion -->

```python id="Y-oqYbHn_6F2"
# TODO: fill-in values
model.most_similar(positive=[], negative=[])
```

<!-- #region id="mw4vDdlN_6F6" -->
### Verify: Is it, really?
<!-- #endregion -->

```python id="-AMCMD_m_6F7"
# TODO: do analogy algebra
result = model[''] - model[''] + model['']

# Similarity between result and 'nurse'
cosine_similarity(result.reshape(1, -1), model['nurse'].reshape(1, -1))
```

```python id="0f5JXIsF_6F9"

# Similarity between result and 'doctor'
cosine_similarity(result.reshape(1, -1), model['doctor'].reshape(1, -1))
```
