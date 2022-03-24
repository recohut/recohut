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

<!-- #region id="JoPWS06PA-VL" -->
# Concept - Visualizing Embeddings Using TSNE
> This notebook demostrates visualization of embeddings using TSNE.
We will use the embedings we trained in the "Training_embeddings_using_gensim.ipnb notebook. We are using the word2vec_cbow model. 

- toc: false
- badges: true
- comments: true
- categories: [Concept, Embedding, TSNE, Gensim, Visualization]
- author: "<a href='https://notebooks.quantumstat.com/'>Quantum Stat</a>"
- image:
<!-- #endregion -->

```python id="pxZ2e9cRT1v6"
# FOR GOOGLE COLAB USERS
# upload the "word2vec.bin" file in form the repository which can be found in the same folder as this notebook.
try:
    from google.colab import files
    uploaded = files.upload()
except ModuleNotFoundError:
    print("Not using colab")
```

```python id="84Dvp9B7_bDZ"
from gensim.models import Word2Vec, KeyedVectors #To load the model
import warnings
warnings.filterwarnings('ignore') #ignore any generated warnings

import numpy as np
import matplotlib.pyplot as plt #to generate the t-SNE plot
from sklearn.manifold import TSNE #scikit learn's TSNE 

import os

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

```python id="qKeqgvn5VEFs"
# load model
cwd=os.getcwd() 
model = KeyedVectors.load_word2vec_format(cwd+'\Models\word2vec_cbow.bin', binary=True)
```

<!-- #region id="KeQBFWezGJL9" -->
## TSNE
t-SNE stands for t-distributed Stochastic Neighbouring Entities. Its a technique used for visualizing high dimensional data by reducing it to a 2 or 3 dimensions.
 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 141} id="jM8IFPOfYRVD" outputId="808ff5c9-9689-4572-b0df-f5a9962e7c4b"
#Preprocessing our models vocabulary to make better visualizations

words_vocab= list(model.wv.vocab)#all the words in the vocabulary. 
print("Size of Vocabulary:",len(words_vocab))
print("Few words in Vocabulary",words_vocab[:50])

#Let us remove the stop words from this it will help making the visualization cleaner
stopwords_en = stopwords.words()
words_vocab_without_sw = [word.lower() for word in words_vocab if not word in stopwords_en]
print("Size of Vocabulary without stopwords:",len(words_vocab_without_sw))
print("Few words in Vocabulary without stopwords",words_vocab_without_sw[:30])
#The size didnt reduce much after removing the stop words so lets try visualizing only a selected subset of words
```

```python id="gM4UhX7pXby7"
#With the increase in the amount of data, it becomes more and more difficult to visualize and interpret
#In practice, similar words are combined into groups for further visualization.

keys = ['school', 'year', 'college', 'city', 'states', 'university', 'team', 'film']
embedding_clusters = []
word_clusters = []

for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)#apending access vector of all similar words
    word_clusters.append(words)#appending list of all smiliar words
```

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="CioQ1ce73qFl" outputId="33c1b8a5-29bc-49a9-e1a7-171c58cfdf9c"
print("Embedding clusters:",embedding_clusters[0][0])#Access vector of the first word only
print("Word Clousters:",word_clusters[:2])
```

```python id="1vdOZBXtDeT3"
from sklearn.manifold import TSNE
import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape #geting the dimensions
tsne_model_en_2d = TSNE(perplexity=5, n_components=2, init='pca', n_iter=1500, random_state=2020) 
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2) #reshaping it into 2d so we can visualize it
```

<!-- #region id="pqUW6lyeyowC" -->
### Hyperparameters of TSNE

1. n_components: The number of components, i.e., the dimension of the value space
2. perplexity: The number of effective neighbours
3. n_iter: Maximum number of iterations for the optimization.
4. init: Initialization of embedding.

t-SNE requires good amount of hyperparameter tuning to give effective results. More details on the hyperparameters can be found in the official [docs](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). It is very easy to misread tsne too. This [article](https://distill.pub/2016/misread-tsne/) provides more deatils about it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 537} id="LtOVUzKuhfAu" outputId="84290061-6c70-4ecf-a238-9ed93a4d232e"
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
%matplotlib inline  

#script for constructing two-dimensional graphics using Matplotlib
def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    

    for label, embeddings, words in zip(labels, embedding_clusters, word_clusters):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()

tsne_plot_similar_words(words_vocab_without_sw, embeddings_en_2d, word_clusters)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 537} id="MCIcu2hkJDX3" outputId="3ca34718-ba5f-4092-ac91-d8be3edf3b74"
tsne_model_en_2d = TSNE(perplexity=25, n_components=2, init='pca', n_iter=1500, random_state=2020) 
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
tsne_plot_similar_words(words_vocab_without_sw, embeddings_en_2d, word_clusters)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 537} id="dDcfoRvq70xx" outputId="6bdb1873-68f3-4b3e-82e3-6db0d5988c3a"
tsne_model_en_2d = TSNE(perplexity=5, n_components=2, init='pca', n_iter=1500, random_state=2020) 
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
tsne_plot_similar_words(words_vocab_without_sw, embeddings_en_2d, word_clusters)
```

<!-- #region id="kVZ97Cdf8Qrm" -->
Take a look at the above 3 graphs. We cannto say a higher or lower perplexity is good. It depends on the problem at hand. Here the plots of perplexity 5,10 are much better defined than the one with 25
<!-- #endregion -->
