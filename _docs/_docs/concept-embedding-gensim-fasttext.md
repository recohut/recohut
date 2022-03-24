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

<!-- #region id="8G1t37lcGSKK" -->
# Training Embeddings Using Gensim and FastText
> Word embeddings are an approach to representing text in NLP. In this notebook we will demonstrate how to train embeddings both CBOW and SkipGram methods using Genism and Fasttext.

- toc: true
- badges: true
- comments: true
- categories: [Concept, Embedding, Gensim, FastText]
- author: "<a href='https://notebooks.quantumstat.com/'>Quantum Stat</a>"
- image:
<!-- #endregion -->

```python id="TBw9OCYcYQ_n"
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
```

```python id="5qWptd54ZcfV"
# define training data
#Genism word2vec requires that a format of ‘list of lists’ be provided for training where every document contained in a list.
#Every list contains lists of tokens of that document.
corpus = [['dog','bites','man'], ["man", "bites" ,"dog"],["dog","eats","meat"],["man", "eats","food"]]

#Training the model
model_cbow = Word2Vec(corpus, min_count=1,sg=0) #using CBOW Architecture for trainnig
model_skipgram = Word2Vec(corpus, min_count=1,sg=1)#using skipGram Architecture for training 
```

<!-- #region id="0QjSxefPl4mh" -->
## Continuous Bag of Words (CBOW) 
In CBOW, the primary task is to build a language model that correctly predicts the center word given the context words in which the center word appears.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 486} id="nyZY8ME4lUjd" outputId="bd00e825-c11a-4b36-dbf5-80f32c659956"
#Summarize the loaded model
print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.vocab)
print(words)

#Acess vector for one word
print(model_cbow['dog'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="gMuHv52GeuoR" outputId="b498032d-6f9d-485b-a3cc-5a21300bfb06"
#Compute similarity 
print("Similarity between eats and bites:",model_cbow.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_cbow.similarity('eats', 'man'))
```

<!-- #region id="twhTZfPOezTU" -->
From the above similarity scores we can conclude that eats is more similar to bites than man.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 104} id="5Lv0V7WofmsB" outputId="00600b23-d9a6-4f14-bacd-395be85076c8"
#Most similarity
model_cbow.most_similar('meat')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="WA783nrSalgs" outputId="80d6e23f-2bed-47d7-f925-4aaa87ec5f9e"
# save model
model_cbow.save('model_cbow.bin')

# load model
new_model_cbow = Word2Vec.load('model_cbow.bin')
print(new_model_cbow)
```

<!-- #region id="deReLSI7mQyr" -->
## SkipGram
In skipgram, the task is to predict the context words from the center word.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 486} id="9QtUtsLglvY0" outputId="6d19902b-66aa-4b0f-9f12-be18f37d40d1"
#Summarize the loaded model
print(model_skipgram)

#Summarize vocabulary
words = list(model_skipgram.wv.vocab)
print(words)

#Acess vector for one word
print(model_skipgram['dog'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="8YUsblEOfFWf" outputId="14cd759c-d5fc-465f-ed20-8fd1a1949168"
#Compute similarity 
print("Similarity between eats and bites:",model_skipgram.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_skipgram.similarity('eats', 'man'))
```

<!-- #region id="gdXVDePKnBpv" -->
From the above similarity scores we can conclude that eats is more similar to bites than man.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 104} id="lpF4qtwpmuM3" outputId="f3bc68f6-3768-4a4d-e5bc-bb3dff6f654f"
#Most similarity
model_skipgram.most_similar('meat')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} id="aNDCEXRTnAnj" outputId="402f77b6-0625-4b37-e135-3650df626007"
# save model
model_skipgram.save('model_skipgram.bin')

# load model
new_model_skipgram = Word2Vec.load('model_skipgram.bin')
print(new_model_skipgram)
```

<!-- #region id="b0MiqJ_1M0mX" -->
## Training Your Embedding on Wiki Corpus

##### The corpus download page : https://dumps.wikimedia.org/enwiki/20200120/
The entire wiki corpus as of 28/04/2020 is just over 16GB in size.
We will take a part of this corpus due to computation constraints and train our word2vec and fasttext embeddings.

The file size is 294MB so it can take a while to download.

Source for code which downloads files from Google Drive: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
<!-- #endregion -->

```python id="60UO41DfGPL0" outputId="262cce44-03e5-46c8-861a-c9da76306c23"
import os
import requests

os.makedirs('data/en', exist_ok= True)
file_name = "data/en/enwiki-latest-pages-articles-multistream14.xml-p13159683p14324602.bz2"
file_id = "11804g0GcWnBIVDahjo5fQyc05nQLXGwF"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if not os.path.exists(file_name):
    download_file_from_google_drive(file_id, file_name)
else:
    print("file already exists, skipping download")

print(f"File at: {file_name}")
```

```python id="wX1kx96JLYvt"
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import time
```

```python id="rJgsEUmRPppc"
#Preparing the Training data
wiki = WikiCorpus(file_name, lemmatize=False, dictionary={})
sentences = list(wiki.get_texts())

#if you get a memory error executing the lines above
#comment the lines out and uncomment the lines below. 
#loading will be slower, but stable.
# wiki = WikiCorpus(file_name, processes=4, lemmatize=False, dictionary={})
# sentences = list(wiki.get_texts())

#if you still get a memory error, try settings processes to 1 or 2 and then run it again.
```

<!-- #region id="xsIrgt_gPQda" -->
### Hyperparameters


1.   sg - Selecting the training algorithm: 1 for skip-gram else its 0 for CBOW. Default is CBOW.
2.   min_count-  Ignores all words with total frequency lower than this.<br>
There are many more hyperparamaeters whose list can be found in the official documentation [here.](https://radimrehurek.com/gensim/models/word2vec.html)

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="idmfbr_8LvoN" outputId="f505a46e-025d-4169-f996-06c672008f81"
#CBOW
start = time.time()
word2vec_cbow = Word2Vec(sentences,min_count=10, sg=0)
end = time.time()

print("CBOW Model Training Complete.\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 471} id="mMdGn08-RkhM" outputId="efb34148-3fb4-435c-f070-8493708fc07a"
#Summarize the loaded model
print(word2vec_cbow)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_cbow.wv.vocab)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_cbow['film'])}")
print(word2vec_cbow['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_cbow.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_cbow.similarity('film', 'tiger'))
print("-"*30)
```

```python id="rXrDOrKskcHX"
# save model
from gensim.models import Word2Vec, KeyedVectors   
word2vec_cbow.wv.save_word2vec_format('word2vec_cbow.bin', binary=True)

# load model
# new_modelword2vec_cbow = Word2Vec.load('word2vec_cbow.bin')
# print(word2vec_cbow)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="dX0U0CbQOK30" outputId="b9bfcf2b-91cb-40d9-ca92-791ec346aef4"
#SkipGram
start = time.time()
word2vec_skipgram = Word2Vec(sentences,min_count=10, sg=1)
end = time.time()

print("SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 471} id="LXnY9YInSvnI" outputId="26f1dab7-27a6-4655-81c7-ac6f08fe1f9c"
#Summarize the loaded model
print(word2vec_skipgram)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_skipgram.wv.vocab)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_skipgram['film'])}")
print(word2vec_skipgram['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_skipgram.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_skipgram.similarity('film', 'tiger'))
print("-"*30)
```

```python id="o8U7bfPSVB04"
# save model
word2vec_skipgram.wv.save_word2vec_format('word2vec_sg.bin', binary=True)

# load model
# new_model_skipgram = Word2Vec.load('model_skipgram.bin')
# print(model_skipgram)
```

<!-- #region id="kExlA8kfrKml" -->
## FastText
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="JPd2VhMEk8gL" outputId="55c44bdd-d7d8-4df2-8140-cdd442bbd68c"
#CBOW
start = time.time()
fasttext_cbow = FastText(sentences, sg=0, min_count=10)
end = time.time()

print("FastText CBOW Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 471} id="FlQFl8-Zsost" outputId="6472e944-e6de-4d64-8c6f-14475ef1eac5"
#Summarize the loaded model
print(fasttext_cbow)
print("-"*30)

#Summarize vocabulary
words = list(fasttext_cbow.wv.vocab)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(fasttext_cbow['film'])}")
print(fasttext_cbow['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",fasttext_cbow.similarity('film', 'drama'))
print("Similarity between film and tiger:",fasttext_cbow.similarity('film', 'tiger'))
print("-"*30)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="UgSOxsNklAvh" outputId="f491f83c-17b8-42ad-a225-479df8419578"
#SkipGram
start = time.time()
fasttext_skipgram = FastText(sentences, sg=1, min_count=10)
end = time.time()

print("FastText SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 610} id="vFiTAP0PsQwi" outputId="a29ae2e3-5dbc-453a-f66b-ceca255a8652"
#Summarize the loaded model
print(fasttext_skipgram)
print("-"*30)

#Summarize vocabulary
words = list(fasttext_skipgram.wv.vocab)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(fasttext_skipgram['film'])}")
print(fasttext_skipgram['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",fasttext_skipgram.similarity('film', 'drama'))
print("Similarity between film and tiger:",fasttext_skipgram.similarity('film', 'tiger'))
print("-"*30)
```

<!-- #region id="oArMIJzYOmUR" -->
An interesting obeseravtion if you noticed is that CBOW trains faster than SkipGram in both cases.
We will leave it to the user to figure out why. A hint would be to refer the working of CBOW and skipgram.
<!-- #endregion -->
