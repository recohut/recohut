---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python [conda env:myenv]
    language: python
    name: conda-env-myenv-py
---

<!-- #region id="bJLQoimyVyQ8" -->
> Note: Uncomment and run the following cells if you work on Google Colab :) Don't forget to change your runtime type to GPU!
<!-- #endregion -->

```python id="rVV81xc3VyQ9"
# !git clone https://github.com/kstathou/vector_engine
```

```python id="C0lSFLw3VyRG"
# cd vector_engine
```

```python id="5sOhWL6UVyRQ"
# pip install -r requirements.txt
```

<!-- #region id="vbnscDwgVyRW" -->
### Let's begin!
<!-- #endregion -->

```python id="v7ftrzzmVyRX"
%load_ext autoreload
```

```python id="fU2i4vlCVyRc"
%autoreload 2
# Used to import data from S3.
import pandas as pd
import s3fs

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer

# Used to create and store the Faiss index.
import faiss
import numpy as np
import pickle
from pathlib import Path

# Used to do vector searches and display the results.
from vector_engine.utils import vector_search, id2details
```

<!-- #region id="Kz5YBwU5VyRi" -->
Stored and processed data in s3
<!-- #endregion -->

```python id="VEANywYAVyRi"
# Use pandas to read files from S3 buckets!
df = pd.read_csv('s3://vector-search-blog/misinformation_papers.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="HJXljSbYVyRn" outputId="1c180fbc-42a4-441a-da47-14f5cc21d826"
df.head(3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MljadlGpVyRs" outputId="8f2f5205-772f-4c6b-f445-5dd32350d45e"
print(f"Misinformation, disinformation and fake news papers: {df.id.unique().shape[0]}")
```

<!-- #region id="VyRG1wZLVyRw" -->
The [Sentence Transformers library](https://github.com/UKPLab/sentence-transformers) offers pretrained transformers that produce SOTA sentence embeddings. Checkout this [spreadsheet](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/) with all the available models.

In this tutorial, we will use the `distilbert-base-nli-stsb-mean-tokens` model which has the best performance on Semantic Textual Similarity tasks among the DistilBERT versions. Moreover, although it's slightly worse than BERT, it is quite faster thanks to having a smaller size.

I use the same model in [Orion's semantic search engine](https://www.orion-search.org/)!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PjF6CrwUVyRx" outputId="db338335-b032-45f2-db21-8e3f53640b86"
# Instantiate the sentence-level DistilBERT
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
# Check if GPU is available and use it
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
print(model.device)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["7a7e927567024c578b33b15000d5e531", "da5011a6565e45e2b5d0b37634498648", "3e9c24c8488b431fb88a0d045d85b700", "7aaf8a423e7f48bbac24d6970ed4dfe9", "f9c22552944b4e2dafe1f98170665293", "15c5bbc768db41e4bc06c9405539091c", "1e1408cddd284bc4a4b5484be0561991", "7f1cd9dbb4b742ab8d78e073fb9b0f90"]} id="Y_GS0_CWVyR1" outputId="4deb0814-1ce9-4ea8-8e50-72992e0ea303"
# Convert abstracts to vectors
embeddings = model.encode(df.abstract.to_list(), show_progress_bar=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gE7w-RJbVyR6" outputId="0451849a-88ef-4aee-be2d-e6ff173782f3"
print(f'Shape of the vectorised abstract: {embeddings[0].shape}')
```

<!-- #region id="YGV4Je1EVyR_" -->
## Vector similarity search with Faiss
[Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, even ones that do not fit in RAM. 
    
Faiss is built around the `Index` object which contains, and sometimes preprocesses, the searchable vectors. Faiss has a large collection of [indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes). You can even create [composite indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes-(composite)). Faiss handles collections of vectors of a fixed dimensionality d, typically a few 10s to 100s.

**Note**: Faiss uses only 32-bit floating point matrices. This means that you will have to change the data type of the input before building the index.

To learn more about Faiss, you can read their paper on [arXiv](https://arxiv.org/abs/1702.08734).

Here, we will the `IndexFlatL2` index:
- It's a simple index that performs a brute-force L2 distance search
- It scales linearly. It will work fine with our data but you might want to try [faster indexes](https://github.com/facebookresearch/faiss/wiki/Faster-search) if you work will millions of vectors.

To create an index with the `misinformation` abstract vectors, we will:
1. Change the data type of the abstract vectors to float32.
2. Build an index and pass it the dimension of the vectors it will operate on.
3. Pass the index to IndexIDMap, an object that enables us to provide a custom list of IDs for the indexed vectors.
4. Add the abstract vectors and their ID mapping to the index. In our case, we will map vectors to their paper IDs from MAG.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8kkUDtwHVyR_" outputId="0f668d02-ef33-4123-ab77-18f72a93ab80"
# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, df.id.values)

print(f"Number of vectors in the Faiss index: {index.ntotal}")
```

<!-- #region id="yt1z-433VySE" -->
### Searching the index
The index we built will perform a k-nearest-neighbour search. We have to provide the number of neighbours to be returned. 

Let's query the index with an abstract from our dataset and retrieve the 10 most relevant documents. **The first one must be our query!**

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 106} id="eEeJt7lYVySN" outputId="771571b3-0200-48b8-f2de-4e8cdbd5ec98"
# Paper abstract
df.iloc[5415, 1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="BSuRcH85VySQ" outputId="cec93a12-e79c-4f0a-fc15-45048a3469aa"
# Retrieve the 10 nearest neighbours
D, I = index.search(np.array([embeddings[5415]]), k=10)
print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="SiO1pa4oVySU" outputId="24c9d1ea-3951-4b4a-e436-c077f3528d7e"
# Fetch the paper titles based on their index
id2details(df, I, 'original_title')
```

```python colab={"base_uri": "https://localhost:8080/"} id="p29pEtGrWUMV" outputId="b6448614-0a6a-4673-c841-2bfb0340607e"
# Fetch the paper abstracts based on their index
id2details(df, I, 'abstract')
```

<!-- #region id="gFKvRb4QY-DL" -->
## Putting all together

So far, we've built a Faiss index using the misinformation abstract vectors we encoded with a sentence-DistilBERT model. That's helpful but in a real case scenario, we would have to work with unseen data. To query the index with an unseen query and retrieve its most relevant documents, we would have to do the following:

1. Encode the query with the same sentence-DistilBERT model we used for the rest of the abstract vectors.
2. Change its data type to float32.
3. Search the index with the encoded query.

Here, we will use the introduction of an article published on [HKS Misinformation Review](https://misinforeview.hks.harvard.edu/article/can-whatsapp-benefit-from-debunked-fact-checked-stories-to-reduce-misinformation/).

<!-- #endregion -->

```python id="iDhftkrhX99T"
user_query = """
WhatsApp was alleged to have been widely used to spread misinformation and propaganda 
during the 2018 elections in Brazil and the 2019 elections in India. Due to the 
private encrypted nature of the messages on WhatsApp, it is hard to track the dissemination 
of misinformation at scale. In this work, using public WhatsApp data from Brazil and India, we 
observe that misinformation has been largely shared on WhatsApp public groups even after they 
were already fact-checked by popular fact-checking agencies. This represents a significant portion 
of misinformation spread in both Brazil and India in the groups analyzed. We posit that such 
misinformation content could be prevented if WhatsApp had a means to flag already fact-checked 
content. To this end, we propose an architecture that could be implemented by WhatsApp to counter 
such misinformation. Our proposal respects the current end-to-end encryption architecture on WhatsApp, 
thus protecting users’ privacy while providing an approach to detect the misinformation that benefits 
from fact-checking efforts.
"""
```

```python colab={"base_uri": "https://localhost:8080/"} id="6AFhbGnWZpWN" outputId="b8a02af0-2f0d-4740-984a-e804405b3e6a"
# For convenience, I've wrapped all steps in the vector_search function.
# It takes four arguments: 
# A query, the sentence-level transformer, the Faiss index and the number of requested results
D, I = vector_search([user_query], model, index, num_results=10)
print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="tbanjBhBZtWZ" outputId="9a5d6d97-8983-4253-8095-b7d899e33ac8"
# Fetching the paper titles based on their index
id2details(df, I, 'original_title')
```

```python colab={"base_uri": "https://localhost:8080/"} id="rbxFKF-DZxg0" outputId="d78cdc03-41f6-469d-bf67-8f32001a7415"
# Define project base directory
# Change the index from 1 to 0 if you run this on Google Colab
project_dir = Path('notebooks').resolve().parents[1]
print(project_dir)

# Serialise index and store it as a pickle
with open(f"{project_dir}/models/faiss_index.pickle", "wb") as h:
    pickle.dump(faiss.serialize_index(index), h)
```
