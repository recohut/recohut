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

# FinBERT QA

<!-- #region colab_type="text" id="2NoVkHwBZ0al" -->
### Data preprocessing
1. Loads and cleans the raw data
2. Prepares the data for the retriever
3. Pre-processes and tokenizes the raw cleaned data
4. Creates vocabulary from the corpus
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 18589, "status": "ok", "timestamp": 1586846488566, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="HZbtelfeZ1Eh" outputId="d7ec962b-a365-4208-e99c-c30bbf53dc19"
!git clone https://github.com/sparsh9012/FinBERT-QA
%cd FinBERT-QA
from src.utils import *
from src.process_data import *
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 1131, "status": "ok", "timestamp": 1586846785687, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="auStHyZJbjZW" outputId="9fbed57a-7a7b-45f9-8a3e-3e3dd62c5578"
# document dataset
collection = load_answers_to_df("data/raw/FiQA_train_doc_final.tsv")
collection.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 992, "status": "ok", "timestamp": 1586846825806, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="rnyNrkDscLfb" outputId="757e7e9b-9f13-4f1c-83dd-cb7697496374"
# question dataset
queries = load_questions_to_df("data/raw/FiQA_train_question_final.tsv")
queries.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 2772, "status": "ok", "timestamp": 1586847088813, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ggn06XfLcRNk" outputId="332bed08-4a1f-4c79-e70e-b32fa5a6b77e"
# question to document mapping
qid_docid = load_qid_docid_to_df("data/raw/FiQA_train_question_doc_final.tsv")
qid_rel = label_to_dict(qid_docid)
qid_docid.head()
```

```python colab={} colab_type="code" id="64KODs6IayrR"
# Cleaning data
empty_docs, empty_id = get_empty_docs(collection)
# Remove empty answers from collection of answers
collection_cleaned = collection.drop(empty_id)
# Remove empty answers from qa pairs
qid_docid = qid_docid[~qid_docid['docid'].isin(empty_docs)]
```

```python colab={} colab_type="code" id="bCTFQDuWaypk"
# Write collection df to file
save_tsv("retriever/collection_cleaned.tsv", collection_cleaned)

# Convert collection df to JSON file for document indexer
collection_to_json("retriever/collection_json/docs.json", "retriever/collection_cleaned.tsv")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 1318, "status": "ok", "timestamp": 1586847258239, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="yAVNCGLpayoo" outputId="6a1a8373-5995-4606-d529-14ee9604f9eb"
# process questions
processed_questions = process_questions(queries)
processed_questions.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 7358, "status": "ok", "timestamp": 1586847264434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="T-XeINoWdyh2" outputId="7f97a0eb-9698-41a9-a6c0-6a2369056c14"
# process answers
processed_answers = process_answers(collection_cleaned)
processed_answers.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" executionInfo={"elapsed": 963, "status": "ok", "timestamp": 1586847333713, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="79QIDrv0eIK-" outputId="068df270-bfeb-4821-a079-51682157aa15"
# statistics
avg_ans_count = processed_answers['ans_len'].mean()
avg_q_count = processed_questions['q_len'].mean()

print("Average answer length: {}".format(round(avg_ans_count)))
print("Average question length: {}".format(round(avg_q_count)))

print("Total answers: {}".format(len(processed_answers)))
print("Number of answers with length greater than 512: {}".format(len(processed_answers[processed_answers['ans_len'] > 512])))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" executionInfo={"elapsed": 23088, "status": "ok", "timestamp": 1586847419161, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="RXHaXycWeRHI" outputId="965d8438-66e0-475b-c8e8-7f85a82b64c4"
# Create vocabulary
word2index, word2count = create_vocab(processed_answers, processed_questions)
print("Vocab size: {}".format(len(word2index)))
print("Top {} common words: {}".format(35, Counter(word2count).most_common(35)))

qid_to_text, docid_to_text = id_to_text(collection, queries)
qid_to_tokenized_text, docid_to_tokenized_text = id_to_tokenized_text(processed_answers, processed_questions)

# Save objects to pickle
save_pickle("data/qa_lstm_tokenizer/word2index.pickle", word2index)
save_pickle("data/qa_lstm_tokenizer/word2count.pickle", word2count)

# id map to raw text
save_pickle("data/id_to_text/qid_to_text.pickle", qid_to_text)
save_pickle("data/id_to_text/docid_to_text.pickle", docid_to_text)

# id map to tokenized text
save_pickle("data/qa_lstm_tokenizer/qid_to_tokenized_text.pickle", qid_to_tokenized_text)
save_pickle("data/qa_lstm_tokenizer/docid_to_tokenized_text.pickle", docid_to_tokenized_text)
```

<!-- #region colab_type="text" id="pMqQqrUW9OnV" -->
## **FinBERT-QA**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 170} colab_type="code" executionInfo={"elapsed": 24968, "status": "ok", "timestamp": 1586845752938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="gHEgA6gHInfl" outputId="682b9394-9298-4a91-b878-4e3db77d6110"
!git clone https://github.com/sparsh9012/FinBERT-QA
%cd FinBERT-QA
from src.utils import *
```

```python colab={"base_uri": "https://localhost:8080/", "height": 683} colab_type="code" executionInfo={"elapsed": 34147, "status": "ok", "timestamp": 1586845762124, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="HDtGwlyRX4Yb" outputId="e8758e47-17b0-4aea-d344-def91bd99fe6"
!pip install transformers
```

```python colab={} colab_type="code" id="6SS1U17EX2s8"
import torch
import pickle
import csv
import regex as re
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 1472, "status": "ok", "timestamp": 1586845847964, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mpGn-tbIX_0r" outputId="02f89f5b-8d9a-4eec-99f3-792c9f34a43d"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1234)
```

```python colab={} colab_type="code" id="Kvte60CEY1Zj"
# Collection of answers - docid, text
collection = pd.read_csv("retriever/collection_cleaned.tsv", sep="\t", header=None)
collection = collection.rename(columns={0: 'docid', 1: 'doc'})
# Questions - qid, text
query_df = pd.read_csv("data/raw/FiQA_train_question_final.tsv", sep="\t")
queries = query_df[['qid', 'question']]

# List of empty docs
empty_docs = load_pickle('data/id_to_text/empty_docs.pickle')

# docid to text mapping
docid_to_text = load_pickle('data/id_to_text/docid_to_text.pickle')
# qid to text mapping
qid_to_text = load_pickle('data/id_to_text/qid_to_text.pickle')
```

```python colab={} colab_type="code" id="Jsy6l-KNYgtQ"
# Load and process dataset
dataset = pd.read_csv("data/raw/FiQA_train_question_doc_final.tsv", sep="\t")
dataset = dataset[["qid", "docid"]]
dataset = dataset[~dataset['docid'].isin(empty_docs)]
dataset['question'] = dataset['qid'].apply(lambda x: qid_to_text[x])
dataset['answer'] = dataset['docid'].apply(lambda x: docid_to_text[x])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 1378, "status": "ok", "timestamp": 1586845952167, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="scR-zXm3YyVw" outputId="34edc5de-c439-4e91-ede5-40c82d1896f7"
dataset.head()
```

```python colab={} colab_type="code" id="YF5ODrFQY5Ta"
def add_ques_token(string):
    question = string + " [SEP] "
    return question
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} colab_type="code" executionInfo={"elapsed": 1417, "status": "ok", "timestamp": 1586846057873, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="qLznv3-2ZSNj" outputId="573a961d-1bce-4284-bf52-b7473239bc91"
# Concatenate question and answer with a separator
dataset['question'] = dataset['question'].apply(add_ques_token)
dataset['seq'] = dataset['question'] + dataset['answer']
dataset = dataset[['seq']]

dataset.at[17081, "seq"]
```

```python colab={} colab_type="code" id="NLAGT0pHZT9r"
# Write data to file
dataset.to_csv('data/data.txt',index=False,header=False, sep="\t", quoting=csv.QUOTE_NONE)
```

```python colab={} colab_type="code" id="6xLHf_iOZaE8"

```
