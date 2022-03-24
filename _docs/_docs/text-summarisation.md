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

<!-- #region id="T9wVKg4w8ZbO" -->
### Connecting to Git
<!-- #endregion -->

```python id="mFH4MkyR7BE_" colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"status": "ok", "timestamp": 1601954844175, "user_tz": -330, "elapsed": 2022, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8de98ed1-af88-4a97-bf2f-c4f46fdad996"
!mkdir text_summarisation
%cd text_summarisation
!git config --global user.email "<email>"
!git config --global user.name  "sparsh-ai"
!git init
!git remote add origin https://d6c793d71f84791b7383a5e8107be037ba873558:x-oauth-basic@github.com/sparsh-ai/text_summarisation.git
```

<!-- #region id="U2JlUidO80uk" -->
### TextRank (Extractive)
<!-- #endregion -->

<!-- #region id="5T4p8Q_CwCba" -->
<img src='https://cdn.analyticsvidhya.com/wp-content/uploads/2018/10/block_3.png'>
<!-- #endregion -->

```python id="dKokpVgd_OWl" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1601955527285, "user_tz": -330, "elapsed": 1328, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b26de198-7e2d-4dd6-f1ec-bdfeba40681d"
!mkdir extractive_textRank
%cd extractive_textRank
```

```python id="QF7BU1MO-dd7" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1601955542028, "user_tz": -330, "elapsed": 2083, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="af09bebd-84ff-431e-c8b5-8d58de998ee9"
%%writefile fb.txt
For years, Facebook gave some of the world's largest technology companies more intrusive access to users' personal data than it has disclosed, effectively exempting those business partners from its usual privacy rules, according to internal records and interviews. The special arrangements are detailed in hundreds of pages of Facebook documents obtained by The New York Times. The records, generated in 2017 by the company's internal system for tracking partnerships, provide the most complete picture yet of the social network's data-sharing practices. They also underscore how personal data has become the most prized commodity of the digital age, traded on a vast scale by some of the most powerful companies in Silicon Valley and beyond. The exchange was intended to benefit everyone. Pushing for explosive growth, Facebook got more users, lifting its advertising revenue. Partner companies acquired features to make their products more attractive. Facebook users connected with friends across different devices and websites. But Facebook also assumed extraordinary power over the personal information of its 2 billion users - control it has wielded with little transparency or outside oversight.Facebook allowed Microsoft's Bing search engine to see the names of virtually all Facebook user's friends without consent, the records show, and gave Netflix and Spotify the ability to read Facebook users' private messages.
```

```python id="NXInkwgm8wVT" colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"status": "ok", "timestamp": 1601955735253, "user_tz": -330, "elapsed": 1477, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0e28370b-5097-495b-d931-5656d7daa2cb"
#!/usr/bin/env python
# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pprint
import networkx as nx

nltk.download("stopwords")
stop_words = stopwords.words('english')
 
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize text
    # print("Summarize Text: \n", ". ".join(summarize_text))
    return summarize_text
```

```python id="OY3OCdxVACaA" colab={"base_uri": "https://localhost:8080/", "height": 210} executionInfo={"status": "ok", "timestamp": 1601955855882, "user_tz": -330, "elapsed": 1090, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d67f6d4-cfdf-4966-942a-175de75c3523"
# let's begin
summarize_text = generate_summary( "fb.txt", 2)
print(summarize_text[0])
```

```python id="o6O45v87_Ku1"
'''
For years, Facebook gave some of the world's largest technology companies more 
intrusive access to users' personal data than it has disclosed, effectively 
exempting those business partners from its usual privacy rules, according to 
internal records and interviews. The special arrangements are detailed in 
hundreds of pages of Facebook documents obtained by The New York Times. 
The records, generated in 2017 by the company's internal system for tracking 
partnerships, provide the most complete picture yet of the social network's 
data-sharing practices. They also underscore how personal data has become the 
most prized commodity of the digital age, traded on a vast scale by some of the 
most powerful companies in Silicon Valley and beyond. The exchange was intended 
to benefit everyone. Pushing for explosive growth, Facebook got more users, 
lifting its advertising revenue. Partner companies acquired features to make 
their products more attractive. Facebook users connected with friends across 
different devices and websites. But Facebook also assumed extraordinary power 
over the personal information of its 2 billion users - control it has wielded 
with little transparency or outside oversight.Facebook allowed Microsoft's Bing 
search engine to see the names of virtually all Facebook user's friends without 
consent, the records show, and gave Netflix and Spotify the ability to read 
Facebook users' private messages.
'''
```

<!-- #region id="DtwT1KxxTi-B" -->
### Summarisation with Sumy Library
<!-- #endregion -->

```python id="cw5F8I6dTquY" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1601960884019, "user_tz": -330, "elapsed": 1248, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3065cda7-4b9e-45b1-c532-68fdf0c71037"
!mkdir -p /content/text_summarisation/sumy
%cd /content/text_summarisation/sumy
```

```python id="NoxqjvQM_DmA"
!pip install sumy
```

```python id="exWcdCn8T0Kx" colab={"base_uri": "https://localhost:8080/", "height": 245} executionInfo={"status": "ok", "timestamp": 1601961188035, "user_tz": -330, "elapsed": 2539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f804372-8a27-4c2b-de87-a3e9e9a831e8"
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import nltk; nltk.download('punkt')


LANGUAGE = "english"
SENTENCES_COUNT = 10

url = "https://en.wikipedia.org/wiki/Automatic_summarization"
parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
# or for plain text files
# parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
stemmer = Stemmer(LANGUAGE)

summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

summary = []
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
```

```python id="0QtXrHg2UdYO" colab={"base_uri": "https://localhost:8080/", "height": 210} executionInfo={"status": "ok", "timestamp": 1601961294804, "user_tz": -330, "elapsed": 6224, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0a49183f-3da8-4853-ca99-9a200fb4241e"
!sumy lex-rank --length=10 --url=http://en.wikipedia.org/wiki/Automatic_summarization
```

```python id="2-elsXAXVWKE" colab={"base_uri": "https://localhost:8080/", "height": 225} executionInfo={"status": "ok", "timestamp": 1601961312015, "user_tz": -330, "elapsed": 4332, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0932066a-fcff-4a26-b031-beb5731406d3"
!sumy luhn --language=czech --url=http://www.zdrojak.cz/clanky/automaticke-zabezpeceni/
```

<!-- #region id="X3iD7K8WvbYt" -->
### Bert Extractive Summarizer
<!-- #endregion -->

```python id="awpV_QYvVaz_"
%%shell
pip install bert-extractive-summarizer
pip install spacy==2.1.3
pip install transformers==2.2.2
pip install neuralcoref

python -m spacy download en_core_web_md
```

```python id="v3jtH2UyvfjA" colab={"base_uri": "https://localhost:8080/", "height": 182, "referenced_widgets": ["7b15d53551514c09b555b07353790c37", "d496297567224f6299892c6a60069838", "4ca83571a72e4b5eb9d59d01bb10b10d", "09f4c2c9764a4282a50d1f452867b8ed", "33b926d340b845f3ab9e63948b5d6897", "fadb9a9ea4944d43991ce43c1a0bcbf0", "eac7a32e3ecb48e993a931c3eedd360c", "c073971e59d244579f6d1e07d7298c60", "206ccdf3b2a04187a0d74530765f894c", "2d8960cb5285498bb9b719d17d3cb08e", "483059f93c294721a65d2188453f811f", "cefce4d8910a49e3b7a5ddc1f70eb9ae", "dd27581985ce428c8a7bc444119a81ab", "4c0c2c1edb06433c9be71e00bc37eca7", "11160706722542a7a4121fc0b6382d3b", "0bb39579bde34d2382bc5e84425129e8", "e2eac12ce52b45aca8740b5c68f488f5", "d9a0ecbb8c444af1a10dd5f0a44fa628", "7a2f85b8cbe74041b5f81370534340cc", "e8f3330f497744d7ad1d1670ff127e4a", "b6a0072e605e479c9e514770a1a443db", "711095b6d8b444408c1eed650754bf83", "c2b97403c14a4f119c8ced3020a6babf", "fb82298784da44a198456df7e36981d2"]} executionInfo={"status": "ok", "timestamp": 1601969056098, "user_tz": -330, "elapsed": 48641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79cc7d2f-8e8f-4540-8207-b4d1f8d7fb41"
from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT. Something else you want to summarize with BERT.'

model = Summarizer()

model(body)
```

```python id="3gZw1aOXvlm1"
result = model(body, ratio=0.2)  # Specified with ratio
result = model(body, num_sentences=3)  # Will return 3 sentences 
```

```python id="x0kHV_4GzNqz" colab={"base_uri": "https://localhost:8080/", "height": 69} executionInfo={"status": "ok", "timestamp": 1601969374485, "user_tz": -330, "elapsed": 25580, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="67c75cad-878d-4a82-9870-9ea774f2cde2"
from summarizer import Summarizer

body = '''
The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
Real estate firm Tishman Speyer had owned the other 10%.
The buyer is RFR Holding, a New York real estate company.
Officials with Tishman and RFR did not immediately respond to a request for comments.
It's unclear when the deal will close.
The building sold fairly quickly after being publicly placed on the market only two months ago.
The sale was handled by CBRE Group.
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
Meantime, rents in the building itself are not rising nearly that fast.
While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
Still the building is among the best known in the city, even to people who have never been to New York.
It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
Blackstone Group (BX) bought it for $1.3 billion 2015.
The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.
'''

model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
full
```

```python id="m85bgroRzV3_"

```
