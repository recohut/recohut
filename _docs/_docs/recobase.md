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

```python colab={"base_uri": "https://localhost:8080/"} id="77RmB1QyEvAL" executionInfo={"status": "ok", "timestamp": 1631354271762, "user_tz": -330, "elapsed": 544, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="064c3ce4-2f12-4f97-8fe0-ce26e4d65075"
import os
project_name = "recobase"; branch = "US390984"; account = "recohut"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="psh8202bV5Nu" executionInfo={"status": "ok", "timestamp": 1631362140618, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b67cd190-4745-43c0-b5fb-321114a2c4e1"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="iLdFo-nrEvAR" executionInfo={"status": "ok", "timestamp": 1631362145465, "user_tz": -330, "elapsed": 1836, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="81a2e5cd-227d-4549-b55e-b0e1da810a98"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python id="151nHKcbWg5X"
!dvc status
!dvc add
!dvc commit
!dvc push
```

```python colab={"base_uri": "https://localhost:8080/"} id="ggSVwBfbE0gb" executionInfo={"status": "ok", "timestamp": 1631352917454, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f781f803-3487-45c8-f989-bf4f2ba8e58d"
!unzip ./data/redial_dataset.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="dthM43-INDBl" executionInfo={"status": "ok", "timestamp": 1631354090373, "user_tz": -330, "elapsed": 612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3b03722a-660b-4c8a-b6f3-1059504a9790"
%%writefile /content/US390984/src/preprocess_dialogs.py
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import simplejson as json


class Dataset:
    def __init__(self):
        self.data = None
        self.text_messages_raw = []

    def read_input_json_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as json_file:
            self.data = json.load(json_file)

    def parse_dialogues(self):
        dialogs = self.data['foo']
        counter = 0
        for key, d in enumerate(dialogs):
            messages = dialogs[key]['messages']
            seeker_id = dialogs[key]['initiatorWorkerId']
            recommender_id = dialogs[key]['respondentWorkerId']
            seeker_text = ''
            gt_text = ''
            counter = counter +1
            self.text_messages_raw.append('CONVERSATION:'+ str(counter))
            for msgid, msg in enumerate(messages):

                senderId = messages[msgid]['senderWorkerId']
                if senderId == seeker_id:
                    if gt_text:
                        self.text_messages_raw.append('GT~' + gt_text)
                        gt_text = ''
                        seeker_text =  seeker_text +' '+ messages[msgid]['text']
                    else:
                        seeker_text =  seeker_text +' ' + messages[msgid]['text']

                elif senderId == recommender_id:
                    if seeker_text:
                        self.text_messages_raw.append('SKR~' + seeker_text)
                        seeker_text = ''
                        gt_text = gt_text+' '  + messages[msgid]['text']
                    else:
                        gt_text = gt_text +' ' + messages[msgid]['text']

            if gt_text:
                self.text_messages_raw.append('GT~' + gt_text)
            elif seeker_text:
                self.text_messages_raw.append('SKR~' + seeker_text)

    def write_data(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as filehandle:
            for line in self.text_messages_raw:
                filehandle.write("%s\n" % line)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.read_input_json_file('data/bronze/dialog_data/unparsed_train_data.txt')
    dataset.parse_dialogues()
    dataset.write_data('data/silver/dialog_data/training_data_parsed_con.txt')
    print('data exported')
```

```python colab={"base_uri": "https://localhost:8080/"} id="sqjrnGkpR36k" executionInfo={"status": "ok", "timestamp": 1631354278297, "user_tz": -330, "elapsed": 1757, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="570ee18c-a505-4816-fbcf-a98274475367"
!python /content/US390984/src/preprocess_dialogs.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="iLxrLhK8SELD" executionInfo={"status": "ok", "timestamp": 1631355079382, "user_tz": -330, "elapsed": 3487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a9e775a7-3174-4eac-d7d8-8f08984ca13d"
!dvc run -n preprocess_dialogs \
          -d src/preprocess_dialogs.py -d data/bronze/dialog_data/unparsed_train_data.txt \
          -o data/silver/dialog_data/parsed_dialogs \
          python src/preprocess_dialogs.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="oKq1565oUUHA" executionInfo={"status": "ok", "timestamp": 1631359975435, "user_tz": -330, "elapsed": 643, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="523c9984-50c2-484a-f2aa-7eb203579f4d"
%%writefile ./src/prepare_sentences.py
#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


def contraction_handle(filepath='data/bronze/dialog_data/contractions.txt'):
    contraction_dict = {}
    with open(filepath) as f:
        for key_line in f:
            (key, val) = key_line.split(':')
            contraction_dict[key] = val
    return contraction_dict

class PrepareSentences:
    def __init__(self):
        self.contraction_dict = contraction_handle()
        self.sentence_data = []

    @staticmethod
    def seeker_sentences_parser(line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line

    @staticmethod
    def gt_sentence_parser(line):
        try:
            if not line == '\n':
                p = re.compile("GROUND TRUTH:(.*)").search(str(line))
                temp_line = p.group(1)
                m = re.compile('<s>(.*?)</s>').search(temp_line)
                gt_line = m.group(1)
                gt_line = gt_line.lower().strip()
                # gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
            else:
                gt_line = ""
        except AttributeError as err:
                # print('exception accured while parsing ground truth.. \n')
                # print(line)
                # print(err)
                return gt_line

    @staticmethod
    def replace_movieIds_withPL(line):
        try:
            if "@" in line:
                ids = re.findall(r'@\S+', line)
                for id in ids:
                    line = line.replace(id,'movieid')
                    #id = re.sub('[^0-9@]+', 'movieid', id)
        except:
            lines.append(line)
            # print('exception occured here')
        return line
        # print('execution ends here')

    @staticmethod
    def remove_stopwords(line):
        text_tokens = word_tokenize(line)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        print(filtered_sentence)
        return filtered_sentence

    @staticmethod
    def convert_contractions(line):
        #line = "What's the best way to ensure this?"
        for word in line.split():
            if word.lower() in self.contraction_dict:
                line = line.replace(word, contraction_dict[word.lower()])
        return line

    #read and retrive dialogs from the input file
    def read_sentences(self, file_name):
        counter =0
        previous_line = ''
        counter = 0
        with open(file_name, 'r', encoding='utf-8') as input:
            for line in input:
                try:
                    #if line.__contains__('~') and line.__contains__('SKR~'):
                    if line:
                        if line.__contains__('CONVERSATION:'):
                            self.sentence_data.append(line.replace('\n',''))
                            continue
                        else:
                            previous_line = line
                            line = self.replace_movieIds_withPL(line)
                            line = line.split('~')[1].strip().lower()
                            line = self.convert_contractions(line)
                            line = re.sub('[^A-Za-z0-9]+', ' ', line)
                            line = line.replace('im','i am').strip()
                            line = self.remove_stopwords(line)
                            if len(line) < 1:
                                self.sentence_data.append('**')
                            else:
                                self.sentence_data.append(line)
                    else:
                        #print('not found')
                        #print(line)
                        #print('previous line is ...' +previous_line)
                        # print('line issue')
                        counter = counter+1
                except:
                    # print((previous_line))
                    # print(line)
                    continue

    def write_data(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as filehandle:
            for line in self.sentence_data:
                filehandle.write("%s\n" % line)



if __name__ == '__main__':
    prep = PrepareSentences()
    prep.read_sentences('data/silver/dialog_data/parsed_dialogs/training_data_parsed_con.txt')
    prep.write_data('data/gold/dialog_data/dialog_sentences/training_data_plsw.txt')
    print('Dialogs have been preprocessed successfully.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="1mLqk2K-V1oj" executionInfo={"status": "ok", "timestamp": 1631359978951, "user_tz": -330, "elapsed": 1480, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="340188d2-2d12-495b-8e6f-819b65d2e1c2"
!python ./src/prepare_sentences.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="jtQNpr_VWM_4" executionInfo={"status": "ok", "timestamp": 1631360103130, "user_tz": -330, "elapsed": 4491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e72c161a-d620-4b4d-90bf-b311593206a3"
!dvc run -n prepare_sentences \
          -d src/prepare_sentences.py -d data/silver/dialog_data/parsed_dialogs/training_data_parsed_con.txt \
          -o data/gold/dialog_data/dialog_sentences \
          python src/prepare_sentences.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="dintjCEkoySj" executionInfo={"status": "ok", "timestamp": 1631360174265, "user_tz": -330, "elapsed": 12176, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e49225c6-6cbe-461a-89b8-ed3c885a9169"
!dvc commit && dvc push
```

```python colab={"base_uri": "https://localhost:8080/"} id="bx48X58JpCcI" executionInfo={"status": "ok", "timestamp": 1631361602811, "user_tz": -330, "elapsed": 435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a64c86ab-265a-443a-dbcb-e9f74c8b43fd"
%%writefile ./src/calculate_mle_probs.py

from __future__ import division
from collections import Counter
import math as calc
import os


class CacululateMLEProbs:
    """A program which creates n-Gram (1-5) Maximum Likelihood Probabilistic Language Model with Laplace Add-1 smoothing
    and stores it in hash-able dictionary form.
    n: number of bigrams (supports up to 5)
    corpus_file: relative path to the corpus file.
    cache: saves computed values if True
"""

    def __init__(self, n=1, corpus_file=None, cache=False):
        """Constructor method which loads the corpus from file and creates ngrams based on imput parameters."""
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data')
        self.DATA_path = os.path.join(self.DATA_path, 'dialog_data', "")
        self.words = []
        self.load_corpus(corpus_file)
        self.unigram = self.bigram = self.trigram = self.quadrigram = self.pentigram = None
        self.create_unigram(cache)
        if n >= 2:
            self.create_bigram(cache)
        if n >= 3:
            self.create_trigram(cache)
        if n >= 4:
            self.create_quadrigram(cache)
        if n >= 5:
            self.create_pentigram(cache)
        return

    def load_corpus(self, file_name):
        """Method to load external file which contains raw corpus."""
        print("Loading Corpus from data file")
        if file_name is None:
            file_name = self.DATA_path+'GT_corpus_tokens.txt'
        corpus_file = open(file_name, 'r')
        corpus = corpus_file.read()
        corpus_file.close()
        print("Processing Corpus")
        self.words = corpus.split('\n')

    def create_unigram(self, cache):
        """Method to create Unigram Model for words loaded from corpus."""
        print("Creating Unigram Model")
        unigram_file = None
        if cache:
            unigram_file = open(self.DATA_path+ 'unigram.data', 'w')
        print("Calculating Count for Unigram Model")
        unigram = Counter(self.words)
        if cache:
            unigram_file.write(str(unigram))
            unigram_file.close()
        self.unigram = unigram

    def create_bigram(self, cache):
        """Method to create Bigram Model for words loaded from corpus."""
        print("Creating Bigram Model")
        words = self.words
        biwords = []
        for index, item in enumerate(words):
            if index == len(words)-1:
                break
            biwords.append(item+' '+words[index+1])
        print("Calculating Count for Bigram Model")
        bigram_file = None
        if cache:
            bigram_file = open(self.DATA_path + 'bigram.data', 'w')
        bigram = Counter(biwords)
        if cache:
            bigram_file.write(str(bigram))
            bigram_file.close()
        self.bigram = bigram

    def create_trigram(self, cache):
        """Method to create Trigram Model for words loaded from corpus."""
        print("Creating Trigram Model")
        words = self.words
        triwords = []
        for index, item in enumerate(words):
            if index == len(words)-2:
                break
            triwords.append(item+' '+words[index+1]+' '+words[index+2])
        print("Calculating Count for Trigram Model")
        if cache:
            trigram_file = open('trigram.data', 'w')
        trigram = Counter(triwords)
        if cache:
            trigram_file.write(str(trigram))
            trigram_file.close()
        self.trigram = trigram

    def create_quadrigram(self, cache):
        """Method to create Quadrigram Model for words loaded from corpus."""
        print("Creating Quadrigram Model")
        words = self.words
        quadriwords = []
        for index, item in enumerate(words):
            if index == len(words)-3:
                break
            quadriwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3])
        print("Calculating Count for Quadrigram Model")
        if cache:
            quadrigram_file = open('fourgram.data', 'w')
        quadrigram = Counter(quadriwords)
        if cache:
            quadrigram_file.write(str(quadrigram))
            quadrigram_file.close()
        self.quadrigram = quadrigram

    def create_pentigram(self, cache):
        """Method to create Pentigram Model for words loaded from corpus."""
        print("Creating pentigram Model")
        words = self.words
        pentiwords = []
        for index, item in enumerate(words):
            if index == len(words)-4:
                break
            pentiwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3]+' '+words[index+4])
        print("Calculating Count for pentigram Model")
        if cache:
            pentigram_file = open('pentagram.data', 'w')
        pentigram = Counter(pentiwords)
        if cache:
            pentigram_file.write(str(pentigram))
            pentigram_file.close()
        self.pentigram = pentigram

    def probability(self, word, words="", n=1):
        """Method to calculate the Maximum Likelihood Probability of n-Grams on the basis of various parameters."""
        if n == 1:
            return calc.log((self.unigram[word]+1)/(len(self.words)+len(self.unigram)))
        elif n == 2:
            return calc.log((self.bigram[words]+1)/(self.unigram[word]+len(self.unigram)))
        elif n == 3:
            return calc.log((self.trigram[words]+1)/(self.bigram[word]+len(self.unigram)))
        elif n == 4:
            return calc.log((self.quadrigram[words]+1)/(self.trigram[word]+len(self.unigram)))
        elif n == 5:
            return calc.log((self.pentigram[words]+1)/(self.quadrigram[word]+len(self.unigram)))

    def sentence_probability(self, sentence, n=1):
        """Method to calculate cumulative n-gram Maximum Likelihood Probability of a phrase or sentence."""
        words = sentence.lower().split()
        P = 0
        if n == 1:
            for index, item in enumerate(words):
                P += self.probability(item)
        if n == 2:
            for index, item in enumerate(words):
                if index >= len(words) - 1:
                    break
                P += self.probability(item, item+' '+words[index+1], 2)
        if n == 3:
            for index, item in enumerate(words):
                if index >= len(words) - 2:
                    break
                P += self.probability(item+' '+words[index+1], item+' '+words[index+1]+' '+words[index+2], 3)
        if n == 4:
            for index, item in enumerate(words):
                if index >= len(words) - 3:
                    break
                P += self.probability(item+' '+words[index+1]+' '+words[index+2], item+' '+words[index+1]+' ' +
                                      words[index+2]+' '+words[index+3], 4)
        if n == 5:
            for index, item in enumerate(words):
                if index >= len(words) - 4:
                    break
                P += self.probability(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3], item+' ' +
                                      words[index+1]+' '+words[index+2]+' '+words[index+3]+' '+words[index+4], 5)

        return P


if __name__ == '__main__':
        ng = CacululateMLEProbs(n=2, corpus_file=None, cache=True)
        sentence = 'what kind of movies do you like'
        print(ng.sentence_probability(sentence, n=2))
```

```python colab={"base_uri": "https://localhost:8080/"} id="2LkJuhQfupAs" executionInfo={"status": "ok", "timestamp": 1631361616819, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c5042fb4-9b41-4df7-febf-c15cb57a0e00"
!python ./src/calculate_mle_probs.py
```
