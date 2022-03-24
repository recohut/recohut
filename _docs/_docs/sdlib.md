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

<!-- #region id="X7ppGxG8tVvE" -->
# Shilling simulated attacks and detection methods
<!-- #endregion -->

<!-- #region id="SYGvUUUTtjph" -->
## Setup
<!-- #endregion -->

```python id="N2ItYIT7-FDW"
!mkdir -p results
```

<!-- #region id="qTLZ7TT5vMPN" -->
### Imports
<!-- #endregion -->

```python id="9QQSmT2-vNZk"
from collections import defaultdict
import numpy as np
import random
import os
import os.path
from os.path import abspath
from os import makedirs,remove
from re import compile,findall,split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from numpy.linalg import norm
from scipy.stats.stats import pearsonr
from math import sqrt,exp

import sys
from re import split
from multiprocessing import Process,Manager
from time import strftime,localtime,time
import re

from os.path import abspath
from time import strftime,localtime,time
from sklearn.metrics import classification_report
from re import split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
import time as tm

from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict
from math import log,exp
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from random import choice
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict
from math import log,exp
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import metrics
import scipy
from scipy.sparse import csr_matrix

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import math
from sklearn.naive_bayes import GaussianNB
```

<!-- #region id="UOwMLh6_9ok0" -->
## Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3a41XYOT-DZg" executionInfo={"status": "ok", "timestamp": 1634217832326, "user_tz": -330, "elapsed": 1409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d09d3bf4-6143-40f1-cf37-2812480c4713"
!mkdir -p dataset/amazon
!cd dataset/amazon && wget -q --show-progress https://github.com/Coder-Yu/SDLib/raw/master/dataset/amazon/profiles.txt
!cd dataset/amazon && wget -q --show-progress https://github.com/Coder-Yu/SDLib/raw/master/dataset/amazon/labels.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="JV8I8iqLy8-W" executionInfo={"status": "ok", "timestamp": 1634217826906, "user_tz": -330, "elapsed": 1267, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4c40d1db-265e-46fe-f788-f84edf11ccb6"
!mkdir -p dataset/averageattack
!cd dataset/averageattack && wget -q --show-progress https://github.com/Coder-Yu/SDLib/raw/master/dataset/averageattack/ratings.txt
!cd dataset/averageattack && wget -q --show-progress https://github.com/Coder-Yu/SDLib/raw/master/dataset/averageattack/labels.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="mPu_agBp-R9D" executionInfo={"status": "ok", "timestamp": 1634217866087, "user_tz": -330, "elapsed": 1220, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="457130bc-9e7b-458d-af44-bcf7c5907817"
!mkdir -p dataset/filmtrust
!cd dataset/filmtrust && wget -q --show-progress https://github.com/Coder-Yu/SDLib/raw/master/dataset/filmtrust/ratings.txt
!cd dataset/filmtrust && wget -q --show-progress https://github.com/Coder-Yu/SDLib/raw/master/dataset/filmtrust/trust.txt
```

<!-- #region id="OQCzsVZRt2ZB" -->
## Config
<!-- #endregion -->

<!-- #region id="5WkBfSket3c-" -->
### Configure the Detection Method

<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th width="12%" scope="col"> Entry</th>
    <th width="16%" class="conf" scope="col">Example</th>
    <th width="72%" class="conf" scope="col">Description</th>
  </tr>
  <tr>
    <td>ratings</td>
    <td>dataset/averageattack/ratings.txt</td>
    <td>Set the path to the dirty recommendation dataset. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
 <tr>
    <td>label</td>
    <td>dataset/averageattack/labels.txt</td>
    <td>Set the path to labels (for users). Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
  <tr>
    <td scope="row">ratings.setup</td>
    <td>-columns 0 1 2</td>
    <td>-columns: (user, item, rating) columns of rating data are used;
      -header: to skip the first head line when reading data<br>
    </td>
  </tr>

  <tr>
    <td scope="row">MethodName</td>
    <td>DegreeSAD/PCASelect/etc.</td>
    <td>The name of the detection method<br>
    </td>
  </tr>
  <tr>
    <td scope="row">evaluation.setup</td>
    <td>-testSet dataset/testset.txt</td>
    <td>Main option: -testSet, -ap, -cv <br>
      -testSet path/to/test/file   (need to specify the test set manually)<br>
      -ap ratio   (ap means that the user set (including items and ratings) are automatically partitioned into training set and test set, the number is the ratio of test set. e.g. -ap 0.2)<br>
      -cv k   (-cv means cross validation, k is the number of the fold. e.g. -cv 5)<br>
     </td>
  </tr>

  <tr>
    <td scope="row">output.setup</td>
    <td>on -dir Results/</td>
    <td>Main option: whether to output recommendation results<br>
      -dir path: the directory path of output results.
       </td>
  </tr>
  </table>
</div>
<!-- #endregion -->

<!-- #region id="pC7aeK-audZW" -->
### Configure the Shilling Model

<div>
 <table class="table table-hover table-bordered">

  <tr>
    <th width="12%" scope="col"> Entry</th>
    <th width="16%" class="conf" scope="col">Example</th>
    <th width="72%" class="conf" scope="col">Description</th>
  </tr>
   <tr>
    <td>ratings</td>
    <td>dataset/averageattack/ratings.txt</td>
    <td>Set the path to the recommendation dataset. Format: each row separated by empty, tab or comma symbol. </td>
  </tr>
  <tr>
    <td scope="row">ratings.setup</td>
    <td>-columns 0 1 2</td>
    <td>-columns: (user, item, rating) columns of rating data are used;
      -header: to skip the first head line when reading data<br>
    </td>
  </tr>
  <tr>
    <td>attackSize</td>
    <td>0.01</td>
    <td>The ratio of the injected spammers to genuine users</td>
  </tr>
 <tr>
    <td>fillerSize</td>
    <td>0.01</td>
    <td>The ratio of the filler items to all items </td>
  </tr>
 <tr>
    <td>selectedSize</td>
    <td>0.001</td>
    <td>The ratio of the selected items to all items </td>
 </tr>
  <tr>
    <td>linkSize</td>
    <td>0.01</td>
    <td>The ratio of the users maliciously linked by a spammer to all user </td>
 </tr>
   <tr>
    <td>targetCount</td>
    <td>20</td>
    <td>The count of the targeted items </td>
  </tr>

   <tr>
    <td>targetScore</td>
    <td>5.0</td>
    <td>The score given to the target items</td>
  </tr>
  <tr>
    <td>threshold</td>
    <td>3.0</td>
    <td>Item has an average score lower than threshold may be chosen as one of the target items</td>
  </tr>

  <tr>
    <td>minCount</td>
    <td>3</td>
    <td>Item has a ratings count larger than minCount may be chosen as one of the target items</td>
  </tr>

  <tr>
    <td>maxCount</td>
    <td>50</td>
    <td>Item has a rating count smaller that maxCount may be chosen as one of the target items</td>
  </tr>

  <tr>
    <td scope="row">outputDir</td>
    <td>data/</td>
    <td> User profiles and labels will be output here     </td>
  </tr>
  </table>
</div>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cXC3PBoey0Vy" executionInfo={"status": "ok", "timestamp": 1634217508419, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ae7604b-ae74-431f-bd16-3f4198d59928"
%%writefile BayesDetector.conf
ratings=dataset/amazon/profiles.txt
ratings.setup=-columns 0 1 2
label=dataset/amazon/labels.txt
methodName=BayesDetector
evaluation.setup=-cv 5
item.ranking=off -topN 50
num.max.iter=100
learnRate=-init 0.03 -max 0.1
reg.lambda=-u 0.3 -i 0.3
BayesDetector=-k 10 -negCount 256 -gamma 1 -filter 4 -delta 0.01
output.setup=on -dir results/
```

```python colab={"base_uri": "https://localhost:8080/"} id="pOBzPax48pyk" executionInfo={"status": "ok", "timestamp": 1634217536217, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="77f23523-2602-4ec5-a7ae-5ac6c5f98b75"
%%writefile CoDetector.conf
ratings=dataset/amazon/profiles.txt
ratings.setup=-columns 0 1 2
label=dataset/amazon/labels.txt
methodName=CoDetector
evaluation.setup=-ap 0.3
item.ranking=on -topN 50
num.max.iter=200
learnRate=-init 0.01 -max 0.01
reg.lambda=-u 0.8 -i 0.4
CoDetector=-k 10 -negCount 256 -gamma 1 -filter 4
output.setup=on -dir results/amazon/
```

```python colab={"base_uri": "https://localhost:8080/"} id="hASGWg768p14" executionInfo={"status": "ok", "timestamp": 1634215085313, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5829e02-e847-456a-ee01-8131ad429d48"
%%writefile DegreeSAD.conf
ratings=dataset/amazon/profiles.txt
ratings.setup=-columns 0 1 2
label=dataset/amazon/labels.txt
methodName=DegreeSAD
evaluation.setup=-cv 5
output.setup=on -dir results/
```

```python colab={"base_uri": "https://localhost:8080/"} id="xlbnbUFT8p6j" executionInfo={"status": "ok", "timestamp": 1634217562478, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c3891e2b-c649-4937-eb61-ba4edd3445d3"
%%writefile FAP.conf
ratings=dataset/averageattack/ratings.txt
ratings.setup=-columns 0 1 2
label=dataset/averageattack/labels.txt
methodName=FAP
evaluation.setup=-ap 0.000001
seedUser=350
topKSpam=1557
output.setup=on -dir results/
```

```python colab={"base_uri": "https://localhost:8080/"} id="Dr17WXks8p9A" executionInfo={"status": "ok", "timestamp": 1634217585257, "user_tz": -330, "elapsed": 465, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed7252eb-6a8a-4d5e-85a8-14a776f38d45"
%%writefile PCASelectUsers.conf
ratings=dataset/averageattack/ratings.txt
ratings.setup=-columns 0 1 2
label=dataset/averageattack/labels.txt
methodName=PCASelectUsers
evaluation.setup=-ap 0.00001
kVals=3
attackSize=0.1
output.setup=on -dir results/
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZCR6LD748qO_" executionInfo={"status": "ok", "timestamp": 1634217607813, "user_tz": -330, "elapsed": 427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a72e2b67-2392-43c6-a9aa-d3be593e4373"
%%writefile SemiSAD.conf
ratings=dataset/averageattack/ratings.txt
ratings.setup=-columns 0 1 2
label=dataset/averageattack/labels.txt
methodName=SemiSAD
evaluation.setup=-ap 0.2
Lambda=0.5
topK=28
output.setup=on -dir results/
```

<!-- #region id="Dcb8Xwds-hbU" -->
## Baseclass
<!-- #endregion -->

```python id="ZOJBdRaX0s54"
class SDetection(object):

    def __init__(self,conf,trainingSet=None,testSet=None,labels=None,fold='[1]'):
        self.config = conf
        self.isSave = False
        self.isLoad = False
        self.foldInfo = fold
        self.labels = labels
        self.dao = RatingDAO(self.config, trainingSet, testSet)
        self.training = []
        self.trainingLabels = []
        self.test = []
        self.testLabels = []

    def readConfiguration(self):
        self.algorName = self.config['methodName']
        self.output = LineConfig(self.config['output.setup'])


    def printAlgorConfig(self):
        "show algorithm's configuration"
        print('Algorithm:',self.config['methodName'])
        print('Ratings dataSet:',abspath(self.config['ratings']))
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:',abspath(LineConfig(self.config['evaluation.setup']).getOption('-testSet')))
        #print 'Count of the users in training set: ',len()
        print('Training set size: (user count: %d, item count %d, record count: %d)' %(self.dao.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' %(self.dao.testSize()))
        print('='*80)

    def initModel(self):
        pass

    def buildModel(self):
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self):
        pass

    def execute(self):
        self.readConfiguration()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        # load model from disk or build model
        if self.isLoad:
            print('Loading model %s...' % (self.foldInfo))
            self.loadModel()
        else:
            print('Initializing model %s...' % (self.foldInfo))
            self.initModel()
            print('Building Model %s...' % (self.foldInfo))
            self.buildModel()

        # preict the ratings or item ranking
        print('Predicting %s...' % (self.foldInfo))
        prediction = self.predict()
        report = classification_report(self.testLabels, prediction, digits=4)
        currentTime = currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        FileIO.writeFile(self.output['-dir'],self.algorName+'@'+currentTime+self.foldInfo,report)
        # save model
        if self.isSave:
            print('Saving model %s...' % (self.foldInfo))
            self.saveModel()
        print(report)
        return report
```

```python id="lydTjwcQ-kwU"
class SSDetection(SDetection):

    def __init__(self,conf,trainingSet=None,testSet=None,labels=None,relation=list(),fold='[1]'):
        super(SSDetection, self).__init__(conf,trainingSet,testSet,labels,fold)
        self.sao = SocialDAO(self.config, relation)  # social relations access control
```

<!-- #region id="fUmrP4xLvaqW" -->
## Utils
<!-- #endregion -->

```python id="no_eNBw8vbsg"
class Config(object):
    def __init__(self,fileName):
        self.config = {}
        self.readConfiguration(fileName)

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.config[item]

    def getOptions(self,item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.config[item]

    def contains(self,key):
        return key in self.config

    def readConfiguration(self,fileName):
        if not os.path.exists(abspath(fileName)):
            print('config file is not found!')
            raise IOError
        with open(fileName) as f:
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        key,value=line.strip().split('=')
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d'%(ind))


class LineConfig(object):
    def __init__(self,content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1


    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def getOption(self,key):
        if not self.contains(key):
            print('parameter '+key+' is invalid!')
            exit(-1)
        return self.options[key]

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options
```

```python id="Ec4gpP9Kvsvv"
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if type(content)=='str':
            with open(dir + file, op) as f:
                f.write(content)
        else:
            with open(dir+file,op) as f:
                f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        trainingData = defaultdict(dict)
        testData = defaultdict(dict)
        ratingConfig = LineConfig(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()

        for lineNo, line in enumerate(ratings):
            items = split(' |,|\t', line.strip())
            if not bTest and len(order) < 3:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                userId = items[int(order[0])]
                itemId = items[int(order[1])]
                if bTest and len(order)<3:
                    rating = 1 #default value
                else:
                    rating  = items[int(order[2])]

            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if not bTest:
                trainingData[userId][itemId]=float(rating)
            else:
                testData[userId][itemId] = float(rating)
        if not bTest:
            return trainingData
        else:
            return testData

    @staticmethod
    def loadRelationship(conf, filePath):
        socialConfig = LineConfig(conf['social.setup'])
        relation = []
        print('loading social data...')
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if socialConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = socialConfig['-columns'].strip().split()
        if len(order) <= 2:
            print('The social file is not in a correct format.')
        for lineNo, line in enumerate(relations):
            items = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The social file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            userId1 = items[int(order[0])]
            userId2 = items[int(order[1])]
            if len(order) < 3:
                weight = 1
            else:
                weight = float(items[int(order[2])])
            relation.append([userId1, userId2, weight])
        return relation


    @staticmethod
    def loadLabels(filePath):
        labels = {}
        with open(filePath) as f:
            for line in f:
                items = split(' |,|\t', line.strip())
                labels[items[0]] = items[1]
        return labels
```

```python id="5j_Er3_ovqAS"
class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def dataSplit(data,test_ratio = 0.3,output=False,path='./',order=1):
        if test_ratio>=1 or test_ratio <=0:
            test_ratio = 0.3
        testSet = {}
        trainingSet = {}
        for user in data:
            if random.random() < test_ratio:
                testSet[user] = data[user].copy()
            else:
                trainingSet[user] = data[user].copy()

        if output:
            FileIO.writeFile(path,'testSet['+str(order)+']',testSet)
            FileIO.writeFile(path, 'trainingSet[' + str(order) + ']', trainingSet)
        return trainingSet,testSet

    @staticmethod
    def crossValidation(data,k,output=False,path='./',order=1):
        if k<=1 or k>10:
            k=3
        for i in range(k):
            trainingSet = {}
            testSet = {}
            for ind,user in enumerate(data):
                if ind%k == i:
                    testSet[user] = data[user].copy()
                else:
                    trainingSet[user] = data[user].copy()
            yield trainingSet,testSet
```

```python id="P8-Yh3snxF3o"
def drawLine(x,y,labels,xLabel,yLabel,title):
    f, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)

    #f.tight_layout()
    #sns.set(style="darkgrid")

    palette = ['blue','orange','red','green','purple','pink']
    # for i in range(len(ax)):
    #     x1 = range(0, len(x))
        #ax.set_xlim(min(x1)-0.2,max(x1)+0.2)
        # mini = 10000;max = -10000
        # for label in labels:
        #     if mini>min(y[i][label]):
        #         mini = min(y[i][label])
        #     if max<max(y[i][label]):
        #         max = max(y[i][label])
        # ax[i].set_ylim(mini-0.25*(max-mini),max+0.25*(max-mini))
        # for j,label in enumerate(labels):
        #     if j%2==1:
        #         ax[i].plot(x1, y[i][label], color=palette[j/2], marker='.', label=label, markersize=12)
        #     else:
        #         ax[i].plot(x1, y[i][label], color=palette[j/2], marker='.', label=label,markersize=12,linestyle='--')
        # ax[0].set_ylabel(yLabel,fontsize=20)

    for xdata,ydata,lab,c in zip(x,y,labels,palette):
        ax.plot(xdata,ydata,color = c,label=lab)
    ind = np.arange(0,60,10)
    ax.set_xticks(ind)
    #ax.set_xticklabels(x)
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.tick_params(labelsize=16)
    #ax.tick_params(axs='y', labelsize=20)

    ax.set_title(title,fontsize=24)
    plt.grid(True)
    handles, labels1 = ax.get_legend_handles_labels()

    #ax[i].legend(handles, labels1, loc=2, fontsize=20)
    # ax.legend(loc=2,
    #        ncol=6,  borderaxespad=0.,fontsize=20)
    #ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=20)
    ax.legend(loc='upper right',fontsize=20,shadow=True)
    plt.show()
    plt.close()

paths = ['SVD.txt','PMF.txt','EE.txt','RDML.txt']
files = ['EE['+str(i)+'] iteration.txt' for i in range(2,9)]
x = []
y = []

data = []
def normalize():
    for file in files:
        xdata = []
        with open(file) as f:
            for line in f:
                items = line.strip().split()
                rmse = items[2].split(':')[1]
                xdata.append(float(rmse))
        data.append(xdata)
    average = []
    for i in range(len(data[0])):
        total = 0
        for k in range(len(data)):
            total += data[k][i]
        average.append(str(i+1)+':'+str(float(total)/len(data))+'\n')
    with open('EE.txt','w') as f:
        f.writelines(average)



def readData():
    for file in paths:
        xdata = []
        ydata = []
        with open(file) as f:
            for line in f:
                items = line.strip().split(':')
                xdata.append(int(items[0]))
                rmse = float(items[1])
                ydata.append(float(rmse))
        x.append(xdata)
        y.append(ydata)




# x = [[1,2,3],[1,2,3]]
# y = [[1,2,3],[4,5,6]]
#normalize()
readData()
labels = ['SVD','PMF','EE','RDML',]
xlabel = 'Iteration'
ylabel = 'RMSE'

drawLine(x,y,labels,xlabel,ylabel,'')
```

```python id="MJqVx9j6xTGF"
def l1(x):
    return norm(x,ord=1)

def l2(x):
    return norm(x)

def common(x1,x2):
    # find common ratings
    common = (x1!=0)&(x2!=0)
    new_x1 = x1[common]
    new_x2 = x2[common]
    return new_x1,new_x2

def cosine_sp(x1,x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    denom1 = 0
    denom2 =0
    for k in x1:
        if k in x2:
            total+=x1[k]*x2[k]
            denom1+=x1[k]**2
            denom2+=x2[k]**2
    try:
        return (total + 0.0) / (sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        return 0


def cosine(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1,x2)
    #compute the cosine similarity between two vectors
    sum = new_x1.dot(new_x2)
    denom = sqrt(new_x1.dot(new_x1)*new_x2.dot(new_x2))
    try:
        return float(sum)/denom
    except ZeroDivisionError:
        return 0

    #return cosine_similarity(x1,x2)[0][0]

def pearson_sp(x1,x2):
    total = 0
    denom1 = 0
    denom2 = 0
    overlapped=False
    try:
        mean1 = sum(x1.values())/(len(x1)+0.0)
        mean2 = sum(x2.values()) / (len(x2) + 0.0)
        for k in x1:
            if k in x2:
                total += (x1[k]-mean1) * (x2[k]-mean2)
                denom1 += (x1[k]-mean1) ** 2
                denom2 += (x2[k]-mean2) ** 2
                overlapped=True

        return (total + 0.0) / (sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        if overlapped:
            return 1
        else:
            return 0

def euclidean(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1, x2)
    #compute the euclidean between two vectors
    diff = new_x1-new_x2
    denom = sqrt((diff.dot(diff)))
    try:
        return 1/denom
    except ZeroDivisionError:
        return 0


def pearson(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1, x2)
    #compute the pearson similarity between two vectors
    ind1 = new_x1 > 0
    ind2 = new_x2 > 0
    try:
        mean_x1 = float(new_x1.sum())/ind1.sum()
        mean_x2 = float(new_x2.sum())/ind2.sum()
        new_x1 = new_x1 - mean_x1
        new_x2 = new_x2 - mean_x2
        sum = new_x1.dot(new_x2)
        denom = sqrt((new_x1.dot(new_x1))*(new_x2.dot(new_x2)))
        return float(sum) / denom
    except ZeroDivisionError:
        return 0


def similarity(x1,x2,sim):
    if sim == 'pcc':
        return pearson_sp(x1,x2)
    if sim == 'euclidean':
        return euclidean(x1,x2)
    else:
        return cosine_sp(x1, x2)


def normalize(vec,maxVal,minVal):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return float(vec-minVal)/(maxVal-minVal)+0.01
    elif maxVal==minVal:
        return vec/maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError

def sigmoid(val):
    return 1/(1+exp(-val))


def denormalize(vec,maxVal,minVal):
    return minVal+(vec-0.01)*(maxVal-minVal)
```

<!-- #region id="EpwnOGs5uxrP" -->
## Shilling models
<!-- #endregion -->

<!-- #region id="Duy61lrku9Yc" -->
### Attack base class
<!-- #endregion -->

```python id="VNCgq5B-u_Rm"
class Attack(object):
    def __init__(self,conf):
        self.config = Config(conf)
        self.userProfile = FileIO.loadDataSet(self.config,self.config['ratings'])
        self.itemProfile = defaultdict(dict)
        self.attackSize = float(self.config['attackSize'])
        self.fillerSize = float(self.config['fillerSize'])
        self.selectedSize = float(self.config['selectedSize'])
        self.targetCount = int(self.config['targetCount'])
        self.targetScore = float(self.config['targetScore'])
        self.threshold = float(self.config['threshold'])
        self.minCount = int(self.config['minCount'])
        self.maxCount = int(self.config['maxCount'])
        self.minScore = float(self.config['minScore'])
        self.maxScore = float(self.config['maxScore'])
        self.outputDir = self.config['outputDir']
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        for user in self.userProfile:
            for item in self.userProfile[user]:
                self.itemProfile[item][user] = self.userProfile[user][item]
        self.spamProfile = defaultdict(dict)
        self.spamItem = defaultdict(list) #items rated by spammers
        self.targetItems = []
        self.itemAverage = {}
        self.getAverageRating()
        self.selectTarget()
        self.startUserID = 0

    def getAverageRating(self):
        for itemID in self.itemProfile:
            li = list(self.itemProfile[itemID].values())
            self.itemAverage[itemID] = float(sum(li)) / len(li)


    def selectTarget(self,):
        print('Selecting target items...')
        print('-'*80)
        print('Target item       Average rating of the item')
        itemList = list(self.itemProfile.keys())
        itemList.sort()
        while len(self.targetItems) < self.targetCount:
            target = np.random.randint(len(itemList)) #generate a target order at random

            if len(self.itemProfile[str(itemList[target])]) < self.maxCount and len(self.itemProfile[str(itemList[target])]) > self.minCount \
                    and str(itemList[target]) not in self.targetItems \
                    and self.itemAverage[str(itemList[target])] <= self.threshold:
                self.targetItems.append(str(itemList[target]))
                print(str(itemList[target]),'                  ',self.itemAverage[str(itemList[target])])

    def getFillerItems(self):
        mu = int(self.fillerSize*len(self.itemProfile))
        sigma = int(0.1*mu)
        markedItemsCount = abs(int(round(random.gauss(mu, sigma))))
        markedItems = np.random.randint(len(self.itemProfile), size=markedItemsCount)
        return markedItems.tolist()

    def insertSpam(self,startID=0):
        pass

    def loadTarget(self,filename):
        with open(filename) as f:
            for line in f:
                self.targetItems.append(line.strip())

    def generateLabels(self,filename):
        labels = []
        path = self.outputDir + filename
        with open(path,'w') as f:
            for user in self.spamProfile:
                labels.append(user+' 1\n')
            for user in self.userProfile:
                labels.append(user+' 0\n')
            f.writelines(labels)
        print('User profiles have been output to '+abspath(self.config['outputDir'])+'.')

    def generateProfiles(self,filename):
        ratings = []
        path = self.outputDir+filename
        with open(path, 'w') as f:
            for user in self.userProfile:
                for item in self.userProfile[user]:
                    ratings.append(user+' '+item+' '+str(self.userProfile[user][item])+'\n')

            for user in self.spamProfile:
                for item in self.spamProfile[user]:
                    ratings.append(user + ' ' + item + ' ' + str(self.spamProfile[user][item])+'\n')
            f.writelines(ratings)
        print('User labels have been output to '+abspath(self.config['outputDir'])+'.')
```

<!-- #region id="Frmemg_xCNnv" -->
### Relation attack
<!-- #endregion -->

```python id="gpCpBPg7CNkk"
class RelationAttack(Attack):
    def __init__(self,conf):
        super(RelationAttack, self).__init__(conf)
        self.spamLink = defaultdict(list)
        self.relation = FileIO.loadRelationship(self.config,self.config['social'])
        self.trustLink = defaultdict(list)
        self.trusteeLink = defaultdict(list)
        for u1,u2,t in self.relation:
            self.trustLink[u1].append(u2)
            self.trusteeLink[u2].append(u1)
        self.activeUser = {}  # 关注了虚假用户的正常用户
        self.linkedUser = {}  # 被虚假用户种植过链接的用户

    # def reload(self):
    #     super(RelationAttack, self).reload()
    #     self.spamLink = defaultdict(list)
    #     self.trustLink, self.trusteeLink = loadTrusts(self.config['social'])
    #     self.activeUser = {}  # 关注了虚假用户的正常用户
    #     self.linkedUser = {}  # 被虚假用户种植过链接的用户

    def farmLink(self):
        pass

    def getReciprocal(self,target):
        #当前目标用户关注spammer的概率，依赖于粉丝数和关注数的交集
        reciprocal = float(2 * len(set(self.trustLink[target]).intersection(self.trusteeLink[target])) + 0.1) \
                     / (len(set(self.trustLink[target]).union(self.trusteeLink[target])) + 1)
        reciprocal += (len(self.trustLink[target]) + 0.1) / (len(self.trustLink[target]) + len(self.trusteeLink[target]) + 1)
        reciprocal /= 2
        return reciprocal

    def generateSocialConnections(self,filename):
        relations = []
        path = self.outputDir + filename
        with open(path, 'w') as f:
            for u1 in self.trustLink:
                for u2 in self.trustLink[u1]:
                    relations.append(u1 + ' ' + u2 + ' 1\n')

            for u1 in self.spamLink:
                for u2 in self.spamLink[u1]:
                    relations.append(u1 + ' ' + u2 + ' 1\n')
            f.writelines(relations)
        print('Social relations have been output to ' + abspath(self.config['outputDir']) + '.')
```

<!-- #region id="XmE9vaZvCark" -->
### Random relation attack
<!-- #endregion -->

```python id="e3SH9fJiCanT"
class RandomRelationAttack(RelationAttack):
    def __init__(self,conf):
        super(RandomRelationAttack, self).__init__(conf)
        self.scale = float(self.config['linkSize'])

    def farmLink(self):  # 随机注入虚假关系

        for spam in self.spamProfile:

            #对购买了目标项目的用户种植链接
            for item in self.spamItem[spam]:
                if random.random() < 0.01:
                    for target in self.itemProfile[item]:
                        self.spamLink[spam].append(target)
                        response = np.random.random()
                        reciprocal = self.getReciprocal(target)
                        if response <= reciprocal:
                            self.trustLink[target].append(spam)
                            self.activeUser[target] = 1
                        else:
                            self.linkedUser[target] = 1
            #对其它用户以scale的比例种植链接
            for user in self.userProfile:
                if random.random() < self.scale:
                    self.spamLink[spam].append(user)
                    response = np.random.random()
                    reciprocal = self.getReciprocal(user)
                    if response < reciprocal:
                        self.trustLink[user].append(spam)
                        self.activeUser[user] = 1
                    else:
                        self.linkedUser[user] = 1
```

<!-- #region id="Agh_VTYVCgSM" -->
### Random attack
<!-- #endregion -->

```python id="7AyNNzk5CgPR"
class RandomAttack(Attack):
    def __init__(self,conf):
        super(RandomAttack, self).__init__(conf)


    def insertSpam(self,startID=0):
        print('Modeling random attack...')
        itemList = list(self.itemProfile.keys())
        if startID == 0:
            self.startUserID = len(self.userProfile)
        else:
            self.startUserID = startID

        for i in range(int(len(self.userProfile)*self.attackSize)):
            #fill 装填项目
            fillerItems = self.getFillerItems()
            for item in fillerItems:
                self.spamProfile[str(self.startUserID)][str(itemList[item])] = random.randint(self.minScore,self.maxScore)

            #target 目标项目
            for j in range(self.targetCount):
                target = np.random.randint(len(self.targetItems))
                self.spamProfile[str(self.startUserID)][self.targetItems[target]] = self.targetScore
                self.spamItem[str(self.startUserID)].append(self.targetItems[target])
            self.startUserID += 1
```

```python id="TAjlWWrXC389"
class RR_Attack(RandomRelationAttack,RandomAttack):
    def __init__(self,conf):
        super(RR_Attack, self).__init__(conf)
```

<!-- #region id="uQ11CRfGxetw" -->
### Average attack
<!-- #endregion -->

```python id="uITd6vjhxgHV"
class AverageAttack(Attack):
    def __init__(self,conf):
        super(AverageAttack, self).__init__(conf)

    def insertSpam(self,startID=0):
        print('Modeling average attack...')
        itemList = list(self.itemProfile.keys())
        if startID == 0:
            self.startUserID = len(self.userProfile)
        else:
            self.startUserID = startID

        for i in range(int(len(self.userProfile)*self.attackSize)):
            #fill
            fillerItems = self.getFillerItems()
            for item in fillerItems:
                self.spamProfile[str(self.startUserID)][str(itemList[item])] = round(self.itemAverage[str(itemList[item])])
            #target
            for j in range(self.targetCount):
                target = np.random.randint(len(self.targetItems))
                self.spamProfile[str(self.startUserID)][self.targetItems[target]] = self.targetScore
                self.spamItem[str(self.startUserID)].append(self.targetItems[target])
            self.startUserID += 1
```

<!-- #region id="VsaKjYo1DPeY" -->
### Random average relation
<!-- #endregion -->

```python id="nTuGveE9DS85"
class RA_Attack(RandomRelationAttack,AverageAttack):
    def __init__(self,conf):
        super(RA_Attack, self).__init__(conf)
```

<!-- #region id="aO63OdcrCgMZ" -->
### Bandwagon attack
<!-- #endregion -->

```python id="Hx0HaDEYC4AO"
class BandWagonAttack(Attack):
    def __init__(self,conf):
        super(BandWagonAttack, self).__init__(conf)
        self.hotItems = sorted(iter(self.itemProfile.items()), key=lambda d: len(d[1]), reverse=True)[
                   :int(self.selectedSize * len(self.itemProfile))]


    def insertSpam(self,startID=0):
        print('Modeling bandwagon attack...')
        itemList = list(self.itemProfile.keys())
        if startID == 0:
            self.startUserID = len(self.userProfile)
        else:
            self.startUserID = startID

        for i in range(int(len(self.userProfile)*self.attackSize)):
            #fill 装填项目
            fillerItems = self.getFillerItems()
            for item in fillerItems:
                self.spamProfile[str(self.startUserID)][str(itemList[item])] = random.randint(self.minScore,self.maxScore)
            #selected 选择项目
            selectedItems = self.getSelectedItems()
            for item in selectedItems:
                self.spamProfile[str(self.startUserID)][item] = self.targetScore
            #target 目标项目
            for j in range(self.targetCount):
                target = np.random.randint(len(self.targetItems))
                self.spamProfile[str(self.startUserID)][self.targetItems[target]] = self.targetScore
                self.spamItem[str(self.startUserID)].append(self.targetItems[target])
            self.startUserID += 1

    def getFillerItems(self):
        mu = int(self.fillerSize*len(self.itemProfile))
        sigma = int(0.1*mu)
        markedItemsCount = int(round(random.gauss(mu, sigma)))
        if markedItemsCount < 0:
            markedItemsCount = 0
        markedItems = np.random.randint(len(self.itemProfile), size=markedItemsCount)
        return markedItems

    def getSelectedItems(self):

        mu = int(self.selectedSize * len(self.itemProfile))
        sigma = int(0.1 * mu)
        markedItemsCount = abs(int(round(random.gauss(mu, sigma))))
        markedIndexes =  np.random.randint(len(self.hotItems), size=markedItemsCount)
        markedItems = [self.hotItems[index][0] for index in markedIndexes]
        return markedItems
```

<!-- #region id="b7fb5jn3DqLI" -->
### Random bandwagon relation
<!-- #endregion -->

```python id="ibXNF8aAC35s"
class RB_Attack(RandomRelationAttack,BandWagonAttack):
    def __init__(self,conf):
        super(RB_Attack, self).__init__(conf)
```

<!-- #region id="1SXLHTH2C32B" -->
### Hybrid attack
<!-- #endregion -->

```python id="KKNV8OoJD4Ee"
class HybridAttack(Attack):
    def __init__(self,conf):
        super(HybridAttack, self).__init__(conf)
        self.aveAttack = AverageAttack(conf)
        self.bandAttack = BandWagonAttack(conf)
        self.randAttack = RandomAttack(conf)


    def insertSpam(self,startID=0):
        self.aveAttack.insertSpam()
        self.bandAttack.insertSpam(self.aveAttack.startUserID+1)
        self.randAttack.insertSpam(self.bandAttack.startUserID+1)
        self.spamProfile = {}
        self.spamProfile.update(self.aveAttack.spamProfile)
        self.spamProfile.update(self.bandAttack.spamProfile)
        self.spamProfile.update(self.randAttack.spamProfile)

    def generateProfiles(self,filename):

        ratings = []
        path = self.outputDir + filename
        with open(path, 'w') as f:
            for user in self.userProfile:
                for item in self.userProfile[user]:
                    ratings.append(user + ' ' + item + ' ' + str(self.userProfile[user][item]) + '\n')

            for user in self.spamProfile:
                for item in self.spamProfile[user]:
                    ratings.append(user + ' ' + item + ' ' + str(self.spamProfile[user][item]) + '\n')
            f.writelines(ratings)
        print('User labels have been output to ' + abspath(self.config['outputDir']) + '.')

    def generateLabels(self,filename):
        labels = []
        path = self.outputDir + filename
        with open(path,'w') as f:
            for user in self.spamProfile:
                labels.append(user+' 1\n')
            for user in self.userProfile:
                labels.append(user+' 0\n')
            f.writelines(labels)
        print('User profiles have been output to '+abspath(self.config['outputDir'])+'.')
```

<!-- #region id="60j93rEWD4Bn" -->
### Generate data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="g54ZYLoSEEiQ" executionInfo={"status": "ok", "timestamp": 1634220261087, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a77852a5-f1fa-4a8c-9b03-c5ff1f19a230"
%%writefile config.conf
ratings=dataset/filmtrust/ratings.txt
ratings.setup=-columns 0 1 2
social=dataset/filmtrust/trust.txt
social.setup=-columns 0 1 2
attackSize=0.1
fillerSize=0.05
selectedSize=0.005
targetCount=20
targetScore=4.0
threshold=3.0
maxScore=4.0
minScore=1.0
minCount=5
maxCount=50
linkSize=0.001
outputDir=output/
```

```python colab={"base_uri": "https://localhost:8080/"} id="4A-JTXPHD39X" executionInfo={"status": "ok", "timestamp": 1634219617594, "user_tz": -330, "elapsed": 460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b4ec00a9-6efb-4e21-abc4-cd1b02833825"
attack = RR_Attack('config.conf')
attack.insertSpam()
attack.farmLink()
attack.generateLabels('labels.txt')
attack.generateProfiles('profiles.txt')
attack.generateSocialConnections('relations.txt')
```

<!-- #region id="nL3xQOPY_Dc0" -->
## Data access objects
<!-- #endregion -->

```python id="M3o1XWEc0kFM"
class RatingDAO(object):
    'data access control'
    def __init__(self,config, trainingData, testData):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.user = {} #used to store the order of users in the training set
        self.item = {} #used to store the order of items in the training set
        self.id2user = {}
        self.id2item = {}
        self.all_Item = {}
        self.all_User = {}
        self.userMeans = {} #used to store the mean values of users's ratings
        self.itemMeans = {} #used to store the mean values of items's ratings


        self.globalMean = 0
        self.timestamp = {}
        # self.trainingMatrix = None
        # self.validationMatrix = None
        self.testSet_u = testData.copy() # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = defaultdict(dict) # used to store the test set by hierarchy item:[user,rating]
        self.trainingSet_u = trainingData.copy()
        self.trainingSet_i = defaultdict(dict)
        #self.rScale = []

        self.trainingData = trainingData
        self.testData = testData
        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()



    def __generateSet(self):
        scale = set()
        # find the maximum rating and minimum value
        # for i, entry in enumerate(self.trainingData):
        #     userName, itemName, rating = entry
        #     scale.add(float(rating))
        # self.rScale = list(scale)
        # self.rScale.sort()

        for i,user in enumerate(self.trainingData):
            for item in self.trainingData[user]:

                # makes the rating within the range [0, 1].
                #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
                #self.trainingSet_u[userName][itemName] = float(rating)
                self.trainingSet_i[item][user] = self.trainingData[user][item]
                # order the user
                if user not in self.user:
                    self.user[user] = len(self.user)
                    self.id2user[self.user[user]] = user
                # order the item
                if item not in self.item:
                    self.item[item] = len(self.item)
                    self.id2item[self.item[item]] = item
                self.trainingSet_i[item][user] = self.trainingData[user][item]
                # userList.append
        #     triple.append([self.user[userName], self.item[itemName], rating])
        # self.trainingMatrix = new_sparseMatrix.SparseMatrix(triple)

        self.all_User.update(self.user)
        self.all_Item.update(self.item)

        for i, user in enumerate(self.testData):
            # order the user
            if user not in self.user:
                self.all_User[user] = len(self.all_User)
            for item in self.testData[user]:
                # order the item
                if item not in self.item:
                    self.all_Item[item] = len(self.all_Item)
                #self.testSet_u[userName][itemName] = float(rating)
                self.testSet_i[item][user] = self.testData[user][item]


    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        # for u in self.user:
        #     n = self.row(u) > 0
        #     mean = 0
        #
        #     if not self.containsUser(u):  # no data about current user in training set
        #         pass
        #     else:
        #         sum = float(self.row(u)[0].sum())
        #         try:
        #             mean =  sum/ n[0].sum()
        #         except ZeroDivisionError:
        #             mean = 0
        #     self.userMeans[u] = mean
        for u in self.trainingSet_u:
            self.userMeans[u] = sum(self.trainingSet_u[u].values())/(len(list(self.trainingSet_u[u].values()))+0.0)
        for u in self.testSet_u:
            self.userMeans[u] = sum(self.testSet_u[u].values())/(len(list(self.testSet_u[u].values()))+0.0)

    def __computeItemMean(self):
        # for c in self.item:
        #     n = self.col(c) > 0
        #     mean = 0
        #     if not self.containsItem(c):  # no data about current user in training set
        #         pass
        #     else:
        #         sum = float(self.col(c)[0].sum())
        #         try:
        #             mean = sum / n[0].sum()
        #         except ZeroDivisionError:
        #             mean = 0
        #     self.itemMeans[c] = mean
        for item in self.trainingSet_i:
            self.itemMeans[item] = sum(self.trainingSet_i[item].values())/(len(list(self.trainingSet_i[item].values())) + 0.0)
        for item in self.testSet_i:
            self.itemMeans[item] = sum(self.testSet_i[item].values())/(len(list(self.testSet_i[item].values())) + 0.0)

    def getUserId(self,u):
        if u in self.user:
            return self.user[u]
        else:
            return -1

    def getItemId(self,i):
        if i in self.item:
            return self.item[i]
        else:
            return -1

    def trainingSize(self):
        recordCount = 0
        for user in self.trainingData:
            recordCount+=len(self.trainingData[user])
        return (len(self.trainingSet_u),len(self.trainingSet_i),recordCount)


    def testSize(self):
        recordCount = 0
        for user in self.testData:
            recordCount += len(self.testData[user])
        return (len(self.testSet_u),len(self.testSet_i),recordCount)

    def contains(self,u,i):
        'whether user u rated item i'
        if u in self.trainingSet_u and i in self.trainingSet_u[u]:
            return True
        return False

    def containsUser(self,u):
        'whether user is in training set'
        return u in self.trainingSet_u

    def containsItem(self,i):
        'whether item is in training set'
        return i in self.trainingSet_i

    def allUserRated(self, u):
        if u in self.user:
            return list(self.trainingSet_u[u].keys()), list(self.trainingSet_u[u].values())
        else:
            return list(self.testSet_u[u].keys()), list(self.testSet_u[u].values())
    # def userRated(self,u):
    #     if self.trainingMatrix.matrix_User.has_key(self.getUserId(u)):
    #         itemIndex =  self.trainingMatrix.matrix_User[self.user[u]].keys()
    #         rating = self.trainingMatrix.matrix_User[self.user[u]].values()
    #         return (itemIndex,rating)
    #     return ([],[])
    #
    # def itemRated(self,i):
    #     if self.trainingMatrix.matrix_Item.has_key(self.getItemId(i)):
    #         userIndex = self.trainingMatrix.matrix_Item[self.item[i]].keys()
    #         rating = self.trainingMatrix.matrix_Item[self.item[i]].values()
    #         return (userIndex,rating)
    #     return ([],[])

    # def row(self,u):
    #     return self.trainingMatrix.row(self.getUserId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getItemId(c))
    #
    # def sRow(self,u):
    #     return self.trainingMatrix.sRow(self.getUserId(u))
    #
    # def sCol(self,c):
    #     return self.trainingMatrix.sCol(self.getItemId(c))
    #
    # def rating(self,u,c):
    #     return self.trainingMatrix.elem(self.getUserId(u),self.getItemId(c))
    #
    # def ratingScale(self):
    #     return (self.rScale[0],self.rScale[1])

    # def elemCount(self):
    #     return self.trainingMatrix.elemCount()
```

```python id="6t6Ec7Ve_Syj"
class SocialDAO(object):
    def __init__(self,conf,relation=list()):
        self.config = conf
        self.user = {} #used to store the order of users
        self.relation = relation
        self.followees = {}
        self.followers = {}
        self.trustMatrix = self.__generateSet()

    def __generateSet(self):
        #triple = []
        for line in self.relation:
            userId1,userId2,weight = line
            #add relations to dict
            if userId1 not in self.followees:
                self.followees[userId1] = {}
            self.followees[userId1][userId2] = weight
            if userId2 not in self.followers:
                self.followers[userId2] = {}
            self.followers[userId2][userId1] = weight
            # order the user
            if userId1 not in self.user:
                self.user[userId1] = len(self.user)
            if userId2 not in self.user:
                self.user[userId2] = len(self.user)
            #triple.append([self.user[userId1], self.user[userId2], weight])
        #return new_sparseMatrix.SparseMatrix(triple)

    # def row(self,u):
    #     #return user u's followees
    #     return self.trustMatrix.row(self.user[u])
    #
    # def col(self,u):
    #     #return user u's followers
    #     return self.trustMatrix.col(self.user[u])
    #
    # def elem(self,u1,u2):
    #     return self.trustMatrix.elem(u1,u2)

    def weight(self,u1,u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    # def trustSize(self):
    #     return self.trustMatrix.size

    def getFollowers(self,u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def getFollowees(self,u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def hasFollowee(self,u1,u2):
        if u1 in self.followees:
            if u2 in self.followees[u1]:
                return True
            else:
                return False
        return False

    def hasFollower(self,u1,u2):
        if u1 in self.followers:
            if u2 in self.followers[u1]:
                return True
            else:
                return False
        return False
```

<!-- #region id="8UD1B1i70O7T" -->
## Methods
<!-- #endregion -->

<!-- #region id="dAQny9vu_dDQ" -->
### BayesDetector
<!-- #endregion -->

```python id="yFpZGu1F_dA4"
#BayesDetector: Collaborative Shilling Detection Bridging Factorization and User Embedding
class BayesDetector(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(BayesDetector, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(BayesDetector, self).readConfiguration()
        extraSettings = LineConfig(self.config['BayesDetector'])
        self.k = int(extraSettings['-k'])
        self.negCount = int(extraSettings['-negCount'])  # the number of negative samples
        if self.negCount < 1:
            self.negCount = 1

        self.regR = float(extraSettings['-gamma'])
        self.filter = int(extraSettings['-filter'])
        self.delta = float(extraSettings['-delta'])
        learningRate = LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        self.maxIter = int(self.config['num.max.iter'])
        regular = LineConfig(self.config['reg.lambda'])
        self.regU, self.regI = float(regular['-u']), float(regular['-i'])
        # self.delta = float(self.config['delta'])
    def printAlgorConfig(self):
        super(BayesDetector, self).printAlgorConfig()
        print('k: %d' % self.negCount)
        print('regR: %.5f' % self.regR)
        print('filter: %d' % self.filter)
        print('=' * 80)

    def initModel(self):
        super(BayesDetector, self).initModel()
        # self.c = np.random.rand(len(self.dao.all_User) + 1) / 20  # bias value of context
        self.G = np.random.rand(len(self.dao.all_User)+1, self.k) / 100  # context embedding
        self.P = np.random.rand(len(self.dao.all_User)+1, self.k) / 100  # latent user matrix
        self.Q = np.random.rand(len(self.dao.all_Item)+1, self.k) / 100  # latent item matrix

        # constructing SPPMI matrix
        self.SPPMI = defaultdict(dict)
        D = len(self.dao.user)
        print('Constructing SPPMI matrix...')
        # for larger data set has many items, the process will be time consuming
        occurrence = defaultdict(dict)
        for user1 in self.dao.all_User:
            iList1, rList1 = self.dao.allUserRated(user1)
            if len(iList1) < self.filter:
                continue
            for user2 in self.dao.all_User:
                if user1 == user2:
                    continue
                if user2 not in occurrence[user1]:
                    iList2, rList2 = self.dao.allUserRated(user2)
                    if len(iList2) < self.filter:
                        continue
                    count = len(set(iList1).intersection(set(iList2)))
                    if count > self.filter:
                        occurrence[user1][user2] = count
                        occurrence[user2][user1] = count

        maxVal = 0
        frequency = {}
        for user1 in occurrence:
            frequency[user1] = sum(occurrence[user1].values()) * 1.0
        D = sum(frequency.values()) * 1.0
        # maxx = -1
        for user1 in occurrence:
            for user2 in occurrence[user1]:
                try:
                    val = max([log(occurrence[user1][user2] * D / (frequency[user1] * frequency[user2]), 2) - log(
                        self.negCount, 2), 0])
                except ValueError:
                    print(self.SPPMI[user1][user2])
                    print(self.SPPMI[user1][user2] * D / (frequency[user1] * frequency[user2]))
                if val > 0:
                    if maxVal < val:
                        maxVal = val
                    self.SPPMI[user1][user2] = val
                    self.SPPMI[user2][user1] = self.SPPMI[user1][user2]

        # normalize
        for user1 in self.SPPMI:
            for user2 in self.SPPMI[user1]:
                self.SPPMI[user1][user2] = self.SPPMI[user1][user2] / maxVal

    def buildModel(self):
        self.dao.ratings = dict(self.dao.trainingSet_u, **self.dao.testSet_u)
        #suspicous set
        print('Preparing sets...')
        self.sSet = defaultdict(dict)
        #normal set
        self.nSet = defaultdict(dict)
        # self.NegativeSet = defaultdict(list)

        for user in self.dao.user:
            for item in self.dao.ratings[user]:
                # if self.dao.ratings[user][item] >= 5 and self.labels[user]=='1':
                if self.labels[user] =='1':
                    self.sSet[item][user] = 1
                # if self.dao.ratings[user][item] >= 5 and self.labels[user] == '0':
                if self.labels[user] == '0':
                    self.nSet[item][user] = 1
        # Jointly decompose R(ratings) and SPPMI with shared user latent factors P
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0

            for item in self.sSet:
                i = self.dao.all_Item[item]
                if item not in self.nSet:
                    continue
                normalUserList = list(self.nSet[item].keys())
                for user in self.sSet[item]:
                    su = self.dao.all_User[user]
                    # if len(self.NegativeSet[user]) > 0:
                    #     item_j = choice(self.NegativeSet[user])
                    # else:
                    normalUser = choice(normalUserList)
                    nu = self.dao.all_User[normalUser]

                    s = sigmoid(self.P[su].dot(self.Q[i]) - self.P[nu].dot(self.Q[i]))
                    self.Q[i] += (self.lRate * (1 - s) * (self.P[su] - self.P[nu]))
                    self.P[su] += (self.lRate * (1 - s) * self.Q[i])
                    self.P[nu] -= (self.lRate * (1 - s) * self.Q[i])

                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.P[su] -= self.lRate * self.regU * self.P[su]
                    self.P[nu] -= self.lRate * self.regU * self.P[nu]

                    self.loss += (-log(s))
            #
            # for item in self.sSet:
            #     if not self.nSet.has_key(item):
            #         continue
            #     for user1 in self.sSet[item]:
            #         for user2 in self.sSet[item]:
            #             su1 = self.dao.all_User[user1]
            #             su2 = self.dao.all_User[user2]
            #             self.P[su1] += (self.lRate*(self.P[su1]-self.P[su2]))*self.delta
            #             self.P[su2] -= (self.lRate*(self.P[su1]-self.P[su2]))*self.delta
            #
            #             self.loss += ((self.P[su1]-self.P[su2]).dot(self.P[su1]-self.P[su2]))*self.delta


            for user in self.dao.ratings:
                for item in self.dao.ratings[user]:
                    rating = self.dao.ratings[user][item]
                    if rating < 5:
                        continue
                    error = rating - self.predictRating(user,item)
                    u = self.dao.all_User[user]
                    i = self.dao.all_Item[item]
                    p = self.P[u]
                    q = self.Q[i]
                    # self.loss += (error ** 2)*self.b
                    # update latent vectors
                    self.P[u] += (self.lRate * (error * q - self.regU * p))
                    self.Q[i] += (self.lRate * (error * p - self.regI * q))


            for user in self.SPPMI:
                u = self.dao.all_User[user]
                p = self.P[u]
                for context in self.SPPMI[user]:
                    v = self.dao.all_User[context]
                    m = self.SPPMI[user][context]
                    g = self.G[v]
                    diff = (m - p.dot(g))
                    self.loss += (diff ** 2)
                    # update latent vectors
                    self.P[u] += (self.lRate * diff * g)
                    self.G[v] += (self.lRate * diff * p)
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()  + self.regR * (self.G * self.G).sum()
            iteration += 1
            print('iteration:',iteration)

        # preparing examples
        self.training = []
        self.trainingLabels = []
        self.test = []
        self.testLabels = []

        for user in self.dao.trainingSet_u:
            self.training.append(self.P[self.dao.all_User[user]])
            self.trainingLabels.append(self.labels[user])
        for user in self.dao.testSet_u:
            self.test.append(self.P[self.dao.all_User[user]])
            self.testLabels.append(self.labels[user])
        #
        # tsne = TSNE(n_components=2)
        # self.Y = tsne.fit_transform(self.P)
        #
        # self.normalUsers = []
        # self.spammers = []
        # for user in self.labels:
        #     if self.labels[user] == '0':
        #         self.normalUsers.append(user)
        #     else:
        #         self.spammers.append(user)
        #
        #
        # print len(self.spammers)
        # self.normalfeature = np.zeros((len(self.normalUsers), 2))
        # self.spamfeature = np.zeros((len(self.spammers), 2))
        # normal_index = 0
        # for normaluser in self.normalUsers:
        #     if normaluser in self.dao.all_User:
        #         self.normalfeature[normal_index] = self.Y[self.dao.all_User[normaluser]]
        #         normal_index += 1
        #
        # spam_index = 0
        # for spamuser in self.spammers:
        #     if spamuser in self.dao.all_User:
        #         self.spamfeature[spam_index] = self.Y[self.dao.all_User[spamuser]]
        #         spam_index += 1
        # self.randomNormal = np.zeros((500,2))
        # self.randomSpam = np.zeros((500,2))
        # # for i in range(500):
        # #     self.randomNormal[i] = self.normalfeature[random.randint(0,len(self.normalfeature)-1)]
        # #     self.randomSpam[i] = self.spamfeature[random.randint(0,len(self.spamfeature)-1)]
        # plt.scatter(self.normalfeature[:, 0], self.normalfeature[:, 1], c='red',s=8,marker='o',label='NormalUser')
        # plt.scatter(self.spamfeature[:, 0], self.spamfeature[:, 1], c='blue',s=8,marker='o',label='Spammer')
        # plt.legend(loc='lower left')
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('9.png',dpi=500)


    def predictRating(self,user,item):
        u = self.dao.all_User[user]
        i = self.dao.all_Item[item]
        return self.P[u].dot(self.Q[i])

    def predict(self):
        classifier =  RandomForestClassifier(n_estimators=12)
        # classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        return pred_labels
```

<!-- #region id="oMYfGVsl_c-Z" -->
### CoDetector
<!-- #endregion -->

```python id="2r1_ZpIT_c77"
#CoDetector: Collaborative Shilling Detection Bridging Factorization and User Embedding
class CoDetector(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(CoDetector, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(CoDetector, self).readConfiguration()
        extraSettings = LineConfig(self.config['CoDetector'])
        self.k = int(extraSettings['-k'])
        self.negCount = int(extraSettings['-negCount'])  # the number of negative samples
        if self.negCount < 1:
            self.negCount = 1

        self.regR = float(extraSettings['-gamma'])
        self.filter = int(extraSettings['-filter'])

        learningRate = LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        self.maxIter = int(self.config['num.max.iter'])
        regular = LineConfig(self.config['reg.lambda'])
        self.regU, self.regI = float(regular['-u']), float(regular['-i'])

    def printAlgorConfig(self):
        super(CoDetector, self).printAlgorConfig()
        print('k: %d' % self.negCount)
        print('regR: %.5f' % self.regR)
        print('filter: %d' % self.filter)
        print('=' * 80)

    def initModel(self):
        super(CoDetector, self).initModel()
        self.w = np.random.rand(len(self.dao.all_User)+1) / 20  # bias value of user
        self.c = np.random.rand(len(self.dao.all_User)+1)/ 20  # bias value of context
        self.G = np.random.rand(len(self.dao.all_User)+1, self.k) / 20  # context embedding
        self.P = np.random.rand(len(self.dao.all_User)+1, self.k) / 20  # latent user matrix
        self.Q = np.random.rand(len(self.dao.all_Item)+1, self.k) / 20  # latent item matrix


        # constructing SPPMI matrix
        self.SPPMI = defaultdict(dict)
        D = len(self.dao.user)
        print('Constructing SPPMI matrix...')
        # for larger data set has many items, the process will be time consuming
        occurrence = defaultdict(dict)
        for user1 in self.dao.all_User:
            iList1, rList1 = self.dao.allUserRated(user1)
            if len(iList1) < self.filter:
                continue
            for user2 in self.dao.all_User:
                if user1 == user2:
                    continue
                if user2 not in occurrence[user1]:
                    iList2, rList2 = self.dao.allUserRated(user2)
                    if len(iList2) < self.filter:
                        continue
                    count = len(set(iList1).intersection(set(iList2)))
                    if count > self.filter:
                        occurrence[user1][user2] = count
                        occurrence[user2][user1] = count

        maxVal = 0
        frequency = {}
        for user1 in occurrence:
            frequency[user1] = sum(occurrence[user1].values()) * 1.0
        D = sum(frequency.values()) * 1.0
        # maxx = -1
        for user1 in occurrence:
            for user2 in occurrence[user1]:
                try:
                    val = max([log(occurrence[user1][user2] * D / (frequency[user1] * frequency[user2]), 2) - log(
                        self.negCount, 2), 0])
                except ValueError:
                    print(self.SPPMI[user1][user2])
                    print(self.SPPMI[user1][user2] * D / (frequency[user1] * frequency[user2]))
                if val > 0:
                    if maxVal < val:
                        maxVal = val
                    self.SPPMI[user1][user2] = val
                    self.SPPMI[user2][user1] = self.SPPMI[user1][user2]

        # normalize
        for user1 in self.SPPMI:
            for user2 in self.SPPMI[user1]:
                self.SPPMI[user1][user2] = self.SPPMI[user1][user2] / maxVal

    def buildModel(self):
        # Jointly decompose R(ratings) and SPPMI with shared user latent factors P
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0

            self.dao.ratings = dict(self.dao.trainingSet_u, **self.dao.testSet_u)
            for user in self.dao.ratings:
                for item in self.dao.ratings[user]:
                    rating = self.dao.ratings[user][item]
                    error = rating - self.predictRating(user,item)
                    u = self.dao.all_User[user]
                    i = self.dao.all_Item[item]
                    p = self.P[u]
                    q = self.Q[i]
                    self.loss += error ** 2
                    # update latent vectors
                    self.P[u] += self.lRate * (error * q - self.regU * p)
                    self.Q[i] += self.lRate * (error * p - self.regI * q)


            for user in self.SPPMI:
                u = self.dao.all_User[user]
                p = self.P[u]
                for context in self.SPPMI[user]:
                    v = self.dao.all_User[context]
                    m = self.SPPMI[user][context]
                    g = self.G[v]
                    diff = (m - p.dot(g) - self.w[u] - self.c[v])
                    self.loss += diff ** 2
                    # update latent vectors
                    self.P[u] += self.lRate * diff * g
                    self.G[v] += self.lRate * diff * p
                    self.w[u] += self.lRate * diff
                    self.c[v] += self.lRate * diff
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()  + self.regR * (self.G * self.G).sum()
            iteration += 1
            print('iteration:',iteration)

        # preparing examples
        self.training = []
        self.trainingLabels = []
        self.test = []
        self.testLabels = []

        for user in self.dao.trainingSet_u:
            self.training.append(self.P[self.dao.all_User[user]])
            self.trainingLabels.append(self.labels[user])
        for user in self.dao.testSet_u:
            self.test.append(self.P[self.dao.all_User[user]])
            self.testLabels.append(self.labels[user])

    def predictRating(self,user,item):
        u = self.dao.all_User[user]
        i = self.dao.all_Item[item]
        return self.P[u].dot(self.Q[i])

    def predict(self):
        classifier =  DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        return pred_labels
```

<!-- #region id="aYM6GuD-0QgA" -->
### DegreeSAD
<!-- #endregion -->

```python id="SFuhZGNP0SCw"
class DegreeSAD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(DegreeSAD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def buildModel(self):
        self.MUD = {}
        self.RUD = {}
        self.QUD = {}
        # computing MUD,RUD,QUD for training set
        sList = sorted(iter(self.dao.trainingSet_i.items()), key=lambda d: len(d[1]), reverse=True)
        maxLength = len(sList[0][1])
        for user in self.dao.trainingSet_u:
            self.MUD[user] = 0
            for item in self.dao.trainingSet_u[user]:
                self.MUD[user] += len(self.dao.trainingSet_i[item]) #/ float(maxLength)
            self.MUD[user]/float(len(self.dao.trainingSet_u[user]))
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]

            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.trainingSet_u[user]]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]

        # computing MUD,RUD,QUD for test set
        for user in self.dao.testSet_u:
            self.MUD[user] = 0
            for item in self.dao.testSet_u[user]:
                self.MUD[user] += len(self.dao.trainingSet_i[item]) #/ float(maxLength)
        for user in self.dao.testSet_u:
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.testSet_u[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]
        for user in self.dao.testSet_u:
            lengthList = [len(self.dao.trainingSet_i[item]) for item in self.dao.testSet_u[user]]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]

        # preparing examples

        for user in self.dao.trainingSet_u:
            self.training.append([self.MUD[user], self.RUD[user], self.QUD[user]])
            self.trainingLabels.append(self.labels[user])

        for user in self.dao.testSet_u:
            self.test.append([self.MUD[user], self.RUD[user], self.QUD[user]])
            self.testLabels.append(self.labels[user])

    def predict(self):
        # classifier = LogisticRegression()
        # classifier.fit(self.training, self.trainingLabels)
        # pred_labels = classifier.predict(self.test)
        # print 'Logistic:'
        # print classification_report(self.testLabels, pred_labels)
        #
        # classifier = SVC()
        # classifier.fit(self.training, self.trainingLabels)
        # pred_labels = classifier.predict(self.test)
        # print 'SVM:'
        # print classification_report(self.testLabels, pred_labels)

        classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        return pred_labels
```

<!-- #region id="Qw1J9EML_c4B" -->
### FAP
<!-- #endregion -->

```python id="GI3xwj1V_c05"
class FAP(SDetection):

    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(FAP, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(FAP, self).readConfiguration()
        # # s means the number of seedUser who be regarded as spammer in training
        self.s =int( self.config['seedUser'])
        # preserve the real spammer ID
        self.spammer = []
        for i in self.dao.user:
            if self.labels[i] == '1':
                self.spammer.append(self.dao.user[i])
        sThreshold = int(0.5 * len(self.spammer))
        if self.s > sThreshold :
            self.s = sThreshold
            print('*** seedUser is more than a half of spammer, so it is set to', sThreshold, '***')

        # # predict top-k user as spammer
        self.k = int(self.config['topKSpam'])
        # 0.5 is the ratio of spammer to dataset, it can be changed according to different datasets
        kThreshold = int(0.5 * (len(self.dao.user) - self.s))
        if self.k > kThreshold:
            self.k = kThreshold
            print('*** the number of top-K users is more than threshold value, so it is set to', kThreshold, '***')
    # product transition probability matrix self.TPUI and self.TPIU

    def __computeTProbability(self):
        # m--user count; n--item count
        m, n, tmp = self.dao.trainingSize()
        self.TPUI = np.zeros((m, n))
        self.TPIU = np.zeros((n, m))

        self.userUserIdDic = {}
        self.itemItemIdDic = {}
        tmpUser = list(self.dao.user.values())
        tmpUserId = list(self.dao.user.keys())
        tmpItem = list(self.dao.item.values())
        tmpItemId = list(self.dao.item.keys())
        for users in range(0, m):
            self.userUserIdDic[tmpUser[users]] = tmpUserId[users]
        for items in range(0, n):
            self.itemItemIdDic[tmpItem[items]] = tmpItemId[items]
        for i in range(0, m):
            for j in range(0, n):
                user = self.userUserIdDic[i]
                item = self.itemItemIdDic[j]
                # if has edge in graph,set a value ;otherwise set 0
                if (user not in self.bipartiteGraphUI) or (item not in self.bipartiteGraphUI[user]):
                    continue
                else:
                    w = float(self.bipartiteGraphUI[user][item])
                    # to avoid positive feedback and reliability problem,we should Polish the w
                    otherItemW = 0
                    otherUserW = 0
                    for otherItem in self.bipartiteGraphUI[user]:
                        otherItemW += float(self.bipartiteGraphUI[user][otherItem])
                    for otherUser in self.dao.trainingSet_i[item]:
                        otherUserW += float(self.bipartiteGraphUI[otherUser][item])
                    # wPrime = w*1.0/(otherUserW * otherItemW)
                    wPrime = w
                    self.TPUI[i][j] = wPrime / otherItemW
                    self.TPIU[j][i] = wPrime / otherUserW
            if i % 100 == 0:
                print('progress: %d/%d' %(i,m))

    def initModel(self):
        # construction of the bipartite graph
        print("constructing bipartite graph...")
        self.bipartiteGraphUI = {}
        for user in self.dao.trainingSet_u:
            tmpUserItemDic = {}  # user-item-point
            for item in self.dao.trainingSet_u[user]:
                # tmpItemUserDic = {}#item-user-point
                recordValue = float(self.dao.trainingSet_u[user][item])
                w = 1 + abs((recordValue - self.dao.userMeans[user]) / self.dao.userMeans[user]) + abs(
                    (recordValue - self.dao.itemMeans[item]) / self.dao.itemMeans[item]) + abs(
                    (recordValue - self.dao.globalMean) / self.dao.globalMean)
                # tmpItemUserDic[user] = w
                tmpUserItemDic[item] = w
            # self.bipartiteGraphIU[item] = tmpItemUserDic
            self.bipartiteGraphUI[user] = tmpUserItemDic
        # we do the polish in computing the transition probability
        print("computing transition probability...")
        self.__computeTProbability()

    def isConvergence(self, PUser, PUserOld):
        if len(PUserOld) == 0:
            return True
        for i in range(0, len(PUser)):
            if (PUser[i] - PUserOld[i]) > 0.01:
                return True
        return False

    def buildModel(self):
        # -------init--------
        m, n, tmp = self.dao.trainingSize()
        PUser = np.zeros(m)
        PItem = np.zeros(n)
        self.testLabels = [0 for i in range(m)]
        self.predLabels = [0 for i in range(m)]

        # preserve seedUser Index
        self.seedUser = []
        randDict = {}
        for i in range(0, self.s):
            randNum = random.randint(0, len(self.spammer) - 1)
            while randNum in randDict:
                randNum = random.randint(0, len(self.spammer) - 1)
            randDict[randNum] = 0
            self.seedUser.append(int(self.spammer[randNum]))
            # print len(randDict), randDict

        #initial user and item spam probability
        for j in range(0, m):
            if j in self.seedUser:
                #print type(j),j
                PUser[j] = 1
            else:
                PUser[j] = random.random()
        for tmp in range(0, n):
            PItem[tmp] = random.random()

        # -------iterator-------
        PUserOld = []
        iterator = 0
        while self.isConvergence(PUser, PUserOld):
        #while iterator < 100:
            for j in self.seedUser:
                PUser[j] = 1
            PUserOld = PUser
            PItem = np.dot(self.TPIU, PUser)
            PUser = np.dot(self.TPUI, PItem)
            iterator += 1
            print(self.foldInfo,'iteration', iterator)

        PUserDict = {}
        userId = 0
        for i in PUser:
            PUserDict[userId] = i
            userId += 1
        for j in self.seedUser:
            del PUserDict[j]

        self.PSort = sorted(iter(PUserDict.items()), key=lambda d: d[1], reverse=True)


    def predict(self):
        # predLabels
        # top-k user as spammer
        spamList = []
        sIndex = 0
        while sIndex < self.k:
            spam = self.PSort[sIndex][0]
            spamList.append(spam)
            self.predLabels[spam] = 1
            sIndex += 1

        # trueLabels
        for user in self.dao.trainingSet_u:
            userInd = self.dao.user[user]
            # print type(user), user, userInd
            self.testLabels[userInd] = int(self.labels[user])

        # delete seedUser labels
        differ = 0
        for user in self.seedUser:
            user = int(user - differ)
            # print type(user)
            del self.predLabels[user]
            del self.testLabels[user]
            differ += 1

        return self.predLabels
```

<!-- #region id="Gdk4fDUP_cvp" -->
### PCASelectUsers
<!-- #endregion -->

```python id="R8d1wxY2_csN"
class PCASelectUsers(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]', k=None, n=None ):
        super(PCASelectUsers, self).__init__(conf, trainingSet, testSet, labels, fold)


    def readConfiguration(self):
        super(PCASelectUsers, self).readConfiguration()
        # K = top-K vals of cov
        self.k = int(self.config['kVals'])
        self.userNum = len(self.dao.trainingSet_u)
        self.itemNum = len(self.dao.trainingSet_i)
        if self.k >= min(self.userNum, self.itemNum):
            self.k = 3
            print('*** k-vals is more than the number of user or item, so it is set to', self.k)

        # n = attack size or the ratio of spammers to normal users
        self.n = float(self.config['attackSize'])


    def buildModel(self):
        #array initialization
        dataArray = np.zeros([self.userNum, self.itemNum], dtype=float)
        self.testLabels = np.zeros(self.userNum)
        self.predLabels = np.zeros(self.userNum)

        #add data
        print('construct matrix')
        for user in self.dao.trainingSet_u:
            for item in list(self.dao.trainingSet_u[user].keys()):
                value = self.dao.trainingSet_u[user][item]
                a = self.dao.user[user]
                b = self.dao.item[item]
                dataArray[a][b] = value

        sMatrix = csr_matrix(dataArray)
        # z-scores
        sMatrix = preprocessing.scale(sMatrix, axis=0, with_mean=False)
        sMT = np.transpose(sMatrix)
        # cov
        covSM = np.dot(sMT, sMatrix)
        # eigen-value-decomposition
        vals, vecs = scipy.sparse.linalg.eigs(covSM, k=self.k, which='LM')

        newArray = np.dot(dataArray**2, np.real(vecs))

        distanceDict = {}
        userId = 0
        for user in newArray:
            distance = 0
            for tmp in user:
                distance += tmp
            distanceDict[userId] = float(distance)
            userId += 1

        print('sort distance ')
        self.disSort = sorted(iter(distanceDict.items()), key=lambda d: d[1], reverse=False)


    def predict(self):
        print('predict spammer')
        spamList = []
        i = 0
        while i < self.n * len(self.disSort):
            spam = self.disSort[i][0]
            spamList.append(spam)
            self.predLabels[spam] = 1
            i += 1

        # trueLabels
        for user in self.dao.trainingSet_u:
            userInd = self.dao.user[user]
            self.testLabels[userInd] = int(self.labels[user])

        return self.predLabels
```

<!-- #region id="KMHaQzDuAYY-" -->
### SemiSAD
<!-- #endregion -->

```python id="vn7EqiezAYUg"
class SemiSAD(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(SemiSAD, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(SemiSAD, self).readConfiguration()
        # K = top-K vals of cov
        self.k = int(self.config['topK'])
        # Lambda = λ参数
        self.Lambda = float(self.config['Lambda'])

    def buildModel(self):
        self.H = {}
        self.DegSim = {}
        self.LengVar = {}
        self.RDMA = {}
        self.FMTD = {}
        print('Begin feature engineering...')
        # computing H,DegSim,LengVar,RDMA,FMTD for LabledData set
        trainingIndex = 0
        testIndex = 0
        trainingUserCount, trainingItemCount, trainingrecordCount = self.dao.trainingSize()
        testUserCount, testItemCount, testrecordCount = self.dao.testSize()
        for user in self.dao.trainingSet_u:
            trainingIndex += 1
            self.H[user] = 0
            for i in range(10,50,5):
                n = 0
                for item in self.dao.trainingSet_u[user]:
                    if(self.dao.trainingSet_u[user][item]==(i/10.0)):
                        n+=1
                if n==0:
                    self.H[user] += 0
                else:
                    self.H[user] += (-(n/(trainingUserCount*1.0))*math.log(n/(trainingUserCount*1.0),2))

            SimList = []
            self.DegSim[user] = 0
            for user1 in self.dao.trainingSet_u:
                userA, userB, C, D, E, Count = 0,0,0,0,0,0
                for item in list(set(self.dao.trainingSet_u[user]).intersection(set(self.dao.trainingSet_u[user1]))):
                    userA += self.dao.trainingSet_u[user][item]
                    userB += self.dao.trainingSet_u[user1][item]
                    Count += 1
                if Count==0:
                    AverageA = 0
                    AverageB = 0
                else:
                    AverageA = userA/Count
                    AverageB = userB/Count
                for item in list(set(self.dao.trainingSet_u[user]).intersection(set(self.dao.trainingSet_u[user1]))):
                    C += (self.dao.trainingSet_u[user][item]-AverageA)*(self.dao.trainingSet_u[user1][item]-AverageB)
                    D += np.square(self.dao.trainingSet_u[user][item]-AverageA)
                    E += np.square(self.dao.trainingSet_u[user1][item]-AverageB)
                if C==0:
                    SimList.append(0.0)
                else:
                    SimList.append(C/(math.sqrt(D)*math.sqrt(E)))
            SimList.sort(reverse=True)
            for i in range(1,self.k+1):
                self.DegSim[user] += SimList[i] / (self.k)

            GlobalAverage = 0
            F = 0
            for user2 in self.dao.trainingSet_u:
                GlobalAverage += len(self.dao.trainingSet_u[user2]) / (len(self.dao.trainingSet_u) + 0.0)
            for user3 in self.dao.trainingSet_u:
                F += pow(len(self.dao.trainingSet_u[user3])-GlobalAverage,2)
            self.LengVar[user] = abs(len(self.dao.trainingSet_u[user])-GlobalAverage)/(F*1.0)

            Divisor = 0
            for item1 in self.dao.trainingSet_u[user]:
                Divisor += abs(self.dao.trainingSet_u[user][item1]-self.dao.itemMeans[item1])/len(self.dao.trainingSet_i[item1])
            self.RDMA[user] = Divisor/len(self.dao.trainingSet_u[user])

            Minuend, index1, Subtrahend, index2 = 0, 0, 0, 0
            for item3 in self.dao.trainingSet_u[user]:
                if(self.dao.trainingSet_u[user][item3]==5.0 or self.dao.trainingSet_u[user][item3]==1.0) :
                    Minuend += sum(self.dao.trainingSet_i[item3].values())
                    index1 += len(self.dao.trainingSet_i[item3])
                else:
                    Subtrahend += sum(self.dao.trainingSet_i[item3].values())
                    index2 += len(self.dao.trainingSet_i[item3])
            if index1 == 0 and index2 == 0:
                self.FMTD[user] = 0
            elif index1 == 0:
                self.FMTD[user] = abs(Subtrahend / index2)
            elif index2 == 0:
                self.FMTD[user] = abs(Minuend / index1)
            else:
                self.FMTD[user] = abs(Minuend / index1 - Subtrahend / index2)

            if trainingIndex==(trainingUserCount/5):
                print('trainingData Done 20%...')
            elif trainingIndex==(trainingUserCount/5*2):
                print('trainingData Done 40%...')
            elif trainingIndex==(trainingUserCount/5*3):
                print('trainingData Done 60%...')
            elif trainingIndex==(trainingUserCount/5*4):
                print('trainingData Done 80%...')
            elif trainingIndex==(trainingUserCount):
                print('trainingData Done 100%...')

        # computing H,DegSim,LengVar,RDMA,FMTD for UnLabledData set
        for user in self.dao.testSet_u:
            testIndex += 1
            self.H[user] = 0
            for i in range(10,50,5):
                n = 0
                for item in self.dao.testSet_u[user]:
                    if(self.dao.testSet_u[user][item]==(i/10.0)):
                        n+=1
                if n==0:
                    self.H[user] += 0
                else:
                    self.H[user] += (-(n/(testUserCount*1.0))*math.log(n/(testUserCount*1.0),2))

            SimList = []
            self.DegSim[user] = 0
            for user1 in self.dao.testSet_u:
                userA, userB, C, D, E, Count = 0,0,0,0,0,0
                for item in list(set(self.dao.testSet_u[user]).intersection(set(self.dao.testSet_u[user1]))):
                    userA += self.dao.testSet_u[user][item]
                    userB += self.dao.testSet_u[user1][item]
                    Count += 1
                if Count==0:
                    AverageA = 0
                    AverageB = 0
                else:
                    AverageA = userA/Count
                    AverageB = userB/Count
                for item in list(set(self.dao.testSet_u[user]).intersection(set(self.dao.testSet_u[user1]))):
                    C += (self.dao.testSet_u[user][item]-AverageA)*(self.dao.testSet_u[user1][item]-AverageB)
                    D += np.square(self.dao.testSet_u[user][item]-AverageA)
                    E += np.square(self.dao.testSet_u[user1][item]-AverageB)
                if C==0:
                    SimList.append(0.0)
                else:
                    SimList.append(C/(math.sqrt(D)*math.sqrt(E)))
            SimList.sort(reverse=True)
            for i in range(1,self.k+1):
                self.DegSim[user] += SimList[i] / self.k

            GlobalAverage = 0
            F = 0
            for user2 in self.dao.testSet_u:
                GlobalAverage += len(self.dao.testSet_u[user2]) / (len(self.dao.testSet_u) + 0.0)
            for user3 in self.dao.testSet_u:
                F += pow(len(self.dao.testSet_u[user3])-GlobalAverage,2)
            self.LengVar[user] = abs(len(self.dao.testSet_u[user])-GlobalAverage)/(F*1.0)

            Divisor = 0
            for item1 in self.dao.testSet_u[user]:
                Divisor += abs(self.dao.testSet_u[user][item1]-self.dao.itemMeans[item1])/len(self.dao.testSet_i[item1])
            self.RDMA[user] = Divisor/len(self.dao.testSet_u[user])

            Minuend, index1, Subtrahend, index2= 0,0,0,0
            for item3 in self.dao.testSet_u[user]:
                if(self.dao.testSet_u[user][item3]==5.0 or self.dao.testSet_u[user][item3]==1.0):
                    Minuend += sum(self.dao.testSet_i[item3].values())
                    index1 += len(self.dao.testSet_i[item3])
                else:
                    Subtrahend += sum(self.dao.testSet_i[item3].values())
                    index2 += len(self.dao.testSet_i[item3])
            if index1 == 0 and index2 == 0:
                self.FMTD[user] = 0
            elif index1 == 0:
                self.FMTD[user] = abs(Subtrahend / index2)
            elif index2 == 0:
                self.FMTD[user] = abs(Minuend / index1)
            else:
                self.FMTD[user] = abs(Minuend / index1 - Subtrahend / index2)

            if testIndex == testUserCount / 5:
                 print('testData Done 20%...')
            elif testIndex == testUserCount / 5 * 2:
                print('testData Done 40%...')
            elif testIndex == testUserCount / 5 * 3:
                print('testData Done 60%...')
            elif testIndex == testUserCount / 5 * 4:
                print('testData Done 80%...')
            elif testIndex == testUserCount:
                print('testData Done 100%...')

        # preparing examples training for LabledData ,test for UnLableData

        for user in self.dao.trainingSet_u:
            self.training.append([self.H[user], self.DegSim[user], self.LengVar[user],self.RDMA[user],self.FMTD[user]])
            self.trainingLabels.append(self.labels[user])

        for user in self.dao.testSet_u:
            self.test.append([self.H[user], self.DegSim[user], self.LengVar[user],self.RDMA[user],self.FMTD[user]])
            self.testLabels.append(self.labels[user])

    def predict(self):
            ClassifierN = 0
            classifier = GaussianNB()
            X_train,X_test,y_train,y_test = train_test_split(self.training,self.trainingLabels,test_size=0.75,random_state=33)
            classifier.fit(X_train, y_train)
            # predict UnLabledData
            #pred_labelsForTrainingUn = classifier.predict(X_test)
            print('Enhanced classifier...')
            while 1:
                if len(X_test)<=5: # min
                    break         #min
                proba_labelsForTrainingUn = classifier.predict_proba(X_test)
                X_test_labels = np.hstack((X_test, proba_labelsForTrainingUn))
                X_test_labels0_sort = sorted(X_test_labels,key=lambda x:x[5],reverse=True)
                if X_test_labels0_sort[4][5]>X_test_labels0_sort[4][6]:
                    a = [x[:5] for x in X_test_labels0_sort]
                    b = a[0:5]
                    classifier.partial_fit(b, ['0','0','0','0','0'], classes=['0', '1'],sample_weight=np.ones(len(b), dtype=np.float) * self.Lambda)
                    X_test_labels = X_test_labels0_sort[5:]
                    X_test = a[5:]
                if len(X_test)<6: # min
                    break         #min

                X_test_labels0_sort = sorted(X_test_labels, key=lambda x: x[5], reverse=True)
                if X_test_labels0_sort[4][5]<=X_test_labels0_sort[4][6]: #min
                    a = [x[:5] for x in X_test_labels0_sort]
                    b = a[0:5]
                    classifier.partial_fit(b, ['1', '1', '1', '1', '1'], classes=['0', '1'],sample_weight=np.ones(len(b), dtype=np.float) * 1)
                    X_test_labels = X_test_labels0_sort[5:]  # min
                    X_test = a[5:]
                if len(X_test)<6:
                    break
            # while 1 :
            #     p1 = pred_labelsForTrainingUn
            #     # 将带λ参数的无标签数据拟合入分类器
            #     classifier.partial_fit(X_test, pred_labelsForTrainingUn,classes=['0','1'], sample_weight=np.ones(len(X_test),dtype=np.float)*self.Lambda)
            #     pred_labelsForTrainingUn = classifier.predict(X_test)
            #     p2 = pred_labelsForTrainingUn
            #     # 判断分类器是否稳定
            #     if list(p1)==list(p2) :
            #         ClassifierN += 1
            #     elif ClassifierN > 0:
            #         ClassifierN = 0
            #     if ClassifierN == 20:
            #         break
            pred_labels = classifier.predict(self.test)
            print('naive_bayes with EM algorithm:')
            return pred_labels
```

<!-- #region id="BJs1It7axh30" -->
## Main
<!-- #endregion -->

```python id="X7XKs5izyAzV"
class SDLib(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.relation = []
        self.measure = []
        self.config =config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.labels = FileIO.loadLabels(config['label'])

        if self.config.contains('evaluation.setup'):
            self.evaluation = LineConfig(config['evaluation.setup'])
            
            if self.evaluation.contains('-testSet'):
                #specify testSet
                self.trainingData = FileIO.loadDataSet(config, config['ratings'])
                self.testData = FileIO.loadDataSet(config, self.evaluation['-testSet'], bTest=True)

            elif self.evaluation.contains('-ap'):
                #auto partition
                self.trainingData = FileIO.loadDataSet(config,config['ratings'])
                self.trainingData,self.testData = DataSplit.\
                    dataSplit(self.trainingData,test_ratio=float(self.evaluation['-ap']))

            elif self.evaluation.contains('-cv'):
                #cross validation
                self.trainingData = FileIO.loadDataSet(config, config['ratings'])
                #self.trainingData,self.testData = DataSplit.crossValidation(self.trainingData,int(self.evaluation['-cv']))

        else:
            print('Evaluation is not well configured!')
            exit(-1)

        if config.contains('social'):
            self.socialConfig = LineConfig(self.config['social.setup'])
            self.relation = FileIO.loadRelationship(config,self.config['social'])
        print('preprocessing...')


    def execute(self):
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k <= 1 or k > 10:
                k = 3
            #create the manager used to communication in multiprocess
            manager = Manager()
            m = manager.dict()
            i = 1
            tasks = []
            for train,test in DataSplit.crossValidation(self.trainingData,k):
                fold = '['+str(i)+']'
                if self.config.contains('social'):
                    method = self.config['methodName'] + "(self.config,train,test,self.labels,self.relation,fold)"
                else:
                    method = self.config['methodName'] + "(self.config,train,test,self.labels,fold)"
               #create the process
                p = Process(target=run,args=(m,eval(method),i))
                tasks.append(p)
                i+=1
            #start the processes
            for p in tasks:
                p.start()
            #wait until all processes are completed
            for p in tasks:
                p.join()
            #compute the mean error of k-fold cross validation
            self.measure = [dict(m)[i] for i in range(1,k+1)]
            res = []
            pattern = re.compile('(\d+\.\d+)')
            countPattern = re.compile('\d+\\n')
            labelPattern = re.compile('\s\d{1}[^\.|\n|\d]')
            labels = re.findall(labelPattern, self.measure[0])
            values = np.array([0]*9,dtype=float)
            count = np.array([0,0,0],dtype=int)
            for report in self.measure:
                patterns = np.array(re.findall(pattern,report),dtype=float)
                values += patterns[:9]
                patterncounts = np.array(re.findall(countPattern,report),dtype=int)
                count += patterncounts[:3]
            values/=k
            values=np.around(values,decimals=4)
            res.append('             precision  recall  f1-score  support\n\n')
            res.append('         '+labels[0]+'  '+'    '.join(np.array(values[0:3],dtype=str).tolist())+'   '+str(count[0])+'\n')
            res.append('         '+labels[1]+'  '+'    '.join(np.array(values[3:6],dtype=str).tolist())+'   '+str(count[1])+'\n\n')
            res.append('  avg/total   ' + '    '.join(np.array(values[6:9], dtype=str).tolist()) + '   ' + str(count[2]) + '\n')
            print('Total:')
            print(''.join(res))
                # for line in lines[1:]:
                #
                # measure = self.measure[0][i].split(':')[0]
                # total = 0
                # for j in range(k):
                #     total += float(self.measure[j][i].split(':')[1])
                # res.append(measure+':'+str(total/k)+'\n')
            #output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = LineConfig(self.config['output.setup'])['-dir']
            fileName = self.config['methodName'] +'@'+currentTime+'-'+str(k)+'-fold-cv' + '.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The results have been output to '+abspath(LineConfig(self.config['output.setup'])['-dir'])+'\n')
        else:
            if self.config.contains('social'):
                method = self.config['methodName'] + '(self.config,self.trainingData,self.testData,self.labels,self.relation)'
            else:
                method = self.config['methodName'] + '(self.config,self.trainingData,self.testData,self.labels)'
            eval(method).execute()


def run(measure,algor,order):
    measure[order] = algor.execute()
```

```python colab={"base_uri": "https://localhost:8080/"} id="BUfH4niR1ZMR" executionInfo={"status": "ok", "timestamp": 1634216477304, "user_tz": -330, "elapsed": 2766, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba9bfc59-3f50-49f9-95d0-67145a1f9e73"
conf = Config('DegreeSAD.conf')
sd = SDLib(conf)
sd.execute()
```

```python colab={"base_uri": "https://localhost:8080/"} id="YmqtChCayQJa" executionInfo={"status": "ok", "timestamp": 1634220587417, "user_tz": -330, "elapsed": 302460, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="095593d8-d3f2-4018-c213-1e1e8bb14f8f"
print('='*80)
print('Supervised Methods:')
print('1. DegreeSAD   2.CoDetector   3.BayesDetector\n')
print('Semi-Supervised Methods:')
print('4. SemiSAD\n')
print('Unsupervised Methods:')
print('5. PCASelectUsers    6. FAP   7.timeIndex\n')
print('-'*80)
order = eval(input('please enter the num of the method to run it:'))

algor = -1
conf = -1

s = tm.clock()

if order == 1:
    conf = Config('DegreeSAD.conf')

elif order == 2:
    conf = Config('CoDetector.conf')

elif order == 3:
    conf = Config('BayesDetector.conf')

elif order == 4:
    conf = Config('SemiSAD.conf')

elif order == 5:
    conf = Config('PCASelectUsers.conf')

elif order == 6:
    conf = Config('FAP.conf')
elif order == 7:
    conf = Config('timeIndex.conf')

else:
    print('Error num!')
    exit(-1)

# conf = Config('DegreeSAD.conf')

sd = SDLib(conf)
sd.execute()
e = tm.clock()
print("Run time: %f s" % (e - s))
```

```python colab={"base_uri": "https://localhost:8080/"} id="XitDi26yz0U7" executionInfo={"status": "ok", "timestamp": 1634220613458, "user_tz": -330, "elapsed": 8805, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2c69731b-269b-4123-9516-dab8397f096d"
print('='*80)
print('Supervised Methods:')
print('1. DegreeSAD   2.CoDetector   3.BayesDetector\n')
print('Semi-Supervised Methods:')
print('4. SemiSAD\n')
print('Unsupervised Methods:')
print('5. PCASelectUsers    6. FAP   7.timeIndex\n')
print('-'*80)
order = eval(input('please enter the num of the method to run it:'))

algor = -1
conf = -1

s = tm.clock()

if order == 1:
    conf = Config('DegreeSAD.conf')

elif order == 2:
    conf = Config('CoDetector.conf')

elif order == 3:
    conf = Config('BayesDetector.conf')

elif order == 4:
    conf = Config('SemiSAD.conf')

elif order == 5:
    conf = Config('PCASelectUsers.conf')

elif order == 6:
    conf = Config('FAP.conf')
elif order == 7:
    conf = Config('timeIndex.conf')

else:
    print('Error num!')
    exit(-1)

# conf = Config('DegreeSAD.conf')

sd = SDLib(conf)
sd.execute()
e = tm.clock()
print("Run time: %f s" % (e - s))
```

```python colab={"base_uri": "https://localhost:8080/"} id="l8ErpBwoILs-" executionInfo={"status": "ok", "timestamp": 1634220649332, "user_tz": -330, "elapsed": 21280, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="89be0fb4-a135-4c5e-99a6-6a840260fd1d"
print('='*80)
print('Supervised Methods:')
print('1. DegreeSAD   2.CoDetector   3.BayesDetector\n')
print('Semi-Supervised Methods:')
print('4. SemiSAD\n')
print('Unsupervised Methods:')
print('5. PCASelectUsers    6. FAP   7.timeIndex\n')
print('-'*80)
order = eval(input('please enter the num of the method to run it:'))

algor = -1
conf = -1

s = tm.clock()

if order == 1:
    conf = Config('DegreeSAD.conf')

elif order == 2:
    conf = Config('CoDetector.conf')

elif order == 3:
    conf = Config('BayesDetector.conf')

elif order == 4:
    conf = Config('SemiSAD.conf')

elif order == 5:
    conf = Config('PCASelectUsers.conf')

elif order == 6:
    conf = Config('FAP.conf')
elif order == 7:
    conf = Config('timeIndex.conf')

else:
    print('Error num!')
    exit(-1)

# conf = Config('DegreeSAD.conf')

sd = SDLib(conf)
sd.execute()
e = tm.clock()
print("Run time: %f s" % (e - s))
```

```python colab={"base_uri": "https://localhost:8080/"} id="pdBA7drNI9GB" executionInfo={"status": "ok", "timestamp": 1634220727329, "user_tz": -330, "elapsed": 71038, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ce699c86-cc84-41d6-866c-afcee8e8c418"
print('='*80)
print('Supervised Methods:')
print('1. DegreeSAD   2.CoDetector   3.BayesDetector\n')
print('Semi-Supervised Methods:')
print('4. SemiSAD\n')
print('Unsupervised Methods:')
print('5. PCASelectUsers    6. FAP   7.timeIndex\n')
print('-'*80)
order = eval(input('please enter the num of the method to run it:'))

algor = -1
conf = -1

s = tm.clock()

if order == 1:
    conf = Config('DegreeSAD.conf')

elif order == 2:
    conf = Config('CoDetector.conf')

elif order == 3:
    conf = Config('BayesDetector.conf')

elif order == 4:
    conf = Config('SemiSAD.conf')

elif order == 5:
    conf = Config('PCASelectUsers.conf')

elif order == 6:
    conf = Config('FAP.conf')
elif order == 7:
    conf = Config('timeIndex.conf')

else:
    print('Error num!')
    exit(-1)

# conf = Config('DegreeSAD.conf')

sd = SDLib(conf)
sd.execute()
e = tm.clock()
print("Run time: %f s" % (e - s))
```
