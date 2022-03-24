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

<!-- #region id="8-aWDAVNdOAd" -->
# Attribute to Feature Mappings for Cold-Start Recommendations
<!-- #endregion -->

<!-- #region id="FK65Ha4JdMRr" -->
## Datasets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XqKHrO5gMvYN" executionInfo={"status": "ok", "timestamp": 1635664489529, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4bb480cf-3e42-4a67-a237-e44f2f37ea98"
%%writefile attribute.txt
0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0
0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0
0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0
0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0
0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0
0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0
0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0
```

```python colab={"base_uri": "https://localhost:8080/"} id="nqvcTHM_M2NG" executionInfo={"status": "ok", "timestamp": 1635664508883, "user_tz": -330, "elapsed": 492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fd83669d-c9e4-4c2c-d106-c79407e501ec"
%%writefile feedback.txt
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0
1 0 0 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 1 0 1 1 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0
0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 0 1 0 1 0 1 1 1 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 0 0 1 1 0 1 1 0 0 1 0 0 0 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0
0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 0 1 1 0 1 1 1 0 1 1 1 1 0 0 1 0 0 0 1 1 1 1 1 0 1 1 1 0 0 1 1 0 0 0 1 1 1 1 0 1 0 0 1
0 0 0 0 0 0 1 0 1 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
1 0 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 0 1 1 1 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0
1 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 1 0 1 1 0 0 1 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1
0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

<!-- #region id="PqZgnqfbNrWW" -->
## Imports
<!-- #endregion -->

```python id="JUWFMgwFM7Jx"
import numpy as np
from math import exp, log
from copy import deepcopy
import random
import scipy.sparse as sp
import sys
from math import sqrt, exp
import scipy.sparse as sp
from copy import copy
import functools
```

<!-- #region id="rMst_GnmNs5k" -->
## BPR
<!-- #endregion -->

```python id="qKEWHiZzNAXI"
class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors
```

```python id="aF2us0JNNDiR"
class BPR(object):

    def __init__(self,D,args):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors

    def train(self,dataidx,num_items,sampler,num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        self.init(dataidx,num_items)

        #print 'initial loss = {0}'.format(self.loss())
        for it in range(num_iters):
            #print 'starting iteration {0}'.format(it)
            for u,i,j in sampler.generate_samples(self.dataidx, self.num_items):
                self.update_factors(u,i,j)
            print('iteration {0}: loss = {1}'.format(it,self.loss()))

    def init(self,dataidx,num_items):
        self.dataidx = dataidx
        self.num_users = len(dataidx)
        self.num_items = num_items

        self.item_bias = np.zeros(self.num_items)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))

        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)

        sampler = UniformUserUniformItem()
        self.loss_samples = [t for t in sampler.generate_samples(self.dataidx,self.num_items,num_loss_samples)]

    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = self.update_negative_item_factors

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])

        #XXX: maybe it should be exp(-x)/(1.0+exp(-x))
        #z = 1.0/(1.0+exp(x))
        z = 1.0 - 1.0/(1.0+exp(-x))

        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            #it should be ln(1.0/(1.0+exp(-x)) according to thesis)
            #ranking_loss += 1.0/(1.0+exp(x))
            ranking_loss += log(1.0/(1.0+exp(-x)))

        complexity = 0;
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        #XXX: where does 0.5 come from? returns negative BPR-OPT so that it looks we are minimizing it
        #return ranking_loss + 0.5*complexity
        return -ranking_loss + complexity

    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])
```

<!-- #region id="LJwogO6VNueC" -->
## Sampling
<!-- #endregion -->

```python id="RFrUnJzrNFZY"
class Sampler(object):

    def __init__(self):
        pass

    def init(self,dataidx,num_items,max_samples=None):
        self.dataidx = dataidx
        self.num_users = len(dataidx)
        self.num_items = num_items
        self.max_samples = max_samples
        self.datannz = 0
        for u in range(self.num_users):
            self.datannz += len(dataidx[u])

    def sample_user(self):
        u = self.uniform_user()
        num_pos = len(self.dataidx[u])
        assert(num_pos > 0 and num_pos != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = random.randint(0,self.num_items-1)
        while j in user_items:
            j = random.randint(0,self.num_items-1)
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)
```

```python id="iOtjZlvzNO8G"
class UniformUserUniformItem(Sampler):

    def generate_samples(self,dataidx,num_items,max_samples=None):
        self.init(dataidx,num_items,max_samples)
        for _ in range(self.num_samples(self.datannz)):
            u = self.uniform_user()
            indices = self.dataidx[u]
            # sample positive item
            num_pos = len(indices)
            if (num_pos<=0 or num_pos==self.num_items):
                #throw bad user samples out
                continue
            i = random.choice(indices)
            j = self.sample_negative_item(indices)
            yield u,i,j
```

```python id="w7IzDuepNTFk"
class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self,dataidx,num_items,max_samples=None):
        self.init(dataidx,num_items,max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_dataidx = deepcopy(self.dataidx)
        for _ in range(self.num_samples(self.datannz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_dataidx[u]
            if user_items.size == 0:
                if self.dataidx[u].size == 0:
                    continue
                # reset user data if it's all been sampled
                self.local_dataidx[u] = self.dataidx[u].copy()
                user_items = self.local_dataidx[u]
            i = random.randint(0,user_items.size-1)
            # forget this item so we don't sample it again for the same user
            self.local_dataidx[u] = np.delete(user_items,i)
            j = self.sample_negative_item(user_items)
            yield u,i,j
```

```python id="1MfpIS0rNRyT"
class ExternalSchedule(Sampler):

    def __init__(self,filepath,index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self,dataidx,num_items,max_samples=None):
        self.init(dataidx,num_items,max_samples)
        f = open(self.filepath)
        samples = [map(int,line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u,i,j in samples[:num_samples]:
            yield u-self.index_offset,i-self.index_offset,j-self.index_offset
```

<!-- #region id="0vbXSv2wNUGP" -->
## Splitting
<!-- #endregion -->

```python id="GvCmyDkPN2ZQ"
class DataSplitter(object):

    def __init__(self, datamat, attrmat, k):
        assert sp.isspmatrix_csc(datamat)
        self.datamat = datamat
        self.attrmat = attrmat
        self.k = k
        _, self.num_items = datamat.shape
        assert self.k<=self.num_items
        self.index = [i for i in range(self.num_items)]
        #random.shuffle(self.index)

    def split_data(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(int(min(self.num_items-base, self.num_items/self.k))):
                tmp.append(self.datamat.getcol(self.index[base+j]))
            base = base + int(self.num_items/self.k)
            result.append(sp.hstack(tmp,"csc"))
        return result

    def split_attr(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(int(min(self.num_items-base, self.num_items/self.k))):
                tmp.append(self.attrmat[self.index[base+j]])
            base = base + int(self.num_items/self.k)
            result.append(np.vstack(tmp))
        return result
```

<!-- #region id="uzmbWdO1N-JB" -->
## Parsing
<!-- #endregion -->

```python id="rFd4dK5uN3GN"
import sys

class DataParser(object):

    def __init__(self):
        pass

    def init(self, filename, k):
        self.filename = filename
        self.attrfile = attrfile
        self.k = k

class Ml_100_Parser(DataParser):
    
    def parse(self, filename, k):
        pass
                
    def split(self, data):
        pass

class Ml_1M_Parser(DataParser):

    def parse(self, filename, k):
        pass

    def split(self, data):
        pass

if __name__ == '__main__':
    pass
    # example of training and testing with mapping functions
```

<!-- #region id="hTX-lVdPON2T" -->
## Mapping
<!-- #endregion -->

```python id="srDPFqpEVbey"
import operator as op

def cmp(a, b):
    # return (a > b) - (a < b) 
    x = int(op.gt(a,b))#.astype(np.float32)
    y = int(op.lt(a,b))#.astype(np.float32)
    return x - y
```

```python id="dnrSp7IEOOut"
class Mapper(object):

    def __init__(self):
        pass

    def init(self, data, attr, bpr_k=None, bpr_args=None, bpr_model=None):
        assert sp.isspmatrix_csc(data)
        self.data = data
        self.num_users, self.num_items = data.shape
        self.attr = attr
        assert attr.shape[0] >= self.num_items
        _, self.num_attrs = attr.shape
        if bpr_model==None:
            self.bpr_k = [self.num_users/5,bpr_k][bpr_k!=None]
            if bpr_args==None:
                self.bpr_args = BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
            else:
                self.bpr_args = bpr_args
            self.bpr_model = BPR(self.bpr_k, self.bpr_args)
        else:
            self.bpr_model = bpr_model
            self.bpr_k = bpr_model.D
            self.bpr_args = BPRArgs(bpr_model.learning_rate, \
                bpr_model.bias_regularization, \
                bpr_model.user_regularization, \
                bpr_model.positive_item_regularization, \
                bpr_model.negative_item_regularization, \
                bpr_model.update_negative_item_factors)
        self.sampler = UniformUserUniformItem()
    
    def train_init(self):
        tmp = self.data.tocsr()
        self.dataidx = []
        for u in range(self.num_users):
            self.dataidx.append(tmp[u].indices)

    def test_init(self, test_data, test_attr):
        assert sp.isspmatrix_csc(test_data)
        self.num_test_items, _ = test_attr.shape
        tmp = test_data.tocsr()
        self.test_attr = test_attr
        self.test_dataidx = []
        for u in range(self.num_users):
            self.test_dataidx.append(tmp[u].indices)

    def cos_similarity(self, i):
        try:
            assert len(self.attr_sqr_cache) == len(self.attr)
            assert len(self.tattr_sqr_cache) == len(self.test_attr)
        except:
            self.attr_sqr_cache = []
            self.tattr_sqr_cache = []
            for j in range(self.num_items):
                self.attr_sqr_cache.append(sqrt(np.dot(self.attr[j], self.attr[j])))
            for j in range(self.num_test_items):
                self.tattr_sqr_cache.append(sqrt(np.dot(self.test_attr[j], self.test_attr[j])))

        similarity = []
        for j in range(self.num_items):
            similarity.append(np.dot(self.test_attr[i], self.attr[j]) / (self.tattr_sqr_cache[i] * self.attr_sqr_cache[j]))
        return similarity

    def accuracy(self, threshold=0.5):
    #XXX: bpr models have no range bound, while its focus are pair-wise relationships, so it's hard to set a threshold and test accuracy
        result = 0.0
        for i in range(self.num_test_items):
            pred_i = self.test_predict(i)
            for u in range(self.num_users):
                posidx = self.test_dataidx[u]
                if (pred_i[u]>=threshold and (i in posidx)) or (pred_i[u]<threshold and (not i in posidx)):
                    result += 1
        result /= (self.num_items * self.num_users)
        return result

    def prec_at_n(self, prec_n):
        #precision of top-n recommended results, average across users
        assert prec_n <= self.num_test_items
        result = 0
        
        cand = [[] for i in range(self.num_users)]
        for i in range(self.num_test_items):
            pred_i = self.test_predict(i)
            for u in range(self.num_users):
                cand[u].append((pred_i[u], i))
        for u in range(self.num_users):
            keyfunc = functools.cmp_to_key(lambda x,y : cmp(x[0],y[0]))
            cand[u].sort(key=keyfunc, reverse=True)
            tmp = 0.0
            row_u = self.test_dataidx[u]
            for i in range(prec_n):
                if cand[u][i][1] in row_u:
                    tmp += 1
            result += tmp/prec_n
        result /= self.num_users
        return result

    def auc(self):
        #area under ROC curve, compute , average across users
        result = 0
        pred = [[] for i in range(self.num_users)]
        for i in range(self.num_test_items):
            pred_i = self.test_predict(i)
            for u in range(self.num_users):
                pred[u].append(pred_i[u])
        for u in range(self.num_users):
            tmp = 0.0
            posidx = self.test_dataidx[u]
            for j in range(self.num_test_items):
                if j in posidx:
                    continue
                for i in posidx:
                    if pred[u][i]-pred[u][j]>=0:
                        tmp += 1
            real_pos = len(posidx)
            result += tmp/max(real_pos, 1)/max(self.num_test_items-real_pos, 1)
        result /= self.num_users
        return result 

    def cross_validation(self, cv_num_iters, cv_set, cv_folds):
        origin_data = self.data
        origin_attr = self.attr
        origin_model = self.bpr_model
        splitter = DataSplitter(origin_data, origin_attr, cv_folds)
        datamats = splitter.split_data()
        attrmats = splitter.split_attr()
        bestscore = 0.0
        bestpara = None
        for para in cv_set:
            self.set_parameter(para)
            avg_score = 0.0
            print("Cross-validating parameter",para,".........")
            for i in range(cv_folds):
                tmp_data = copy(datamats)
                tmp_data.pop(i)
                tmp_attr = copy(attrmats)
                tmp_attr.pop(i)
                self.init(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), self.bpr_k, self.bpr_args)
                self.train(cv_num_iters)
                self.test_init(datamats[i], attrmats[i])
                #avg_score += self.accuracy()
                cur_score = self.prec_at_n(5)
                print("prec@5 of cross-validation fold",i,":",cur_score)
                avg_score += cur_score
            avg_score /= cv_folds
            print("Average score for parameter after cross-validation",para,":",avg_score)
            if (avg_score > bestscore):
                bestpara = para
                bestscore = avg_score
        #print("best parameter in cross-validation :", bestpara, "with accuracy", bestscore)
        print("best parameter in cross-validation :", bestpara, "with prec@n", bestscore)
        self.init(origin_data, origin_attr, None, None, origin_model)
        return para

class Map_KNN(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, k=1):
        self.init(data, attr, bpr_k, bpr_args)
        self.k = k

    def set_parameter(self, k):
        self.k = k

    def train(self, num_iters):
        self.train_init()
        self.bpr_model.train(self.dataidx, self.num_items, self.sampler, num_iters)
        
    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr)
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        cos_sim = self.cos_similarity(i)
        cand = [(cos_sim[i], i) for i in range(self.num_items)]
        keyfunc = functools.cmp_to_key(lambda x,y: cmp(x[0],y[0]))
        cand.sort(key=keyfunc, reverse=True)
        #average new h from top-k h vectors, and predict with bpr
        i_factors = np.zeros(self.bpr_k)
        i_bias = 0
        for j in range(self.k):
            i_factors += cand[j][0] * self.bpr_model.item_factors[cand[j][1],:]
            i_bias += cand[j][0] * self.bpr_model.item_bias[cand[j][1]]
        sim_sum = sum(cand[j][0] for j in range(self.k))
        i_factors /= sim_sum
        i_bias /= sim_sum
        for u in range(self.num_users):
            result.append(i_bias + np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

class Map_Linear(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

    def train(self, num_iters):
        self.train_init()
        self.bpr_model.train(self.dataidx, self.num_items, self.sampler, num_iters)
        #train linear models for bpr_k column across attributes(X=attrs, Y=H[u])
        self.mapper_factors = np.random.random_sample((self.bpr_k, self.num_attrs))
        self.mapper_bias = np.zeros(self.bpr_k)
        self.mapper_factors_b = np.random.random_sample(self.num_attrs)
        self.mapper_bias_b = 0
        for it in range(num_iters):
            print("Mapper Map_Linear trainning for iteration",it,"...")
            diff = np.dot(self.attr, self.mapper_factors.transpose()) + np.dot(np.ones((self.num_items,1)), self.mapper_bias.reshape((1,self.bpr_k))) \
                - self.bpr_model.item_factors 
            self.mapper_factors -= self.learning_rate/self.num_items*(np.dot(diff.transpose(), self.attr)+self.penalty_factor*self.mapper_factors)
            self.mapper_bias -= self.learning_rate/self.num_items*(np.dot(diff.transpose(), np.ones(self.num_items)))
            diff_b = np.dot(self.attr, self.mapper_factors_b) + self.mapper_bias_b*np.ones(self.num_items)-self.bpr_model.item_bias
            self.mapper_factors_b -= self.learning_rate/self.num_items*(np.dot(diff_b, self.attr) + self.penalty_factor*self.mapper_factors_b)
            self.mapper_bias_b -= self.learning_rate/self.num_items*(np.dot(diff_b, np.ones(self.num_items)))

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr) 
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        i_factors = self.mapper_bias + np.dot(self.mapper_factors, self.test_attr[i])
        i_bias = self.mapper_bias_b + np.dot(self.mapper_factors_b, self.test_attr[i])
        for u in range(self.num_users):
            result.append(i_bias + np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

    def set_parameter(self, para_set):
        self.learning_rate = para_set[0]
        self.penalty_factor = para_set[1]

class Map_BPR(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

    def train(self, num_iters):
        self.train_init()
        self.bpr_model.train(self.dataidx, self.num_items, self.sampler, num_iters)
        #train linear models for bpr_k column across attributes(X=attrs, Y=H[u])
        self.mapper_factors = np.random.random_sample((self.bpr_k, self.num_attrs))
        #self.mapper_bias = np.zeros((self.bpr_k, 1))
        #self.mapper_factors_b = np.random.random_sample(self.num_attrs)
        #self.mapper_bias_b = np.zeros(1)
        for it in range(num_iters):
            print("Mapper Map_BPR trainning for iteration",it,"...")
            for u,i,j in self.sampler.generate_samples(self.dataidx, self.num_items):
                x_uij = self.predict(u,i) - self.predict(u,j)
                #XXX: maybe it should be exp(-x)/(1.0+exp(-x))
                #z = 1.0/(1.0+exp(x_uij))
                z = 1.0 - 1.0/(1.0+exp(-x_uij))
                u_factor = (self.bpr_model.user_factors[u,:]).reshape((self.bpr_k, 1))
                ij_diff = (self.attr[i]-self.attr[j]).reshape((1, self.num_attrs))

                gradient = z * np.dot(u_factor, ij_diff) 
                self.mapper_factors = self.learning_rate * ( \
                    gradient - self.penalty_factor * self.mapper_factors )
                #self.mapper_bias = self.learning_rate * ( \
                #    z * u_factor \
                #    - self.penalty_factor * self.mapper_bias )

    def predict(self, u, i):
        return np.dot( self.bpr_model.user_factors[u,:] \
            , np.dot(self.mapper_factors, self.attr[i]) )
            #\+self.mapper_bias )

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr) 
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        i_factors = np.dot(self.mapper_factors, self.test_attr[i])
            #\+self.mapper_bias
        #no i_bias here because we didn't use actual h_i in trainning
        for u in range(self.num_users):
            result.append(np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

    def set_parameter(self, para_set):
        self.learning_rate = para_set[0]
        self.penalty_factor = para_set[1]

class CBF_KNN(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, k=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.k = k

    def set_parameter(self, k):
        self.k = k

    def train(self, num_iters):
        # underlying bpr model is useless, so no need to train
        self.train_init()
        pass 

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr)
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        cos_sim = self.cos_similarity(i)
        if self.k==None:
            # k is infinity by default
            for u in range(self.num_users):
                pred_j = 0
                for j in self.dataidx[u]:
                    pred_j += cos_sim[j]
                result.append(pred_j)
        else:
            for u in range(self.num_users):
                cand = []
                for j in self.dataidx[u]:
                    cand.append(cos_sim[j])
                cand.sort()
                pred_j = 0
                for j in range(self.k):
                    pred_j += cand[j]
                result.append(pred_j)
        return result

class Map_Random(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None):
        self.init(data, attr, bpr_k, bpr_args)

    def train(self, num_iters):
        #no need to train
        pass

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr)
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i, max_score=1.0):
        return [(random.random() * max_score) for i in range(self.num_users)]
```

<!-- #region id="18rynne1Q35g" -->
## Main
<!-- #endregion -->

```python id="FwCVHDGGby56"
def main(model_id):
    #all parameters needed setting are here
    num_folds = 3
    bpr_args = BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
    bpr_k = 24
    cv_iters = 10
    cv_folds = 3
    num_iters = 20

    data = sp.csc_matrix(np.loadtxt('feedback.txt'))
    attr = np.loadtxt('attribute.txt')
    splitter = DataSplitter(data, attr, num_folds)
    datamats = splitter.split_data()
    attrmats = splitter.split_attr()

    assert num_folds>1
    assert cv_folds>1
    avg_prec = 0
    avg_auc = 0

    #training & testing
    for i in range(num_folds):
        tmp_data = copy(datamats)
        tmp_data.pop(i)
        tmp_attr = copy(attrmats)
        tmp_attr.pop(i)

        if (model_id == 0):
            cv_parameter_set = [(0.03,0.03), (0.03,0.1), (0.1,0.03), (0.1,0.1)]
            model = Map_BPR(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 1):
            cv_parameter_set = [(0.03,0.03), (0.03,0.1), (0.1,0.03), (0.1,0.1)]
            model = Map_Linear(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 2):
            cv_parameter_set = [1, 2, 3]
            model = Map_KNN(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 3):
            model = CBF_KNN(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 4):
            model = Map_Random(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)

        if (model_id<3):
            para = model.cross_validation(cv_iters, cv_parameter_set, cv_folds)
            model.set_parameter(para)
        model.train(num_iters)

        prec, auc = model.test(datamats[i], attrmats[i])
        print("Test for fold",i,": Prec@n =",prec,"auc =",auc)
        print("------------------------------------------------")
        avg_prec += prec
        avg_auc += auc
    print("avg_prec = ", avg_prec/num_folds, ", avg_auc = ", avg_auc/num_folds)
```

<!-- #region id="DATZJ_XbcIau" -->
## Runs
<!-- #endregion -->

<!-- #region id="2uCSTKw9OiYr" -->
0=Map_BPR 1=Map_Linear 2=Map_KNN 3=CBF_KNN 4=Random
<!-- #endregion -->

<!-- #region id="nceJJCR5cKyj" -->
### Random
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gLLzDBTPVj-V" executionInfo={"status": "ok", "timestamp": 1635668537125, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7fdcbe36-6942-4777-b76a-1fb7c30b9f54"
main(model_id=4)
```

<!-- #region id="iYum03z-QVfY" -->
### CBF-KNN
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GddVLNjWTm2P" executionInfo={"status": "ok", "timestamp": 1635668581409, "user_tz": -330, "elapsed": 586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="222097f6-6a59-4813-f67b-c75d3dcabd26"
main(model_id=3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="yQJcklOCcdcl" executionInfo={"status": "ok", "timestamp": 1635668689047, "user_tz": -330, "elapsed": 3898, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d7bd7ad5-4851-43e4-8bd3-66140106256c"
### Map-KNN
main(model_id=2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="vcML2MnUciyz" executionInfo={"status": "ok", "timestamp": 1635668710033, "user_tz": -330, "elapsed": 5428, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dc54929b-feca-481f-eeb5-271b4a58479b"
### Map-Linear
main(model_id=1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hqnw3Wo2cjQq" executionInfo={"status": "ok", "timestamp": 1635668721170, "user_tz": -330, "elapsed": 7501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6105e36b-f928-4d21-b888-e0b59242c3ac"
### Map-BPR
main(model_id=0)
```

<!-- #region id="_3HapHXLEvIR" -->
## Extra Notes
<!-- #endregion -->

<!-- #region id="zYRLWJ3bExZM" -->
### Example
<!-- #endregion -->

<!-- #region id="bgv8w1qEEyoG" -->
Training a hypothetical factorization model with k = 2 yields two matrices consisting of the user and item factor vectors, respectively:
<!-- #endregion -->

<!-- #region id="9pLxFvzAmZBC" -->
<p><center><img src='_images/T847725_1.png'></center></p>
<!-- #endregion -->

<!-- #region id="9-W2IlQqE3Kl" -->
### Loss function

The general form of score estimation by mapping from item attributes to item factors is:

$$\hat{y}_{ui} := \sum_{f=1}^k w_{uf}\phi_f(a_i^I) = \langle w_u,\phi(a_i^I) \rangle$$
<!-- #endregion -->

<!-- #region id="nVG0k-P-CuX8" -->
## Citations

Learning Attribute to Feature Mappings for Cold-Start Recommendations. Lucas Drumond, Christoph Freudenthaler, Steffen Rendle, Lars Schmidt-Thieme. 2010. ICDM. [https://bit.ly/3Eh4NEK](https://bit.ly/3Eh4NEK)
<!-- #endregion -->
