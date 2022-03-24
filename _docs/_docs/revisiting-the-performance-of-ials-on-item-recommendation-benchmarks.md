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

```python id="a6dr9dm975ah"
!wget https://github.com/hexiangnan/neural_collaborative_filtering/archive/master.zip
!unzip master.zip
!mv neural_collaborative_filtering-master/* ./
```

```python id="uSqdAibP8wF9" executionInfo={"status": "ok", "timestamp": 1635510315611, "user_tz": -330, "elapsed": 874, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import scipy.sparse as sp
import numpy as np
import concurrent.futures
import argparse
from collections import defaultdict
import math
import heapq # for retrieval topK
import multiprocessing
from time import time
```

```python id="2t7yWU929anA" executionInfo={"status": "ok", "timestamp": 1635510524082, "user_tz": -330, "elapsed": 888, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
```

```python id="1BPDa4kN78OC" executionInfo={"status": "ok", "timestamp": 1635510525107, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Dataset(object):

    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
```

```python id="T301-0Aq9BBu" executionInfo={"status": "ok", "timestamp": 1635510525108, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class IALSDataset():
  """A class holding the train and test data."""

  def __init__(self, train_by_user, train_by_item, test, num_batches):
    """Creates a DataSet and batches it.
    Args:
      train_by_user: list of (user, items)
      train_by_item: list of (item, users)
      test: list of (user, history_items, target_items)
      num_batches: partitions each set using this many batches.
    """
    self.train_by_user = train_by_user
    self.train_by_item = train_by_item
    self.test = test
    self.num_users = len(train_by_user)
    self.num_items = len(train_by_item)
    self.user_batches = self._batch(train_by_user, num_batches)
    self.item_batches = self._batch(train_by_item, num_batches)
    self.test_batches = self._batch(test, num_batches)

  def _batch(self, xs, num_batches):
    batches = [[] for _ in range(num_batches)]
    for i, x in enumerate(xs):
      batches[i % num_batches].append(x)
    return batches


def map_parallel(fn, xs, *args):
  """Applies a function to a list, equivalent to [fn(x, *args) for x in xs]."""
  if len(xs) == 1:
    return [fn(xs[0], *args)]

  num_threads = len(xs)
  executor = concurrent.futures.ProcessPoolExecutor(num_threads)
  futures = [executor.submit(fn, x, *args) for x in xs]
  concurrent.futures.wait(futures)
  results = [future.result() for future in futures]
  return results


class Recommender():
  """A Recommender class used to evaluate a recommendation algorithm.
  Inheriting classes must implement the score() method.
  """

  def _evaluate_user(self, user_history, ground_truth, exclude):
    """Evaluates one user.
    Args:
      user_history: list of items to use in the projection.
      ground_truth: list of target items.
      exclude: list of items to exclude, usually the same as ground_truth.
    Returns:
      A tuple of (Recall@20, Recall@50 and nDCG@100).
    """
    scores = self.score(user_history)
    scores[exclude] = -np.infty
    topk = np.argsort(scores)[::-1]

    def recall(k, gt_set, topk):
      result = 0.0
      for i in range(k):
        if topk[i] in gt_set:
          result += 1
      return result / min(k, len(gt_set))

    def ndcg(k, gt_set, topk):
      result = 0.0
      norm = 0.0
      for i in range(k):
        if topk[i] in gt_set:
          result += 1.0/np.log2(i+2)
      for i in range(min(k, len(gt_set))):
        norm += 1.0/np.log2(i+2)
      return result / norm

    gt_set = ground_truth
    return np.array([
        recall(20, gt_set, topk), recall(50, gt_set, topk),
        ndcg(100, gt_set, topk)
        ])

  def _evaluate_users(self, users):
    """Evaluates a set of users.
    Args:
      users: a list of users, where each user is a tuple
        (id, history, ground truth).
    Returns:
      A dict mapping user id to a tuple of (Recall@20, Recall@50, nDCG@100).
    """
    metrics = {}
    for user_id, ground_truth, history in users:
      if set(ground_truth) & set(history):
        raise ValueError("The history and ground_truth must be disjoint.")
      metrics[user_id] = self._evaluate_user(history, ground_truth, history)
    return metrics

  def evaluate(self, users_batches):
    results = map_parallel(self._evaluate_users, users_batches)
    all_metrics = []
    for r in results:
      all_metrics.extend(list(r.values()))
    return np.mean(all_metrics, axis=0)


class IALS(Recommender):
  """iALS solver."""

  def __init__(self, num_users, num_items, embedding_dim, reg,
               unobserved_weight, stddev):
    self.embedding_dim = embedding_dim
    self.reg = reg
    self.unobserved_weight = unobserved_weight
    self.user_embedding = np.random.normal(
        0, stddev, (num_users, embedding_dim))
    self.item_embedding = np.random.normal(
        0, stddev, (num_items, embedding_dim))
    self._update_user_gramian()
    self._update_item_gramian()

  def _update_user_gramian(self):
    self.user_gramian = np.matmul(self.user_embedding.T, self.user_embedding)

  def _update_item_gramian(self):
    self.item_gramian = np.matmul(self.item_embedding.T, self.item_embedding)

  def score(self, user_history):
    user_emb = project(
        user_history, self.item_embedding, self.item_gramian, self.reg,
        self.unobserved_weight)
    result = np.dot(user_emb, self.item_embedding.T)
    return result

  def train(self, ds):
    """Runs one iteration of the IALS algorithm.
    Args:
      ds: a DataSet object.
    """
    # Solve for the user embeddings
    self._solve(ds.user_batches, is_user=True)
    self._update_user_gramian()
    # Solve for the item embeddings
    self._solve(ds.item_batches, is_user=False)
    self._update_item_gramian()

  def _solve(self, batches, is_user):
    """Solves one side of the matrix."""
    if is_user:
      embedding = self.user_embedding
      args = (self.item_embedding, self.item_gramian, self.reg,
              self.unobserved_weight)
    else:
      embedding = self.item_embedding
      args = (self.user_embedding, self.user_gramian, self.reg,
              self.unobserved_weight)
    results = map_parallel(solve, batches, *args)
    for r in results:
      for user, emb in r.items():
        embedding[user, :] = emb


def project(user_history, item_embedding, item_gramian, reg, unobserved_weight):
  """Solves one iteration of the iALS algorithm."""
  if not user_history:
    raise ValueError("empty user history in projection")
  emb_dim = np.shape(item_embedding)[1]
  lhs = np.zeros([emb_dim, emb_dim])
  rhs = np.zeros([emb_dim])
  for item in user_history:
    item_emb = item_embedding[item]
    lhs += np.outer(item_emb, item_emb)
    rhs += item_emb

  lhs += unobserved_weight * item_gramian
  lhs = lhs + np.identity(emb_dim) * reg
  return np.linalg.solve(lhs, rhs)


def solve(data_by_user, item_embedding, item_gramian, global_reg,
          unobserved_weight):
  user_embedding = {}
  for user, items in data_by_user:
    reg = global_reg *(len(items) + unobserved_weight * item_embedding.shape[0])
    user_embedding[user] = project(
        items, item_embedding, item_gramian, reg, unobserved_weight)
  return user_embedding
```

```python id="_zs6-GKG_wRX" executionInfo={"status": "ok", "timestamp": 1635510525109, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
```

```python id="L7NRk5sm9f9Z" executionInfo={"status": "ok", "timestamp": 1635510525111, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class MFModel(IALS):

  def _predict_one(self, user, item):
    """Predicts the score of a user for an item."""
    return np.dot(self.user_embedding[user],
                  self.item_embedding[item])

  def predict(self, pairs, batch_size, verbose):
    """Computes predictions for a given set of user-item pairs.
    Args:
      pairs: A pair of lists (users, items) of the same length.
      batch_size: unused.
      verbose: unused.
    Returns:
      predictions: A list of the same length as users and items, such that
      predictions[i] is the models prediction for (users[i], items[i]).
    """
    del batch_size, verbose
    num_examples = len(pairs[0])
    assert num_examples == len(pairs[1])
    predictions = np.empty(num_examples)
    for i in range(num_examples):
      predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
    return predictions
```

```python id="zTIw3ShbAoAO" executionInfo={"status": "ok", "timestamp": 1635510525112, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def evaluate(model, test_ratings, test_negatives, K=10):
  """Helper that calls evaluate from the NCF libraries."""
  (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, K=K,
                                 num_thread=1)
  return np.array(hits).mean(), np.array(ndcgs).mean()
```

```python id="vaVNyFOUAmdu" executionInfo={"status": "ok", "timestamp": 1635510525530, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Args:
    # Path to the dataset
    data = 'Data/ml-1m'
    # Number of training epochs
    epochs = 128
    # Embedding dimensions, the first dimension will be used for the bias
    embedding_dim = 8
    # L2 regularization for user and item embeddings
    regularization = 0.0
    # Weight for unobserved pairs
    unobserved_weight = 1.0
    # Standard deviation for initialization
    stddev = 0.1

args = Args()
```

```python id="uUFXzrqvA5Im" executionInfo={"status": "ok", "timestamp": 1635510525532, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args.epochs = 12
args.embedding_dim = 192
args.regularization = 0.007
args.unobserved_weight = 0.3
args.stddev = 0.1
```

```python colab={"base_uri": "https://localhost:8080/"} id="DefLVhgKAlTh" executionInfo={"status": "ok", "timestamp": 1635513740072, "user_tz": -330, "elapsed": 3214050, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="390e34de-fb36-425e-8dfc-6feeb22d5feb"
# Load the dataset
dataset = Dataset(args.data)
train_pos_pairs = np.column_stack(dataset.trainMatrix.nonzero())
test_ratings, test_negatives = (dataset.testRatings, dataset.testNegatives)
print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' % (
    dataset.num_users, dataset.num_items, train_pos_pairs.shape[0],
    len(test_ratings)))

train_by_user = defaultdict(list)
train_by_item = defaultdict(list)

for u, i in train_pos_pairs:
    train_by_user[u].append(i)
    train_by_item[i].append(u)

train_by_user = list(train_by_user.items())
train_by_item = list(train_by_item.items())

train_ds = IALSDataset(train_by_user, train_by_item, [], 1)

# Initialize the model
model = MFModel(dataset.num_users, dataset.num_items,
                args.embedding_dim, args.regularization,
                args.unobserved_weight,
                args.stddev / np.sqrt(args.embedding_dim))

# Train and evaluate model
hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
    % (0, hr, ndcg))
for epoch in range(args.epochs):
    # Training
    _ = model.train(train_ds)

# Evaluation
hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
        % (epoch+1, hr, ndcg))
```
