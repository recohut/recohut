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

```python id="rxL9QQCZPx8Q"
import os
project_name = "reco-chef"; branch = "30music"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="dwLTtRQK88Yz" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630906802490, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0a8591e-0a53-4a78-f850-e0f5f841d4aa"
if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout "{branch}"
else:
    %cd "{project_path}"
```

```python id="2jrtc9Bg88Y0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630915660024, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="adc9e323-499e-451a-b3a6-3d5b56f3a148"
!git status
```

```python id="2G4iErkK88Y1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630915693929, "user_tz": -330, "elapsed": 1132, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9a11245d-142b-4952-e24c-6535032c46c4"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="o7iCtIG5CZiR"
!dvc status
```

```python colab={"base_uri": "https://localhost:8080/"} id="Klqi_zTLF_1d" executionInfo={"status": "ok", "timestamp": 1630914753985, "user_tz": -330, "elapsed": 5998, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="37e4895f-c578-49e3-a696-9f534560f061"
!pip install -U -e .
```

```python colab={"base_uri": "https://localhost:8080/"} id="h8c4LGrkmbNy" executionInfo={"status": "ok", "timestamp": 1630914719397, "user_tz": -330, "elapsed": 5987, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6318c4b4-c3d4-4c67-dde7-daac23f920f6"
!make setup
```

```python id="Tv4THPUQb3FL"
%reload_ext autoreload
%autoreload 2
```

```python id="5HtCujPxVJXK"
!dvc pull ./data/bronze/30music/sessions_sample_10.parquet.snappy.dvc
```

```python id="722DedyfpTc2"
!dvc repro
```

<!-- #region id="J2B-cCzAVL7p" -->
## Prototyping
<!-- #endregion -->

```python id="NtH-9O7EVN_g"
import operator
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region id="sgtN0EF6V2-q" -->
### Abstract recommender class
<!-- #endregion -->

```python id="YF1EX_ECTZRk"
class ISeqRecommender(object):
    """Abstract Recommender class"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, train_data):
        pass

    def recommend(self, user_profile, user_id=None):
        """
        Given the user profile return a list of recommendation
        :param user_profile: the user profile as a list of item identifiers
        :param user_id: (optional) the user id
        :return: list of recommendations e.g. [([2], 0.875), ([6], 1.0)]
        """
        pass

    @staticmethod
    def get_recommendation_list(recommendation):
        return list(map(lambda x: x[0], recommendation))

    @staticmethod
    def get_recommendation_confidence_list(recommendation):
        return list(map(lambda x: x[1], recommendation))

    def activate_debug_print(self):
        self.logger.setLevel(logging.DEBUG)

    def deactivate_debug_print(self):
        self.logger.setLevel(logging.INFO)
```

<!-- #region id="DxZ4v9rCZQhZ" -->
### Evaluation metrics
<!-- #endregion -->

```python id="XoRes9mlpBYc"
class EvalMetrics:
    def __init__(self):
        pass

    def precision(self, ground_truth, prediction):
        """
        Compute Precision metric
        :param ground_truth: the ground truth set or sequence
        :param prediction: the predicted set or sequence
        :return: the value of the metric
        """
        ground_truth = self._remove_duplicates(ground_truth)
        prediction = self._remove_duplicates(prediction)
        precision_score = self._count_a_in_b_unique(prediction, ground_truth) / float(len(prediction))
        assert 0 <= precision_score <= 1
        return precision_score

    def recall(self, ground_truth, prediction):
        """
        Compute Recall metric
        :param ground_truth: the ground truth set or sequence
        :param prediction: the predicted set or sequence
        :return: the value of the metric
        """
        ground_truth = self._remove_duplicates(ground_truth)
        prediction = self._remove_duplicates(prediction)
        recall_score = 0 if len(prediction) == 0 else self._count_a_in_b_unique(prediction, ground_truth) / float(
            len(ground_truth))
        assert 0 <= recall_score <= 1
        return recall_score

    def mrr(self, ground_truth, prediction):
        """
        Compute Mean Reciprocal Rank metric. Reciprocal Rank is set 0 if no predicted item is in contained the ground truth.
        :param ground_truth: the ground truth set or sequence
        :param prediction: the predicted set or sequence
        :return: the value of the metric
        """
        rr = 0.
        for rank, p in enumerate(prediction):
            if p in ground_truth:
                rr = 1. / (rank + 1)
                break
        return rr

    @staticmethod
    def _count_a_in_b_unique(a, b):
        """
        :param a: list of lists
        :param b: list of lists
        :return: number of elements of a in b
        """
        count = 0
        for el in a:
            if el in b:
                count += 1
        return count

    @staticmethod
    def _remove_duplicates(l):
        return [list(x) for x in set(tuple(x) for x in l)]
```

```python colab={"base_uri": "https://localhost:8080/"} id="PyTxAVOOqeTc" executionInfo={"status": "ok", "timestamp": 1630910820403, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cbf80aa0-18e4-4b1d-ee8f-dbecd01752fc"
gt = [(1,1),(0,1)]
pd = [(1,0),(0,1)]

ev = EvalMetrics()
ev.precision(gt,pd), ev.recall(gt,pd), ev.mrr(gt,pd)
```

<!-- #region id="j9PAfoweatV5" -->
### Sequential evaluator methods
<!-- #endregion -->

```python id="-s1SCRHBrrnz"
class SequentialEvaluator:
    """
    In the evaluation of sequence-aware recommenders, each sequence in the test set is split into:
    - the user profile, used to compute recommendations, is composed by the first k events in the sequence;
    - the ground truth, used for performance evaluation, is composed by the remainder of the sequence.
    
    you can control the dimension of the user profile by assigning a positive value to GIVEN_K,
    which correspond to the number of events from the beginning of the sequence that will be assigned
    to the initial user profile. This ensures that each user profile in the test set will have exactly
    the same initial size, but the size of the ground truth will change for every sequence.

    Alternatively, by assigning a negative value to GIVEN_K, you will set the initial size of the ground truth.
    In this way the ground truth will have the same size for all sequences, but the dimension of the user
    profile will differ.
    """

    def __init__(self, train_data, test_data, recommender):
        self.test_data = test_data
        self.recommender = recommender
        self.train_users = train_data['user_id'].values
        self.evalmetrics = EvalMetrics()
        self.evaluation_functions = {'precision':self.evalmetrics.precision,
                        'recall':self.evalmetrics.recall,
                        'mrr': self.evalmetrics.mrr}

    def get_test_sequences(self, given_k):
        # we can run evaluation only over sequences longer than abs(LAST_K)
        test_sequences = self.test_data.loc[self.test_data['sequence'].map(len) > abs(given_k), 'sequence'].values
        return test_sequences

    def get_test_sequences_and_users(self, given_k):
        # we can run evaluation only over sequences longer than abs(LAST_K)
        mask = self.test_data['sequence'].map(len) > abs(given_k)
        mask &= self.test_data['user_id'].isin(self.train_users)
        test_sequences = self.test_data.loc[mask, 'sequence'].values
        test_users = self.test_data.loc[mask, 'user_id'].values
        return test_sequences, test_users

    def sequential_evaluation(self, test_sequences, users=None, given_k=1,
                              look_ahead=1, top_n=10, scroll=True, step=1):
        """
        Runs sequential evaluation of a recommender over a set of test sequences
        :param recommender: the instance of the recommender to test
        :param test_sequences: the set of test sequences
        :param evaluation_functions: list of evaluation metric functions
        :param users: (optional) the list of user ids associated to each test sequence. Required by personalized models like FPMC.
        :param given_k: (optional) the initial size of each user profile, starting from the first interaction in the sequence.
                        If <0, start counting from the end of the sequence. It must be != 0.
        :param look_ahead: (optional) number of subsequent interactions in the sequence to be considered as ground truth.
                        It can be any positive number or 'all' to extend the ground truth until the end of the sequence.
        :param top_n: (optional) size of the recommendation list
        :param scroll: (optional) whether to scroll the ground truth until the end of the sequence.
                    If True, expand the user profile and move the ground truth forward of `step` interactions. Recompute and evaluate recommendations every time.
                    If False, evaluate recommendations once per sequence without expanding the user profile.
        :param step: (optional) number of interactions that will be added to the user profile at each step of the sequential evaluation.
        :return: the list of the average values for each evaluation metric
        """
        if given_k == 0:
            raise ValueError('given_k must be != 0')

        evaluation_functions = self.evaluation_functions.values()

        metrics = np.zeros(len(evaluation_functions))
        with tqdm(total=len(test_sequences)) as pbar:
            for i, test_seq in enumerate(test_sequences):
                if users is not None:
                    user = users[i]
                else:
                    user = None
                if scroll:
                    metrics += self.sequence_sequential_evaluation(test_seq,
                                                                   user,
                                                                   given_k,
                                                                   look_ahead,
                                                                   top_n,
                                                                   step)
                else:
                    metrics += self.evaluate_sequence(test_seq, 
                                                      user,
                                                      given_k,
                                                      look_ahead,
                                                      top_n)
                pbar.update(1)

        return metrics / len(test_sequences)

    def evaluate_sequence(self, seq, user, given_k, look_ahead, top_n):
        """
        :param recommender: which recommender to use
        :param seq: the user_profile/ context
        :param given_k: last element used as ground truth. NB if <0 it is interpreted as first elements to keep
        :param evaluation_functions: which function to use to evaluate the rec performance
        :param look_ahead: number of elements in ground truth to consider. if look_ahead = 'all' then all the ground_truth sequence is considered
        :return: performance of recommender
        """
        # safety checks
        if given_k < 0:
            given_k = len(seq) + given_k

        user_profile = seq[:given_k]
        ground_truth = seq[given_k:]

        # restrict ground truth to look_ahead
        ground_truth = ground_truth[:look_ahead] if look_ahead != 'all' else ground_truth
        ground_truth = list(map(lambda x: [x], ground_truth))  # list of list format

        user_profile = list(user_profile)
        ground_truth = list(ground_truth)
        evaluation_functions = self.evaluation_functions.values()

        if not user_profile or not ground_truth:
            # if any of the two missing all evaluation functions are 0
            return np.zeros(len(evaluation_functions))

        r = self.recommender.recommend(user_profile, user)[:top_n]

        if not r:
            # no recommendation found
            return np.zeros(len(evaluation_functions))
        reco_list = self.recommender.get_recommendation_list(r)

        tmp_results = []
        for f in evaluation_functions:
            tmp_results.append(f(ground_truth, reco_list))
        return np.array(tmp_results)

    def sequence_sequential_evaluation(self, seq, user, given_k, look_ahead, top_n, step):
        if given_k < 0:
            given_k = len(seq) + given_k

        eval_res = 0.0
        eval_cnt = 0
        for gk in range(given_k, len(seq), step):
            eval_res += self.evaluate_sequence(seq,
                                                user,
                                                gk,
                                                look_ahead,
                                                top_n)
            eval_cnt += 1
        return eval_res / eval_cnt

    def eval_seqreveal(self, user_flg=0, GIVEN_K=1, LOOK_AHEAD=1, STEP=1, TOPN=20):
        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(GIVEN_K)
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            results = self.sequential_evaluation(test_sequences,
                                                users=test_users,
                                                given_k=GIVEN_K,
                                                look_ahead=LOOK_AHEAD,
                                                top_n=TOPN,
                                                scroll=True,  # scrolling averages metrics over all profile lengths
                                                step=STEP)
        else:
            test_sequences = self.get_test_sequences(GIVEN_K)
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            results = self.sequential_evaluation(test_sequences,
                                                given_k=GIVEN_K,
                                                look_ahead=LOOK_AHEAD,
                                                top_n=TOPN,
                                                scroll=True,  # scrolling averages metrics over all profile lengths
                                                step=STEP)
        
        # print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
        # for mname, mvalue in zip(self.evaluation_functions.keys(), results):
        #     print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))
        return [results, GIVEN_K, LOOK_AHEAD, STEP]  


    def eval_staticprofile(self, user_flg=0, GIVEN_K=1, LOOK_AHEAD='all', STEP=1, TOPN=20):
        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(GIVEN_K) # we need user ids now!
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            results = self.sequential_evaluation(test_sequences,
                                                users=test_users,
                                                given_k=GIVEN_K,
                                                look_ahead=LOOK_AHEAD,
                                                top_n=TOPN,
                                                scroll=False  # notice that scrolling is disabled!
                                            )                                
        else:
            test_sequences = self.get_test_sequences(GIVEN_K)
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            results = self.sequential_evaluation(test_sequences,
                                                 given_k=GIVEN_K,
                                                 look_ahead=LOOK_AHEAD,
                                                 top_n=TOPN,
                                                 scroll=False  # notice that scrolling is disabled!
                                                 )
            
        return [results, GIVEN_K, LOOK_AHEAD, STEP] 

    def eval_reclength(self, user_flg=0, GIVEN_K=1, LOOK_AHEAD=1, STEP=1,
                       topn_list=[1,5,10,20,50,100], TOPN=20):
        res_list = []

        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(GIVEN_K) # we need user ids now!
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            for topn in topn_list:
                print('Evaluating recommendation lists with length: {}'.format(topn)) 
                res_tmp = self.sequential_evaluation(test_sequences,
                                                        users=test_users,
                                                        given_k=GIVEN_K,
                                                        look_ahead=LOOK_AHEAD,
                                                        top_n=topn,
                                                        scroll=True,  # here we average over all profile lengths
                                                        step=STEP
                                                )
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((topn, mvalues))                            
        else:
            test_sequences = self.get_test_sequences(GIVEN_K)
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            for topn in topn_list:
                print('Evaluating recommendation lists with length: {}'.format(topn))      
                res_tmp = self.sequential_evaluation(test_sequences,
                                                    given_k=GIVEN_K,
                                                    look_ahead=LOOK_AHEAD,
                                                    top_n=topn,
                                                    scroll=True,  # here we average over all profile lengths
                                                    step=STEP)
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((topn, mvalues))

        # show separate plots per metric
        # fig, axes = plt.subplots(nrows=1, ncols=len(self.evaluation_functions), figsize=(15,5))
        res_list_t = list(zip(*res_list))
        results = []
        for midx, metric in enumerate(self.evaluation_functions):
            mvalues = [res_list_t[1][j][midx][1] for j in range(len(res_list_t[1]))]
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(topn_list, mvalues)
            ax.set_title(metric)
            ax.set_xticks(topn_list)
            ax.set_xlabel('List length')
            fig.tight_layout()
            results.append(fig)
        return [results, GIVEN_K, LOOK_AHEAD, STEP]

    def eval_profilelength(self, user_flg=0, given_k_list=[1,2,3,4], 
                           LOOK_AHEAD=1, STEP=1, TOPN=20):
        res_list = []

        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(max(given_k_list)) # we need user ids now!
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            for gk in given_k_list:
                print('Evaluating profiles having length: {}'.format(gk))
                res_tmp = self.sequential_evaluation(test_sequences,
                                                            users=test_users,
                                                            given_k=gk,
                                                            look_ahead=LOOK_AHEAD,
                                                            top_n=TOPN,
                                                            scroll=False,  # here we stop at each profile length
                                                            step=STEP)
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((gk, mvalues))                          
        else:
            test_sequences = self.get_test_sequences(max(given_k_list))
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            for gk in given_k_list:
                print('Evaluating profiles having length: {}'.format(gk))
                res_tmp = self.sequential_evaluation(test_sequences,
                                                            given_k=gk,
                                                            look_ahead=LOOK_AHEAD,
                                                            top_n=TOPN,
                                                            scroll=False,  # here we stop at each profile length
                                                            step=STEP)
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((gk, mvalues))

        # show separate plots per metric
        # fig, axes = plt.subplots(nrows=1, ncols=len(self.evaluation_functions), figsize=(15,5))
        res_list_t = list(zip(*res_list))
        results = []
        for midx, metric in enumerate(self.evaluation_functions):
            mvalues = [res_list_t[1][j][midx][1] for j in range(len(res_list_t[1]))]
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(given_k_list, mvalues)
            ax.set_title(metric)
            ax.set_xticks(given_k_list)
            ax.set_xlabel('Profile length')
            fig.tight_layout()
            results.append(fig)
        return [results, TOPN, LOOK_AHEAD, STEP]
```

<!-- #region id="t-OhKXShUqHa" -->
### Item Popularity Model
<!-- #endregion -->

```python id="os-W3W4_T4yj"
class PopularityRecommender(ISeqRecommender):

    def __init__(self):
        super(PopularityRecommender, self).__init__()

    def fit(self, train_data):
        sequences = train_data['sequence'].values

        count_dict = {}
        for s in sequences:
            for item in s:
                if item not in count_dict:
                    count_dict[item] = 1
                else:
                    count_dict[item] += 1

        self.top = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.top = [([x[0]], x[1]) for x in self.top]

    def recommend(self, user_profile, user_id=None):
        return self.top

    def get_popular_list(self):
        return self.top
```

<!-- #region id="a7pamD2lZlk0" -->
### Run
<!-- #endregion -->

```python id="vo5zKnisZkYf"
METRICS = {'precision':precision, 
           'recall':recall,
           'mrr': mrr}
TOPN=100 # length of the recommendation list
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="s2Vs_Z6FW62k" executionInfo={"status": "ok", "timestamp": 1630907051452, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a87710bd-e254-440e-ea87-d0088ea41b23"
import os
import pandas as pd

base_path = './data/silver/30music'

train_data = pd.read_parquet(os.path.join(base_path, 'train.parquet.snappy'))
test_data = pd.read_parquet(os.path.join(base_path, 'test.parquet.snappy'))

train_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="WaKdePYpgAE_" executionInfo={"status": "ok", "timestamp": 1630907054635, "user_tz": -330, "elapsed": 685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="63a84f37-c0e2-402d-ea63-6efbf6862a91"
train_data.shape[0], test_data.shape[0]
```

```python id="vg_oRwUxXLBL"
pop_recommender = PopularityRecommender()
```

```python id="knN6FECVWfbq"
pop_recommender.fit(train_data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aVOAnIMrd9-P" executionInfo={"status": "ok", "timestamp": 1630911908447, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d2e8a609-c3d5-4fdd-dad2-3521edbe1bee"
pop_recommender.recommend(user_profile=[1719])[:5]
```

```python id="0nKLBWKY7Oe4"
evaluator = SequentialEvaluator(train_data, test_data, pop_recommender)
```

```python colab={"base_uri": "https://localhost:8080/"} id="NkttJbc07Xzr" executionInfo={"status": "ok", "timestamp": 1630911983294, "user_tz": -330, "elapsed": 2031, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="250c90ac-4d7e-454c-e853-a28b22d55a13"
evaluator.eval_seqreveal()
```

```python colab={"base_uri": "https://localhost:8080/"} id="c3GuewwA7jRy" executionInfo={"status": "ok", "timestamp": 1630912014611, "user_tz": -330, "elapsed": 1454, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2dd37d2d-5887-4eca-e74a-a825f409e4a0"
evaluator.eval_staticprofile()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Q29bpvp07pfT" executionInfo={"status": "ok", "timestamp": 1630912046278, "user_tz": -330, "elapsed": 5252, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e40d9bcd-3744-453a-929d-679fd73d8c01"
evaluator.eval_reclength()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GkgzWfvmo2ki" executionInfo={"status": "ok", "timestamp": 1630912071276, "user_tz": -330, "elapsed": 1621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b46c46c-d1b2-45e3-b625-df51e7047d11"
evaluator.eval_profilelength()
```

<!-- #region id="cVUcdA-NEWwo" -->
## Scripting
<!-- #endregion -->

<!-- #region id="Dal-XgynGaA5" -->
### Models
<!-- #endregion -->

```python id="DdAVomImA5o4"
!mkdir -p ./src/models
!touch ./src/models/__init__.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="zVo98MAD-gRC" executionInfo={"status": "ok", "timestamp": 1630914359724, "user_tz": -330, "elapsed": 682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3fe7ffd8-ece4-498c-cf95-0066e201c347"
%%writefile ./src/models/abstract.py

import logging


class ISeqRecommender(object):
    """Abstract Recommender class"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    def fit(self, train_data):
        pass

    def recommend(self, user_profile, user_id=None):
        """
        Given the user profile return a list of recommendation
        :param user_profile: the user profile as a list of item identifiers
        :param user_id: (optional) the user id
        :return: list of recommendations e.g. [([2], 0.875), ([6], 1.0)]
        """
        pass

    @staticmethod
    def get_recommendation_list(recommendation):
        return list(map(lambda x: x[0], recommendation))

    @staticmethod
    def get_recommendation_confidence_list(recommendation):
        return list(map(lambda x: x[1], recommendation))

    def activate_debug_print(self):
        self.logger.setLevel(logging.DEBUG)

    def deactivate_debug_print(self):
        self.logger.setLevel(logging.INFO)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7Y-4R2cN-rwG" executionInfo={"status": "ok", "timestamp": 1630914794343, "user_tz": -330, "elapsed": 634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b915114-be32-4767-d71b-2832c1bb1462"
%%writefile ./src/models/itempop.py

import operator

from src.models.abstract import ISeqRecommender


class PopularityRecommender(ISeqRecommender):

    def __init__(self):
        super(PopularityRecommender, self).__init__()

    def fit(self, train_data):
        sequences = train_data['sequence'].values

        count_dict = {}
        for s in sequences:
            for item in s:
                if item not in count_dict:
                    count_dict[item] = 1
                else:
                    count_dict[item] += 1

        self.top = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.top = [([x[0]], x[1]) for x in self.top]

    def recommend(self, user_profile, user_id=None):
        return self.top

    def get_popular_list(self):
        return self.top
```

<!-- #region id="84cmlMNV_efS" -->
### Trainer
<!-- #endregion -->

```python id="REEBNdNZ-1rv"
!mkdir -p ./src/train
!touch ./src/train/__init__.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="m5r4huYR_Ghd" executionInfo={"status": "ok", "timestamp": 1630914809183, "user_tz": -330, "elapsed": 428, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="72dc5c28-7a13-44ac-9a8a-529b02c7de4b"
%%writefile ./src/train/itempop.py

import os
import sys
import pickle
import pandas as pd

from src.models.itempop import PopularityRecommender

data_source_path = str(sys.argv[1])
data_source_path_train = os.path.join(data_source_path, 'train.parquet.snappy')

train_data = pd.read_parquet(data_source_path_train)

pop_recommender = PopularityRecommender()

pop_recommender.fit(train_data)

pickle.dump(pop_recommender, open('./artifacts/30music/models/itempop.pkl', 'wb'))
```

```python id="9bpmwco4E8l1"
!python ./src/train/itempop.py ./data/silver/30music
```

<!-- #region id="epViYW-qGs7U" -->
### Evaluator
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lNwNLyydE8jP" executionInfo={"status": "ok", "timestamp": 1630915380674, "user_tz": -330, "elapsed": 709, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3595017a-248f-4d24-fbfc-ef19742b6e5e"
%%writefile ./src/eval/itempop.py

import os
import sys
import pickle
import pandas as pd

from src.eval.seqeval import SequentialEvaluator

data_source_path = str(sys.argv[1])
data_source_path_train = os.path.join(data_source_path, 'train.parquet.snappy')
data_source_path_test = os.path.join(data_source_path, 'test.parquet.snappy')

train_data = pd.read_parquet(data_source_path_train)
test_data = pd.read_parquet(data_source_path_test)

model_path = str(sys.argv[2])
model = pickle.load(open(model_path, 'rb'))

evaluator = SequentialEvaluator(train_data, test_data, model)

results = {}
results['seq_reveal'] = evaluator.eval_seqreveal()
results['static_profile'] = evaluator.eval_staticprofile()
results['rec_length'] = evaluator.eval_reclength()
results['profile_length'] = evaluator.eval_profilelength()

pickle.dump(results, open('./artifacts/30music/results/itempop.pkl', 'wb'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="6PLyQTBkIO3o" executionInfo={"status": "ok", "timestamp": 1630915572933, "user_tz": -330, "elapsed": 7265, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="27c950f8-e0c7-41ad-962c-44a5bfc729ed"
!python ./src/eval/itempop.py \
./data/silver/30music \
./artifacts/30music/models/itempop.pkl
```

```python colab={"base_uri": "https://localhost:8080/"} id="haMEDQ6PIk1R" executionInfo={"status": "ok", "timestamp": 1630915578047, "user_tz": -330, "elapsed": 1714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d14d84d8-0f55-4bb7-ea40-7f16c330eab2"
import pickle

results = pickle.load(open('./artifacts/30music/results/itempop.pkl', 'rb'))
results
```

```python id="m73fMZIsIygS"

```
