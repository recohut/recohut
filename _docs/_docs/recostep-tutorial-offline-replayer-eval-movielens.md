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

<!-- #region id="o0GQArDlKTTk" -->
# Offline Replayer Evaluation V - Movielens
> Evaluating bandits offline with replay method on movielens dataset

- toc: true
- badges: true
- comments: true
- categories: [bandit, movie]
- image: 
<!-- #endregion -->

<!-- #region id="ykntm7xrcqjC" -->
## Environment setup
<!-- #endregion -->

<!-- #region id="D8UfnCqxct2M" -->
### Import libraries
<!-- #endregion -->

```python id="3nfpF4cfcUMf"
import pandas as pd
import numpy as np
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
```

<!-- #region id="gr2RIrMccvlV" -->
### Set variables
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 85} id="psGnSaSGcqCm" executionInfo={"status": "ok", "timestamp": 1624003732123, "user_tz": -330, "elapsed": 589, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03fb25ec-9e90-49ba-e499-8a8beb5699d5"
sns.set(font_scale=2.5)

# using a color-blind friendly palette with 20 colors
color_blind_palette_20 = ['#cfcfcf', '#ffbc79', '#a2c8ec', '#898989', '#c85200',
                          '#5f9ed1', '#595959', '#ababab', '#ff800e', '#006ba4',
                          '#cfcfcf', '#ffbc79', '#a2c8ec', '#898989', '#c85200',
                          '#5f9ed1', '#595959', '#ababab', '#ff800e', '#006ba4']

sns.palplot(color_blind_palette_20)
```

<!-- #region id="vUocdyisc8oW" -->
### Loading data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LlHZArr4dVcI" executionInfo={"status": "ok", "timestamp": 1624003733706, "user_tz": -330, "elapsed": 1592, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c927a87c-fc6b-4ff3-94ad-c43c3e0b5c74"
!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip
```

```python id="jRvI5vc8c1qX"
movie_df = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin1', index_col=False,
                   names=['movie_id',
                          'movie_title',
                          'release_date',
                          'video_release_date',
                          'imdb_url',
                          'unknown',
                          'action',
                          'adventure',
                          'animation',
                          'children',
                          'comedy',
                          'crime',
                          'documentary',
                          'drama',
                          'fantasy',
                          'film_noir',
                          'horror',
                          'musical',
                          'mystery'
                          'romance',
                          'sci_fi',
                          'thriller',
                          'war',
                          'western'])

movie_df.movie_id -= 1 # make this column zero-indexed
```

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="hsfk7Z9pdjn7" executionInfo={"status": "ok", "timestamp": 1624003733709, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2f06296a-7402-4eba-a54d-3afb785265b9"
movie_df.head()
```

```python id="vgHYL7wLdmBq"
rating_df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])
rating_df.user_id -= 1 # make this column zero-indexed
rating_df.movie_id -= 1 # make this column zero-indexed
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="zgbxQ2XIdtXO" executionInfo={"status": "ok", "timestamp": 1624003733712, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6fab8d17-8e46-4d56-f466-e6f3eed5be91"
rating_df.head()
```

<!-- #region id="D_HkxX0Rlgn1" -->
## Create agents
<!-- #endregion -->

```python id="nlNYIT56lkMx"
class ReplaySimulator(object):
    '''
    A class to provide base functionality for simulating the replayer method for online algorithms.
    '''

    def __init__(self, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1, random_seed=1):

        np.random.seed(random_seed)
    
        self.reward_history = reward_history
        self.item_col_name = item_col_name
        self.visitor_col_name = visitor_col_name
        self.reward_col_name = reward_col_name

        # number of visits to replay/simulate
        self.n_visits = n_visits
        
        # number of runs to average over
        self.n_iterations = n_iterations
    
        # items under test
        self.items = self.reward_history[self.item_col_name].unique()
        self.n_items = len(self.items)
        
        # visitors in the historical reward_history (e.g., from ratings df)
        self.visitors = self.reward_history[self.visitor_col_name].unique()
        self.n_visitors = len(self.visitors)
        

    def reset(self):
        # number of times each item has been sampled (previously n_sampled)
        self.n_item_samples = np.zeros(self.n_items)
        
        # fraction of time each item has resulted in a reward (previously movie_clicks)
        self.n_item_rewards = np.zeros(self.n_items)
        
    
    def replay(self):
        
        results = []

        for iteration in tqdm(range(0, self.n_iterations)):
        
            self.reset()
            
            total_rewards = 0
            fraction_relevant = np.zeros(self.n_visits)

            for visit in range(0, self.n_visits):
            
                found_match = False
                while not found_match:
                
                    # choose a random visitor
                    visitor_idx = np.random.randint(self.n_visitors)
                    visitor_id = self.visitors[visitor_idx]

                    # select an item to offer the visitor
                    item_idx = self.select_item()
                    item_id = self.items[item_idx]
                    
                    # if this interaction exists in the history, count it
                    reward = self.reward_history.query(
                        '{} == @item_id and {} == @visitor_id'.format(self.item_col_name, self.visitor_col_name))[self.reward_col_name]
                    
                    found_match = reward.shape[0] > 0
                
                reward_value = reward.iloc[0]
                
                self.record_result(visit, item_idx, reward_value)
                
                ## record metrics
                total_rewards += reward_value
                fraction_relevant[visit] = total_rewards * 1. / (visit + 1)
                
                result = {}
                result['iteration'] = iteration
                result['visit'] = visit
                result['item_id'] = item_id
                result['visitor_id'] = visitor_id
                result['reward'] = reward_value
                result['total_reward'] = total_rewards
                result['fraction_relevant'] = total_rewards * 1. / (visit + 1)
                
                results.append(result)
        
        return results
        
    def select_item(self):
        return np.random.randint(self.n_items)
        
    def record_result(self, visit, item_idx, reward):
    
        self.n_item_samples[item_idx] += 1
        
        alpha = 1./self.n_item_samples[item_idx]
        self.n_item_rewards[item_idx] += alpha * (reward - self.n_item_rewards[item_idx])


class ABTestReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on an A/B test.
    '''
    
    def __init__(self, n_visits, n_test_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(ABTestReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
        
        # TODO: validate that n_test_visits <= n_visits
    
        self.n_test_visits = n_test_visits
        
        self.is_testing = True
        self.best_item_id = None
        
    def reset(self):
        super(ABTestReplayer, self).reset()
        
        self.is_testing = True
        self.best_item_idx = None
    
    def select_item(self):
        if self.is_testing:
            return super(ABTestReplayer, self).select_item()
        else:
            return self.best_item_idx
            
    def record_result(self, visit, item_idx, reward):
        super(ABTestReplayer, self).record_result(visit, item_idx, reward)
        
        if (visit == self.n_test_visits - 1): # this was the last visit during the testing phase
            
            self.is_testing = False
            self.best_item_idx = np.argmax(self.n_item_rewards)
        

class EpsilonGreedyReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on an epsilon-Greedy bandit algorithm.
    '''

    def __init__(self, epsilon, n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations=1):
        super(EpsilonGreedyReplayer, self).__init__(n_visits, reward_history, item_col_name, visitor_col_name, reward_col_name, n_iterations)
    
        # parameter to control exploration vs exploitation
        self.epsilon = epsilon
    
    def select_item(self):
        
        # decide to explore or exploit
        if np.random.uniform() < self.epsilon: # explore
            item_id = super(EpsilonGreedyReplayer, self).select_item()
            
        else: # exploit
            item_id = np.argmax(self.n_item_rewards)
            
        return item_id
    

class ThompsonSamplingReplayer(ReplaySimulator):
    '''
    A class to provide functionality for simulating the replayer method on a Thompson Sampling bandit algorithm.
    '''

    def reset(self):
        self.alphas = np.ones(self.n_items)
        self.betas = np.ones(self.n_items)

    def select_item(self):
    
        samples = [np.random.beta(a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(samples)

    def record_result(self, visit, item_idx, reward):
        
        ## update value estimate
        if reward == 1:
            self.alphas[item_idx] += 1
        else:
            self.betas[item_idx] += 1
```

<!-- #region id="HrhWAY9PdyOF" -->
## Simulation
<!-- #endregion -->

```python id="gAlMaoC5dwbv"
# select top-20 most rated movies
n_movies = 20
movie_counts = rating_df.groupby('movie_id')['rating'].count().sort_values(ascending=False)
top_n_movies = rating_df.query('movie_id in {}'.format((movie_counts[:n_movies].index.values.tolist())))
```

```python id="ZIYGWZWSeCh8"
# add the movie title for ease of access when plotting
# remove the timestamp column because we don't need it
top_n_movies = top_n_movies.merge(movie_df[['movie_id','movie_title']], on='movie_id', how='left') \
                           .drop(columns=['timestamp'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="efCPpl6jhWyr" executionInfo={"status": "ok", "timestamp": 1624003735842, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="990e783a-13a0-41b6-f65a-c7e8a091ee5d"
top_n_movies.head()
```

```python id="N1gR_rakhU6N"
# remove the year from the title
top_n_movies.movie_title = top_n_movies.movie_title.str.replace('\s+\(.+\)', '').str.strip()
```

```python colab={"base_uri": "https://localhost:8080/"} id="nx04ZxzjhYE1" executionInfo={"status": "ok", "timestamp": 1624003736373, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ecc4b66f-a59b-4f17-9d7b-0612150c5844"
top_n_movies.movie_title.unique().tolist()
```

```python colab={"base_uri": "https://localhost:8080/"} id="cKgZGty2BvxI" executionInfo={"status": "ok", "timestamp": 1624003738562, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb26e2a0-3e30-4e05-f23e-9c3cfa04329f"
# create a dictionary between movie_id and movie_title (since the results output doesn't have titles)
movie_titles = top_n_movies.groupby(['movie_id','movie_title']).size().to_frame() \
                                    .reset_index('movie_title').movie_title \
                                    .to_dict()
movie_titles
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q6ECfDxghf88" executionInfo={"status": "ok", "timestamp": 1624003739022, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ced771b0-74b8-437c-dc5e-d43e0a69bd2d"
# Create a dictionary to map these movies to colors for consistency in plots
color_map = dict(zip(top_n_movies.movie_title.sort_values().unique().tolist(), color_blind_palette_20))
color_map
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="T3j0b-BkhigQ" executionInfo={"status": "ok", "timestamp": 1624003740515, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="213ef034-44ba-43fb-9fab-62bccbf55d8a"
# What are the actual rating distributions of the top N movies?
rating_counts_by_movie = top_n_movies.groupby(['movie_id','movie_title','rating'], as_index=False).size()
rating_counts_by_movie.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 860} id="5vh6_8Mah3lH" executionInfo={"status": "ok", "timestamp": 1624003743341, "user_tz": -330, "elapsed": 2390, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e3c91d72-c312-4506-e40f-b91d50a7a527"
sns.set_palette(color_blind_palette_20)

rating_v_count_ax = sns.catplot(data=rating_counts_by_movie, x='rating', y='size',
                                   hue='movie_title', hue_order=color_map.keys(),
                                   height=20, aspect=1.5, kind='bar', legend=False)

plt.title('Rating Histogram for the 20 Movies with the Most Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')

plt.legend(title='')

plt.show()
```

```python id="IGgg_40EirEY"
rating_v_count_ax.savefig('rating_distributions.png', transparent=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="4YRC6fPGjsKl" executionInfo={"status": "ok", "timestamp": 1624003744182, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="712e7730-28bb-413b-dd14-a28985808e1a"
# Let's transform the ratings from a discrete range into a binary "like" or "dislike"
# movie ratings above this threshold will be counted as a "like"
reward_threshold = 4
rating_counts_by_movie.eval('liked = rating > @reward_threshold') \
                      .groupby(['movie_title','liked']) \
                      .sum().head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="bBbDS5a9jyYN" executionInfo={"status": "ok", "timestamp": 1624003745197, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4058feca-d51f-41e4-9ab5-60eb68d1a502"
rating_counts_by_movie.eval('liked = rating > @reward_threshold') \
                      .groupby(['movie_title','liked'], as_index=False) \
                      .sum().head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="WbRc7Welj0E_" executionInfo={"status": "ok", "timestamp": 1624003745976, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98544751-4521-44b7-a2ac-0ab6914bb95f"
total_reward_counts = rating_counts_by_movie.eval('liked = rating > @reward_threshold') \
    .groupby(['movie_id','movie_title','liked'])['size'] \
    .sum() \
    .unstack('liked') \
    .reset_index() \
    .rename(columns={False:'disliked', True:'liked'}) \
    .eval('total = disliked + liked') \
    .eval('like_pct = 100 * liked / total')

total_reward_counts.sort_values('like_pct', ascending=False, inplace=True)
total_reward_counts.head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 873} id="pKSIX37Aj9kd" executionInfo={"status": "ok", "timestamp": 1624003749083, "user_tz": -330, "elapsed": 1871, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d3be823f-f076-4cb0-80a3-bd890ceabd85"
like_probability_ax = sns.catplot(height=20, aspect=1.5, x='like_pct', y='movie_title', kind='bar',
                                     palette=total_reward_counts['movie_title'].apply(lambda x: color_map[x]),
                                     data=total_reward_counts)

plt.title('The 20 Movies with the Most Ratings')
like_probability_ax.set_axis_labels('Probability of Being "Liked" (%)','')

plt.show()
```

```python id="DOdRhObvk3vx"
like_probability_ax.savefig('like_probabilities.png', transparent=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="j5Y7oqzalEfL" executionInfo={"status": "ok", "timestamp": 1624003749786, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="310cff7b-7324-4d59-d91c-379d9b0a0468"
# Translate ratings into a binary reward value
reward_threshold = 4
rating_df['reward'] = rating_df.eval('rating > @reward_threshold').astype(int)
rating_df.head()
```

<!-- #region id="L6vLsAP9nB1P" -->
## Running the simulation
<!-- #endregion -->

```python id="dYwn-hMoltcN"
# Set the parameters for the simulations
n_visits = 50 # 20000 for more realistic results, will take time time to run
n_iterations = 10 #20

reward_history = rating_df[:1000]
item_col_name = 'movie_id'
visitor_col_name = 'user_id'
reward_col_name = 'reward'
```

<!-- #region id="FqnZYbHnl1DY" -->
## A/B test simulation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="R_I-w4irlyZ4" executionInfo={"status": "ok", "timestamp": 1624003925420, "user_tz": -330, "elapsed": 174426, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="05499697-68b8-4c6d-c2ec-dc754730c035"
n_test_visits = 30 #1000
ab_results_1 = ABTestReplayer(n_visits, n_test_visits, reward_history,
                               item_col_name, visitor_col_name, reward_col_name,
                               n_iterations=n_iterations).replay()

ab_results_1_df = pd.DataFrame(ab_results_1)
ab_results_1_df.head()
```

```python id="YkFsKEXfmFm3"
# save the output
ab_results_1_df.to_csv('ab_results_1.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="Uadm6jpSmPh9" executionInfo={"status": "ok", "timestamp": 1624004129750, "user_tz": -330, "elapsed": 204364, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f1f4e0f1-6f2d-4496-ce8c-8b79275dfadc"
n_test_visits = 50 #5000
ab_results_2 = ABTestReplayer(n_visits, n_test_visits, reward_history,
                               item_col_name, visitor_col_name, reward_col_name,
                               n_iterations=n_iterations).replay()

ab_results_2_df = pd.DataFrame(ab_results_2)
ab_results_2_df.head()
```

```python id="qcy-YrYXmpVc"
# save the output
ab_results_2_df.to_csv('ab_results_2.csv')
```

<!-- #region id="fGE5FnEPnHOj" -->
### ϵ-Greedy simulation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="mW0OX7kImpSN" executionInfo={"status": "ok", "timestamp": 1624004267912, "user_tz": -330, "elapsed": 138197, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b9db2355-7335-4dc6-e5e7-aa467119ae8f"
epsilon = 0.05
epsilon_05_results = EpsilonGreedyReplayer(epsilon, n_visits, reward_history,
                                           item_col_name, visitor_col_name, reward_col_name,
                                           n_iterations=n_iterations).replay()

epsilon_05_results_df = pd.DataFrame(epsilon_05_results)
epsilon_05_results_df.head()
```

```python id="ahipFwdempNt"
# save the output
epsilon_05_results_df.to_csv('epsilon_greedy_05.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="yCzU_yepnUdq" executionInfo={"status": "ok", "timestamp": 1624004401990, "user_tz": -330, "elapsed": 134111, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="787d3ceb-667e-49b7-e03a-5ebc8e6cbe9d"
epsilon = 0.10
epsilon_10_results = EpsilonGreedyReplayer(epsilon, n_visits, reward_history,
                                           item_col_name, visitor_col_name, reward_col_name,
                                           n_iterations=n_iterations).replay()

epsilon_10_results_df = pd.DataFrame(epsilon_10_results)
epsilon_10_results_df.head()
```

```python id="r3vQLUv3nf1z"
# save the output
epsilon_10_results_df.to_csv('epsilon_greedy_10.csv')
```

<!-- #region id="00_rnpXsnl8a" -->
### Thompson sampling simulation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bEuGzKz5nn3F" executionInfo={"status": "ok", "timestamp": 1624004675455, "user_tz": -330, "elapsed": 273499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="be8e8647-685a-4b09-a086-639c0ac148e7"
thompson_results = ThompsonSamplingReplayer(n_visits, reward_history,
                                            item_col_name, visitor_col_name, reward_col_name,
                                            n_iterations=n_iterations).replay()

thompson_results_df = pd.DataFrame(thompson_results)
thompson_results_df.head()
```

```python id="EdXink-anuqb"
# save the output
thompson_results_df.to_csv('thompson_sampling.csv')
```

<!-- #region id="AlLPHV_Nnz4w" -->
## Result analysis
<!-- #endregion -->

<!-- #region id="6kEROZRiprGa" -->
### Thompson sampling output
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="AGMCIib2DbN0" executionInfo={"status": "ok", "timestamp": 1624004675462, "user_tz": -330, "elapsed": 67, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7160e790-5b9d-4bdf-b316-57829e9d2a62"
thompson_results_df.query('iteration == 0')
```

```python colab={"base_uri": "https://localhost:8080/"} id="XRx_BhEECaeL" executionInfo={"status": "ok", "timestamp": 1624004675468, "user_tz": -330, "elapsed": 69, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="54975c21-c97f-46ab-848a-3452c011fef0"
thompson_results_df.query('iteration == 0') \
                                .eval('selected = 1') \
                                .pivot(index='visit', columns='item_id', values='selected') \
                                .fillna(0) \
                                .cumsum(axis=0) \
                                .reset_index() \
                                .rename(columns=movie_titles).shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 275} id="qp3qbVSQpPwR" executionInfo={"status": "ok", "timestamp": 1624004675470, "user_tz": -330, "elapsed": 62, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1525c1ad-d427-4c8e-89c3-8aaeff01d020"
# Let's visualize the Thompson Sampling output for a single run
# We'll look at each movie's percentage of the total recommendations as the simulation progressed

# create a dataframe with running totals of how many times each recommendation was taken
thompson_running_ttl = thompson_results_df.query('iteration == 0') \
                                .eval('selected = 1') \
                                .pivot(index='visit', columns='item_id', values='selected') \
                                .fillna(0) \
                                .cumsum(axis=0) \
                                .reset_index() \
                                .rename(columns=movie_titles)

# scale the value by the visit number to get turn the running total into a percentage
thompson_running_ttl.iloc[:,1:] = thompson_running_ttl.iloc[:,1:].div((thompson_running_ttl.visit + 1)/100, axis=0)

thompson_running_ttl.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 847} id="abfFT-EOphnM" executionInfo={"status": "error", "timestamp": 1624004676528, "user_tz": -330, "elapsed": 1116, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="607638af-a90f-47b1-c926-f4b97da86636"
fig, ax = plt.subplots(figsize=(10,8))

ax.stackplot(thompson_running_ttl.visit,
             thompson_running_ttl.iloc[:,1:21].T,
             labels=thompson_running_ttl.iloc[:,1:21].columns.values.tolist(),
             colors=[color_map[x] for x in thompson_running_ttl.iloc[:,1:].columns.values]
            )

ax.set_xlim(0,1000)
ax.set_xticks(range(0, 1100, 250))

ax.set_title('Thompson Sampling Algorithm')
ax.set_xlabel('Recommendation #')
ax.set_ylabel('% of Recommendations')

lgd = plt.legend(bbox_to_anchor=(1.02, 0.15), loc=2, borderaxespad=0., labelspacing=-2.3)

ax.set_facecolor('w')

plt.tight_layout()
plt.show()
```

```python id="dfio-JCMp-p3"
fig.savefig('bandit_results.png', transparent=False, bbox_extra_artists=(lgd,), bbox_inches='tight')
```

<!-- #region id="iDL2mIUyn3gC" -->
### Average the results across all runs
<!-- #endregion -->

```python id="vusjA-h6nx9N"
ab_1k_avg_results_df = ab_1k_results_df.groupby('visit', as_index=False).mean()

ab_5k_avg_results_df = ab_5k_results_df.groupby('visit', as_index=False).mean()

epsilon_05_avg_results_df = epsilon_05_results_df.groupby('visit', as_index=False).mean()

epsilon_10_avg_results_df = epsilon_10_results_df.groupby('visit', as_index=False).mean()

thompson_avg_results_df = thompson_results_df.groupby('visit', as_index=False).mean()
```

<!-- #region id="8WSCZPnSpxqF" -->
### Compare the results
<!-- #endregion -->

```python id="Pfmv8omMoLaI"
fig, ax = plt.subplots(figsize=(12,10))

for (avg_results_df, style) in [(ab_1k_avg_results_df, 'r-'),
                                (ab_5k_avg_results_df, 'r--'),
                                (epsilon_05_avg_results_df, 'b-'),
                                (epsilon_10_avg_results_df, 'b--'),
                                (thompson_avg_results_df, 'tab:brown')]:
    
    ax.plot(avg_results_df.visit, avg_results_df.fraction_relevant, style, linewidth=3.5)

# add a line for the optimal value -- 0.5575 for Star Wars
ax.axhline(y=0.5575, color='k', linestyle=':', linewidth=2.5)

ax.set_title('Percentage of Liked Recommendations')
ax.set_xlabel('Recommendation #')
ax.set_ylabel('% of Recs Liked')

ax.set_xticks(range(0,22000,5000))
ax.set_ylim(0.2, 0.6)
ax.set_yticks(np.arange(0.2, 0.7, 0.1))

# rescale the y-axis tick labels to show them as a percentage
ax.set_yticklabels((ax.get_yticks()*100).astype(int))

ax.legend(['A/B Test (1k Recs)',
           'A/B Test (5k Recs)',
           '$\epsilon$ = 0.05',
           '$\epsilon$ = 0.10',
           'Thompson Sampling',
           'Optimal (Star Wars)'
          ],
          loc='lower right'
         )

plt.tight_layout()
plt.show()
```

```python id="vb_VE53rovQF"
fig.savefig('pct_liked_recs.png', transparent=False)
```
