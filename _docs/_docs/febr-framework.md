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

<!-- #region id="tos46oQJjpua" -->
# FEBR framework
<!-- #endregion -->

```python id="IEhfX4aAXGOk" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636650246893, "user_tz": -330, "elapsed": 20011, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a7a2037-9c02-40e6-da82-62f137eac04e"
!pip install gym==0.17.2
!pip install -U dopamine-rl
!pip install tensorflow-estimator==1.15.1
# !pip install recsim==0.2.4
```

```python id="9XGJsSFFXCcg" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636650250862, "user_tz": -330, "elapsed": 3980, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="29a290c9-2399-4800-e668-e58a5af295d2"
!git clone -b T608854 https://github.com/sparsh-ai/drl-recsys.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="Nh4PBG7zI29W" executionInfo={"status": "ok", "timestamp": 1636650250864, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4b4a2d85-0b56-479c-95df-88873a5788c9"
%tensorflow_version 1.x
```

```python colab={"base_uri": "https://localhost:8080/"} id="YWJMqb7kX8Jf" executionInfo={"status": "ok", "timestamp": 1636650250867, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c4718560-afd7-4a6e-c214-eb0c9cde7481"
%cd /content/drl-recsys
```

<!-- #region id="HA6ngcOif12S" -->
## Expert Environment Test
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3ystEudbXj1y" executionInfo={"status": "ok", "timestamp": 1636642517310, "user_tz": -330, "elapsed": 5852, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c27bdfa7-e4b5-40b8-b81f-291e885ff46f"
from recsim.agents import full_slate_q_agent, random_agent
from rl import *
from utils import *
from recsim.environments import interest_evolution
from recsim.simulator import runner_lib

def clicked_quality_reward(responses):
    """Calculates the total clicked watchtime from a list of responses.
    Args:
      responses: A list of IEvResponse objects
    Returns:
      reward: A float representing the total watch time from the responses
    """
    qual = 0.0
    watch = 0.0
    for response in responses:
        if response.clicked:
            qual += float(response.quality)
            watch += float(response.watch_time)
    return [qual, watch]


def create_agent_full_slate(sess, environment, eval_mode, summary_writer=None):
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
    }
    return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)


def create_agent_random(slate_size, random_seed=0):
    action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
    return random_agent.RandomAgent(action_space, random_seed)


def clicked_evaluation_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            if response.evaluated:
                reward += 1
    return reward


if __name__ == '__main__':

    slate_size = 2
    num_candidates = 5
    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function': clicked_quality_reward,
    }

    expertEnv = environment.Environment(
        ExpertModel(slate_size),
        EEVideoSampler(0),
        num_candidates,
        slate_size,
        resample_documents=True)

    lts_gym_env = recsim_gym.RecSimGymEnv(expertEnv, clicked_evaluation_reward)
    recsim_gym_env = interest_evolution.create_environment(env_config)
    observation_0 = recsim_gym_env.reset()

    for i in range(3):
        recommendation_slate_0 = [0,1]
        print(recommendation_slate_0)
        observation_1, reward, done, scores, _ = recsim_gym_env.step(recommendation_slate_0)
        print('Observation ' + str(i))
        print('Available documents')
        doc_strings = ['doc_id ' + key + str(value) for key, value
                   in observation_1['doc'].items()]
        print('\n'.join(doc_strings))
        rsp_strings = [str(response) for response in observation_1['response']]
        print('User responses to documents in the slate')
        print('\n'.join(rsp_strings))
        print('Reward: ', reward)
        print("User observation noise:", observation_1['user'][0], " interests features: ", observation_1['user'][1:])
        print("*******************************************")
```

<!-- #region id="lnsh3ciMhp-r" -->
## Policy Agent Test
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2KW32Y2bdUFU" executionInfo={"status": "ok", "timestamp": 1636642996436, "user_tz": -330, "elapsed": 418, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0f6f48e5-70e2-42b1-ede5-6c3fa14d0781"
from rl import *
from utils import *
import random
from recsim.agents import full_slate_q_agent
from irl_agent import InverseRLAgent
import tensorflow.compat.v1 as tf
import time
from numpy import load


def run_one_episode(env, agent, max_steps_per_episode=100):
    observation = env.reset()
    action = agent.begin_episode(observation)
    step_number = 0
    total_watch = 0.
    total_qual = 0
    start_time = time.time()
    total_length_videos = 0
    while True:
        observation, reward, done, info, _ = env.step(action)

        for j in range(len(observation['response'])):
            if observation['response'][j]['click'] == 1:
                index = action[j]
                total_length_videos += list(observation['doc'].values())[index][-1]
                break

        # Update environment-specific metrics with responses to the slate.
        env.update_metrics(observation['response'], info)

        total_watch += reward[1]
        total_qual += reward[0]
        step_number += 1

        if done:
            break
        elif step_number == max_steps_per_episode:
            # Stop the run loop once we reach the true end of episode.
            break
        else:
            action = agent.step(reward[1], observation)

    agent.end_episode(reward[1], observation)
    time_diff = time.time() - start_time
    print("hhhhhhhhhhh",total_length_videos,"fffff",total_watch)

    return step_number, total_watch/total_length_videos, time_diff, total_qual


def clicked_quality_reward(responses):
    """Calculates the total clicked watchtime from a list of responses.
    Args:
      responses: A list of IEvResponse objects
    Returns:
      reward: A float representing the total watch time from the responses
    """
    qual = 0.0
    watch = 0.0
    for response in responses:
        if response.clicked:
            qual += float(response.quality)
            watch += float(response.watch_time)
    return [qual, watch]

if __name__ == '__main__':
    def clicked_evaluation_reward(responses):
        reward = 0.0
        for response in responses:
            if response.clicked:
                if response.evaluated:
                    reward += 1
        return reward


    slate_size = 2
    num_candidates = 5
    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function':clicked_quality_reward
    }

    expertEnv = environment.Environment(
        ExpertModel(slate_size),
        EEVideoSampler(1),
        num_candidates,
        slate_size,
        resample_documents=True)


    states = load('./datasets_states/statesV1.npy', allow_pickle=True)
    policy_ = load('./datasets_states/policyV1.npy', allow_pickle=True)

    lts_gym_env = recsim_gym.RecSimGymEnv(expertEnv, clicked_evaluation_reward)
    recsim_gym_env = interest_evolution.create_environment(env_config)
    agent = full_slate_q_agent.FullSlateQAgent(tf.Session(config=tf.ConfigProto(allow_soft_placement=True)),
                                               recsim_gym_env.observation_space, recsim_gym_env.action_space)

    agent_irl = InverseRLAgent(recsim_gym_env, states, policy_, num_cand=num_candidates,
                               slate_size=slate_size, max_steps_per_episode=100)

    max_episode = 5
    results = []
    for i in range(max_episode):
        # steps, watch, time_, q = run_one_episode(recsim_gym_env, agent_irl)
        steps, watch, time_, q = run_one_episode(recsim_gym_env, agent)
        results += ["episode "+str(i)+", total_steps: "+ str(steps) +", total_watch_time: "+
                    str(watch)+", time_episode: "+ str(time_) + ", total qual: ",str(q)]

    for i in results:
        print(i)
```

<!-- #region id="O3kTfjJyiIWK" -->
## Learned Optimal Policy, Rewards, and States
<!-- #endregion -->

```python id="9cethMnJibl-"
from numpy import load

states_ = load('./datasets_states/statesV1.npy', allow_pickle=True)
policy_ = load('./datasets_states/policyV1.npy', allow_pickle=True)
rewards_ = load('./datasets_states/rewardsV1.npy', allow_pickle=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="P8SDJNYGiNTi" executionInfo={"status": "ok", "timestamp": 1636643270269, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="744b9823-8445-4a36-b09a-c4ede201a36b"
states_.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="38v9TZlZiQsp" executionInfo={"status": "ok", "timestamp": 1636643344833, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c7c6a623-0115-48f5-999c-b4ba979e67ca"
states_[0,0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="EX9imsWJiQqF" executionInfo={"status": "ok", "timestamp": 1636643352953, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ee45cd0-ebc9-4bd3-cf8d-0d54c4d742ed"
states_[0,1]
```

```python colab={"base_uri": "https://localhost:8080/"} id="9by2DZu-i6w3" executionInfo={"status": "ok", "timestamp": 1636643372475, "user_tz": -330, "elapsed": 504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cb979765-4977-483a-d97f-dd3a48582e04"
states_[0,2]
```

```python colab={"base_uri": "https://localhost:8080/"} id="8GVSifZki_kD" executionInfo={"status": "ok", "timestamp": 1636643385413, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bcbc5418-968d-4155-b933-c9f932ce26b5"
states_[0,3]
```

```python colab={"base_uri": "https://localhost:8080/"} id="RZ4_FSyvjC0L" executionInfo={"status": "ok", "timestamp": 1636643397498, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5bf376c3-43fd-4d4e-b097-254be73e5d33"
policy_.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="qoTjfcLbjFh-" executionInfo={"status": "ok", "timestamp": 1636643432816, "user_tz": -330, "elapsed": 499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bcb4fde3-3351-4621-c1df-650e21d7a119"
policy_[0,]
```

```python colab={"base_uri": "https://localhost:8080/"} id="hfGZ5OH7jI20" executionInfo={"status": "ok", "timestamp": 1636643445094, "user_tz": -330, "elapsed": 432, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="82e44d3f-a18a-467b-b41e-950f8af1b542"
rewards_.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="OrkYsWZOjL4Z" executionInfo={"status": "ok", "timestamp": 1636643456108, "user_tz": -330, "elapsed": 401, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="db08d444-4ed2-4f5c-91f0-2d7b60c4454f"
rewards_[0]
```

<!-- #region id="uH1F2-xsyuZ9" -->
## Model Evaluations
<!-- #endregion -->

```python id="a-8orgTeywIO" executionInfo={"status": "ok", "timestamp": 1636650266294, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from rl import *
from recsim.agents import  full_slate_q_agent
# from irl_agent import InverseRLAgent
import time
import pickle
from utils import *
import tensorflow.compat.v1 as tf
import time
from tqdm.notebook import tqdm
from numpy import load
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```

```python id="4cgBTNpSyzqz" executionInfo={"status": "ok", "timestamp": 1636650266786, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# define reward functions

def v1_clicked_evaluation_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            if response.evaluated:
                reward += 1
    return reward

def v2_clicked_evaluation_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            if response.evaluated:
                reward += int((response.accuracy_eval + response.importance_eval
                                   + response.pedagogy_eval + response.entertainment_eval) / 4)
    return reward

def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.watch_time
    return reward
```

```python id="RDo088bKzAp2" executionInfo={"status": "ok", "timestamp": 1636650267457, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# useful plotting function for result visualisation

def plot_chart1(x_,y_,c, x_label, y_label):
    plt.plot(x_, y_, c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def plot_chart2(x_,y_, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.stem(x_, y_)
    plt.show()
    
def plot_3a(t,data1,data2,axe1,axe2,ro1="ro",bo2="bo",x_="episodes"):
    # Create some mock data
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(x_)
    ax1.set_ylabel(axe1, color=color)
    ax1.plot(t, data1, ro1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(axe2, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, bo2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
```

```python id="UeC9-dCC0A54" executionInfo={"status": "ok", "timestamp": 1636650269656, "user_tz": -330, "elapsed": 729, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class InverseRLAgent():
    """A  recommender system agent that implements the policy learned by
    Maximum Entropy inverse reinforcement learning model."""

    def __init__(self, env, states_list, policy, num_cand, slate_size, max_steps_per_episode=100, 
                ml_model=False,filename='expert_bestModel.sav'):
        
        """
        Args:
            env: The RecSim Gym user environment, e.g. interest evolution environment
            states_list: Dataset of expert states
            policy: The learned expert policy produced by MaxEnt-IRL model
            num_cand: The size of the corpus video candidate
            slate_size: The number of recommended videos 
            max_steps_per_episode: The length of a user session (size of an episode)
            ml_model: Should be set to True in the case you want to use a classification 
                    model (learned on the dataset) for prediction. Otherwise, you use the 
                    comparaison model with similarities margins
            filename: If ml_model=True, you have to specify your serialized machine learning model

        """
        self.env = env
        self.states = states_list
        self.policy = policy
        self.list_actions = generatesIndexSlates(num_cand, slate_size)
        self.max_steps_per_episode = max_steps_per_episode
        # number of user states that matched a given expert state
        self.count = 0
        # total watch time by the expert during a user episode
        self.expert_watch = 0
        # total quality delivered by following the expert policy
        self.deleg_quality = 0

        self.deleg_q_temp = 0

        self.ml_model = ml_model

        self.loaded_model = None
        # list to save all state's qualities through a given session (episode)
        self.list_quality_videos = []
        # list to save watching time of selected videos through a given session (episode)
        self.list_watching_time = []

    def step(self, observation):

        """Receives observations of environment and returns a slate.

        Args:
        observation: A dictionary that stores all the observations including:
            - user: A list of floats representing the user's observed state
            - doc: A list of observations of document features
            - response: A vector valued response signal that represent user's
            response to each document

        Returns:
        slate: An integer array of size _slate_size, where each element is an
            index in the list of document observvations.
        """


        if not self.ml_model:
            for i in range(len(self.states)):
                if self.find_state_(observation, self.states[i], margin_score=0.5, margin_interests=0.5):
                    #print("state found ")
                    for j in self.states[i][1]:
                        if j[0] == 1:
                            self.expert_watch += j[5]
                            self.deleg_q_temp = (j[1] + j[2] + j[3] + j[4]) / 4
                            self.deleg_quality += self.deleg_q_temp

                    self.count += 1
                    return self.list_actions[np.argmax(self.policy[i])], 1
            #print("state not found")
            return self.list_actions[np.random.randint(0, len(self.list_actions), dtype=int)], 0
        else:
            self.loaded_model = pickle.load(open(filename, 'rb'))
            s = self.loaded_model.predict([observation['user']])

            return self.list_actions[int(s)], 1



    def find_state(self, user_state, expert_state, margin_features=0.1):
        for j in range(0, len(user_state)):
            if abs(user_state['user'][j] - expert_state[0][j + 1]) > margin_features:
                return False
        return True

    def find_state_(self, user_state, expert_state, margin_score=0.1, margin_interests=0.5):

        """
        Implements a simple classification algorithm to compare states accordings to some margins.
        This function could be overwritten or modified depending on what criteria or similarities 
        you want to evaluate.
        Args:
            user_state:  A list of floats representing the user's observed state from the user observation
            expert_satte: A list of floats representing the expert's observed state from the  dataset
            margin_score: to compare video's quality
            margin_interests: to compare user and expert's interests.

        Returns:
            True if the state has been found, and False if it is not;


        """

        assert (len(expert_state[0][1:]) == len(user_state['user'])
                ), 'user interests size does not match'

        for i in range(len(list(user_state['doc'].values()))):
            if not (list(user_state['doc'].values())[i][:len(expert_state[0][1:])] == list(expert_state[2].values())[i][
                                                                        :len(expert_state[0][1:])]).all():
                #print("false1 ")
                return False
        if (abs(user_state['user'][1] - expert_state[0][2]) > margin_interests):
            #print("false2 ",user_state['user'], expert_state[0])
            return False
        for i in range(len(list(user_state['doc'].values()))):
            a = np.dot(user_state['user'], list(user_state['doc'].values())[i][:len(expert_state[0][1:])]) * \
                list(user_state['doc'].values())[i][-2]
            b = np.dot(expert_state[0][1:], list(expert_state[2].values())[i][:len(expert_state[0][1:])]) * \
                (sum(list(expert_state[2].values())[i][len(expert_state[0][1:]):]) /
                len(list(expert_state[2].values())[i][len(expert_state[0][1:]):]))
            if abs(a - b) > margin_score:
                #print("false3 ", abs(a - b))
                return False
        return True

    def run_one_episode(self):

        """
        Runs one episode with the given configuration 

        Returns:

        step_number: length of the episode
        total_reward: total watching_time and quality for this episode
        time_dif: execution time of this episode
        c_found: number of user states that have been matched to expert states by the classifier
        expert_watch: total watching time by state experts for user states that are similar to those experts states
        total_clicked: number of clicked videos throughout this episode
        total_length_videos: total duration time of the clicked videos
        total_deleg_q = total quality calculated by slates recommended by the expert policy
        total_episode_q = total quality of the episode for all the clicked watched videos 

        """
        # Initialize the envronment
        observation = self.env.reset()
        action, test = self.step(observation)
        step_number = 1
        total_reward = 0.
        self.count = 0
        self.expert_watch = 0
        self.deleg_quality = 0
        total_quality_exp = 0
        total_quality_not_found = 0
        total_clicked = 0
        total_length_videos = 0
        start_time = time.time()
        c_found = 0
        while True:
            # execute the action and receives the reward and the new observation from the environment
            observation, reward, done, info, _ = self.env.step(action)
            for j in range(len(observation['response'])):
                # if the user has clicked on the video
                if observation['response'][j]['click'] == 1:
                    # user state is found ( matched to a similar expert state)
                    if test == 1:
                        c_found += 1
                        if reward != None:
                            self.list_watching_time += [reward]
                            self.list_quality_videos += [self.deleg_q_temp]
                    elif test == 0:
                        total_quality_not_found += float(observation['response'][j]['quality'])
                        if reward != None:
                            self.list_quality_videos += [float(observation['response'][j]['quality'])]
                            self.list_watching_time += [reward]
                    index = action[j]
                    total_length_videos += list(observation['doc'].values())[index][-1]

                    total_clicked += 1

                    total_reward += reward

                    break
            self.env.update_metrics(observation['response'], info)

            step_number += 1

            if done:
                break
            elif step_number == self.max_steps_per_episode:
                # Stop the run loop once we reach the true end of episode.
                break
            else:
                # receive the new slate from the agent (classifier)
                action, test = self.step(observation)

        time_diff = time.time() - start_time
        total_delga_q = self.deleg_quality/c_found
        total_episode_q = (total_quality_not_found + self.deleg_quality)/total_clicked

        return step_number, total_reward, time_diff, c_found, self.expert_watch,\
            [total_clicked, total_length_videos, total_delga_q, total_episode_q]
        

    def videos_info(self):
        """ Returns the lists of watching time and the delivered 
            quality for states of the associated episode"""
        return self.list_watching_time, self.list_quality_videos

def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.watch_time
    return reward
```

<!-- #region id="DsUju-_NzCK4" -->
### Evaluation of the AL/IRL model
In this section, we evaluate the performance of the expert policy generated by AL/IRL component, which saved for our simulation in policyV1.npy. We also use its associated states dataset stored in file statesV1.npy .
<!-- #endregion -->

```python id="YPgbr5t1zJgQ" executionInfo={"status": "ok", "timestamp": 1636650317299, "user_tz": -330, "elapsed": 499, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
max_episode = 500
slate_size = 2
num_candidates = 5

env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function':clicked_engagement_reward
    }

# User simulation environment: interest evolution model presented in the paper of SlateQ
recsim_gym_env = interest_evolution.create_environment(env_config)

# Load the learned policy from the Expert IRL model as well as the resulted/associated states
states = load('datasets_states/statesV1.npy', allow_pickle=True)
policy_ = load('datasets_states/policyV1.npy', allow_pickle=True)

# Instanciate the Expert IRL agent 
agent_irl = InverseRLAgent(recsim_gym_env, states, policy_, num_cand=num_candidates,
                               slate_size=slate_size, max_steps_per_episode=100, ml_model=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["22a53f52ca2047698ae05864e762ef8a", "0d10299108354dfa9928570dff499655", "9d2861d225f94d13957494e1bd59f66c", "1ed38f3e2e084043b474f0da5a12ff6a", "4ad72bce8ab5463c9380beaaedffc7af", "6d6530aa93bb44e8bbc3812c9b6be4e9", "0721165dccd84229b7d7238c9c293891", "413a3bb71c4e471687cb6b57c0e572c2", "eb14fecf938f42689453fd9f79420796", "2fc4d2a8cd7c42a8ac617e6aa8fe9957", "46093cc1f649436a8400ad91e4e203e3"]} id="8SaoBS4rzOI6" executionInfo={"status": "ok", "timestamp": 1636648747153, "user_tz": -330, "elapsed": 842641, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b0424ab7-3d64-46ef-df43-89593b7ef1e2"
results = []
# generate independent episodes
for i in tqdm(range(max_episode)):
    steps, reward, time_, found, exp_watch, user_metrics = agent_irl.run_one_episode()
    results += [[i,steps, reward, time_, found, exp_watch, user_metrics]]
```

```python id="xwkP1woLzevD"
episodes = np.linspace(0,max_episode-1, max_episode)
user_reward = []
expert_reward = []
episode_steps = []
episode_found_state = []
episode_clicked = []
episode_videos_length = []
episode_total_quality = []
episode_expert_quality = []

for i in range(len(results)):
    episode_steps += [results[i][1]]
    user_reward += [results[i][2]]
    episode_found_state += [results[i][4]]
    expert_reward += [results[i][5]]
    episode_clicked += [results[i][6][0]]
    episode_videos_length += [results[i][6][1]]
    episode_expert_quality += [results[i][6][2]]
    episode_total_quality += [results[i][6][3]]
    
ratio_watch = [user_reward[i] for i in range(len(episode_videos_length))]
```

```python id="6uHHV1o20eJO" executionInfo={"status": "ok", "timestamp": 1636648862900, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from numpy import save
a = results
save('results.npy', a)
```

<!-- #region id="mOMLrXSR0nPb" -->
The table below shows the following simulation metrics:

- #episode_steps: length of the episode ( number of states)
- #episode_match_state: number of user states that have been matched by the classification algorithm to similar expert states
- T_clicked videos: Total number of videos that have been clicked by users within the episode (session)
- T_episode_quality: total quality calculated for the episode
expert_deleg_quality: quality calculated for watched videos that belongs to states of #episode_match_state
- T_videos_length: duration of all videos watched in a given episode
- T_watching_time: Total watching time of clicked videos

> Note 1: In some episodes, T_watching_time values are greater than T_videos_length while it is supposed to be the contrary. However, this situation may happen because of some videos that are watched more than one time.

> Note 2: Metrics T_episode_quality and expert_deleg_quality gives an indea about how much it is beneficial to follow expert rated videos, which are recommended based on our approach. For instance, in the episode 0, we have 30 videos that have been seen through the expert policy of our approach. These videos are delivering a quality of 0.449174. However, for the rest 32 videos (62-30=32), which are watched by a random recommendation because their states are not matched to any expert state from the states dataset, they then led to dicrease the quality (by negative quality values) ending up by a total quality of 0.32.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 346} id="nusGHi3Q0x0u" executionInfo={"status": "ok", "timestamp": 1636648866169, "user_tz": -330, "elapsed": 954, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a2fc836c-ab1c-44d9-e883-422d3ba83de6"
import pandas as pd
data = np.array([episode_steps, episode_found_state,  episode_clicked, 
            episode_total_quality, episode_expert_quality, episode_videos_length, user_reward, ratio_watch])
pd.DataFrame(data, columns=np.arange(max_episode), index=["#episode_steps", "#episode_match_state",  "T_clicked videos",
                             "T_episode_quality", "expert_deleg_quality", "T_videos_length", "T_watching_time"," ratio_watching"])
```

<!-- #region id="v8h2z1XY1AJQ" -->
Figure below shows that the delegated expert quality values positively increase the total quality as explained in note 2 above.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 569} id="gC1xWTMA1DSK" executionInfo={"status": "ok", "timestamp": 1636648871755, "user_tz": -330, "elapsed": 1562, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7eb9b989-4b0a-4d9a-8e6b-95a96f1febe7"
#plt.plot(episodes, ratio_watch, color='blue', label = "watching_time", marker='o')
plt.figure(figsize=(12,9))
plt.scatter(episodes, episode_total_quality, color='red',  s=5, label = "$Q_T$", marker='o')
plt.scatter(episodes, episode_expert_quality, color='green', s=5, label = "$Q_d$", marker='o')
plt.ylim(-1, 1)
#plt.xticks(np.arange(min(episodes), max(episodes)+1, 1.0))
plt.xlabel("episodes")
plt.ylabel("quality rate")
plt.legend(loc='upper right')
plt.title("Figure 1: expert delegated quality $Q_d$ and total quality $Q_T$ by episodes")
plt.savefig("q_eval.png", bbox_inches='tight')
plt.show()
```

<!-- #region id="BDdvf8C51bO6" -->
## Comparison of recFEBR with the baseline methodes: recFSQ and recNaive
<!-- #endregion -->

<!-- #region id="OmWgmFSr1enN" -->
### Training
<!-- #endregion -->

```python id="ybMSD5Lx1gRO" executionInfo={"status": "ok", "timestamp": 1636650293636, "user_tz": -330, "elapsed": 623, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from recsim.agents import  full_slate_q_agent, random_agent
# from irl_agent import InverseRLAgent
import tensorflow.compat.v1 as tf
import time
from numpy import load


def run_one_episode(env, agent, max_steps_per_episode=100):
    observation = env.reset()
    action = agent.begin_episode(observation)
    step_number = 0
    total_watch = 0.
    q_videos = []
    w_videos = []
    total_qual = 0
    start_time = time.time()
    total_length_videos = 0
    while True:
        observation, reward, done, info, _ = env.step(action)

        for j in range(len(observation['response'])):
            if observation['response'][j]['click'] == 1:
                index = action[j]
                total_length_videos += list(observation['doc'].values())[index][-1]
                total_watch += reward[1]
                total_qual += reward[0]
                q_videos += [reward[0]]
                w_videos += [reward[1]]

        # Update environment-specific metrics with responses to the slate.
        env.update_metrics(observation['response'], info)
        step_number += 1

        if done:
            break
        elif step_number == max_steps_per_episode:
            # Stop the run loop once we reach the true end of episode.
            break
        else:
            action = agent.step(reward[1], observation)

    agent.end_episode(reward[1], observation)
    time_diff = time.time() - start_time

    return step_number, total_watch, time_diff, total_qual/step_number, q_videos, w_videos


def clicked_quality_reward(responses):
    """Calculates the total clicked watchtime from a list of responses.

    Args:
      responses: A list of IEvResponse objects

    Returns:
      reward: A float representing the total watch time from the responses
    """
    qual = 0.0
    watch = 0.0
    for response in responses:
        if response.clicked:
            qual += float(response.quality)
            watch += float(response.watch_time)
    return [qual, watch]


def create_agent_random(slate_size, random_seed=0):
    action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
    return random_agent.RandomAgent(action_space, random_seed)
```

<!-- #region id="lkT_HTFX1_FS" -->
### recNaive simulation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["b8b5e81a95754c5d95d39d159039dcdf", "dfedfa2679864d34a479e44ab0fe4044", "793d0bf67420463396f243b075dbef25", "3b1f28a8fb0641cfb09e319168d7b2cc", "36fec969c7734f1794529e9ecd632b99", "30b84bb0ce384b029b031fdec0f49975", "7a560b55c875407c8e0a685ebcf21ad5", "9f20563f50fc44ddbb028751a5cbec1b", "bbcaad3baa9841d58cf930a3bc8d598f", "1ed082baf73a49229c7764f5be1db7ec", "5fc7e4da6f2a4498b5c6276ebf99a0e1"]} id="hySMxpqP1mh4" executionInfo={"status": "ok", "timestamp": 1636650369519, "user_tz": -330, "elapsed": 27564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8fe56106-4c0c-4a8b-8189-79a65086ec8d"
slate_size = 2
num_candidates = 5
env_config1 = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function':clicked_quality_reward
    }

recsim_gym_env = interest_evolution.create_environment(env_config1)
agent = create_agent_random(slate_size)
results_r = []

for i in tqdm(range(max_episode)):
    steps_r, watch, time_r, q, q_vid, w_vid = run_one_episode(recsim_gym_env, agent)
    results_r += [[i,steps_r, watch, time_r, q, q_vid, w_vid ]]
    
episode_steps_r = []
episode_ratio_watch_r = []
episode_total_quality_r = []
episodes_qv = []
episodes_wv = []
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["20db74448497490887c476f9ac00c3e3", "5158fe83133c4849bfedca8651d04192", "951176dd041d4512975a65f937a0a361", "9dffff92dcc84f0e84f93674ab66d028", "ec838107d81842dd81e7a8185410d8db", "a6c56a83f81847ad9b4753255e539fea", "4c4b73b84b9348eaa411c2eba7bad07f", "e246ef712bc34db9bc2b6d7478f70fa8", "454ba4378ea544db9a58effa35f305a8", "b387b3c5300b41e28b73f97fe323c5cf", "b1210b71f13c40d1b56b94c32d05f02e"]} id="OoO4Ronp1shV" executionInfo={"status": "ok", "timestamp": 1636650369522, "user_tz": -330, "elapsed": 46, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a5e90fbd-6eef-4c0e-a3c0-9a9c4eefe3a4"
for i in tqdm(range(len(results_r))):
    episode_steps_r += [results_r[i][1]]
    episode_ratio_watch_r += [results_r[i][2]]
    episode_total_quality_r += [results_r[i][4]]
    episodes_qv += [results_r[i][5]]
    episodes_wv += [results_r[i][6]]
```

<!-- #region id="IzduKMS72CqA" -->
### recFSQ simulation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["d1431ba94998486a92a26847d92d6d5e", "b7f9e4b30a144adc93f348a9b2e7cf4b", "ac55c48b776a4e3e93e309cb960b58e9", "093c3017bbea4508ac504f2898d51b32", "b8d4309a172f4ab9a5fcc48dbefeafee", "26aa981e8f804516a6390ea08f4c8893", "c479a4aebf4b41f8abfe0e0ce68083fc", "aac33c8017af4fc8a8bb9c7b35886633", "bb13cff673624467bd40f0fab99a38ac", "ffc2db41ebbc4c72aed2c7e5f93c56ab", "d4f2055e0ca44fff8a2f33bc3d75a855"]} id="wjeIMgsF2DRD" executionInfo={"status": "ok", "timestamp": 1636653572468, "user_tz": -330, "elapsed": 3161951, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7811e480-8941-4767-9e2d-d9b93f6dd2d7"
slate_size = 2
num_candidates = 5
env_config1 = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function':clicked_quality_reward
    }

recsim_gym_env = interest_evolution.create_environment(env_config1)
results_f = []

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    
    agent = full_slate_q_agent.FullSlateQAgent(sess, recsim_gym_env.observation_space, recsim_gym_env.action_space)
    
    for i in tqdm(range(max_episode)):
        steps_f, watch, time_f, q, q_vid, w_vid = run_one_episode(recsim_gym_env, agent)
        results_f += [[i, steps_f, watch, time_f, q, q_vid, w_vid]]
        sess.run(tf.global_variables_initializer())
    
episode_steps_f = []
episode_ratio_watch_f = []
episode_total_quality_f = []
episodes_qvf = []
episodes_wvf = []
```

```python colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["4b0dea9702424e2fb4ac008134bfe355", "5039f734a47447449de08e72742f1c34", "37302b9fb25b4ae4a6840ac4846cd664", "9f1bfcc78fc84de1b760c12da77c8c09", "da5ee7ab8e3343f5b3200aa61a376285", "ee5d01fde43a48ecb2164fbc434c656e", "5345c139565e459ea666c0adf57ae6ee", "006ba7778e23405786d591aa8c27de98", "8b740fc1522848d09994b4f4e8f7e606", "0dac8457501c40f9834b8d26e1207ee5", "32d723e547ef490abeb80b20f1a5a77f"]} id="efbbSI422KOv" executionInfo={"status": "ok", "timestamp": 1636653589031, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b9fe8f53-19e9-49bb-9a5a-b4f539cfe676"
for i in tqdm(range(len(results_f))):
    episode_steps_f += [results_f[i][1]]
    episode_ratio_watch_f += [results_f[i][2]]
    episode_total_quality_f += [results_f[i][4]]
    episodes_qvf += [results_f[i][5]]
    episodes_wvf += [results_f[i][6]]
```

<!-- #region id="lgZKMLh22PDZ" -->
### Statistics and comparison
<!-- #endregion -->

```python id="1L5ri25aKrbj" outputId="986edf69-4f51-47ce-fb9a-794a0ea50dd4"
import numpy as np
import matplotlib.pyplot as plt

n_bins = 10

# Generate a normal distribution, center at x=0 and y=5

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(episode_total_quality, bins=n_bins)
axs[0].set_title("recFEBR")
axs[1].hist(episode_total_quality_f, bins=n_bins)
axs[1].set_title("recFSQ")
axs[2].hist(episode_total_quality_r, bins=n_bins)
axs[2].set_title("revNaive")
st=fig.suptitle("Figure 2: Empirical distribution of total quality $Q_T$", fontsize="x-large")
st.set_y(1.1)
```

```python id="Fg2TMUOKKrbk" outputId="a01d7006-1316-4bfe-a181-61a0120ad7e3"
import pandas as pd
data_quality = [sum(episode_total_quality_f) / max_episode, sum(episode_total_quality_r) / max_episode, sum(episode_total_quality) / max_episode]
data_watch_time = [sum(episode_ratio_watch_f), sum(episode_ratio_watch_r), float(sum(user_reward))]
data = np.array([data_quality, data_watch_time])
pd.DataFrame(data, columns=["recFSQ", "recNaive", "recFEBR"], index=["Average total quality $Q_T$","Total watching time $W_T$"])
```

```python id="c8XCTJbIKrbl" outputId="4ffe7df7-6449-44e3-85e4-47859511af48"
df_quality = pd.DataFrame([episode_total_quality, episode_total_quality_f, episode_total_quality_r], index=["FEBR","Full_Q","Naive_random_"])
data1 = df_quality.T
data1.describe()
```

```python id="marhlcrmKrbm" outputId="0f2bbc11-f16a-460b-f43b-0f3f7e9c2183"
name_file = "w_t_compar.png"
approaches = ["RecFSQ","RecNaive", "RecFEBR"]
plt.bar(approaches, data[1])
plt.ylabel('Total watch time (s)')
plt.title("Figure 3: Comparison of the total watch time $W_T$ (seconds)")
# saving file into result folder
#plt.savefig("../eval_results/"+name_file,, bbox_inches='tight')
```

```python id="jB9UljutKrbn" outputId="fdb55c24-e1fc-4f7d-8b28-3841cfa1571c"
name_file = "q_t_compar.png"
plt.bar(approaches, data[0])
plt.ylabel('Average total quality')
plt.yticks(np.arange(-0.2, 0.5, 0.05))
plt.title("Figure 4: Comparison of the average total quality $Q_T$ ")
# saving file into result folder
#plt.savefig("../eval_results/"+name_file, bbox_inches='tight')
```
