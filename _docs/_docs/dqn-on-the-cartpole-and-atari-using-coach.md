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

<!-- #region id="ywbetM9kk5XQ" -->
# DQN on the CartPole and Atari using Coach
<!-- #endregion -->

<!-- #region id="c1_k7c_gk_zM" -->
The Cartpole environment is a popular simple environment with a continuous state space and a discrete action space. Nervana Systems coach provides a simple interface to experiment with a variety of algorithms and environments. In this workshop you will use coach to train an agent to balance a pole.
<!-- #endregion -->

<!-- #region id="vgySGucdl911" -->
## Setup
<!-- #endregion -->

<!-- #region id="q9vFtLX5l5v8" -->
### Installations
<!-- #endregion -->

```python id="sa0FY3SFpfI2" executionInfo={"status": "ok", "timestamp": 1634464041566, "user_tz": -330, "elapsed": 773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import minio
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="m-Yq7j2rpgkE" executionInfo={"status": "ok", "timestamp": 1634464048538, "user_tz": -330, "elapsed": 749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fccf643c-8f1a-4947-b5c2-68d3b87e3a96"
minio.__version__
```

```python colab={"base_uri": "https://localhost:8080/"} id="gRAKqVcdpLZV" executionInfo={"status": "ok", "timestamp": 1634463975138, "user_tz": -330, "elapsed": 1282, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a47299f-3425-4d8b-f62a-fcc7e9287c0f"
%tensorflow_version 1.x
```

```python id="Z3X1433Ql9TQ"
!sudo -E apt-get install python3-pip cmake zlib1g-dev python3-tk python-opencv -y
!sudo -E apt-get install libboost-all-dev -y
!sudo -E apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y
!sudo -E apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev
!libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y
!sudo -E apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev 
!freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev
!libgstreamer-plugins-base1.0-dev -y
!sudo -E apt-get install libav-tools libsdl2-dev swig cmake -y
!pip install rl_coach
```

```python id="N8c06puFl9Cs"
!sudo -E apt-get install python3-pip cmake zlib1g-dev python3-tk python-opencv -y
!sudo -E apt-get install libboost-all-dev -y
!sudo -E apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y
!sudo -E apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y
!sudo -E apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev -y
!sudo -E apt-get install libav-tools libsdl2-dev swig cmake -y
!pip install setuptools==41.4.0
!pip install rl-coach==1.0.1 gym==0.12.5
!pip install gym[atari]
```

<!-- #region id="eVcNBfS2l5tE" -->
### Imports
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 510} id="wNgSWZTxm4aB" executionInfo={"status": "error", "timestamp": 1634463980032, "user_tz": -330, "elapsed": 830, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1d82be4-664f-4522-903e-62dd4d74d0a3"
import math
import random
from collections import defaultdict
from typing import Union
import numpy as np

from rl_coach.agents.agent import Agent
from rl_coach.base_parameters import AgentParameters, AlgorithmParameters
from rl_coach.core_types import ActionInfo, EnvironmentSteps
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_crop_filter import ObservationCropFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import ConstantSchedule
from rl_coach.agents.agent import Agent
from rl_coach.base_parameters import AlgorithmParameters, AgentParameters
from rl_coach.core_types import ActionInfo
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import ConstantSchedule

from CustomObservationFilters import ObservationRoundingFilter, ObservationScalingFilter
```

<!-- #region id="fPaA_9SJl3P0" -->
## Environment
The environment simulates balancing a pole on a cart. The agent can nudge the cart left or right; these are the actions. It represents the state with a position on the x-axis, the velocity of the cart, the velocity of the tip of the pole and the angle of the pole (0° is straight up). The agent receives a reward of 1 for every step taken. The episode ends when the pole angle is more than ±12°, the cart position is more than ±2.4 (the edge of the display) or the episode length is greater than 200 steps. To solve the environment you need an average reward greater than or equal to 195 over 100 consecutive trials.

## Coach Presets
Coach has a concept of presets which are settings for algorithms that are known to work.

## DQN Preset
The CartPole_DQN preset has a solution to solve the CartPole environment with a DQN. I took this preset and made a few alterations to leave the following parameters:

- It copies the target weights to the online weights every 100 steps of the environment
- The discount factor is set to 0.99
- The maximum size of the memory is 40,000 experiences
- It uses a constant greedy schedule of 0.05 (to make the plots consistent)
- The NN uses a mean-squared error (MSE) based loss, rather than the default Huber loss.
- No environment “warmup” to pre-populate the memory (to obtain a result from the beginning)

You can see the full DQN preset I used below:
<!-- #endregion -->

```python id="kVvPU_KPlarw"
# dqn_preset.py
# Adapted from https://github.com/NervanaSystems/coach/blob/master/rl_coach/presets/CartPole_DQN.py

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(200)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = DQNAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(
    100)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = ConstantSchedule(0.05)

################
#  Environment #
################
env_params = GymVectorEnvironment(level='CartPole-v0')

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())
```

<!-- #region id="FLAXgPFombHS" -->
### Q-Learning Agent and Preset
Coach doesn’t have a basic Q-learning algorithm or preset, so I implemented my own. You can see the code below.
<!-- #endregion -->

```python id="N34-YiIOmzs4"
class QLearningAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.discount = 0.99
        self.num_consecutive_playing_steps = EnvironmentSteps(1)


class QLearningAgentParameters(AgentParameters):
    def __init__(self, default_q=0, alpha=0.1):
        super().__init__(algorithm=QLearningAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={})
        self.default_q = default_q
        self.alpha = alpha

    @property
    def path(self):
        return 'q_learning_agent:QLearningAgent'


class QLearningAgent(Agent):
    def __init__(self, agent_parameters,
                 parent: Union['LevelManager', 'CompositeAgent'] = None):
        super().__init__(agent_parameters, parent)
        self.default_q = self.ap.default_q
        self.q_func = defaultdict(lambda: defaultdict(lambda: self.default_q))

    def train(self) -> float:
        loss = 0
        if self._should_train():
            # Required: State, action, reward
            transition = self.current_episode_buffer.get_last_transition()
            if transition is None:
                return loss
            state = tuple(transition.state["observation"])
            action = transition.action
            reward = transition.reward
            actions_q_values = self.get_all_q_values_for_states(
                transition.next_state)
            max_q_next_state = np.max(actions_q_values)
            delta = (reward + self.ap.algorithm.discount *
                     max_q_next_state - self.q_func[state][action])
            self.q_func[state][action] += self.ap.alpha * delta

            # Coach want's me to return the total training loss, but we're not
            # really training. Instead, I will return the TD error.
            loss = np.abs(delta)
        return loss

    def get_all_q_values_for_states(self, state):
        # This is almost a replica of the ValueIterationAgent. Probably could
        # be refactored to use that.
        state = tuple(state["observation"])
        l = np.array([self.q_func[state][a]
                      for a in self.spaces.action.actions])
        # Add a little random noise to all q_values to prevent ties
        # See https://github.com/NervanaSystems/coach/issues/414
        l = l + np.random.normal(loc=0, scale=0.000000001, size=l.shape)
        return l

    def choose_action(self, curr_state):
        actions_q_values = self.get_all_q_values_for_states(curr_state)
        action, action_probabilities = self.exploration_policy.get_action(
            actions_q_values)
        action_info = ActionInfo(action=action,
                                 action_value=actions_q_values[action],
                                 max_action_value=np.max(actions_q_values),
                                 all_action_probabilities=action_probabilities)
        return action_info
```

<!-- #region id="URKopX1qms9r" -->
Then the preset looks like:
<!-- #endregion -->

```python id="uX2l7NRtmu9E"
####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(200)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = QLearningAgentParameters(alpha=0.5)
agent_params.algorithm.discount = 0.99

# Simplify the observations. I want to only use the angle and angular velocity.
# And I want to place the continuous observations into bins. This is achieved
# by multiplying by 10 and rounding to an integer. This limits the total number
# of states to about 150.
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_observation_filter(
    "observation",
    "cropping",
    ObservationCropFilter(crop_low=np.array([2]), crop_high=np.array([4])),
)
agent_params.input_filter.add_observation_filter(
    "observation", "scaling", ObservationScalingFilter(10.0)
)

agent_params.input_filter.add_observation_filter(
    "observation", "rounding", ObservationRoundingFilter()
)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = ConstantSchedule(0.05)

################
#  Environment #
################
env_params = GymVectorEnvironment(level="CartPole-v0")

graph_manager = BasicRLGraphManager(
    agent_params=agent_params, env_params=env_params, schedule_params=schedule_params
)
```

<!-- #region id="3jAx1oECmkpD" -->
### Random Agent and Preset
To provide a baseline for the other algorithms, I implemented a random agent and preset because coach doesn’t provide one out of the box.
<!-- #endregion -->

```python id="SGWbfRCPml1e"
class RandomAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=AlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={})

    @property
    def path(self):
        return 'random_agent:RandomAgent'


class RandomAgent(Agent):
    def __init__(self, agent_parameters,
                 parent: Union['LevelManager', 'CompositeAgent'] = None):
        super().__init__(agent_parameters, parent)

    def train(self):
        return 0

    def choose_action(self, curr_state):
        action_info = ActionInfo(
            action=self.exploration_policy.action_space.sample())
        return action_info
```

<!-- #region id="JTY9G2RGmc7y" -->
And the preset:
<!-- #endregion -->

```python id="O8DC62AvmeBW"
####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(200)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(200)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = RandomAgentParameters()

################
#  Environment #
################
env_params = GymVectorEnvironment(level='CartPole-v0')
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params)
```
