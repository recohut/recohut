# TensorFlow 2 Reinforcement Learning Cookbook

## Process flow

![https://github.com/RecoHut-Projects/drl-recsys/raw/main/images/S990517_process_flow.svg](https://github.com/RecoHut-Projects/drl-recsys/raw/main/images/S990517_process_flow.svg)

## Environments

### Simple Gridworld

This is a simple environment where the world is represented as a grid. Each location on the grid can be referred to as a cell. The goal of an agent in this environment is to find its way to the goal state in a grid like the one shown here:

![Untitled](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled.png)

The agent's location is represented by the blue cell in the grid, while the goal and a mine/bomb/obstacle's location is represented in the grid using green and red cells, respectively. The agent (blue cell) needs to find its way through the grid to reach the goal (green cell) without running over the mine/bomb (red cell).

### Simple Gridworld v2

This is a simplified version of the grid environment. This simplification would help in understanding and visualizing the learning processes of various algorithms. As shown in the figure below, it is a 3x4 gridworld, where the goal is at [0,3] and the bomb is at [1,3]. (so near to each other, that our RL agent need to learn moving with care 😇).

![Untitled](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-1.png)

### Stochastic Maze

To train RL agents for the real world, we need learning environments that are stochastic, since real-world problems are stochastic in nature. This recipe will walk you through the steps for building a Maze learning environment to train RL agents. The Maze is a simple, stochastic environment where the world is represented as a grid. Each location on the grid can be referred to as a cell. The goal of an agent in this environment is to find its way to the goal state. Consider the maze shown in the following diagram, where the black cells represent walls:

![Untitled](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-2.png)

The agent's location is initialized to be at the top-left cell in the Maze. The agent needs to find its way around the grid to reach the goal located at the top-right cell in the Maze, collecting a maximum number of coins along the way while avoiding walls. The location of the goal, coins, walls, and the agent's starting location can be modified in the environment's code.

The reward is based on the number of coins that are collected by the agent before they reach the goal state. Because the environment is stochastic, the action that's taken by the environment has a slight (0.1) probability of "slipping" wherein the actual action that's executed will be altered stochastically.

### CartPole

In this environment, a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

![CartPole before training. [source](https://gsurma.medium.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288).](https://miro.medium.com/max/600/1*LnQ5sRu-tJmlvRWmDsdSvw.gif)

CartPole before training. [source](https://gsurma.medium.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288).

![CartPole after training. [source](https://gsurma.medium.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288).](https://miro.medium.com/max/600/1*jLj9SYWI7e6RElIsI3DFjg.gif)

CartPole after training. [source](https://gsurma.medium.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288).

### MountainCar

In this environment, a car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

### Pendulum

In this environment, the pendulum starts in a random position, and the goal is to swing it up so it stays upright.

<iframe width="727" height="409" src="https://www.youtube.com/embed/XbXZ9dmKG_s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Cryptocurrency Trading

This environment simulates a Bitcoin trading exchange based on real-world data from the Gemini cryptocurrency exchange. In this environment, your RL agent can place buy/sell/hold trades and get rewards based on the profit/loss it makes, starting with an initial cash balance in the agent's trading account.

In continuous action space, instead of allowing the Agent to only take discrete actions, such as buying/selling/holding a pre-set amount of Bitcoin or Ethereum tokens, we allow the Agent to decide how many crypto coins/tokens it would like to buy or sell.

### Stock Trading

The stock market provides anyone with a highly lucrative opportunity to participate and make profits. While it is easily accessible, not all humans can make consistently profitable trades due to the dynamic nature of the market and the emotional aspects that can impair people's actions. RL agents take emotion out of the equation and can be trained to make profits consistently.

This stock market trading environment enable RL agents to trade stocks using real stock market data. When you have trained them enough, you can deploy them so that they automatically make trades (and profits) for you!

### WebGym World-of-Bits (WoB)

WebGym is a **World of Bits** (**WoB**)-based OpenAI Gym compatible learning platform for training RL Agents for world wide web-based real-world tasks. It provides learning environments for agents to perceive the world-wide-web like how we (humans) perceive – using the pixels rendered on to the display screen. The agent interacts with the environment using keyboard and mouse events as actions. This allows the agent to experience the world-wide-web like how we do thereby require no new additional modifications for the agents to train. This allows us to train RL agents that can directly work with the web-based pages and applications to complete real-world tasks. For more information about WoB, check out [this](http://proceedings.mlr.press/v70/shi17a/shi17a.pdf) link.

### Building a simple Gridworld Environment

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T533231_Building_a_simple_Gridworld_Environment.ipynb)

In this, we are building the gridworld environment as a python class object `GridworldEnv` by subclassing `gym.Env` module.

### Building a simple Gridworld v2 Environment

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T195475_Building_a_simple_Gridworld_v2_Environment.ipynb)

In this, we are building a simple 3x4 gridworld environment as a python class object `GridworldV2Env` by subclassing `gym.Env` module.

### Building a Stochastic Maze Gridworld Environment

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T495794_Building_a_Stochastic_Maze_Gridworld_Environment.ipynb)

In this, we are building a stochastic maze environment as a python class object `MazeEnv` by subclassing `gym.Env` module. The slip probability is set to 10%.

### Training RL Agent in Gridworld with MLP Model

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T490651_Training_RL_Agent_in_Gridworld_Environment_with_MLP_Model.ipynb)

In this, we are first building the simple gridworld environment. Then we are building the agent model `Brain` by subclassing `keras.Model`, and a wrapper class `Agent`. Then we train the agent in the given gridworld environment.

### Training RL Agent in Maze Gridworld with Value-iteration

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T920001_Training_RL_Agent_in_Maze_Gridworld_with_Value_iteration_method.ipynb)

In this, we are first building the stochastic maze environment. Then we apply value-iteration to learn the optimal actions in each and every state.

![Agent started with a random policy.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-3.png)

Agent started with a random policy.

![And after few iterations, learned the optimal policy.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-4.png)

And after few iterations, learned the optimal policy.

```
Action mapping:[0 - UP; 1 - DOWN; 2 - LEFT; 3 - RIGHT
Optimal actions:
[1. 1. 1. 1. 1. 1. 1. 1. 3. 3. 3. 3. 3. 3. 3. 3. 1. 1. 3. 1. 3. 1. 3. 3.
 3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 0. 0. 3. 3. 0. 0. 2. 2. 0. 2. 0. 2. 0. 0.
 0. 1. 0. 0. 1. 1. 0. 1. 0. 3. 0. 0. 3. 3. 0. 3. 0. 3. 0. 0. 3. 0. 0. 0.
 1. 1. 2. 2. 3. 3. 3. 3. 1. 1. 1. 2. 1. 0. 0. 0. 1. 1. 1. 0. 1. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 2. 0. 2. 0. 0. 0. 0. 0.]
```

### Training RL Agent in Gridworld with Temporal Difference

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T122762_Training_RL_Agent_in_Gridworld_with_Temporal_Difference_learning_method.ipynb)

In this, we are first building the gridworld v2 environment. Then we learn the optimal policy by applying temporal-difference learning method.

![Ground-truth state.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-1.png)

Ground-truth state.

![optimal state values learned by the TD algorithm.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-5.png)

optimal state values learned by the TD algorithm.

### Training RL Agent in Gridworld with Monte-Carlo

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T303629_Training_RL_Agent_in_Gridworld_with_Monte_Carlo_Prediction_and_Control_method.ipynb)

In this, we are first building the gridworld v2 environment. Then we applied the monte-carlo prediction method to learn the optimal state values and later on, we also applied monte-carlo control method to learn the optimal action values.

![Monte Carlo Prediction.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-6.png)

Monte Carlo Prediction.

![Monte Carlo Control.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-7.png)

Monte Carlo Control.

### Training RL Agent in Gridworld with SARSA

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T515396_Training_RL_Agent_in_Gridworld_with_SARSA_method.ipynb)

In this, we are first building the gridworld v2 environment. Then we applied the SARSA algorithm to learn the optimal action-values.

![Action values learned by SARSA.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-8.png)

Action values learned by SARSA.

### Training RL Agent in Gridworld with Q-learning

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T453493_Training_RL_Agent_in_Gridworld_with_Q_learning_method.ipynb)

In this, we are first building the gridworld v2 environment. Then we applied the Q-learning algorithm to learn the optimal action-values.

![Action values learned by Q-learning.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-9.png)

Action values learned by Q-learning.

### Training RL Agent in CartPole Env. with Actor-Critic

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T043789_Training_RL_Agent_in_CartPole_Environment_with_Actor_Critic_method.ipynb)

In this, we first create the CartPole environment using `gym.make` api call. Then we built the agent model `ActorCritic` by subclassing `keras.Model`, and a wrapper class `Agent`. Then we train the agent in this CartPole environment.

### Training RL Agent in CartPole Env. with DQN

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T473399_Training_RL_Agent_in_CartPole_Environment_with_DQN_method.ipynb)

In this, we first create the CartPole environment using `gym.make` api call. Then we built the `ReplayBuffer` and `DQN` class objects, and a wrapper class `Agent`. Then we train the agent in this CartPole environment.

### Training RL Agent in CartPole Env. with Dueling DQN

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T432381_Training_RL_Agent_in_CartPole_Environment_with_Dueling_DQN_method.ipynb)

In this, we first create the CartPole environment using `gym.make` api call. Then we built the `ReplayBuffer` and `DuelingDQN` class objects, and a wrapper class `Agent`. Then we train the agent in this CartPole environment.

### Training RL Agent in CartPole Env. with DRQN

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T244614_Training_RL_Agent_in_CartPole_Environment_with_DRQN_method.ipynb)

In this, we first create the CartPole environment using `gym.make` api call. Then we built the `ReplayBuffer` and `DRQN` class objects, and a wrapper class `Agent`. Then we train the agent in this CartPole environment.

### Training RL Agent in MountainCar Env. with Policy Gradient

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T611861_Training_RL_Agent_in_Mountain_Car_Environment_with_Policy_gradient_method.ipynb)

In this, we first create the MountainCar environment using `gym.make` api call. Then we built the agent model `PolicyNet` by subclassing `keras.Model`, and a wrapper class `Agent`. Then we train the agent in this MountainCar environment.

### Training RL Agent in MountainCar Env. with A3C Continuous

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T307891_Training_RL_Agent_in_Mountain_Car_Environment_with_A3C_Continuous_method.ipynb)

In this, we first create the MountainCar-continuous environment using `gym.make` api call. Then we built the `Actor` and `Critic` class objects, and a wrapper class `Agent`. For A3C algorithm, we also built `A3CWorker` object by subclassing `thread.Thread` module. Then we train the agent in this MountainCar-continuous environment.

### Training RL Agent in Pendulum Env. with PPO Continuous

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T626473_Training_RL_Agent_in_Pendulum_Environment_with_PPO_Continuous_method.ipynb)

In this, we first create the Pendulum environment using `gym.make` api call. Then we built the `Actor` and `Critic` class objects, and a wrapper class `Agent`. Then we train the agent in this Pendulum environment.

### Training RL Agent in Pendulum Env. with DDPG

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T559464_Training_RL_Agent_in_Pendulum_Environment_with_DDPG_method.ipynb)

In this, we first create the Pendulum environment using `gym.make` api call. Then we built the `ReplayBuffer`, the `Actor` and `Critic` class objects, and a wrapper class `Agent`. Then we train the agent in this Pendulum environment.

### Building Bitcoin and Ethereum Cryptocurrency Trading Env.

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T350011_Building_Bitcoin_and_Ethereum_Cryptocurrency_based_Trading_RL_Environment.ipynb)

In this, we implemented custom OpenAI Gym-compatible learning environments for cryptocurrency trading with both discrete and continuous-value action spaces.

- Building a Bitcoin trading RL platform using real market data (`CryptoTradingEnv` class object)
- Building an Ethereum trading RL platform using price charts (`CryptoTradingVisualEnv` class object)
- Building an advanced cryptocurrency trading platform for RL agents (`CryptoTradingContinuousEnv`, and `CryptoTradingVisualContinuousEnv` class objects)

We used the Bitcoin (`Gemini_BTCUSD_d.csv`) and Ethereum (`Gemini_ETHUSD_d.csv`) data from Gemini in building the environment.

### Training RL Agent for Trading Cryptocurrencies with SAC

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T778350_Training_an_RL_Agent_for_Trading_Cryptocurrencies_using_SAC_method.ipynb)

In this, we are first building the crypto trading environment `CryptoTradingContinuousEnv`, and then using the SAC method to train the RL agent to trade the Bitcoins in the trading RL environment.

![Untitled](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-10.png)

### Building Stock Trading RL Environment

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T344654_Building_Stock_Trading_RL_Environment.ipynb)

In this, we are:

- Building a stock market trading RL platform using real stock exchange data
- Building a stock market trading RL platform using price charts
- Building an advanced stock trading RL platform to train agents to mimic professional traders

We used the Tesla (`TSLA.csv`) and Microsoft (`MSFT.csv`) stocks data from Gemini in building the environment.

### Training RL Agent for Trading Stocks with SAC

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T836251_Training_an_RL_Agent_for_Trading_Stocks_using_SAC_method.ipynb)

In this, we are first building the stocks trading environment `StockTradingContinuousEnv`, and then using the SAC method to train the RL agent to trade the Tesla stocks in the trading RL environment.

### Building RL Agent to complete tasks on the web – Call to Action

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T702798_Building_an_RL_Agent_to_complete_tasks_on_the_web_%E2%80%93_Call_to_Action.ipynb)

In this, we are first building the `MiniWoBClickButtonVisualEnv` environment, and then training a PPO agent to handle **Call-To-Action** (**CTA**) type tasks for you. CTA buttons are the actionable buttons that you typically find on web pages that you need to click in order to proceed to the next step. While there are several CTA button examples available, some common examples include the **OK**/**Cancel** dialog boxes, where you need you to click to acknowledge/dismiss the pop-up notification, and the **Click to learn more** button. 

The following image illustrates a set of observations from a randomized CTA environment (with different seeds) so that you understand the task that the Agent will be solving:

![Screenshot of the Agent's observations from a randomized CTA environment.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-11.png)

Screenshot of the Agent's observations from a randomized CTA environment.

Note that for simplicity, we used one instance of the environment, though the code can scale for a greater number of environment instances to speed up training.

To understand how the Agent training progresses, consider the following sequence of images. During the initial stages of training, when the Agent is trying to understand the task and the objective of the task, the Agent may just be executing random actions (exploration) or even clicking outside the screen, as shown in the following screenshot:

![Agent clicking outside the screen (no visible blue dot) during initial exploration.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-12.png)

Agent clicking outside the screen (no visible blue dot) during initial exploration.

As the Agent learns by stumbling upon the correct button to click, it starts to make progress. The following screenshot shows the Agent making some progress:

![Deep PPO Agent making progress in the CTA task.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-13.png)

Deep PPO Agent making progress in the CTA task.

Finally, when the episode is complete or ends (due to a time limit), the Agent receives an observation similar to the one shown in the following screenshot (left):

![End of episode observation (left) and summary of performance (right).](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-14.png)

End of episode observation (left) and summary of performance (right).

### Building RL Agent to auto-login on the web

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T769395_Building_an_RL_Agent_to_auto_login_on_the_web.ipynb)

Imagine that you have an Agent or a bot that watches what you are doing and automatically logs you into websites whenever you click on a login screen. While browser plugins exist that can automatically log you in, they do so using hardcoded scripts that only work on the pre-programmed website's login URLs. But what if you had an Agent that only relied on the rendered web page – just like you do to perform a task – and worked even when the URL changes and when you are on a new website with no prior saved data? How cool would that be?!

In this, we are first building the `MiniWoBLoginUserVisualEnv` environment, and then training a PPO agent to log in on a web page! You will learn how to randomize, customize, and increase the generality of the Agent to get it to work on any login screen. An example of randomizing and customizing the usernames and passwords for a task can be seen in the following image:

![Sample observations from a randomized user login task.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-15.png)

Sample observations from a randomized user login task.

The login task involves clicking on the correct form field and typing in the correct username and/or password. For an Agent to be able to do this, it needs to master how to use a mouse and keyboard, in addition to processing the visual web page to understand the task and the web login form. With enough samples, the deep RL Agent will learn a policy to complete this task. Let's take a look at the state of the Agent's progress, snapshotted at different stages.

The following image shows the Agent successfully entering the username and correctly clicking on the password field to enter the password, but not being able to complete the task yet:

![Screenshot of a trained Agent successfully entering the username but not a password.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-16.png)

Screenshot of a trained Agent successfully entering the username but not a password.

In the following image, you can see that the Agent has learned to enter both the username and password, but they are not quite right for the task to be classed as complete:

![Agent entering both the username and password but incorrectly.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-17.png)

Agent entering both the username and password but incorrectly.

The same Agent with a different checkpoint, after several thousand more episodes of learning, is close to completing the task, as shown in the following image

![A well-trained Agent model about to complete the login task successfully.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-18.png)

A well-trained Agent model about to complete the login task successfully.

Now that you understand how the Agent works and behaves, you can customize it to your liking and use use cases to train the Agent to automatically log into any custom website you want!

### Building RL Agent to book flights on the web

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T462163_Building_an_RL_Agent_to_book_flights_on_the_web.ipynb)

In this, we are first building the `MiniWoBBookFlightVisualEnv` environment, and then training a PPO agent to visually operate flight booking websites using a keyboard and mouse to book flights! This task is quite useful but complicated due to the varying amount of task parameters we need to implement, such as source city, destination, date, and more. The following image shows a sample of the start states from a randomized **MiniWoBBookFlightVisualEnv** flight booking environment:

![Sample start-state observations from the randomized MiniWoBBookFlightVisualEnv environment.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-19.png)

Sample start-state observations from the randomized MiniWoBBookFlightVisualEnv environment.

The flight booking environment is quite complex as it requires the Agent to master both the keyboard and the mouse, in addition to understanding the task by looking at visual images of the task description (visual text parsing), inferring the intended task objective, and executing the actions in the correct sequence. The following screenshot shows the performance of the Agent upon completing a sufficiently large number of episodes of training:

![A screenshot of the Agent performing the flight booking task at different stages of learning.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-20.png)

A screenshot of the Agent performing the flight booking task at different stages of learning.

The following screenshot shows the Agent's screen after the Agent progressed to the final stage of the task (although it's not close to completing the task):

![Screenshot of the Agent progressing all the way to the final stage of the flight booking task.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-21.png)

Screenshot of the Agent progressing all the way to the final stage of the flight booking task.

### Building RL Agent to manage emails on the web

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T515244_Building_an_RL_Agent_to_manage_emails_on_the_web.ipynb)

Email has become an integral part of many people's lives. The number of emails that an average working professional goes through in a workday is growing daily. While a lot of email filters exist for spam control, how nice would it be to have an intelligent Agent that can perform a series of email management tasks that just provide a task description (through text or speech via speech-to-text) and are not limited by any APIs that have rate limits?

In this, we are first building the `MiniWoBEmailInboxImportantVisualEnv` environment, and then training a PPO agent to manage emails on the web.

A set of sample tasks can be seen in the following image:

![A sample set of observations from the randomized MiniWoBEmailInboxImportantVisualEnv environment.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-22.png)

A sample set of observations from the randomized MiniWoBEmailInboxImportantVisualEnv environment.

The email management environment poses as a nice sequential decision-making problem for the deep RL Agent. First, the Agent has to choose the correct email from a series of emails in an inbox and then perform the desired action (starring the email and so on). The Agent only has access to the visual rendering of the inbox, so it needs to extract the task specification details, interpret the task specification, and then plan and execute the actions!

The following is a screenshot of the Agent's performance at different stages of learning (loaded from different checkpoints):

![A series of screenshots showing the Agent's learning progress.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-23.png)

A series of screenshots showing the Agent's learning progress.

### Building RL Agent to manage social media accounts on the web

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T098537_Building_an_RL_Agent_to_manage_social_media_accounts_on_the_web.ipynb)

In this, we are first building 3 environments:

- `MiniWoBSocialMediaReplyVisualEnv` to train a social media like & reply agent with PPO.
- `MiniWoBSocialMediaMuteUserVisualEnv` to train a social media user mute agent with PPO.
- `MiniWoBSocialMediaMuteUserVisualEnv` to train a social media user mute agent with DDPG.

Then we are building an RL Agent that is trained to perform management tasks on the social media account! The following image shows a series of (randomized) tasks from the environment that we will be training the Agent in:

![A sample set of social media account management tasks that the Agent has been asked to solve.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-24.png)

A sample set of social media account management tasks that the Agent has been asked to solve.

Note that there is a scroll bar in this task that the Agent needs to learn how to use! The tweet that's relevant to this task may be hidden from the visible part of the screen, so the Agent will have to actively explore (by sliding the scroll bar up/down) in order to progress!

Let's visually explore how a well-trained Agent progresses through social media management tasks. The following screenshot shows the Agent learning to use the scroll bar to "navigate" in this environment:

![The Agent learning to navigate using the scroll bar.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-25.png)

The Agent learning to navigate using the scroll bar.

Note that the task specification does not imply anything related to the scroll bar or the navigation, and that the Agent was able to explore and figure out that it needs to navigate in order to progress with the task! The following screenshot shows the Agent progressing much further by choosing the correct tweet but clicking on the wrong action; that is, **Embed Tweet** instead of the **Mute** button:

![The Agent clicking on Embed Tweet when the goal was to click on Mute.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-26.png)

The Agent clicking on Embed Tweet when the goal was to click on Mute.

After 96 million episodes of training, the Agent was sufficiently able to solve the task. The following screenshot shows the Agent's performance on an evaluation episode (the Agent was loaded from a checkpoint)

![The Agent loaded from trained parameters about to complete the task successfully.](/img/content-tutorials-raw-tensorflow-2-reinforcement-learning-cookbook-untitled-27.png)

The Agent loaded from trained parameters about to complete the task successfully.

### Training Stock Trading RL Agent and Deploying as a Service

[Link to notebook →](https://github.com/RecoHut-Projects/drl-recsys/blob/main/tutorials/T219631_Training_Stock_Trading_RL_Agent_using_SAC_and_Deploying_as_a_Service.ipynb)

In this, we first implemented the essential components for SAC method. Then we built an RL environment simulator as a service and deployed using flask (at `0.0.0.0:6666`). It contain two core modules – the tradegym server and the tradegym client, which are built based on the OpenAI Gym HTTP API. We first defined a minimum set of custom environments exposed as part of the tradegym library and then built the server and client modules.

Then we trained a deep RL agent using remote simulator. And then evaluated the trained RL agent. After evaluation, we packaged this trained RL agent to be deployed as a service and deployed using flask (at `0.0.0.0:5555`). And after deployment, performed a simple testing to make sure that RL agent as a service is successfully deployed.

## References

1. ["TensorFlow 2 Reinforcement Learning Cookbook" by Praveen Palanisamy (Packt, 2021)](https://learning.oreilly.com/library/view/tensorflow-2-reinforcement/9781838982546)
2. [https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook](https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook)
3. [https://github.com/RecoHut-Projects/drl-recsys](https://github.com/RecoHut-Projects/drl-recsys)