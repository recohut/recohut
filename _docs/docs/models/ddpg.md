# DDPG

**Deterministic Policy Gradient (DPG)** is a type of Actor-Critic RL algorithm that uses two neural networks: one for estimating the action value function, and the other for estimating the optimal target policy. The **Deep Deterministic Policy Gradient** (**DDPG**) agent builds upon the idea of DPG and is quite efficient compared to vanilla Actor-Critic agents due to the use of deterministic action policies.

DDPG, or Deep Deterministic Policy Gradient, is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. It combines the actor-critic approach with insights from DQNs: in particular, the insights that 1) the network is trained off-policy with samples from a replay buffer to minimize correlations between samples, and 2) the network is trained with a target Q network to give consistent targets during temporal difference backups. DDPG makes use of the same ideas along with batch normalization.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces.

## Algorithm

![Untitled](/img/content-models-raw-mp2-ddpg-untitled.png)

As far as the recommended scenario is concerned, discrete actions are a more natural idea, and each action corresponds to each item. However, in reality, the number of items may be at least one million, which means that the action space is large and the calculation complexity with softmax is very high. For continuous actions, DDPG is a more general choice. For more details, articles by Jingdong [[1]](https://arxiv.org/abs/1801.00209) [[2]](https://arxiv.org/pdf/1805.02343.pdf), Ali [[1]](https://arxiv.org/pdf/1803.00710.pdf) , Huawei [[1]](https://arxiv.org/pdf/1810.12027.pdf) can be referenced.

Then the core of the algorithm is to optimize these two objective functions through gradient ascent (descent) to obtain the final parameters, and then to obtain the optimal strategy. Some other implementation details of DDPG such as target network, soft update, etc. will not be repeated here. Since we are using a fixed data set, we only need to convert the data into a format that the DDPG algorithm can input, and then batch training like supervised learning.

## Links

- [https://spinningup.openai.com/en/latest/algorithms/ddpg.html](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)