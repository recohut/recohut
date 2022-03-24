# A3C

A3C stands for Asynchronous Advantage Actor-Critic. The A3C algorithm builds upon the Actor-Critic class of algorithms by using a neural network to approximate the actor (and critic). The actor learns the policy function using a deep neural network, while the critic estimates the value function. The asynchronous nature of the algorithm allows the agent to learn from different parts of the state space, allowing parallel learning and faster convergence. Unlike DQN agents, which use an experience replay memory, the A3C agent uses multiple workers to gather more samples for learning.

In simple terms, the crux of the A3C algorithm can be summarized in the following sequence of steps for each iteration:

![Untitled](/img/content-models-raw-mp1-a3c-untitled.png)

The steps repeat again from top to bottom for the next iteration and so on until convergence.