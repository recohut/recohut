# Dueling DQN

A Dueling DQN agent explicitly estimates two quantities through a modified network architecture:

- State values, V(*s*)
- Advantage values, A(*s*, *a*)

The state value estimates the value of being in state s, and the advantage value represents the advantage of taking action *a* in state *s*. This key idea of explicitly and separately estimating the two quantities enables the Dueling DQN to perform better in comparison to DQN.

The Dueling-DQN agent differs from the DQN agent in terms of the neural network architecture.

The differences are summarized in the following diagram:

![Untitled](/img/content-models-raw-mp1-dueling-dqn-untitled.png)

The DQN (top half of the diagram) has a linear architecture and predicts a single quantity (Q(s, a)), whereas the Dueling-DQN has a bifurcation in the last layer and predicts multiple quantities.