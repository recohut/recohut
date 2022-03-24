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

<!-- #region id="j0usJhNim6cB" -->
# Deep Reinforcement Learning in Large Discrete Action Spaces
<!-- #endregion -->

<!-- #region id="lxtTIv-Qmqd_" -->
Advanced AI systems will likely need to reason with a large number of possible actions at every step. Recommender systems used in large systems such as YouTube and Amazon must reason about hundreds of millions of items every second, and control systems for large industrial processes may have millions of possible actions that can be applied at every time step.

We deal with these large action spaces by leveraging prior information about the actions to embed them in a continuous space upon which the actor can generalize. The policy produces a continuous action within this space, and then uses an approximate nearest neighbor search to find the set of closest discrete actions in logarithmic time.

<p><center><figure><img src='_images/T734685_1.png'><figcaption>Wolpertinger Architecture</figcaption></figure></center></p>

This architecture avoids the heavy cost of evaluating all actions while retaining generalization over actions. This policy builds upon the actor-critic (Sutton & Barto, 1998) framework. We define both an efficient action-generating actor, and utilize the critic to refine our actor’s choices for the full policy. We use multi-layer neural networks as function approximators for both our actor and critic functions. We train this policy using Deep Deterministic Policy Gradient (Lillicrap et al., 2015).

## Wolpertinger

<p><center><img src='_images/T734685_2.png'></center></p>

Pass the current states through the actor network, and get a proto action μ. While in training phase, use a continuous exploration policy, such as the a gaussian noise, to add exploration noise to the proto action. Then, pass the proto action to a k-NN tree to find actual valid action candidates, which are in the surrounding neighborhood of the proto action. Those actions are then passed to the critic to evaluate their goodness, and eventually the discrete index of the action with the highest Q value is chosen. When testing, the same flow is used, but no exploration noise is added.

## Training procedure

Training the network is exactly the same as in DDPG. Unlike when choosing the action, the proto action is not passed through the k-NN tree. It is being passed directly to the critic.

Start by sampling a batch of transitions from the experience replay.

- To train the **critic network**, use the following targets:
    
    $$y_t=r(s_t,a_t )+\gamma \cdot Q(s_{t+1},\mu(s_{t+1} ))$$
    
    First run the actor target network, using the next states as the inputs, and get μ(st+1)μ(st+1). Next, run the critic target network using the next states and μ(st+1)μ(st+1), and use the output to calculate ytyt according to the equation above. To train the network, use the current states and actions as the inputs, and ytyt as the targets.
    
- To train the **actor network**, use the following equation:
    
    $$\nabla_{\theta^\mu } J \approx E_{s_t \tilde{} \rho^\beta } [\nabla_a Q(s,a)|_{s=s_t,a=\mu (s_t ) } \cdot \nabla_{\theta^\mu} \mu(s)|_{s=s_t} ]$$
    
    Use the actor’s online network to get the action mean values using the current states as the inputs. Then, use the critic online network in order to get the gradients of the critic output with respect to the action mean values $\nabla _a Q(s,a)|_{s=s_t,a=\mu(s_t ) }$. Using the chain rule, calculate the gradients of the actor’s output, with respect to the actor weights, given $\nabla_a Q(s,a)$. Finally, apply those gradients to the actor network.
    

After every training step, do a soft update of the critic and actor target networks’ weights from the online networks.
<!-- #endregion -->

<!-- #region id="lIiKP508VyaZ" -->
## Setup
<!-- #endregion -->

```python id="IWcDY4b6RuXA"
!pip install setproctitle
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="6X4BwTs_TDfK" executionInfo={"status": "ok", "timestamp": 1634978374471, "user_tz": -330, "elapsed": 511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="51bebdd0-8e93-486e-c1be-4ff550d4a892"
import os

# to know the python path in colab
os.path.abspath(os.__file__)
```

```python colab={"base_uri": "https://localhost:8080/"} id="D1NWFUNKSBnY" executionInfo={"status": "ok", "timestamp": 1634977994905, "user_tz": -330, "elapsed": 2129, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae133509-aabf-4d0c-c76e-8682a08dff71"
!git clone https://github.com/ChangyWen/wolpertinger_ddpg.git
%cd wolpertinger_ddpg
!cp -r pyflann /usr/lib/python3.7
```

<!-- #region id="_G1PN1jCSGPH" -->
## Pendulum with 200K actions

In Pendulum-v0 (continuous control), discretize the continuous action space to a discrete action spaces with 200000 actions.
<!-- #endregion -->

```python id="zujyfkhVSV8H"
!cd src && python main.py --env 'Pendulum-v0' --max-actions 200000
```

<!-- #region id="X1OPxfGoQZ_5" -->
## CartPole

In CartPole-v1 (discrete control), --max-actions is not needed.
<!-- #endregion -->

```python id="uDvQMbQ2Vdwv"
!cd src && python main.py --env 'CartPole-v1'
```

<!-- #region id="rBmJYCv0VvIZ" -->
## Analysis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XUj6I0IWQZ9q" executionInfo={"status": "ok", "timestamp": 1634978952173, "user_tz": -330, "elapsed": 6944, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fc51ef7f-d1f7-47a9-d8ee-7aa3ce054486"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="am2sV6D4QZ7A" executionInfo={"status": "ok", "timestamp": 1634978995521, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b107d50d-02cf-45f7-e1f5-1d4cc58ae5d1"
!tree --du -h -C .
```

<!-- #region id="2udQJevCV1cM" -->
As we can see, we get 2 PyTorch (.pt) models - actor.pt and critic.pt

Let's analyze the implementation details of some essential components
<!-- #endregion -->

```python id="AU847JW_WvPK"
!pip install -U Ipython
```

```python id="kcIm1gIyYIkT"
from IPython.display import Code
import inspect
```

```python id="xSrMGQikYSwY"
import sys
sys.path.insert(0,'./src')
from src import *
```

<!-- #region id="qgoXFQr9Wk1O" -->
### Wolpertinger Agent
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="iCu8ML89X-pN" executionInfo={"status": "ok", "timestamp": 1634979707472, "user_tz": -330, "elapsed": 504, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="68cea442-3514-47bd-c4ff-10f0e7bc3f9c"
from src import wolp_agent
Code(inspect.getsource(wolp_agent), language='python')
```

<!-- #region id="HWbHNLsZYnWn" -->
### Memory
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="q7b_cQlIZFGP" executionInfo={"status": "ok", "timestamp": 1634979876184, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="de6b28a1-a30d-4414-c181-76738548d38f"
from src import memory
Code(inspect.getsource(memory), language='python')
```

<!-- #region id="EPBNF6w-ZcP7" -->
There are two types of memory - Sequential and Episodic.
<!-- #endregion -->

<!-- #region id="fVy7F2T8ZFEC" -->
### Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="V3adAp9kZFCA" executionInfo={"status": "ok", "timestamp": 1634979966293, "user_tz": -330, "elapsed": 478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4faa5530-e46e-476e-ac73-5b0298a1df41"
from src import model
Code(inspect.getsource(model), language='python')
```

<!-- #region id="3tQrbIIGZug9" -->
We have 2 NN models - Actor and Critic.
<!-- #endregion -->

<!-- #region id="JM8PRo1pZE_u" -->
### Action space
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="-hQOvrDOZE9T" executionInfo={"status": "ok", "timestamp": 1634980047793, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c414ab16-51ad-41e7-d15f-f2db3ac41b26"
from src import action_space
Code(inspect.getsource(action_space), language='python')
```

<!-- #region id="mKHxV1EUmzwD" -->
## Additional notebook

[Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/main.ipynb at master · nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces](https://github.com/nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/blob/master/main.ipynb)

## Links

1. [https://github.com/ChangyWen/wolpertinger_ddpg](https://github.com/ChangyWen/wolpertinger_ddpg)
2. [https://github.com/nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces](https://github.com/nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces)
3. [https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces](https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces)
4. [https://intellabs.github.io/coach/components/agents/policy_optimization/wolpertinger.html](https://intellabs.github.io/coach/components/agents/policy_optimization/wolpertinger.html)
<!-- #endregion -->
