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

<!-- #region id="U_iFAN-MxTto" -->
# Introduction to Reinforcement Learning (Playground)
> Learn RL Q-learning by simulating a discrete and continuous toy environment manually as well as with Gym library

- toc: true
- badges: true
- comments: true
- categories: [RL, Gym, Playground, Concept, QLearning]
- image:
<!-- #endregion -->

<!-- #region id="FTSs3Y8Lv5_w" -->
## Introduction
<!-- #endregion -->

<!-- #region id="fh1c5II-xTts" -->
| |  |
| :-: | -:|
| Vision | Build reinforcement learning based recommender agents that can learn and adjust behavior in real-time|
| Mission | To achieve big, we have to start small and this tutorial is one step towards our vision. Our mission is to learn Q-learning policy based method. |
| Scope | Policy-based RL methods, Playground |
| Task | Maximize the reward in discrete environment, Learn continuous reward policy to maximize reward in continuous environment|
| Data | Simulation |
| Tool | Gym, Colab, Python, OpenCV |
| Technique | Q-learning policy in discrete and continuous environments |
| Process | 1) Design a simple 2-D board-based discrete environment, 2) Train RL agent using Randomwalk as well as Q-policy with Bellman, 3) Increase the environment complexity by adding more variables/contraints, 4) Setup Gym Cartpole environment, 5) Train RL agent on Cartpole continuous environment  |
| Takeaway | RL has potential to be used in recommender systems |
| Credit | [Microsoft, Dmitry](https://github.com/microsoft/ML-For-Beginners/tree/main/8-Reinforcement) |
<!-- #endregion -->

<!-- #region id="KtxNqfHhvmbb" -->
### Overview
<!-- #endregion -->

<!-- #region id="6SMJShY-vplY" -->
We will train machine learning algorithms that will help Peter:

- **Explore** the surrounding area and build an optimal navigation map
- **Learn** how to use a skateboard and balance on it, in order to move around faster.

> [Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) is a musical fairy tale written by a Russian composer Sergei Prokofiev. It is a story about young pioneer Peter, who bravely goes out of his house to the forest clearing to chase the wolf.
<!-- #endregion -->

<!-- #region id="DjjDcxODvtLS" -->
<!-- #endregion -->

<!-- #region id="KG0w_lLOvzy5" -->
### Background

Instead of looking for existing game data, **Reinforcement Learning** (RL) is based on the idea of *making the computer play* many times and observing the result. Thus, to apply Reinforcement Learning, we need two things:

- **An environment** and **a simulator** which allow us to play a game many times. This simulator would define all the game rules as well as possible states and actions.
- **A reward function**, which would tell us how well we did during each move or game.

The main difference between other types of machine learning and RL is that in RL we typically do not know whether we win or lose until we finish the game. Thus, we cannot say whether a certain move alone is good or not - we only receive a reward at the end of the game. And our goal is to design algorithms that will allow us to train a model under uncertain conditions. We will learn about one RL algorithm called **Q-learning**.

Reinforcement learning involves three important concepts: the agent, some states, and a set of actions per state. By executing an action in a specified state, the agent is given a reward. Again imagine the computer game Super Mario. You are Mario, you are in a game level, standing next to a cliff edge. Above you is a coin. You being Mario, in a game level, at a specific position ... that's your state. Moving one step to the right (an action) will take you over the edge, and that would give you a low numerical score. However, pressing the jump button would let score a point and you would stay alive. That's a positive outcome and that should award you a positive numerical score.

By using reinforcement learning and a simulator (the game), you can learn how to play the game to maximize the reward which is staying alive and scoring as many points as possible.
<!-- #endregion -->

<!-- #region id="o-7EMawPwCAP" -->
### Environment setup
<!-- #endregion -->

```python id="yvW2bwhJlmSe"
!wget https://github.com/recohut/reco-static/raw/master/media/images/ms8rl/images.zip
!unzip images.zip
```

```python id="XpJ-EiqpwFIg"
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import math
```

<!-- #region id="B3b7vlXdjkEQ" -->
## Section 1

In this section, we will learn how to apply Reinforcement learning to a problem of path finding. The setting is inspired by [Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) musical fairy tale by Russian composer [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). It is a story about young pioneer Peter, who bravely goes out of his house to the forest clearing to chase the wolf. We will train machine learning algorithms that will help Peter to explore the surroinding area and build an optimal navigation map.
<!-- #endregion -->

<!-- #region id="pC-RED8jvY7f" -->
<!-- #endregion -->

<!-- #region id="5NdtWJ5Ivb6P" -->
Peter and his friends need to escape the hungry wolf! Image by [Jen Looper](https://twitter.com/jenlooper)
<!-- #endregion -->

<!-- #region id="GPdkzWlIwQm5" -->
### Basic helper functions
<!-- #endregion -->

```python id="MCIxnOM6wPeT"
def clip(min,max,x):
    if x<min:
        return min
    if x>max:
        return max
    return x

def imload(fname,size):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_LANCZOS4)
    img = img / np.max(img)
    return img

def draw_line(dx,dy,size=50):
    p=np.ones((size-2,size-2,3))
    if dx==0:
        dx=0.001
    m = (size-2)//2
    l = math.sqrt(dx*dx+dy*dy)*(size-4)/2
    a = math.atan(dy/dx)
    cv2.line(p,(int(m-l*math.cos(a)),int(m-l*math.sin(a))),(int(m+l*math.cos(a)),int(m+l*math.sin(a))),(0,0,0),1)
    s = -1 if dx<0 else 1
    cv2.circle(p,(int(m+s*l*math.cos(a)),int(m+s*l*math.sin(a))),3,0)
    return p   

def probs(v):
    v = v-v.min()
    if (v.sum()>0):
        v = v/v.sum()
    return v
```

<!-- #region id="mIfDdsvsqNCd" -->
### The Environment

For simplicity, let's consider Peter's world to be a square board of size `width` x `height`. Each cell in this board can either be:
* **ground**, on which Peter and other creatures can walk
* **water**, on which you obviously cannot walk
* **a tree** or **grass** - a place where you cat take some rest
* **an apple**, which represents something Peter would be glad to find in order to feed himself
* **a wolf**, which is dangerous and should be avoided
<!-- #endregion -->

<!-- #region id="JxjhQ-N4wTwp" -->
### Creating `Board` environment
<!-- #endregion -->

```python id="ZhINF4jNCYmO"
class Board:
    class Cell:
        empty = 0
        water = 1
        wolf = 2
        tree = 3
        apple = 4
    def __init__(self,width,height,size=50):
        self.width = width
        self.height = height
        self.size = size+2
        self.matrix = np.zeros((width,height))
        self.grid_color = (0.6,0.6,0.6)
        self.background_color = (1.0,1.0,1.0)
        self.grid_thickness = 1
        self.grid_line_type = cv2.LINE_AA
        self.pics = {
            "wolf" : imload('images/wolf.png',size-4),
            "apple" : imload('images/apple.png',size-4),
            "human" : imload('images/human.png',size-4)
        }
        self.human = (0,0)
        self.frame_no = 0

    def randomize(self,water_size=5, num_water=3, num_wolves=1, num_trees=5, num_apples=3,seed=None):
        if seed:
            random.seed(seed)
        for _ in range(num_water):
            x = random.randint(0,self.width-1)
            y = random.randint(0,self.height-1)
            for _ in range(water_size):
                self.matrix[x,y] = Board.Cell.water
                x = clip(0,self.width-1,x+random.randint(-1,1))
                y = clip(0,self.height-1,y+random.randint(-1,1))
        for _ in range(num_trees):
            while True:
                x = random.randint(0,self.width-1)
                y = random.randint(0,self.height-1)
                if self.matrix[x,y]==Board.Cell.empty:
                    self.matrix[x,y] = Board.Cell.tree # tree
                    break
        for _ in range(num_wolves):
            while True:
                x = random.randint(0,self.width-1)
                y = random.randint(0,self.height-1)
                if self.matrix[x,y]==Board.Cell.empty:
                    self.matrix[x,y] = Board.Cell.wolf # wolf
                    break
        for _ in range(num_apples):
            while True:
                x = random.randint(0,self.width-1)
                y = random.randint(0,self.height-1)
                if self.matrix[x,y]==Board.Cell.empty:
                    self.matrix[x,y] = Board.Cell.apple
                    break

    def at(self,pos=None):
        if pos:
            return self.matrix[pos[0],pos[1]]
        else:
            return self.matrix[self.human[0],self.human[1]]

    def is_valid(self,pos):
        return pos[0]>=0 and pos[0]<self.width and pos[1]>=0 and pos[1] < self.height

    def move_pos(self, pos, dpos):
        return (pos[0] + dpos[0], pos[1] + dpos[1])

    def move(self,dpos,check_correctness=True):
        new_pos = self.move_pos(self.human,dpos)
        if self.is_valid(new_pos) or not check_correctness:
            self.human = new_pos

    def random_pos(self):
        x = random.randint(0,self.width-1)
        y = random.randint(0,self.height-1)
        return (x,y)

    def random_start(self):
        while True:
            pos = self.random_pos()
            if self.at(pos) == Board.Cell.empty:
                self.human = pos
                break


    def image(self,Q=None):
        img = np.zeros((self.height*self.size+1,self.width*self.size+1,3))
        img[:,:,:] = self.background_color
        # Draw water
        for x in range(self.width):
            for y in range(self.height):
                if (x,y) == self.human:
                    ov = self.pics['human']
                    img[self.size*y+2:self.size*y+ov.shape[0]+2,self.size*x+2:self.size*x+2+ov.shape[1],:] = np.minimum(ov,1.0)
                    continue
                if self.matrix[x,y] == Board.Cell.water:
                    img[self.size*y:self.size*(y+1),self.size*x:self.size*(x+1),:] = (0,0,1.0)
                if self.matrix[x,y] == Board.Cell.wolf:
                    ov = self.pics['wolf']
                    img[self.size*y+2:self.size*y+ov.shape[0]+2,self.size*x+2:self.size*x+2+ov.shape[1],:] = np.minimum(ov,1.0)
                if self.matrix[x,y] == Board.Cell.apple: # apple
                    ov = self.pics['apple']
                    img[self.size*y+2:self.size*y+ov.shape[0]+2,self.size*x+2:self.size*x+2+ov.shape[1],:] = np.minimum(ov,1.0)
                if self.matrix[x,y] == Board.Cell.tree: # tree
                    img[self.size*y:self.size*(y+1),self.size*x:self.size*(x+1),:] = (0,1.0,0)
                if self.matrix[x,y] == Board.Cell.empty and Q is not None:
                    p = probs(Q[x,y])
                    dx,dy = 0,0
                    for i,(ddx,ddy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                        dx += ddx*p[i]
                        dy += ddy*p[i]
                        l = draw_line(dx,dy,self.size)
                        img[self.size*y+2:self.size*y+l.shape[0]+2,self.size*x+2:self.size*x+2+l.shape[1],:] = l

        # Draw grid
        for i in range(self.height+1):
            img[:,i*self.size] = 0.3
            #cv2.line(img,(0,i*self.size),(self.width*self.size,i*self.size), self.grid_color, self.grid_thickness,lineType=self.grid_line_type)
        for j in range(self.width+1):
            img[j*self.size,:] = 0.3
            #cv2.line(img,(j*self.size,0),(j*self.size,self.height*self.size), self.grid_color, self.grid_thickness,lineType=self.grid_line_type)
        return img

    def plot(self,Q=None):
        plt.figure(figsize=(11,6))
        plt.imshow(self.image(Q),interpolation='hanning')

    def saveimage(self,filename,Q=None):
        cv2.imwrite(filename,255*self.image(Q)[...,::-1])

    def walk(self,policy,save_to=None,start=None):
        n = 0
        if start:
            self.human = start
        else:
            self.random_start()

        while True:
            if save_to:
                self.saveimage(save_to.format(self.frame_no))
                self.frame_no+=1
            if self.at() == Board.Cell.apple:
                return n # success!
            if self.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = policy(self)
                new_pos = self.move_pos(self.human,a)
                if self.is_valid(new_pos) and self.at(new_pos)!=Board.Cell.water:
                    self.move(a) # do the actual move
                    break
            n+=1
```

<!-- #region id="VLNTBzV9jkEX" -->
Let's now create a random board and see how it looks:
<!-- #endregion -->

```python id="zBWRzsdejkEY" colab={"base_uri": "https://localhost:8080/", "height": 378} outputId="54d9c446-770d-40d9-fc2f-514bd61434a4"
width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

<!-- #region id="QCnF08TMjkEa" -->
### Actions and Policy

In our example, Peter's goal would be to find an apple, while avoiding the wolf and other obstacles. To do this, he can essentially walk around until he finds an apple. Therefore, at any position he can chose between one of the following actions: up, down, left and right. We will define those actions as a dictionary, and map them to pairs of corresponding coordinate changes. For example, moving right (`R`) would correspond to a pair `(1,0)`.
<!-- #endregion -->

```python id="QgjSRrEcjkEb"
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

<!-- #region id="hkFJeg3RjkEd" -->
The strategy of our agent (Peter) is defined by a so-called **policy**. Let's consider the simplest policy called **random walk**.
<!-- #endregion -->

<!-- #region id="pDl_6XWgqIfc" -->
### Random walk

Let's first solve our problem by implementing a random walk strategy.
<!-- #endregion -->

```python tags=[] id="zpycj37QjkEd" colab={"base_uri": "https://localhost:8080/"} outputId="4ca47030-d79a-47ca-ce43-1239802fe7b5"
def random_policy(m):
    return random.choice(list(actions))

def walk(m,policy,start_position=None):
    n = 0 # number of steps
    # set initial position
    if start_position:
        m.human = start_position 
    else:
        m.random_start()
    while True:
        if m.at() == Board.Cell.apple:
            return n # success!
        if m.at() in [Board.Cell.wolf, Board.Cell.water]:
            return -1 # eaten by wolf or drowned
        while True:
            a = actions[policy(m)]
            new_pos = m.move_pos(m.human,a)
            if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                m.move(a) # do the actual move
                break
        n+=1

walk(m,random_policy)
```

<!-- #region id="X5j0t1ckjkEf" -->
Let's run random walk experiment several times and see the average number of steps taken:
<!-- #endregion -->

```python id="WQRd5UD5jkEf" colab={"base_uri": "https://localhost:8080/"} outputId="7dd6d449-749c-4598-a07e-14bcfbc0bfc1"
def print_statistics(policy):
    s,w,n = 0,0,0
    for _ in range(100):
        z = walk(m,policy)
        if z<0:
            w+=1
        else:
            s += z
            n += 1
    print(f"Average path length = {s/n}, eaten by wolf: {w} times")

print_statistics(random_policy)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 433} id="S6G_1vde1-vF" outputId="dd50097e-5d46-426b-e5db-9c5b70e38b7f"
from IPython.display import Image
Image(open('images/random_walk.gif','rb').read())
```

<!-- #region id="LOUyie9kjkEg" -->
### Reward Function

To make our policy more intelligent, we need to understand which moves are "better" than others.


<!-- #endregion -->

```python id="7JUe6SsUjkEg"
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

<!-- #region id="nGp5RMaLjkEh" -->
### Q-Learning

Build a Q-Table, or multi-dimensional array. Since our board has dimensions `width` x `height`, we can represent Q-Table by a numpy array with shape `width` x `height` x `len(actions)`:
<!-- #endregion -->

```python id="USWH9c0AjkEh"
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

<!-- #region id="4ettSq3VjkEi" -->
Pass the Q-Table to the plot function in order to visualize the table on the board:
<!-- #endregion -->

```python id="16tqn0OvjkEi" colab={"base_uri": "https://localhost:8080/", "height": 378} outputId="37013049-c7e9-4d1d-ab6c-24475202199d"
m.plot(Q)
```

<!-- #region id="evws9q19sVvF" -->
In the center of each cell there is an "arrow" that indicates the preferred direction of movement. Since all directions are equal, a dot is displayed.

Now we need to run the simulation, explore our environment, and learn a better distribution of Q-Table values, which will allow us to find the path to the apple much faster.
<!-- #endregion -->

<!-- #region id="lhQsmbx5jkEi" -->
### Essence of Q-Learning: Bellman Equation and  Learning Algorithm

Let's now write a pseudo-code for our leaning algorithm:

* Initialize Q-Table Q with equal numbers for all states and actions
* Set learning rate $\alpha\leftarrow 1$
* Repeat simulation many times
   1. Start at random position
   1. Repeat
        1. Select an action $a$ at state $s$
        2. Exectute action by moving to a new state $s'$
        3. If we encounter end-of-game condition, or total reward is too small - exit simulation  
        4. Compute reward $r$ at the new state
        5. Update Q-Function according to Bellman equation: $Q(s,a)\leftarrow (1-\alpha)Q(s,a)+\alpha(r+\gamma\max_{a'}Q(s',a'))$
        6. $s\leftarrow s'$
        7. Update total reward and decrease $\alpha$.
<!-- #endregion -->

<!-- #region id="WF1fEh4Hws9F" -->
### Exploit vs. Explore

The best approach is to balance between exploration and exploitation. As we learn more about our environment, we would be more likely to follow the optimal route, however, choosing the unexplored path once in a while.
<!-- #endregion -->

<!-- #region id="d3WOMk4hwvOc" -->
### Python Implementation

Now we are ready to implement the learning algorithm. Before that, we also need some function that will convert arbitrary numbers in the Q-Table into a vector of probabilities for corresponding actions:
<!-- #endregion -->

```python id="7IR4kLtBjkEj"
def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v
```

<!-- #region id="6usMtFQvjkEj" -->
We add a small amount of `eps` to the original vector in order to avoid division by 0 in the initial case, when all components of the vector are identical.

The actual learning algorithm we will run for 10000 experiments, also called **epochs**: 
<!-- #endregion -->

```python id="HZ4XcabCjkEk" colab={"base_uri": "https://localhost:8080/"} outputId="2618f33f-c04e-46f2-bb59-c2f1c6e036a0"
from IPython.display import clear_output

lpath = []

for epoch in range(10000):
    clear_output(wait=True)
    print(f"Epoch = {epoch}",end='')

    # Pick initial point
    m.random_start()
    
    # Start travelling
    n=0
    cum_reward = 0
    while True:
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        dpos = actions[a]
        m.move(dpos,check_correctness=False) # we allow player to move outside the board, which terminates episode
        r = reward(m)
        cum_reward += r
        if r==end_reward or cum_reward < -1000:
            print(f" {n} steps",end='\r')
            lpath.append(n)
            break
        alpha = np.exp(-n / 3000)
        gamma = 0.5
        ai = action_idx[a]
        Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
        n+=1
```

<!-- #region id="sCkoL3jQjkEk" -->
After executing this algorithm, the Q-Table should be updated with values that define the attractiveness of different actions at each step. Visualize the table here:
<!-- #endregion -->

```python id="6GiDSyoajkEl" colab={"base_uri": "https://localhost:8080/", "height": 378} outputId="2233e4d9-741d-4c2d-e9fc-61352bfed3b0"
m.plot(Q)
```

<!-- #region id="QzcMSc9ZjkEl" -->
### Checking the Policy

Since Q-Table lists the "attractiveness" of each action at each state, it is quite easy to use it to define the efficient navigation in our world. In the simplest case, we can just select the action corresponding to the highest Q-Table value:
<!-- #endregion -->

```python id="MerDWi1-jkEl" colab={"base_uri": "https://localhost:8080/"} outputId="b914e69c-242b-4bad-8ebd-40c439b8cbfe"
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

<!-- #region id="5UDtIiMPjkEm" -->
If you try the code above several times, you may notice that sometimes it just "hangs", and you need to press the STOP button in the notebook to interrupt it. 

> **Task 1:** Modify the `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where is has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape. 
<!-- #endregion -->

```python id="6n1K3lv5jkEm" colab={"base_uri": "https://localhost:8080/"} outputId="68c45f4a-fdfe-40c8-cce6-59c8f0107660"
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

<!-- #region id="blDX_QH5jkEn" -->
### Investigating the Learning Process
<!-- #endregion -->

```python id="MnUwchoFjkEn" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="a7d04c66-bd3c-4efe-cd1c-947eead2ca40"
plt.plot(lpath)
```

<!-- #region id="4NIAwaSDjkEn" -->
What we see here is that at first the average path length increased. This is probably due to the fact that when we know nothing about the environment - we are likely to get trapped into bad states, water or wolf. As we learn more and start using this knowledge, we can explore the environment for longer, but we still do not know well where apples are.

Once we learn enough, it becomes easier for the agent to achieve the goal, and the path length starts to decrease. However, we are still open to exploration, so we often diverge away from the best path, and explore new options, making the path longer than optimal.

What we also observe on this graph, is that at some point the length increased abruptly. This indicates stochastic nature of the process, and that we can at some point "spoil" the Q-Table coefficients, by overwriting them with new values. This ideally should be minimized by decreasing learning rate (i.e. towards the end of training we only adjust Q-Table values by a small value).

Overall, it is important to remember that the success and quality of the learning process significantly depends on parameters, such as leaning rate, learning rate decay and discount factor. Those are often called **hyperparameters**, to distinguish them from **parameters** which we optimize during training (eg. Q-Table coefficients). The process of finding best hyperparameter values is called **hyperparameter optimization**, and it deserves a separate topic.
<!-- #endregion -->

<!-- #region id="kMd3CzpjjkEo" -->
### Exercise
#### A More Realistic Peter and the Wolf World

In our situation, Peter was able to move around almost without getting tired or hungry. In a more realistic world, he has to sit down and rest from time to time, and also to feed himself. Let's make our world more realistic by implementing the following rules:

1. By moving from one place to another, Peter loses **energy** and gains some **fatigue**.
2. Peter can gain more energy by eating apples.
3. Peter can get rid of fatigue by resting under the tree or on the grass (i.e. walking into a board location with a tree or grass - green field)
4. Peter needs to find and kill the wolf
5. In order to kill the wolf, Peter needs to have certain levels of energy and fatigue, otherwise he loses the battle.

Modify the reward function above according to the rules of the game, run the reinforcement learning algorithm to learn the best strategy for winning the game, and compare the results of random walk with your algorithm in terms of number of games won and lost.


> **Note**: You may need to adjust hyperparameters to make it work, especially the number of epochs. Because the success of the game (fighting the wolf) is a rare event, you can expect much longer training time.


<!-- #endregion -->

<!-- #region id="_EfnTPdholDN" -->
## Section 2
<!-- #endregion -->

<!-- #region id="YXIH5_BGozIG" -->
Let's implement the above exercise.
<!-- #endregion -->

```python id="ucnJOnt0ozIM" colab={"base_uri": "https://localhost:8080/", "height": 378} outputId="feb90774-61dd-4b5d-ff7a-f1e74039b40a"
width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

```python id="7vlMUy6uozIO"
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

<!-- #region id="xIfsGRZ8ozIO" -->
### Defining state

In our new game rules, we need to keep track of energy and fatigue at each board state. Thus we will create an object `state` that will carry all required information about current problem state, including state of the board, current levels of energy and fatigue, and whether we can win the wolf while at terminal state:
<!-- #endregion -->

```python id="QJJnXKllozIP"
class state:
    def __init__(self,board,energy=10,fatigue=0,init=True):
        self.board = board
        self.energy = energy
        self.fatigue = fatigue
        self.dead = False
        if init:
            self.board.random_start()
        self.update()

    def at(self):
        return self.board.at()

    def update(self):
        if self.at() == Board.Cell.water:
            self.dead = True
            return
        if self.at() == Board.Cell.tree:
            self.fatigue = 0
        if self.at() == Board.Cell.apple:
            self.energy = 10

    def move(self,a):
        self.board.move(a)
        self.energy -= 1
        self.fatigue += 1
        self.update()

    def is_winning(self):
        return self.energy > self.fatigue
```

<!-- #region id="3sxKhMjQozIQ" -->
Let's try to solve the problem using random walk and see if we succeed:
<!-- #endregion -->

```python tags=[] id="E4cH6LjfozIR" colab={"base_uri": "https://localhost:8080/"} outputId="c3269034-1283-4178-ff8d-2286021c8a02"
def random_policy(state):
    return random.choice(list(actions))

def walk(board,policy):
    n = 0 # number of steps
    s = state(board)
    while True:
        if s.at() == Board.Cell.wolf:
            if s.is_winning():
                return n # success!
            else:
                return -n # failure!
        if s.at() == Board.Cell.water:
            return 0 # died
        a = actions[policy(m)]
        s.move(a)
        n+=1

walk(m,random_policy)
```

```python id="y8AAEPvcozIS" colab={"base_uri": "https://localhost:8080/"} outputId="77035353-86b4-441c-a704-afd955886c05"
def print_statistics(policy):
    s,w,n = 0,0,0
    for _ in range(100):
        z = walk(m,policy)
        if z<0:
            w+=1
        elif z==0:
            n+=1
        else:
            s+=1
    print(f"Killed by wolf = {w}, won: {s} times, drown: {n} times")

print_statistics(random_policy)
```

<!-- #region id="ZEgFPeLAozIS" -->
### Reward Function

<!-- #endregion -->

```python id="Me5f2_5kozIT"
def reward(s):
    r = s.energy-s.fatigue
    if s.at()==Board.Cell.wolf:
        return 100 if s.is_winning() else -100
    if s.at()==Board.Cell.water:
        return -100
    return r
```

<!-- #region id="_8fq4SxoozIU" -->
### Q-Learning algorithm

The actual learning algorithm stays pretty much unchanged, we will use `state` instead of just board position.
<!-- #endregion -->

```python id="ow7pamUeozIU"
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

```python id="_NU7h54NozIU"
def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v
```

```python id="4dhfa2ZRozIV" colab={"base_uri": "https://localhost:8080/"} outputId="77d8e8e2-f64d-47e7-f5ab-4e9ded7ad8bc"
from IPython.display import clear_output

lpath = []

for epoch in range(10000):
    clear_output(wait=True)
    print(f"Epoch = {epoch}",end='')

    # Pick initial point
    s = state(m)
    
    # Start travelling
    n=0
    cum_reward = 0
    while True:
        x,y = s.board.human
        v = probs(Q[x,y])
        while True:
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            if s.board.is_valid(s.board.move_pos(s.board.human,dpos)):
                break 
        s.move(dpos)
        r = reward(s)
        if abs(r)==100: # end of game
            print(f" {n} steps",end='\r')
            lpath.append(n)
            break
        alpha = np.exp(-n / 3000)
        gamma = 0.5
        ai = action_idx[a]
        Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
        n+=1
```

```python id="wGi_eDOZozIW" colab={"base_uri": "https://localhost:8080/", "height": 378} outputId="d2760985-fcfa-456c-b5d5-693269dff4d0"
m.plot(Q)
```

<!-- #region id="25dRA0QBozIZ" -->
### Results

Let's see if we were successful training Peter to fight the wolf!
<!-- #endregion -->

```python id="LLOPmQjjozIZ" colab={"base_uri": "https://localhost:8080/"} outputId="be4c2453-fa71-4c4f-8811-b1959c91997d"
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

<!-- #region id="wEZgpUidozIa" -->
We now see much less cases of drowning, but Peter is still not always able to kill the wolf. Try to experiment and see if you can improve this result by playing with hyperparameters.
<!-- #endregion -->

```python id="_Re70UjcozIa" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="b366c062-c685-4334-9ca3-eb9350aeed2f"
plt.plot(lpath)
```

<!-- #region id="lkf6gIILss62" -->
## Section 3
<!-- #endregion -->

<!-- #region id="df-3tGBlozIb" -->
In this section, we will apply the same principles of Q-Learning to a problem with continuous state, i.e. a state that is given by one or more real numbers. We will deal with the following problem:

Problem: If Peter wants to escape from the wolf, he needs to be able to move faster. We will see how Peter can learn to skate, in particular, to keep balance, using Q-Learning.
<!-- #endregion -->

<!-- #region id="nBLZQXmps0il" -->
<!-- #endregion -->

<!-- #region id="kmIQwHJasy0U" -->
> Note: We will use a simplified version of balancing known as a CartPole problem. In the cartpole world, we have a horizontal slider that can move left or right, and the goal is to balance a vertical pole on top of the slider.
<!-- #endregion -->

<!-- #region id="M4cl98EatNIl" -->
### OpenAI Gym
In previous sections, the rules of the game and the state were given by the Board class which we defined ourselves. Here we will use a special simulation environment, which will simulate the physics behind the balancing pole. One of the most popular simulation environments for training reinforcement learning algorithms is called a Gym, which is maintained by OpenAI. By using this gym we can create difference environments from a cartpole simulation to Atari games.
<!-- #endregion -->

```python id="JH115T2htBqT"
!apt-get install -y xvfb x11-utils

!pip install pyvirtualdisplay==0.2.* \
             PyOpenGL==3.1.* \
             PyOpenGL-accelerate==3.1.*

!pip install gym
```

```python id="LSvVZV2ytdbs"
import pyvirtualdisplay

_display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
                                    size=(1400, 900))
_ = _display.start()
```

<!-- #region id="uyQ2VxqquNoZ" -->
### CartPole Skating

> **Problem**: If Peter wants to escape from the wolf, he needs to be able to move faster than him. We will see how Peter can learn to skate, in particular, to keep balance, using Q-Learning.

First, let's install the gym and import required libraries:
<!-- #endregion -->

```python id="XCWZjoGiuNoc"
import sys
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

<!-- #region id="R5K2-a4WuNof" -->
### Create a cartpole environment
<!-- #endregion -->

```python id="MO3OiR-juNof" colab={"base_uri": "https://localhost:8080/"} outputId="91b488c1-ce38-4850-972c-fcfc82d67fe1"
env = gym.make("CartPole-v1")
print(env.action_space)
print(env.observation_space)
print(env.action_space.sample())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 417} id="WSo-Gun_12Wq" outputId="bd9d3e24-6daa-4017-89f4-6c6f6e10565a"
from IPython.display import Image
Image(open('images/cartpole-nobalance.gif','rb').read())
```

<!-- #region id="oXSYrv5fuNog" -->
To see how the environment works, let's run a short simulation for 100 steps.
<!-- #endregion -->

```python id="m-9-6ZSpuNoh" colab={"base_uri": "https://localhost:8080/"} outputId="23931c92-6b18-463f-8354-a4e0838f40c9"
env.reset()

for i in range(100):
   env.render()
   env.step(env.action_space.sample())
env.close()
```

<!-- #region id="S0j7KKYtuNoi" -->
During simulation, we need to get observations in order to decide how to act. In fact, `step` function returns us back current observations, reward function, and the `done` flag that indicates whether it makes sense to continue the simulation or not:
<!-- #endregion -->

```python id="Q3DzNjN8uNoj" colab={"base_uri": "https://localhost:8080/"} outputId="5a7a7536-0c25-4810-9658-9658fea4e5bf"
env.reset()

done = False
while not done:
   env.render()
   obs, rew, done, info = env.step(env.action_space.sample())
   print(f"{obs} -> {rew}")
env.close()
```

<!-- #region id="AGgds2pLuNol" -->
We can get min and max value of those numbers:
<!-- #endregion -->

```python id="v7mpurQGuNom" colab={"base_uri": "https://localhost:8080/"} outputId="66b80ac4-1730-4ebb-ab97-52d895d72bab"
print(env.observation_space.low)
print(env.observation_space.high)
```

<!-- #region id="Q58CGRuuuNon" -->
### State Discretization
<!-- #endregion -->

```python id="RFkszuoUuNoo"
def discretize(x):
    return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
```

<!-- #region id="5-lE7kD0uNoo" -->
Let's also explore other discretization method using bins:
<!-- #endregion -->

```python id="utuRsW2EuNop" colab={"base_uri": "https://localhost:8080/"} outputId="eaa6fce3-b12b-49a8-c7d2-893bbfed484a"
def create_bins(i,num):
    return np.arange(num+1)*(i[1]-i[0])/num+i[0]

print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))

ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [20,20,10,10] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i],bins[i]) for i in range(4))
```

<!-- #region id="azWPrudJuNop" -->
Let's now run a short simulation and observe those discrete environment values.
<!-- #endregion -->

```python id="7WF03T_kuNoq" colab={"base_uri": "https://localhost:8080/"} outputId="4f285684-4d4c-4b1d-f819-4a60029a8fdf"
env.reset()

done = False
while not done:
   #env.render()
   obs, rew, done, info = env.step(env.action_space.sample())
   #print(discretize_bins(obs))
   print(discretize(obs))
env.close()
```

<!-- #region id="c0VxfarVuNoq" -->
### Q-Table Structure
<!-- #endregion -->

```python id="JEP_MAaruNoq"
Q = {}
actions = (0,1)

def qvalues(state):
    return [Q.get((state,a),0) for a in actions]
```

<!-- #region id="uGnBdkfjuNor" -->
### Let's Start Q-Learning!
<!-- #endregion -->

```python id="SMlCWSK2uNor"
# hyperparameters
alpha = 0.3
gamma = 0.9
epsilon = 0.90
```

```python id="1bDVZ5sDuNor" colab={"base_uri": "https://localhost:8080/"} outputId="24bc34fb-d2c9-4382-f84b-74e590e2fb9a"
def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v

Qmax = 0
cum_rewards = []
rewards = []
for epoch in range(100000):
    obs = env.reset()
    done = False
    cum_reward=0
    # == do the simulation ==
    while not done:
        s = discretize(obs)
        if random.random()<epsilon:
            # exploitation - chose the action according to Q-Table probabilities
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions,weights=v)[0]
        else:
            # exploration - randomly chose the action
            a = np.random.randint(env.action_space.n)

        obs, rew, done, info = env.step(a)
        cum_reward+=rew
        ns = discretize(obs)
        Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)
    # == Periodically print results and calculate average reward ==
    if epoch%5000==0:
        print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
        if np.average(cum_rewards) > Qmax:
            Qmax = np.average(cum_rewards)
            Qbest = Q
        cum_rewards=[]
```

<!-- #region id="ZMEBfBeJuNot" -->
### Plotting Training Progress
<!-- #endregion -->

```python id="0WcrMLqSuNot" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="8de083da-53d9-4a29-cda8-2a564396df5a"
plt.plot(rewards)
```

<!-- #region id="2wiNZcfBuNou" -->
From this graph, it is not possible to tell anything, because due to the nature of stochastic training process the length of training sessions varies greatly. To make more sense of this graph, we can calculate **running average** over series of experiments, let's say 100. This can be done conveniently using `np.convolve`:
<!-- #endregion -->

```python id="MhuXZ33xuNov" colab={"base_uri": "https://localhost:8080/", "height": 285} outputId="ebf51aa6-5e79-41f3-d72f-c93b1fd3db0c"
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

<!-- #region id="CsEpGxlPuNow" -->
### Varying Hyperparameters and Seeing the Result in Action

Now it would be interesting to actually see how the trained model behaves. Let's run the simulation, and we will be following the same action selection strategy as during training: sampling according to the probability distribution in Q-Table: 
<!-- #endregion -->

```python id="cQ_1zQN6uNox"
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

<!-- #region id="yYNA9-4nuNox" -->
### Saving result to an animated GIF

If you want to impress your friends, you may want to send them the animated GIF picture of the balancing pole. To do this, we can invoke `env.render` to produce an image frame, and then save those to animated GIF using PIL library:
<!-- #endregion -->

```python id="jCcqOkjwuNoy" colab={"base_uri": "https://localhost:8080/"} outputId="14a8ecac-dedf-457a-a014-93a7ade61ec6"
from PIL import Image
obs = env.reset()
done = False
i=0
ims = []
while not done:
   s = discretize(obs)
   img=env.render(mode='rgb_array')
   ims.append(Image.fromarray(img))
   v = probs(np.array([Qbest.get((s,a),0) for a in actions]))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
   i+=1
env.close()
ims[0].save('images/cartpole-balance.gif',save_all=True,append_images=ims[1::2],loop=0,duration=5)
print(i)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 417} id="1zy5x_Et1fYI" outputId="eb3adc6f-b805-4cb2-c856-34bafd90f59e"
from IPython.display import Image
Image(open('images/cartpole-balance.gif','rb').read())
```
