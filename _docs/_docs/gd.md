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

<!-- #region id="taP_qNMzvJR8" -->
# Gradient Descent from scratch
<!-- #endregion -->

<!-- #region id="FoFBJ9JyI2p8" -->
Frequently when doing data science, we’ll be trying to the find the best model for a certain situation. And usually “best” will mean something like “minimizes the error of its predictions” or “maximizes the likelihood of the data.” In other words, it will represent the solution to some sort of optimization problem.

This means we’ll need to solve a number of optimization problems. And in particular, we’ll need to solve them from scratch. Our approach will be a technique called gradient descent, which lends itself pretty well to a from-scratch treatment.
<!-- #endregion -->

<!-- #region id="qZIxB8BJvFbA" -->
## Linear Algebra
<!-- #endregion -->

```python id="yW9y5ZQMJuoZ"
from typing import List

Vector = List[float]

height_weight_age = [70,  # inches,
                     170, # pounds,
                     40 ] # years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3

import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function

assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))


def distance(v: Vector, w: Vector) -> float:  # type: ignore
    return magnitude(subtract(v, w))

# Another type alias
Matrix = List[List[float]]

A = [[1, 2, 3],  # A has 2 rows and 3 columns
     [4, 5, 6]]

B = [[1, 2],     # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # number of elements in first row
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]             # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j]          # jth element of row A_i
            for A_i in A]   # for each row A_i

from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             # given i, create a list
             for j in range(num_cols)]  #   [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # create one list for each i

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            user 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9

assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are not friends"

# only need to look at one row
friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]
```

<!-- #region id="6KD5-2TEJKgh" -->
Suppose we have some function f that takes as input a vector of real numbers and outputs a single real number. One simple such function is:
<!-- #endregion -->

```python id="5p3Vy6ZyJmyp"
def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)
```

<!-- #region id="IAR3h3omKRgr" -->
We’ll frequently need to maximize or minimize such functions. That is, we need to find the input v that produces the largest (or smallest) possible value.

For functions like ours, the gradient (if you remember your calculus, this is the vector of partial derivatives) gives the input direction in which the function most quickly increases.

Accordingly, one approach to maximizing a function is to pick a random starting point, compute the gradient, take a small step in the direction of the gradient (i.e., the direction that causes the function to increase the most), and repeat with the new starting point. Similarly, you can try to minimize a function by taking small steps in the opposite direction, as shown below:
<!-- #endregion -->

<!-- #region id="_Ksyrv9uKZ0K" -->
<!-- #endregion -->

<!-- #region id="kMaGI14VKYi9" -->
> Note: If a function has a unique global minimum, this procedure is likely to find it. If a function has multiple (local) minima, this procedure might “find” the wrong one of them, in which case you might rerun the procedure from different starting points. If a function has no minimum, then it’s possible the procedure might go on forever.
<!-- #endregion -->

<!-- #region id="TL4kmFVnKgiD" -->
If f is a function of one variable, its derivative at a point x measures how f(x) changes when we make a very small change to x. The derivative is defined as the limit of the difference quotients, as h approaches zero.
<!-- #endregion -->

```python id="i9og8EBkLAaI"
from typing import Callable

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h
```

<!-- #region id="zmlCLWWDLCeV" -->
The derivative is the slope of the tangent line at $(x, f(x))$, while the difference quotient is the slope of the not-quite-tangent line that runs through $(x+h, f(x+h))$. As h gets smaller and smaller, the not-quite-tangent line gets closer and closer to the tangent line:
<!-- #endregion -->

<!-- #region id="Sut4fbGqLwwl" -->
<!-- #endregion -->

<!-- #region id="lboxkLqrLpF4" -->
For many functions it’s easy to exactly calculate derivatives. For example, the square function:
<!-- #endregion -->

```python id="vzE_XKv_MOjB"
def square(x: float) -> float:
    return x * x

def derivative(x: float) -> float:
    return 2 * x
```

<!-- #region id="Yl0XRSQsMlRo" -->
What if you couldn’t (or didn’t want to) find the gradient? Although we can’t take limits in Python, we can estimate derivatives by evaluating the difference quotient for a very small e. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="vAL00ZjJMnAG" executionInfo={"status": "ok", "timestamp": 1630849483402, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd84394e-b17f-4deb-bfb4-b8a97d078a64"
xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

# plot to show they're basically the same
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')       # red  x
plt.plot(xs, estimates, 'b+', label='Estimate')   # blue +
plt.legend(loc=9)
plt.show()
```

<!-- #region id="-2G4xlwVMonQ" -->
When f is a function of many variables, it has multiple partial derivatives, each indicating how f changes when we make small changes in just one of the input variables.

We calculate its ith partial derivative by treating it as a function of just its ith variable, holding the other variables fixed:
<!-- #endregion -->

```python id="qsyLH1g8Mo3U"
def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h
```

<!-- #region id="RDLat4lGNUc0" -->
after which we can estimate the gradient the same way:
<!-- #endregion -->

```python id="Wd-yevbfNWP0"
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]
```

<!-- #region id="E1CoCPNwNgHH" -->
It’s easy to see that the sum_of_squares function is smallest when its input v is a vector of zeros. But imagine we didn’t know that. Let’s use gradients to find the minimum among all three-dimensional vectors. We’ll just pick a random starting point and then take tiny steps in the opposite direction of the gradient until we reach a point where the gradient is very small:
<!-- #endregion -->

```python id="dBJAfqMGN1aZ"
import random

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]
```

```python colab={"base_uri": "https://localhost:8080/"} id="6ePuLtU4Ou8E" executionInfo={"status": "ok", "timestamp": 1630850199160, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5c4b1c14-973f-4c74-bc8f-e97548f8ea67"
# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)    # compute the gradient at v
    v = gradient_step(v, grad, -0.01)    # take a negative gradient step
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001    # v should be close to 0
```

<!-- #region id="UfMuGc7FZ2Hd" -->
## Choosing the Right Step Size

<!-- #endregion -->

<!-- #region id="k1awybLtPBcJ" -->
Although the rationale for moving against the gradient is clear, how far to move is not. Indeed, choosing the right step size is more of an art than a science. Popular options include:

- Using a fixed step size
- Gradually shrinking the step size over time
- At each step, choosing the step size that minimizes the value of the objective function

The last approach sounds great but is, in practice, a costly computation. To keep things simple, we’ll mostly just use a fixed step size. The step size that “works” depends on the problem—too small, and your gradient descent will take forever; too big, and you’ll take giant steps that might make the function you care about get larger or even be undefined. So we’ll need to experiment.
<!-- #endregion -->

<!-- #region id="utw4nIhSZzi6" -->
## Using Gradient Descent to Fit Models
<!-- #endregion -->

<!-- #region id="PEl4FewuZjni" -->
If we think of our data as being fixed, then our loss function tells us how good or bad any particular model parameters are. This means we can use gradient descent to find the model parameters that make the loss as small as possible. Let’s look at a simple example:
<!-- #endregion -->

```python id="aNSrNsyHZ3Wq"
# x ranges from -50 to 49, y is always 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
```

<!-- #region id="j3MHskU-Z5sE" -->
In this case we know the parameters of the linear relationship between x and y, but imagine we’d like to learn them from the data. We’ll use gradient descent to find the slope and intercept that minimize the average squared error.

We’ll start off with a function that determines the gradient based on the error from a single data point:
<!-- #endregion -->

```python id="tZOX-o3XbYOq"
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual).
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad
```

<!-- #region id="2mQsAHr0bcum" -->
Now, that computation was for a single data point. For the whole dataset we’ll look at the mean squared error. And the gradient of the mean squared error is just the mean of the individual gradients.

So, here’s what we’re going to do:

1. Start with a random value for theta.
2. Compute the mean of the gradients.
3. Adjust theta in that direction.
4. Repeat.

After a lot of epochs (what we call each pass through the dataset), we should learn something like the correct parameters:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xwHsXEFrcZqx" executionInfo={"status": "ok", "timestamp": 1630853934285, "user_tz": -330, "elapsed": 926, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aab5f170-fdab-4cd8-cb74-6d62bc6728a8"
# Start with random values for slope and intercept
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001
n_epochs = 5000

for epoch in range(n_epochs):
    # Compute the mean of the gradients
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # Take a step in that direction
    theta = gradient_step(theta, grad, -learning_rate)
    if epoch%(n_epochs//10)==0:
        print('@ epoch {}, slope={:.2f} and intercept={:.2f}'.format(epoch, theta[0], theta[1]))

slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
```

<!-- #region id="8rgQxW9BddVr" -->
## Minibatch and Stochastic Gradient Descent
<!-- #endregion -->

<!-- #region id="ejKfYy23eMuo" -->
One drawback of the preceding approach is that we had to evaluate the gradients on the entire dataset before we could take a gradient step and update our parameters. In this case it was fine, because our dataset was only 100 pairs and the gradient computation was cheap.

Your models, however, will frequently have large datasets and expensive gradient computations. In that case you’ll want to take gradient steps more often.

We can do this using a technique called minibatch gradient descent, in which we compute the gradient (and take a gradient step) based on a “minibatch” sampled from the larger dataset:
<!-- #endregion -->

```python id="eieFwtbmeOb1"
from typing import TypeVar, List, Iterator

T = TypeVar('T')  # this allows us to type "generic" functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
```

<!-- #region id="znkAGdzZeivd" -->
> Note: The TypeVar(T) allows us to create a “generic” function. It says that our dataset can be a list of any single type—strs, ints, lists, whatever—but whatever that type is, the outputs will be batches of it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OwIMDFCCej6U" executionInfo={"status": "ok", "timestamp": 1630854248271, "user_tz": -330, "elapsed": 640, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dae30c9b-41fd-4ae6-e190-3ca3dcc60266"
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

n_epochs = 1000

for epoch in range(n_epochs):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    if epoch%(n_epochs//10)==0:
        print('@ epoch {}, slope={:.2f} and intercept={:.2f}'.format(epoch, theta[0], theta[1]))

slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
```

<!-- #region id="pikeOpmvfPC-" -->
Another variation is stochastic gradient descent, in which you take gradient steps based on one training example at a time:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_ROcV7voflPP" executionInfo={"status": "ok", "timestamp": 1630854359868, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8e3806eb-6d69-45d5-e8f2-abe73cee2c31"
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

n_epochs = 100

for epoch in range(n_epochs):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    if epoch%(n_epochs//10)==0:
        print('@ epoch {}, slope={:.2f} and intercept={:.2f}'.format(epoch, theta[0], theta[1]))
        
slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
```

<!-- #region id="VWsPtqdifYWB" -->
On this problem, stochastic gradient descent finds the optimal parameters in a much smaller number of epochs. But there are always tradeoffs. Basing gradient steps on small minibatches (or on single data points) allows you to take more of them, but the gradient for a single point might lie in a very different direction from the gradient for the dataset as a whole.

In addition, if we weren’t doing our linear algebra from scratch, there would be performance gains from “vectorizing” our computations across batches rather than computing the gradient one point at a time.
<!-- #endregion -->
