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

<!-- #region id="rT2jhaFmeXIu" -->
# Pareto-Efficient algorithm for MOO
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9ifbtdd2ifja" executionInfo={"status": "ok", "timestamp": 1634546106518, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="69ad66eb-789e-436f-ef59-3782ee050beb"
%tensorflow_version 1.x
```

```python id="qpBmKd7Oig3M"
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls
```

```python colab={"base_uri": "https://localhost:8080/"} id="xSiAHrdnin8S" executionInfo={"status": "ok", "timestamp": 1634546145379, "user_tz": -330, "elapsed": 2250, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="25b043ce-d25a-42f2-8948-9e334529d07e"
seed = 3456
tf.set_random_seed(seed)
np.random.seed(seed)

x_data = np.float32(np.random.rand(100, 4))
y_data = np.dot(x_data, [[0.100], [0.200], [0.3], [0.4]]) + 0.300

weight_a = tf.placeholder(tf.float32)
weight_b = tf.placeholder(tf.float32)

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0))
y = tf.matmul(x_data, W) + b

loss_a = tf.reduce_mean(tf.square(y - y_data))
loss_b = tf.reduce_mean(tf.square(W) + tf.square(b))
loss = weight_a * loss_a + weight_b * loss_b

optimizer = tf.train.GradientDescentOptimizer(0.1)

a_gradients = tf.gradients(loss_a, W)
b_gradients = tf.gradients(loss_b, W)

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
```

```python id="_Hx_xqbGiryg" executionInfo={"status": "ok", "timestamp": 1634546151622, "user_tz": -330, "elapsed": 506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="14757873-c810-4322-e39e-b3a9b534d987" colab={"base_uri": "https://localhost:8080/"}
def pareto_step(w, c, G):
    """
    ref:http://ofey.me/papers/Pareto.pdf
    K : the number of task
    M : the dim of NN's params
    :param W: # (K,1)
    :param C: # (K,1)
    :param G: # (K,M)
    :return:
    """
    GGT = np.matmul(G, np.transpose(G))  # (K, K)
    e = np.mat(np.ones(np.shape(w)))  # (K, 1)
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((np.transpose(e), np.mat(np.zeros((1, 1)))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    hat_w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), M), z)  # (K+1, 1)
    hat_w = hat_w[:-1]  # (K, 1)
    hat_w = np.reshape(np.array(hat_w), (hat_w.shape[0],))  # (K,)
    c = np.reshape(np.array(c), (c.shape[0],))  # (K,)
    new_w = ASM(hat_w, c)
    return new_w


def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0., None] for _ in range(len(hat_w))]
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    new_w = min_out.x + c
    return new_w


w_a, w_b = 0.5, 0.5
c_a, c_b = 0.2, 0.2

for step in range(0, 10):
    res = sess.run([a_gradients, b_gradients, train], feed_dict={weight_a: w_a, weight_b: w_b})
    weights = np.mat([[w_a], [w_b]])
    paras = np.hstack((res[0][0], res[1][0]))
    paras = np.transpose(paras)
    w_a, w_b = pareto_step(weights, np.mat([[c_a], [c_b]]), paras)
    la = sess.run(loss_a)
    lb = sess.run(loss_b)
    print("{:0>2d} {:4f} {:4f} {:4f} {:4f} {:4f}".format(step, w_a, w_b, la, lb, la / lb))
    # print(np.reshape(sess.run(W), (4,)), sess.run(b))
```
