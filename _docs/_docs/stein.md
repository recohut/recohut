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

<!-- #region id="rKkRpKUF3Th6" -->
# Stein's Paradox
<!-- #endregion -->

<!-- #region id="NXvBQPn72YxC" -->
[Stein's paradox](https://en.wikipedia.org/wiki/Stein%27s_example)

We will compare the risk of [James–Stein estimator](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) to a naive estimator on a simulated high-dimensional dataset.
<!-- #endregion -->

```python id="c1I5Jpj1U4o5"
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
```

<!-- #region id="tsa2Euoehvvk" -->
We have a single data point $x$ drawn from a $d$-dimensional multivariate normal distribution with mean vector $\theta$ and covariance matrix $I_d$ (the $d\times d$ identity matrix).

$X \sim \mathcal{N}(\theta, I_d)$

We want to obtain an estimate $\hat{\theta}$ of $\theta$ by using only $x$.

We will compare different estimators $\hat{\theta}(x)$ using their "risk", which is basically their mean squared error across trials.
<!-- #endregion -->

<!-- #region id="uGFe5U1kjTkw" -->
The simplest estimator is $\hat{\theta}(x) = x$. We will call this the naive estimator.
<!-- #endregion -->

```python id="vmkCowyUU7sW"
def est_naive(x):
    return x
```

<!-- #region id="k0XbMOBMjZgg" -->
Stein's paradox suggests that we can come up with an alternative estimator that has lower risk: $\hat{\theta}(x) = \left(1 - \frac{d - 2}{||x||_2^2}\right) \times x$. We can think of this as shrinking our estimate $\hat{\theta}$ towards zero, tuning the strength of the shrinkage we apply by estimating something directly from our single data point (namely, it's Euclidean norm).
<!-- #endregion -->

```python id="phu0s0cyhuUX"
def est_stein(x):
    return (1 - (x.shape[1] - 2)/np.linalg.norm(x, axis=1)**2)[:, None] * x
```

<!-- #region id="9k_0iiEAj9_O" -->
We define a function to estimate the risk of an estimator at a particular true value of $\theta$ by averaging the mean squared error of the estimator over $m$ trials.
<!-- #endregion -->

```python id="cxDaxHCfXBFK"
def mean_risk(est, theta, m=int(1e6)):
    rvs = stats.multivariate_normal(theta, 1).rvs(m)
    ests = est(rvs)
    rs = np.linalg.norm((ests - theta), axis=1)**2
    return np.mean(rs)
```

<!-- #region id="sG2ipIRHkgo3" -->
We now evaluate the mean risk for various choices of $\theta$. For simplicity, we just try a sequence of $\theta$'s whose components are all equal and take integer values between 0 and 10 inclusive.
<!-- #endregion -->

```python id="4H4SF0ljXK2T"
d = 10
naive_risk = [mean_risk(est_naive, [t] * d) for t in range(11)]
stein_risk = [mean_risk(est_stein, [t] * d) for t in range(11)]
```

<!-- #region id="HTxa3-Nlk5OL" -->
We can then plot the mean risk.
<!-- #endregion -->

```python id="e6hr1oKrhJaK" colab={"base_uri": "https://localhost:8080/", "height": 283} executionInfo={"status": "ok", "timestamp": 1634920544613, "user_tz": -330, "elapsed": 884, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3aab121c-4310-45c6-b752-98f97c820821"
plt.plot(np.arange(len(naive_risk)), naive_risk, label='naive', color='r')
plt.plot(np.arange(len(stein_risk)), stein_risk, label='Stein', color='b')
plt.xlabel(r'$\theta_i$')
plt.ylabel('risk')
plt.legend()
plt.show()
```

<!-- #region id="wn6i53r_mLQ3" -->
We can see that when the true $\theta$ is close to the zero vector, the Stein estimator has a much lower risk than the naive estimator. This is what we expect to happen if we think of the Stein estimator as performing some kind of Bayesian shrinkage towards a prior distribution over $\theta$ which happens to match the true $\theta$ reasonably well. We could imagine that some property like this might also hold for 1 or 2 dimensions. What is perhaps more surprising is that the Stein estimator has lower risk than the naive estimator even when the true $\theta$ is far from the zero vector (the Stein estimator appears to asymptotically approach the risk of the naive estimator from below as the distance between the true $\theta$ and the zero vector goes to infinity). This suggests that even when the choice of a Bayesian prior is arbitrarily "wrong" in the sense that it is centered very far from the true value of the parameter, it is still better to apply the shrinkage (as long as we are in high-dimensional space, $d \geq 3$).
<!-- #endregion -->
