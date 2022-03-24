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

<!-- #region id="_-o7LqCb_FfO" -->
# Multi-Objective Ranking using Constrained Optimization in GBTs
<!-- #endregion -->

<!-- #region id="ZGVuT8tzhrW5" -->
This method is inspired by [Multi-objective Ranking via Constrained Optimization](https://arxiv.org/pdf/2002.05753.pdf) which introduces multi-objective optimization in Search by incorporating constrained optimization in LambdaMART, a popular LTR algorithm.

For recommendations in general, in a pointwise modeling approach, user-product pairs are scored based on previous interactions between users and products. The outcome of this interaction (say, click or conversion) becomes the target variable and is typically the primary objective of the ML model. However, in most cases, there are other objectives (or constraints) that will need to be handled. For example, cost-of-delivery, time-to-delivery, fairness-to-partners, diversity, repeat-rate, etc. Furthermore, the importance of these additional constraints can vary subject to other factors. Let’s call these constraints sub-objectives.
<!-- #endregion -->

<!-- #region id="NnTurefnhxgc" -->
<tr>
<td>
<p><center><figure><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/55a5b31a-ddf2-45a2-b8a2-4ccd52ee8227/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211027%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211027T065415Z&X-Amz-Expires=86400&X-Amz-Signature=de1b268ce06270b6bf4d4cba28a581b704b490adb7c99d27ef5c6c9fd984a7e4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22' width=400><figcaption>Scatter with respect to the primary objective of whether there was a click on a restaurant ad.</figcaption></figure></center></p>
</td>
<td>
<p><center><figure><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1a619e07-297a-4e99-a973-15b59de1fbaa/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211027%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211027T065611Z&X-Amz-Expires=86400&X-Amz-Signature=048844c3b9cfd9ff8553acd8bb0a29397689105157f9f1beb78e133d4d43a18f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22' width=400><figcaption>Scatter with respect to the sub-objective which is the normalized SLA of delivery from the restaurant.</figcaption></figure></center></p>
</td>
</tr>
<!-- #endregion -->

<!-- #region id="W4pPkHJeifWt" -->
Consider the plots above, representing two target variables with respect to two features f1 and f2. We can model them individually with simple NNs but how to optimize them together? Our goal is to develop a model which optimizes for both the primary objective (say, clicks) and the sub-objective (say, SLA).

Several methods have been explored to handle MOO. Broadly, there are (at least) 3–4 classes of methods:

1. scalarization or post processing (where the additional objectives are not really part of the learning process but are handled post facto). Construct a convex linear combination — λ * *S1* + (1-λ) * S2, where S1 and S2 are scores from individual models trained on different objectives and λ is the weight learned from a Pareto analysis. This method works surprisingly well in practice and is a very credible baseline.
2. changing the training data (where the model is built on data that is [sampled](https://arxiv.org/pdf/2008.10277.pdf) from the ‘desired’ region).
3. changing the way labels are aggregated (for example, [stochastic label aggregation](https://assets.amazon.science/4d/9c/69cbef8346408349385c780cac48/scipub-1195.pdf) method of David Carmel, et al.).
4. changing the loss function itself. To solve for multiple objectives we modify the algorithm to include constraints in it’s loss function, which are specified as upper bound on the cost value for the sub-objectives.

In this tutorial, we will modify the Gradient Boosted Trees (GBT) algorithm to include constraints in it’s loss function. We combine GBT with the [Augmented Lagrangian method](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method), which is a class of algorithms to solve constrained optimization problems. We define the constrained optimization problem as:

$$\min_s C^{pm} (s)\ s.t.\ C^t (s) \le b^t, t=1,...,T.$$

where,

- $C^{pm}$ represents the cost of the primary objective
- $C^t$ represents the cost of sub-objective $t$
- $T$ is total number of sub-objectives
- $b$ represents the upper bound specified for the sub-objective cost during training

The Lagrangian is written as:

$$\mathcal{L}(s,\alpha) = C^{pm}(s) + \sum_t^T\alpha^t(C^t(s)-b^t)$$

where $α = [α1, α2, α3…..]$ is a vector of dual variables. The Lagrangian is solved by minimizing with respect to the primary variables $s$ and maximizing with respect to the dual variables $α$.

Augmented Lagrangian (AL) has an additional penalty term that iteratively solves the constraint optimization problem:

$$\mathcal{L}(s,\alpha) = C^{pm}(s) + \sum_t^T\alpha^t(C^t(s)-b^t) - \sum_t^T\dfrac{1}{2\mu_k}(\alpha^t-\alpha_{k-1}^t)^2$$

where $α_{k−1}^t$ is a solution in the previous iteration and a constant in the current iteration $k$. $μ$ is a sufficiently large constant associated with each dual variable $α$.

We have modified the loss function of scikit-learn [GBT Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [GBT Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor) to optimize for multiple objectives and combine the iterations of GBT (building trees) with the iterative constrained optimization framework (AL) defined in the previous section.

Below is the GBT algorithm for Classification/Regression and how we modified it to serve for multiple objectives. GBT requires a differentiable loss function. We modify a traditional loss function to include the constraint terms and the AL term which is constant for an iteration.
<!-- #endregion -->

<!-- #region id="4B7DwZnqzJ90" -->
We will be expanding https://github.com/Swiggy/Moo-GBT modules in this section. So be ready for 1Ks code lines. We are doing it this way instead of installing the package because 1) we are learning, not working, 2) codebase is 1Ks, not 10K, so our notebook can easily handle it for us, and most importantly, 3) It is optional part because we can directly jump to runs, present in the last quarter of notebook.
<!-- #endregion -->

<!-- #region id="geol2IZdi1c5" -->
## Training Procedure

<p><center><img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4113c08f-4a87-4399-92bd-042209eb405a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211027%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211027T065846Z&X-Amz-Expires=86400&X-Amz-Signature=2b88f10fcba30c171bee2e28eb2d13c8512dc70a8e496a7f9137b434c192c946&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22' width=50%></center></p>

1. GBT uses residuals as target values to build the current tree. Calculation of these residuals changes as we modify the GBT loss function to include sub-objective constraints and the AL term (constant for an iteration)
2. For each leaf we calculate an output value, gamma, such that it minimizes the loss function
3. Modify the dual variables, alphas, which are used in the loss function based on the respective constraint being met or not
<!-- #endregion -->

<!-- #region id="h2EntR_wzs6t" -->
## Setup
<!-- #endregion -->

```python id="0gW9vW8EzwiI"
from abc import ABCMeta, abstractmethod
from typing import List
from joblib import effective_n_jobs
import warnings

from sklearn.base import clone
from sklearn.base import is_classifier, is_regressor
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils import Bunch, _print_elapsed_time
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _BaseComposition
 
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn.utils import deprecated

from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.ensemble._gradient_boosting import predict_stage
from sklearn.ensemble._gradient_boosting import _random_sample_mask

import numbers
import numpy as np

import pandas as pd
import seaborn as sns
from scipy.special import expit, logsumexp

from sklearn.tree._tree import TREE_LEAF
from sklearn.utils.stats import _weighted_percentile
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from time import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE, DOUBLE

from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _deprecate_positional_args

from sklearn.utils.validation import check_X_y
```

<!-- #region id="v2GLKDa_z4bP" -->
## Utils
<!-- #endregion -->

```python id="PiJudKTYz0CB"
def _fit_single_estimator(estimator, X, y, sample_weight=None,
                          message_clsname=None, message=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights."
                    .format(estimator.__class__.__name__)
                ) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y)
    return estimator
```

```python id="kL6fpduVzz_B"
def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.
    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.
    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.
    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:
        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)
```

```python id="eXxW_Fdjzz9A"
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.
    Warning: This class should not be used directly. Use derived classes
    instead.
    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.
    n_estimators : int, default=10
        The number of estimators in the ensemble.
    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(self, base_estimator, *, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.
        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)
```

```python id="VgUIna8qzz60"
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
```

```python id="WhCTWvgezz4o"
class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition,
                                 metaclass=ABCMeta):
    """Base class for heterogeneous ensemble of learners.
    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.
    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    _required_parameters = ['estimators']

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimators))

    @abstractmethod
    def __init__(self, estimators):
        self.estimators = estimators

    def _validate_estimators(self):
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        has_estimator = any(est != 'drop' for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        is_estimator_type = (is_classifier if is_classifier(self)
                             else is_regressor)

        for est in estimators:
            if est != 'drop' and not is_estimator_type(est):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        est.__class__.__name__, is_estimator_type.__name__[3:]
                    )
                )

        return names, estimators

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.
        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.
        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.
        """
        super()._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.
        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.
        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.
        """
        return super()._get_params('estimators', deep=deep)
```

```python id="7z-NO1Rj0tDQ"
class VerboseReporter:
    """Reports verbose output to stdout.
    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """
    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        """Initialize reporter
        Parameters
        ----------
        est : Estimator
            The estimator
        begin_at_stage : int, default=0
            stage at which to begin reporting
        """
        # header fields and line format str
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
        # do oob?
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')
        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')

        # print the header line
        print(('%10s ' + '%16s ' *
               (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt = ' '.join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """Update reporter with new iteration.
        Parameters
        ----------
        j : int
            The new iteration.
        est : Estimator
            The estimator.
        """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = ((est.n_estimators - (j + 1)) *
                              (time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            print(self.verbose_fmt.format(iter=j + 1,
                                          train_score=est.train_score_[j],
                                          oob_impr=oob_impr,
                                          remaining_time=remaining_time))
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10
```

<!-- #region id="ztS3PCYu1SpV" -->
## Gradient Boosting Losses
<!-- #endregion -->

<!-- #region id="Dyp6FpyT1XAt" -->
Losses and corresponding default initial estimators for gradient boosting decision trees.
<!-- #endregion -->

```python id="KUyVsUUB1WFZ"
class LossFunction(metaclass=ABCMeta):
    """Abstract base class for various loss functions.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes, constraints, n_constraints):
        self.K = n_classes
        self.constraints = constraints
        self.n_constraints = n_constraints
        self.losses = []
        self.alphas = []

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    def get_losses(self):
        cols = ['primary_objective']
        for i in range(self.n_constraints):
            cols.append(f'sub_objective_{i+1}')
        
        return pd.DataFrame(self.losses, columns=cols)\
                            .reset_index()\
                            .rename(columns={'index': 'n_estimators'})

    # def plot_losses(self, index=None):
    #     if index == None:
    #         losses = self.get_losses()

    def get_alphas(self):
        cols = []
        for i in range(self.n_constraints):
            cols.append(f'sub-objective-{i+1}')
        
        return pd.DataFrame(self.alphas, columns=cols)\
                            .reset_index()\
                            .rename(columns={'index': 'n_estimators'})

    @abstractmethod
    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the loss.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """

    @abstractmethod
    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """

    def update_terminal_regions(self, tree, X, Y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.
        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n_samples,)
            The weight of each sample.
        sample_mask : ndarray of shape (n_samples,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, Y, residual,
                                         raw_predictions[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        raw_predictions[:, k] += \
            learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)

        # update dual variables in augmentend lagrangian
        if self.n_constraints:
            losses = []
            alphas = []
            y_0 = Y[:,0]
            
            primary_loss = self._get_loss_function_value(y_0, raw_predictions)
            losses.append(primary_loss)
            for i, constraint in enumerate(self.constraints, 1):
                y_i = Y[:, i]

                sub_loss_i = self._get_loss_function_value(y_i, raw_predictions)

                alpha_i = constraint['alpha']
                b_i = constraint['b']
                mu_i = constraint['mu']

                if sub_loss_i < b_i:
                    alpha_i = 0
                else:
                    alpha_i = mu_i * (sub_loss_i - b_i) + alpha_i

                losses.append(sub_loss_i)
                alphas.append(alpha_i)
                
                self.constraints[i-1]['alpha'] = alpha_i

            self.losses.append(losses)
            self.alphas.append(alphas)
        else:
            primary_loss = self._get_loss_function_value(Y, raw_predictions)
            self.losses.append([primary_loss])


    @abstractmethod
    def _get_loss_function_value(self, y, raw_predictions):
        """Template method to return loss function value of GBT at current step"""

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Template method for updating terminal regions (i.e., leaves)."""

    @abstractmethod
    def get_init_raw_predictions(self, X, estimator):
        """Return the initial raw predictions.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data array.
        estimator : object
            The estimator to use to compute the predictions.
        Returns
        -------
        raw_predictions : ndarray of shape (n_samples, K)
            The initial raw predictions. K is equal to 1 for binary
            classification and regression, and equal to the number of classes
            for multiclass classification. ``raw_predictions`` is casted
            into float64.
        """
        pass
```

```python id="O1uzhxcA2Dh4"
class RegressionLossFunction(LossFunction, metaclass=ABCMeta):
    """Base class for regression loss functions."""
    def __init__(self, constraints, n_constraints):
        super().__init__(n_classes=1, constraints=constraints, n_constraints=n_constraints)


    def check_init_estimator(self, estimator):
        """Make sure estimator has the required fit and predict methods.
        Parameters
        ----------
        estimator : object
            The init estimator to check.
        """
        if not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
            raise ValueError(
                "The init parameter must be a valid estimator and "
                "support both fit and predict."
            )

    def get_init_raw_predictions(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1, 1).astype(np.float64)
```

```python id="_HhRqd5r1lUH"
class LeastSquaresError(RegressionLossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions do not need to be updated for least squares.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def init_estimator(self):
        return DummyRegressor(strategy='mean')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the least squares loss.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        if sample_weight is None:
            return np.mean((y - raw_predictions.ravel()) ** 2)
        else:
            return (1 / sample_weight.sum() * np.sum(
                sample_weight * ((y - raw_predictions.ravel()) ** 2)))

    def negative_gradient(self, Y, raw_predictions, **kargs):
        """Compute half of the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples,)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        if self.n_constraints:
            y_0 = Y[:, 0]
            residual = y_0 - raw_predictions.ravel()
            for i, constraint in enumerate(self.constraints, 1):
                y_i = Y[:, i]
                alpha_i = constraint['alpha']
                residual += alpha_i * (y_i - raw_predictions.ravel())
            return residual
        else:
            return Y - raw_predictions.ravel()

    # def update_terminal_regions(self, tree, X, Y, residual, raw_predictions,
    #                             sample_weight, sample_mask,
    #                             learning_rate=0.1, k=0):
    #     """Least squares does not need to update terminal regions.

    #     But it has to update the predictions.

    #     Parameters
    #     ----------
    #     tree : tree.Tree
    #         The tree object.
    #     X : ndarray of shape (n_samples, n_features)
    #         The data array.
    #     y : ndarray of shape (n_samples,)
    #         The target labels.
    #     residual : ndarray of shape (n_samples,)
    #         The residuals (usually the negative gradient).
    #     raw_predictions : ndarray of shape (n_samples, K)
    #         The raw predictions (i.e. values from the tree leaves) of the
    #         tree ensemble at iteration ``i - 1``.
    #     sample_weight : ndarray of shape (n,)
    #         The weight of each sample.
    #     sample_mask : ndarray of shape (n,)
    #         The sample mask to be used.
    #     learning_rate : float, default=0.1
    #         Learning rate shrinks the contribution of each tree by
    #          ``learning_rate``.
    #     k : int, default=0
    #         The index of the estimator being updated.
    #     """
    #     if self.n_constraints:
    #         terminal_regions = tree.apply(X)
    #         y_0 = Y[:, 0]
    #         y_0 = y_0.take(terminal_region, axis=0)
    #     else:
    #         # update predictions
    #         raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, Y,
                                residual, raw_predictions, sample_weight):
        
        terminal_region = np.where(terminal_regions == leaf)[0]
        raw_pred = raw_predictions.take(terminal_region, axis=0)

        if self.n_constraints:
            y_0 = Y[:, 0]
            y_0 = y_0.take(terminal_region, axis=0)

            leaf_val = np.mean(y_0 - raw_pred)

            for i, constraint in enumerate(self.constraints, 1):
                alpha_i = constraint['alpha']
                
                y_i = Y[:, i]
                y_i = y_i.take(terminal_region, axis=0)
                
                leaf_val += alpha_i * np.mean(y_i - raw_pred)
        else:
            Y = Y.take(terminal_region, axis=0)
            leaf_val = np.mean(Y - raw_pred)
        
        tree.value[leaf, 0, 0] = leaf_val
        


    def _get_loss_function_value(self, y, raw_predictions):
        """return binomial deviance value based on current prediction"""
        return np.mean((y - raw_predictions.ravel()) ** 2)
        
        # return -np.mean(y * raw_predictions.ravel() - \
        #     np.log(1 + np.exp(raw_predictions.ravel())))
```

```python id="nlbKI2fT2BbF"
class LeastAbsoluteError(RegressionLossFunction):
    """Loss function for least absolute deviation (LAD) regression.
    Parameters
    ----------
    n_classes : int
        Number of classes
    """
    def init_estimator(self):
        return DummyRegressor(strategy='quantile', quantile=.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the least absolute error.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        if sample_weight is None:
            return np.abs(y - raw_predictions.ravel()).mean()
        else:
            return (1 / sample_weight.sum() * np.sum(
                sample_weight * np.abs(y - raw_predictions.ravel())))

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.
        1.0 if y - raw_predictions > 0.0 else -1.0
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        raw_predictions = raw_predictions.ravel()
        return 2 * (y - raw_predictions > 0) - 1

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """LAD updates terminal regions to median estimates."""
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        diff = (y.take(terminal_region, axis=0) -
                raw_predictions.take(terminal_region, axis=0))
        tree.value[leaf, 0, 0] = _weighted_percentile(diff, sample_weight,
                                                      percentile=50)
```

```python id="kvMrY18p1ohO"
class HuberLossFunction(RegressionLossFunction):
    """Huber loss function for robust regression.
    M-Regression proposed in Friedman 2001.
    Parameters
    ----------
    alpha : float, default=0.9
        Percentile at which to extract score.
    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.gamma = None

    def init_estimator(self):
        return DummyRegressor(strategy='quantile', quantile=.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Huber loss.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        gamma = self.gamma
        if gamma is None:
            if sample_weight is None:
                gamma = np.percentile(np.abs(diff), self.alpha * 100)
            else:
                gamma = _weighted_percentile(np.abs(diff), sample_weight,
                                             self.alpha * 100)

        gamma_mask = np.abs(diff) <= gamma
        if sample_weight is None:
            sq_loss = np.sum(0.5 * diff[gamma_mask] ** 2)
            lin_loss = np.sum(gamma * (np.abs(diff[~gamma_mask]) -
                                       gamma / 2))
            loss = (sq_loss + lin_loss) / y.shape[0]
        else:
            sq_loss = np.sum(0.5 * sample_weight[gamma_mask] *
                             diff[gamma_mask] ** 2)
            lin_loss = np.sum(gamma * sample_weight[~gamma_mask] *
                              (np.abs(diff[~gamma_mask]) - gamma / 2))
            loss = (sq_loss + lin_loss) / sample_weight.sum()
        return loss

    def negative_gradient(self, y, raw_predictions, sample_weight=None,
                          **kargs):
        """Compute the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        if sample_weight is None:
            gamma = np.percentile(np.abs(diff), self.alpha * 100)
        else:
            gamma = _weighted_percentile(np.abs(diff), sample_weight,
                                         self.alpha * 100)
        gamma_mask = np.abs(diff) <= gamma
        residual = np.zeros((y.shape[0],), dtype=np.float64)
        residual[gamma_mask] = diff[gamma_mask]
        residual[~gamma_mask] = gamma * np.sign(diff[~gamma_mask])
        self.gamma = gamma
        return residual

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        gamma = self.gamma
        diff = (y.take(terminal_region, axis=0)
                - raw_predictions.take(terminal_region, axis=0))
        median = _weighted_percentile(diff, sample_weight, percentile=50)
        diff_minus_median = diff - median
        tree.value[leaf, 0] = median + np.mean(
            np.sign(diff_minus_median) *
            np.minimum(np.abs(diff_minus_median), gamma))
```

```python id="-UkCzUY11oeL"
class QuantileLossFunction(RegressionLossFunction):
    """Loss function for quantile regression.
    Quantile regression allows to estimate the percentiles
    of the conditional distribution of the target.
    Parameters
    ----------
    alpha : float, default=0.9
        The percentile.
    """
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.percentile = alpha * 100

    def init_estimator(self):
        return DummyRegressor(strategy='quantile', quantile=self.alpha)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Quantile loss.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        alpha = self.alpha

        mask = y > raw_predictions
        if sample_weight is None:
            loss = (alpha * diff[mask].sum() -
                    (1 - alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            loss = ((alpha * np.sum(sample_weight[mask] * diff[mask]) -
                    (1 - alpha) * np.sum(sample_weight[~mask] *
                                         diff[~mask])) / sample_weight.sum())
        return loss

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        alpha = self.alpha
        raw_predictions = raw_predictions.ravel()
        mask = y > raw_predictions
        return (alpha * mask) - ((1 - alpha) * ~mask)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        diff = (y.take(terminal_region, axis=0)
                - raw_predictions.take(terminal_region, axis=0))
        sample_weight = sample_weight.take(terminal_region, axis=0)

        val = _weighted_percentile(diff, sample_weight, self.percentile)
        tree.value[leaf, 0] = val
```

```python id="fj5LJL7Q1tzZ"
class ClassificationLossFunction(LossFunction, metaclass=ABCMeta):
    """Base class for classification loss functions. """

    def _raw_prediction_to_proba(self, raw_predictions):
        """Template method to convert raw predictions into probabilities.
        Parameters
        ----------
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        Returns
        -------
        probas : ndarray of shape (n_samples, K)
            The predicted probabilities.
        """

    @abstractmethod
    def _raw_prediction_to_decision(self, raw_predictions):
        """Template method to convert raw predictions to decisions.
        Parameters
        ----------
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        Returns
        -------
        encoded_predictions : ndarray of shape (n_samples, K)
            The predicted encoded labels.
        """

    def check_init_estimator(self, estimator):
        """Make sure estimator has fit and predict_proba methods.
        Parameters
        ----------
        estimator : object
            The init estimator to check.
        """
        if not (hasattr(estimator, 'fit') and
                hasattr(estimator, 'predict_proba')):
            raise ValueError(
                "The init parameter must be a valid estimator "
                "and support both fit and predict_proba."
            )
```

```python id="z2FbPYOx1txN"
class BinomialDeviance(ClassificationLossFunction):
    """Binomial deviance loss function for binary classification.
    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    """
    def __init__(self, n_classes, constraints, n_constraints):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1, constraints=constraints, n_constraints=n_constraints)

    def init_estimator(self):
        # return the most common class, taking into account the samples
        # weights
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood).
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        # logaddexp(0, v) == log(1.0 + exp(v))
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -2 * np.mean((y * raw_predictions) -
                                np.logaddexp(0, raw_predictions))
        else:
            return (-2 / sample_weight.sum() * np.sum(
                sample_weight * ((y * raw_predictions) -
                                 np.logaddexp(0, raw_predictions))))

    def negative_gradient(self, Y, raw_predictions, **kargs):
        """Compute half of the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        if self.n_constraints:
            y_0 = Y[:, 0]
            residual = y_0 - expit(raw_predictions.ravel())
            for i, constraint in enumerate(self.constraints, 1):
                y_i = Y[:, i]
                alpha_i = constraint['alpha']
                residual += alpha_i * (y_i - expit(raw_predictions.ravel()))
            return residual
        else:
            return Y - expit(raw_predictions.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, Y,
                                residual, raw_predictions, sample_weight):
        """Make a single Newton-Raphson step.
        our node estimate is given by:
            sum(w * (y - prob)) / sum(w * prob * (1 - prob))
        we take advantage that: y - prob = residual
        """

        terminal_region = np.where(terminal_regions == leaf)[0]
        if self.n_constraints:
            raw_pred = raw_predictions.take(terminal_region, axis=0)
            y_0 = Y[:, 0]
            y_0 = y_0.take(terminal_region, axis=0)

            # residual for primary objective
            numerator = np.sum(y_0 - expit(raw_pred.ravel()))

            # submission (1 - p) * p
            den = np.sum((1- expit(raw_pred.ravel())) * expit(raw_pred.ravel()))
            
            denominator = den.copy()

            for i, constraint in enumerate(self.constraints, 1):
                alpha_i = constraint['alpha']
                y_i = Y[:, i]
                y_i = y_i.take(terminal_region, axis=0)

                numerator += np.sum(alpha_i * (y_i - expit(raw_pred.ravel())))
                denominator += alpha_i * den
        else:
            residual = residual.take(terminal_region, axis=0)
            Y = Y.take(terminal_region, axis=0)
            sample_weight = sample_weight.take(terminal_region, axis=0)

            numerator = np.sum(sample_weight * residual)
            denominator = np.sum(sample_weight *
                                (Y - residual) * (1 - Y + residual))

        # prevents overflow and division by zero 
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator
    
    def _get_loss_function_value(self, y, raw_predictions):
        """return binomial deviance value based on current prediction"""
        return -np.mean(y * raw_predictions.ravel() - \
            np.log(1 + np.exp(raw_predictions.ravel())))

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # log(x / (1 - x)) is the inverse of the sigmoid (expit) function
        raw_predictions = np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)
```

```python id="y7pdrclD1ttw"
class MultinomialDeviance(ClassificationLossFunction):
    """Multinomial deviance loss function for multi-class classification.
    For multi-class classification we need to fit ``n_classes`` trees at
    each stage.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    is_multi_class = True

    def __init__(self, n_classes):
        if n_classes < 3:
            raise ValueError("{0:s} requires more than 2 classes.".format(
                self.__class__.__name__))
        super().__init__(n_classes)

    def init_estimator(self):
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Multinomial deviance.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        # create one-hot label encoding
        Y = np.zeros((y.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            Y[:, k] = y == k

        return np.average(
            -1 * (Y * raw_predictions).sum(axis=1) +
            logsumexp(raw_predictions, axis=1),
            weights=sample_weight
        )

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):
        """Compute negative gradient for the ``k``-th class.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        k : int, default=0
            The index of the class.
        """
        return y - np.nan_to_num(np.exp(raw_predictions[:, k] -
                                        logsumexp(raw_predictions, axis=1)))

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Make a single Newton-Raphson step. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        numerator *= (self.K - 1) / self.K

        denominator = np.sum(sample_weight * (y - residual) *
                             (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        return np.nan_to_num(
            np.exp(raw_predictions -
                   (logsumexp(raw_predictions, axis=1)[:, np.newaxis])))

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions
```

```python id="p-3zP4ex12F0"
class ExponentialLoss(ClassificationLossFunction):
    """Exponential loss function for binary classification.
    Same loss as AdaBoost.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    References
    ----------
    Greg Ridgeway, Generalized Boosted Models: A guide to the gbm package, 2007
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the exponential loss
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return np.mean(np.exp(-(2. * y - 1.) * raw_predictions))
        else:
            return (1.0 / sample_weight.sum() * np.sum(
                sample_weight * np.exp(-(2 * y - 1) * raw_predictions)))

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the residual (= negative gradient).
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        y_ = -(2. * y - 1.)
        return y_ * np.exp(y_ * raw_predictions.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        raw_predictions = raw_predictions.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        y_ = 2. * y - 1.

        numerator = np.sum(y_ * sample_weight * np.exp(-y_ * raw_predictions))
        denominator = np.sum(sample_weight * np.exp(-y_ * raw_predictions))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(2.0 * raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        return (raw_predictions.ravel() >= 0).astype(int)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # according to The Elements of Statistical Learning sec. 10.5, the
        # minimizer of the exponential loss is .5 * log odds ratio. So this is
        # the equivalent to .5 * binomial_deviance.get_init_raw_predictions()
        raw_predictions = .5 * np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)
```

```python id="lirY5ula12Ds"
LOSS_FUNCTIONS = {
    'ls': LeastSquaresError,
    'lad': LeastAbsoluteError,
    'huber': HuberLossFunction,
    'quantile': QuantileLossFunction,
    'deviance': None,  # for both, multinomial and binomial
    'exponential': ExponentialLoss,
}
```

<!-- #region id="CzNFmnLSzz1e" -->
## Gradient Boosted Regression Trees
<!-- #endregion -->

<!-- #region id="QWwIJNr2z-9o" -->
This module contains methods for fitting gradient boosted regression trees for both classification and regression.

The module structure is the following:
- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.
- ``MooGBTClassifier`` implements gradient boosting for
  classification problems.
- ``MooGBTRegressor`` implements gradient boosting for
  regression problems.
<!-- #endregion -->

```python id="XgSLmBFoz-7F"
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting."""

    @abstractmethod
    def __init__(self, *, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, min_impurity_split,
                 init, subsample, max_features, ccp_alpha,
                 random_state, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, constraints):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        
        constraints_ = []
        for constraint in constraints:
            constraint['alpha'] = 0
            constraints_.append(constraint)
        self.constraints = constraints_
        self.n_constraints = len(constraints_)

    @abstractmethod
    def _validate_y(self, y, sample_weight=None):
        """Called by fit to validate y."""

    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
                   random_state, X_csc=None, X_csr=None):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self.loss_
        if self.n_constraints:
            y_0 = y[:, 0]
        else:
            y_0 = y
        original_y = y_0

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y_0 = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)

            # induce regression tree on residuals
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha)

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight,
                     check_input=False)

            # update tree leaves
            loss.update_terminal_regions(
                tree.tree_, X, y, residual, raw_predictions, sample_weight,
                sample_mask, learning_rate=self.learning_rate, k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions

    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input.")

    def _validate_data(self, X, y='no_validation', reset=True,
                       validate_separately=False, **check_params):
        """Validate input data and set or check the `n_features_in_` attribute.
        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default='no_validation'
            The targets.
            - If `None`, `check_array` is called on `X`. If the estimator's
              requires_y tag is True, then an error will be raised.
            - If `'no_validation'`, `check_array` is called on `X` and the
              estimator's requires_y tag is ignored. This is a default
              placeholder and is never meant to be explicitly set.
            - Otherwise, both `X` and `y` are checked with either `check_array`
              or `check_X_y` depending on `validate_separately`.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.
        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """

        if y is None:
            if self._get_tags()['requires_y']:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
            X = check_array(X, **check_params)
            out = X
        elif isinstance(y, str) and y == 'no_validation':
            X = check_array(X, **check_params)
            out = X
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = check_array(X, **check_X_params)
                y = check_array(y, **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)

        return out

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (MultinomialDeviance
                          if len(self.classes_) > 2
                          else BinomialDeviance)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        if is_classifier(self):
            self.loss_ = loss_class(self.n_classes_, self.constraints, self.n_constraints) # change
        elif self.loss in ("huber", "quantile"):
            self.loss_ = loss_class(self.alpha)
        else:
            self.loss_ = loss_class(self.constraints, self.n_constraints)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == 'zero'):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    "Got init={}".format(self.init)
                )

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classifier(self):
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes."""
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_ = np.resize(self.estimators_,
                                     (total_n_estimators, self.loss_.K))
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = np.resize(self.oob_improvement_,
                                                  total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    @abstractmethod
    def _warn_mae_for_criterion(self):
        pass

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        monitor : callable, default=None
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.
        Returns
        -------
        self : object
        """
        if self.criterion == 'mae':
            # TODO: This should raise an error from 1.1
            self._warn_mae_for_criterion()

        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        # Since check_array converts both X and y to the same dtype, but the
        # trees use different types for X and y, checking them separately.

        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'],
                                   dtype=DTYPE, multi_output=True)
        n_samples, self.n_features_ = X.shape

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_sample_weight(sample_weight, X)
        
        if self.n_constraints:
            y = np.asarray(y)
            y_0 = y[:, 0]
            maxi = y_0.max()
            mini = y_0.min()

            for i, constraint in enumerate(self.constraints, 1):
                y_i = y[:, i]
                c_maxi = y_i.max()
                c_mini = y_i.min()

                if c_maxi != maxi:
                    raise ValueError(
                        f"""Multi Objectives have different scale. 
                        Sub-objective at index {i} has different scale than the Primary Objective(index=0)
                        Make sure minimum and maximum values of objectives are same
                        """
                    )
                if c_mini != mini:
                    raise ValueError(
                        f"""Multi Objectives have different scale. 
                        Sub-objective at index {i} has different scale than the Primary Objective(index=0)
                        Make sure minimum and maximum values of objectives are same
                        """
                    )



            # shape = np.shape(y)
            # if shape[1] != self.n_constraints + 1:
            #     raise ValueError("Number of additional y variables\
            #         and constrains not same") 
            y_0 = column_or_1d(y[:, 0], warn=True)
            
        else:
            y_0 = column_or_1d(y, warn=True)


        if is_classifier(self):
            y_0 = self._validate_y(y_0, sample_weight)
        else:
            y_0 = self._validate_y(y_0)

        if self.n_iter_no_change is not None:
            stratify = y_0 if is_classifier(self) else None
            X, X_val, y_0, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y_0, sample_weight,
                                random_state=self.random_state,
                                test_size=self.validation_fraction,
                                stratify=stratify))
            if is_classifier(self):
                if self._n_classes != np.unique(y_0).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y_0, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        'The training data after the early stopping split '
                        'is missing some classes. Try using another random '
                        'seed.'
                    )
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                           dtype=np.float64)
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y_0)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y_0, sample_weight=sample_weight)
                    except TypeError as e:
                        # regular estimator without SW support
                        raise ValueError(msg) from e
                    except ValueError as e:
                        if "pass parameters to specific steps of "\
                           "your pipeline using the "\
                           "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = \
                    self.loss_.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X, y, raw_predictions, sample_weight, self._rng, X_val, y_val,
            sample_weight_val, begin_at_stage, monitor)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state,
                    X_val, y_val, sample_weight_val,
                    begin_at_stage=0, monitor=None):
        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_
        if self.n_constraints:
            y_0 = y[:, 0]
        else:
            y_0 = y

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(y_0[~sample_mask],
                                      raw_predictions[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i, X, y, raw_predictions, sample_weight, sample_mask,
                random_state, X_csc, X_csr)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y_0[sample_mask],
                                             raw_predictions[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score - loss_(y_0[~sample_mask],
                                          raw_predictions[~sample_mask],
                                          sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y_0, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                       dtype=np.float64)
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(
                X, self.init_).astype(np.float64)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate,
                       raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X):
        """Compute raw predictions of ``X`` for each iteration.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate,
                          raw_predictions)
            yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        self._check_initialized()

        relevant_trees = [tree
                          for stage in self.estimators_ for tree in stage
                          if tree.tree_.node_count > 1]
        if not relevant_trees:
            # degenerate case where all trees have only one node
            return np.zeros(shape=self.n_features_, dtype=np.float64)

        relevant_feature_importances = [
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in relevant_trees
        ]
        avg_feature_importances = np.mean(relevant_feature_importances,
                                          axis=0, dtype=np.float64)
        return avg_feature_importances / np.sum(avg_feature_importances)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.
        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features,)
            The set of target features for which the partial dependence
            should be evaluated.
        Returns
        -------
        averaged_predictions : ndarray of shape \
                (n_trees_per_iteration, n_samples)
            The value of the partial dependence function on each grid point.
        """
        if self.init is not None:
            warnings.warn(
                'Using recursion method with a non-constant init predictor '
                'will lead to incorrect partial dependence values. '
                'Got init=%s.' % self.init,
                UserWarning
            )
        grid = np.asarray(grid, dtype=DTYPE, order='C')
        n_estimators, n_trees_per_stage = self.estimators_.shape
        averaged_predictions = np.zeros((n_trees_per_stage, grid.shape[0]),
                                        dtype=np.float64, order='C')
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                tree = self.estimators_[stage, k].tree_
                tree.compute_partial_dependence(grid, target_features,
                                                averaged_predictions[k])
        averaged_predictions *= self.learning_rate

        return averaged_predictions

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves

    def get_loss_function_values(self):
        """Get Loss function values for objectives and subobjectives
        Parameters
        ----------
        
        Returns
        -------
        y : 2d list
        """
        return self.loss_.losses
    
    def get_alpha_values(self):
        """Get Alpha values for subobjectives
        Parameters
        ----------
        
        Returns
        -------
        y : 2d list
        """
        return self.loss_.alphas
```

```python id="yZ9T5IEaz-4_"
class MooGBTClassifier(ClassifierMixin, BaseGradientBoosting):
    """Gradient Boosting for classification.
    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.
    Read more in the :ref:`User Guide <gradient_boosting>`.
    Parameters
    ----------
    loss : {'deviance', 'exponential'}, default='deviance'
        The loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
    criterion : {'friedman_mse', 'mse', 'mae'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria
        are 'friedman_mse' for the mean squared error with improvement
        score by Friedman, 'mse' for mean squared error, and 'mae' for
        the mean absolute error. The default value of 'friedman_mse' is
        generally the best as it can provide a better approximation in
        some cases.
        .. versionadded:: 0.18
        .. deprecated:: 0.24
            `criterion='mae'` is deprecated and will be removed in version
            1.1 (renaming of 0.26). Use `criterion='friedman_mse'` or `'mse'`
            instead, as trees should use a least-square criterion in
            Gradient Boosting.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_depth : int, default=3
        The maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.
    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :meth:`fit` and :meth:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random spliting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If 'auto', then `max_features=sqrt(n_features)`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.
        .. versionadded:: 0.20
    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        .. versionadded:: 0.20
    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        .. versionadded:: 0.20
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.
        .. versionadded:: 0.20
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``
    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.
    loss_ : LossFunction
        The concrete ``LossFunction`` object.
    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.
    estimators_ : ndarray of DecisionTreeRegressor of \
shape (n_estimators, ``loss_.K``)
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_ : int
        The number of data features.
    n_classes_ : int
        The number of classes.
    max_features_ : int
        The inferred value of max_features.
    See Also
    --------
    HistGradientBoostingClassifier : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    RandomForestClassifier : A meta-estimator that fits a number of decision
        tree classifiers on various sub-samples of the dataset and uses
        averaging to improve the predictive accuracy and control over-fitting.
    AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
        on the original dataset and then fits additional copies of the
        classifier on the same dataset where the weights of incorrectly
        classified instances are adjusted such that subsequent classifiers
        focus more on difficult cases.
    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    J. Friedman, Stochastic Gradient Boosting, 1999
    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    Examples
    --------
    The following example shows how to fit a gradient boosting classifier with
    100 decision stumps as weak learners.
    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import MooGBTClassifier
    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]
    >>> clf = MooGBTClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    0.913...
    """

    _SUPPORTED_LOSS = ('deviance', 'exponential')

    @_deprecate_positional_args
    def __init__(self, *, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 validation_fraction=0.1, n_iter_no_change=None, tol=1e-4,
                 ccp_alpha=0.0, constraints=[]):
        
        # Check constraints
        # for constraint in constraints:
        #     print (constraint)

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
            constraints=constraints)

    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError("y contains %d class after sample_weight "
                             "trimmed classes with zero weights, while a "
                             "minimum of 2 classes are required."
                             % n_trim_classes)
        self._n_classes = len(self.classes_)
        # expose n_classes_ attribute
        self.n_classes_ = self._n_classes
        return y

    def _warn_mae_for_criterion(self):
        # TODO: This should raise an error from 1.1
        warnings.warn("criterion='mae' was deprecated in version 0.24 and "
                      "will be removed in version 1.1 (renaming of 0.26). Use "
                      "criterion='friedman_mse' or 'mse' instead, as trees "
                      "should use a least-square criterion in Gradient "
                      "Boosting.", FutureWarning)

    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        score : ndarray of shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            :term:`classes_`. Regression and binary classification produce an
            array of shape (n_samples,).
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        raw_predictions = self.decision_function(X)
        encoded_labels = \
            self.loss_._raw_prediction_to_decision(raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = \
                self.loss_._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        raw_predictions = self.decision_function(X)
        try:
            return self.loss_._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss) from e

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self.loss_._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss) from e
```

```python id="T4v8dU0Dz-2W"
class MooGBTRegressor(RegressorMixin, BaseGradientBoosting):
    """Gradient Boosting for regression.
    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.
    Read more in the :ref:`User Guide <gradient_boosting>`.
    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, default='ls'
        Loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
    criterion : {'friedman_mse', 'mse', 'mae'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.
        .. versionadded:: 0.18
        .. deprecated:: 0.24
            `criterion='mae'` is deprecated and will be removed in version
            1.1 (renaming of 0.26). The correct way of minimizing the absolute
            error is to use `loss='lad'` instead.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.
    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
        initial raw predictions are set to zero. By default a
        ``DummyEstimator`` is used, predicting either the average target value
        (for loss='ls'), or a quantile for the other losses.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random spliting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    alpha : float, default=0.9
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.
        .. versionadded:: 0.20
    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.
        .. versionadded:: 0.20
    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        .. versionadded:: 0.20
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``
    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.
    loss_ : LossFunction
        The concrete ``LossFunction`` object.
    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.
    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        The collection of fitted sub-estimators.
    n_classes_ : int
        The number of classes, set to 1 for regressors.
        .. deprecated:: 0.24
            Attribute ``n_classes_`` was deprecated in version 0.24 and
            will be removed in 1.1 (renaming of 0.26).
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.
    n_features_ : int
        The number of data features.
    max_features_ : int
        The inferred value of max_features.
    See Also
    --------
    HistGradientBoostingRegressor : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.tree.RandomForestRegressor : A random forest regressor.
    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import MooGBTRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = MooGBTRegressor(random_state=0)
    >>> reg.fit(X_train, y_train)
    MooGBTRegressor(random_state=0)
    >>> reg.predict(X_test[1:2])
    array([-61...])
    >>> reg.score(X_test, y_test)
    0.4...
    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    J. Friedman, Stochastic Gradient Boosting, 1999
    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    @_deprecate_positional_args
    def __init__(self, *, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0,
                 constraints=[]):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
            constraints=constraints)

    def _validate_y(self, y, sample_weight=None):
        if y.dtype.kind == 'O':
            y = y.astype(DOUBLE)
        return y

    def _warn_mae_for_criterion(self):
        # TODO: This should raise an error from 1.1
        warnings.warn("criterion='mae' was deprecated in version 0.24 and "
                      "will be removed in version 1.1 (renaming of 0.26). The "
                      "correct way of minimizing the absolute error is to use "
                      " loss='lad' instead.", FutureWarning)

    def predict(self, X):
        """Predict regression target for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.
        This method allows monitoring (i.e. determine error on testing set)
        after each stage.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

    # FIXME: to be removed in 1.1
    # mypy error: Decorated property not supported
    @deprecated("Attribute n_classes_ was deprecated "  # type: ignore
                "in version 0.24 and will be removed in 1.1 "
                "(renaming of 0.26).")
    @property
    def n_classes_(self):
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_classes_ attribute."
                .format(self.__class__.__name__)
            ) from nfe
        return 1
```

<!-- #region id="r9Gi1afFz-zz" -->
## Expedia Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SryAHVQm1EYw" executionInfo={"status": "ok", "timestamp": 1635188880281, "user_tz": -330, "elapsed": 1820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e8d3b717-f3fa-4bd1-a6b3-498711850a33"
!wget -q --show-progress https://github.com/Swiggy/Moo-GBT/raw/master/examples/data/train_data_sample_2.csv
!head train_data_sample_2.csv
```

```python id="nEDfdQcQ1EWi"
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import seaborn as sns
import matplotlib.pyplot as plt 

import sys

pd.set_option('display.max_columns', None)

plt.rcParams['figure.figsize'] = (18, 6)
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams["font.size"] = "20"
plt.rcParams["legend.fontsize"] = 12
```

<!-- #region id="mkc3iIVv1ETg" -->
## Data Preparation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kcDqyNkS206m" executionInfo={"status": "ok", "timestamp": 1635188966166, "user_tz": -330, "elapsed": 1403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="529ab9fb-8a19-411a-dd23-598ea1e3ff8f"
train_data = pd.read_csv('train_data_sample_2.csv')
train_data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="nrhaiNMc22Nu" executionInfo={"status": "ok", "timestamp": 1635188970560, "user_tz": -330, "elapsed": 567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="68f4af24-9681-4590-c54d-1c33a8ab82e7"
train_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ePYqZYsU24nQ" executionInfo={"status": "ok", "timestamp": 1635188984387, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e22e7b2a-ae4c-4ce4-a064-5153affd0a43"
po = 'is_booking' # primary objective
so = 'is_package' # sub-objective

features = list(train_data.columns)
features.remove('is_booking')
outcome_flag = 'is_booking'

len(features)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1Uq3vKMQ277M" executionInfo={"status": "ok", "timestamp": 1635188991425, "user_tz": -330, "elapsed": 574, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d18e8998-0d80-48b5-d567-152efdeba219"
X_train, X_test, y_train, y_test = train_test_split(train_data[features],\
                                                    train_data[outcome_flag],\
                                                    test_size=0.2,\
                                                    stratify=train_data[['is_package', 'is_booking']],\
                                                    random_state=2021
                                                   )
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="IuWWGhYc29qR" executionInfo={"status": "ok", "timestamp": 1635188996850, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e6d47ced-816c-45b0-92c3-879f00143aea"
y_train.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/"} id="oaF8En7i2-yw" executionInfo={"status": "ok", "timestamp": 1635189001033, "user_tz": -330, "elapsed": 650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ae4ba0e-8030-49db-c6a3-39ad262e31d3"
y_test.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/"} id="KU_PVR183ABl" executionInfo={"status": "ok", "timestamp": 1635189008568, "user_tz": -330, "elapsed": 480, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e187b3ad-9c20-4b95-eab2-b3c496895549"
y_train_ = pd.DataFrame()
y_train_['is_booking'] = y_train
y_train_['is_package'] = X_train['is_package']

y_test_ = pd.DataFrame()
y_test_['is_booking'] = y_test
y_test_['is_package'] = X_test['is_package']

y_train_.shape, y_test_.shape
```

```python id="9A9Y6RY93B8q"
# X_train.drop('is_package', axis=1, inplace=True)
# X_test.drop('is_package', axis=1, inplace=True)
```

```python id="pOTMGMwx3EXK"
def evaluate_results(y_prob, y_true, threshold=0.5):
    avg_precision = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    return avg_precision, auc_roc
```

<!-- #region id="OJ6fCziJ3FEe" -->
## Unconstrained GBT on Primary Objective
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="55U0aLQP3gcq" executionInfo={"status": "ok", "timestamp": 1635189202655, "user_tz": -330, "elapsed": 42882, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9120be8c-7ba4-4bcd-d6fb-bbe28265f314"
unconstrained_gbt = MooGBTClassifier(
                        loss='deviance',
                        n_estimators=100,
                        random_state=2021
)

unconstrained_gbt.fit(X_train, y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="0zLycK0u3nRJ" executionInfo={"status": "ok", "timestamp": 1635189202656, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8b6c5f0e-260c-42b8-d04f-e67fb5a675f1"
def get_binomial_deviance_cost(pred, y):
    return -np.mean(y * np.log(pred) + (1-y) * np.log(1-pred))

pred_train = unconstrained_gbt.predict_proba(X_train)[:,1]
pred_test = unconstrained_gbt.predict_proba(X_test)[:,1]

# get sub-objective costs
so_train_cost = get_binomial_deviance_cost(pred_train, y_train_[so])
so_test_cost = get_binomial_deviance_cost(pred_test, y_test_[so])

print (f"""
Sub-objective cost train - {so_train_cost},
Sub-objective cost test  - {so_test_cost}
""")
```

```python colab={"base_uri": "https://localhost:8080/"} id="uWytUK9z3sQz" executionInfo={"status": "ok", "timestamp": 1635189202657, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1dafc579-b4d8-4b64-d281-4ca7091eebf8"
# Training Accuracies
po_auprc, po_auroc = evaluate_results(pred_train, y_train_.iloc[:,0])
so_auprc, so_auroc = evaluate_results(pred_train, y_train_.iloc[:,1])
print (f"""Training Accuracies
Primary Objective AUROC - {po_auroc}
Subobjective AUROC - {so_auroc}
""")
```

```python colab={"base_uri": "https://localhost:8080/"} id="tYZP4FbG3tzI" executionInfo={"status": "ok", "timestamp": 1635189202660, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c47099a0-7d3d-4f94-9b3d-415adf618e3c"
# Validation Accuracies
po_auprc, po_auroc = evaluate_results(pred_test, y_test_.iloc[:,0])
so_auprc, so_auroc = evaluate_results(pred_test, y_test_.iloc[:,1])
print (f"""Validation Accuracies
Primary Objective AUROC - {po_auroc}
Subobjective AUROC - {so_auroc}
""")
```

```python colab={"base_uri": "https://localhost:8080/"} id="KYa8hEIQ3vBd" executionInfo={"status": "ok", "timestamp": 1635189204680, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d428d8f-11c3-4aa0-d91e-00327e3c6738"
# Primary and Sceondary Loss Values for Training
print (f"""Training Losses - 
Primary Loss - {get_binomial_deviance_cost(pred_train, y_train_.iloc[:,0])}
Secondary Loss - {get_binomial_deviance_cost(pred_train, y_train_.iloc[:,1])}""")
```

```python colab={"base_uri": "https://localhost:8080/"} id="DUXK0mVm3x1F" executionInfo={"status": "ok", "timestamp": 1635189207247, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="656921b0-5bde-408f-a381-b1f11ec9f10c"
# Primary and Sceondary Loss Values for Validation
print (f"""Validation Losses -
Primary Loss - {get_binomial_deviance_cost(pred_test, y_test_.iloc[:,0])}
Secondary Loss - {get_binomial_deviance_cost(pred_test, y_test_.iloc[:,1])}""")
```

<!-- #region id="X7JefqZ43yy3" -->
## Constrained GBT
<!-- #endregion -->

```python id="KSrkByh234B9"
b = 0.65 # upper bound on cost
mu = 10
```

```python colab={"base_uri": "https://localhost:8080/"} id="dk1yFLtW33VX" executionInfo={"status": "ok", "timestamp": 1635189281371, "user_tz": -330, "elapsed": 44101, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="00be7dce-ab0c-41fe-f5fc-b6e1e80867cb"
constrained_gbt = MooGBTClassifier(
                    loss='deviance',
                    n_estimators=100,
                    constraints=[{"mu":mu, "b":b}],
                    random_state=2021
)

constrained_gbt.fit(X_train, y_train_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="PzcjoLHI358e" executionInfo={"status": "ok", "timestamp": 1635189281374, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="816badb2-7a88-4b76-d42f-daec3ed2cfae"
losses = constrained_gbt.loss_.get_losses()
losses.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 357} id="P13A0IoN37da" executionInfo={"status": "ok", "timestamp": 1635189316066, "user_tz": -330, "elapsed": 933, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f460326d-91ac-4a3b-cd32-85c6bc58e9f8"
sns.lineplot(data=losses, x='n_estimators', y='primary_objective', label='primary objective')
sns.lineplot(data=losses, x='n_estimators', y='sub_objective_1', label='subobjective')
plt.xlabel("# estimators(trees)")
plt.ylabel("Cost")
plt.legend(loc = "upper right")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 353} id="kn79IeUo39iM" executionInfo={"status": "ok", "timestamp": 1635189320236, "user_tz": -330, "elapsed": 773, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="09106319-c0eb-4f75-98d4-8c80b6f1e55f"
sns.lineplot(data=losses, x='n_estimators', y='primary_objective', label='primary objective')
plt.show()
```

<!-- #region id="Lb9FMOKM3_5m" -->
## Changing hyperparameters b and mu
<!-- #endregion -->

```python id="BT3NcS4p4De_"
bs = [0.9, 0.8, 0.7, 0.65, 0.6]
mus = [1000, 1000, 100, 10, 5]
```

```python id="2maxmpR04GGL"
# bs = [0.9]
# mus = [10000, 1000]
```

```python id="Kuf54Xti4QFj"
gbt_results = []
losses_master = pd.DataFrame()

for b in bs:
    for mu in mus:
        clf = MooGBTClassifier(loss="deviance", \
                                   constraints=[{"mu":mu, "b":b}])
        
        clf.fit(X_train, y_train_)
        
        losses = pd.DataFrame(clf.loss_.losses, columns=['primary',\
                                          'subobjective_1'])\
                                .reset_index()\
                                .rename(columns={'index': 'n_estimators'})
        losses['b'] = b
        losses['mu'] = mu
        losses['alpha'] = pd.DataFrame(clf.loss_.alphas)
        
        losses_master = pd.concat([losses_master, losses])
        
        pred = clf.predict_proba(X_test)[:,1]
        pred_train = clf.predict_proba(X_train)[:,1]
        
        # Validation Results
        po_val_auprc, po_val_auroc = evaluate_results(pred,\
                                                y_test_.iloc[:,0])
        so_val_auprc, so_val_auroc = evaluate_results(pred,\
                                                y_test_.iloc[:,1])
        
        # Training Results
        po_train_auprc, po_train_auroc = evaluate_results(pred_train,\
                                                    y_train_.iloc[:,0])
        so_train_auprc, so_train_auroc = evaluate_results(pred_train,\
                                                    y_train_.iloc[:,1])
        
        
        # Training Losses
        po_loss_train = get_binomial_deviance_cost(pred_train, y_train_.iloc[:,0])
        so_loss_train = get_binomial_deviance_cost(pred_train, y_train_.iloc[:,1])
        
        # Testing Losses
        po_loss_val = get_binomial_deviance_cost(pred, y_test_.iloc[:,0])
        so_loss_val = get_binomial_deviance_cost(pred, y_test_.iloc[:,1])
        
        gbt_res = [b, mu, po_train_auroc, so_train_auroc, po_val_auroc,\
                   so_val_auroc, po_loss_train, so_loss_train,\
                   po_loss_val, so_loss_val]
        
        gbt_results.append(gbt_res)

gbt_results_df = pd.DataFrame(gbt_results, columns=['b', 'mu', 'po_train_auroc',\
                                                    'so_train_auroc', 'po_val_auroc',\
                                                    'so_val_auroc', 'po_loss_train',\
                                                    'so_loss_train', 'po_loss_val',\
                                                    'so_loss_val'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 844} id="RPPPQjVh4S2z" executionInfo={"status": "ok", "timestamp": 1635190431355, "user_tz": -330, "elapsed": 82, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="71621f2f-49b0-4dda-a4a3-6f27865bb429"
gbt_results_df['po_gain'] = ((gbt_results_df['po_val_auroc'] - po_auroc)/po_auroc)*100
gbt_results_df['so_gain'] = ((gbt_results_df['so_val_auroc'] - so_auroc)/so_auroc)*100
gbt_results_df
```

```python id="NpTe_6XR4T_g"
gbt_results_df.to_csv('gbt_results_df.csv', index=False)
losses_master.to_csv('losses_master.csv', index=False)
```

```python id="7i3DxeHm4ZLT"
def plot_losses(b_, mu_, ):
    losses = losses_master[(losses_master['b']==b_) & (losses_master['mu']==mu_)]
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=losses, x='n_estimators', y='primary', label='primary objective')
    sns.lineplot(data=losses, x='n_estimators', y='subobjective_1', label='subobjective')
    plt.legend(loc = "upper left", prop={'size': 2})
    plt.title(f"mu = {mu_}, b = {b_}")
    plt.xlabel("# estimators(trees)")
    plt.ylabel("Cost")
    plt.legend(loc = "upper right")

    plt.subplot(1, 2, 2)
    sns.lineplot(data=losses, x='n_estimators', y='alpha')

    plt.title(f"mu = {mu_}, b = {b_}")
    plt.xlabel("# estimators(trees)")
    plt.ylabel("Alpha")

    plt.show()
```

```python id="SupTPr0K4aWb"
def plot_primary_loss(b_, mu_):
    losses = losses_master[(losses_master['b']==b_) & (losses_master['mu']==mu_)]

    sns.lineplot(data=losses, x='n_estimators', y='primary', label='primary objective')
    plt.legend(loc = "upper left", prop={'size': 2})
    plt.title(f"Primary Objective Cost, mu = {mu_}, b = {b_}")
    plt.xlabel("# estimators(trees)")
    plt.ylabel("Cost")
    plt.legend(loc = "upper left")
    plt.show()
```

```python id="4PRyCCBK4cCk"
def plot_sub_loss(b_, mu_):
    losses = losses_master[(losses_master['b']==b_) & (losses_master['mu']==mu_)]

    sns.lineplot(data=losses, x='n_estimators', y='subobjective_1', label='subobjective')
    plt.legend(loc = "upper left", prop={'size': 2})
    plt.title(f"Primary Objective Cost, mu = {mu_}, b = {b_}")
    plt.xlabel("# estimators(trees)")
    plt.ylabel("Cost")
    plt.legend(loc = "upper left")
    plt.show()
```

```python id="djKq79EV4dE8"
def plot_alphas(b_, mu_):
    losses = losses_master[(losses_master['b']==b_) & (losses_master['mu']==mu_)]

    sns.lineplot(data=losses, x='n_estimators', y='alpha')
    sns.lineplot(data=losses, x='n_estimators', y='alpha', label='subobjective')

    plt.xticks(np.arange(0, losses['n_estimators'].max() + 5, 5))

    plt.title(f"mu = {mu_}, b = {b_}")
    plt.xlabel("# estimators(trees)")
    plt.ylabel("Alpha Value")
    plt.legend()
    plt.show()
```

```python id="p4DrjM3h4eHp"
plt.rcParams['figure.figsize'] = (18, 6)
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams["font.size"] = "20"
plt.rcParams["legend.fontsize"] = 12

# params = {'legend.fontsize': 20,
#           'legend.handlelength': 2}
# plot.rcParams.update(params)
```

```python id="DIJxCkoL4goq"
b_ = 0.9
mu_ = 1000
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="t4RE2n7X4ke3" executionInfo={"status": "ok", "timestamp": 1635190712493, "user_tz": -330, "elapsed": 1218, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b14a180-b3ff-4a41-c9db-d2bd4596fc39"
b_ = 0.9
mu_ = 100
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="84BLHzvC4o9p" executionInfo={"status": "ok", "timestamp": 1635190722316, "user_tz": -330, "elapsed": 2036, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e50653a2-9128-4dde-c564-3be1927d9a58"
b_ = 0.9
mu_ = 10
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python id="YvO7CnqW4qiM"
b_ = 0.8
mu_ = 1000
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="Dz5BykWd4sg8" executionInfo={"status": "ok", "timestamp": 1635190726716, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e58592d6-43a4-4312-dcb1-beaf2e6e580c"
b_ = 0.8
mu_ = 100
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="Dh4YnLRk4seX" executionInfo={"status": "ok", "timestamp": 1635190732843, "user_tz": -330, "elapsed": 1436, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ebb0dbd0-6d4b-4194-a9d9-bd9ece58967f"
b_ = 0.8
mu_ = 10
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="GU-8m91-4sbi" executionInfo={"status": "ok", "timestamp": 1635190734894, "user_tz": -330, "elapsed": 2056, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8adb17b1-03a1-47e0-d0cb-a0ce9a090def"
b_ = 0.8
mu_ = 10
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="fCMhKqmT4sYd" executionInfo={"status": "ok", "timestamp": 1635190734896, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="890981fa-33c0-4b47-bede-22a06a13d276"
b_ = 0.65
mu_ = 100
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="Sod2ZN1o4sWK" executionInfo={"status": "ok", "timestamp": 1635190736072, "user_tz": -330, "elapsed": 1191, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae2288d0-0c7f-4699-cb2a-0baec469dc0d"
b_ = 0.65
mu_ = 10
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python id="3VskIZlo43Tc"
b_ = 0.6
mu_ = 1000
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="tt2hAJwQ44yv" executionInfo={"status": "ok", "timestamp": 1635190741886, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae02addc-a6c0-42ca-804a-de1bf89179fd"
b_ = 0.6
mu_ = 100
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 728} id="PVq5eN9J45yi" executionInfo={"status": "ok", "timestamp": 1635190741888, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="338cb158-cfc7-4c56-a5ff-8a6ea949563b"
b_ = 0.6
mu_ = 5
plot_losses(b_, mu_)
plot_primary_loss(b_, mu_)
```

<!-- #region id="zRWPQpJH46w0" -->
## Scalarization
<!-- #endregion -->

```python id="-hHSS6Um48Au"
from sklearn.ensemble import GradientBoostingClassifier
```

<!-- #region id="p53nt82t4_x0" -->
**Primary Objective Unconstrained GBT**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hMeekCsm5Buh" executionInfo={"status": "ok", "timestamp": 1635190749089, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="69b5f004-c386-4668-f2d0-5ecf9a5ba476"
po, so
```

```python colab={"base_uri": "https://localhost:8080/"} id="_NKP7ss35Dhr" executionInfo={"status": "ok", "timestamp": 1635190751108, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6c4df0db-ee8c-413d-c7ee-05c2b16d5e33"
features_1 = features + ['is_package']
outcome_flag = po

X_train, X_test, y_train, y_test = train_test_split(train_data[features_1],\
                                                    train_data[outcome_flag],\
                                                    test_size=0.2,\
                                                    stratify=train_data[['is_package', 'is_booking']],\
                                                    random_state=2021
                                                   )

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gj9qk_ZO4sQC" executionInfo={"status": "ok", "timestamp": 1635190794431, "user_tz": -330, "elapsed": 42613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2bb3768-b80c-423b-b3fc-4a312f20db45"
model_1 = GradientBoostingClassifier(
                    loss='deviance',
                    n_estimators=100,
                    random_state=2021
)

model_1.fit(X_train, y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="tV7Q9-WZ5I_V" executionInfo={"status": "ok", "timestamp": 1635190794436, "user_tz": -330, "elapsed": 46, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dc469ca6-39ce-467f-e07c-e12930b35310"
pred_train_po = model_1.predict_proba(X_train)[:,1]
pred_test_po = model_1.predict_proba(X_test)[:,1]

po_train_auprc, po_train_auroc = evaluate_results(pred_train_po,\
                                                y_train)

po_val_auprc, po_val_auroc = evaluate_results(pred_test_po,\
                                                y_test)
po_train_auroc, po_val_auroc
```

<!-- #region id="SRszXbRo5Opv" -->
## Primary Objective Unconstrained GBT
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8IE5u0UB5QFd" executionInfo={"status": "ok", "timestamp": 1635190794438, "user_tz": -330, "elapsed": 38, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="12091658-8691-4e48-fc1a-e5963b55dc4a"
len(features)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4s3KxiR25RXV" executionInfo={"status": "ok", "timestamp": 1635190794439, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4b69b509-c5d3-48d6-99c2-caac4ed178e8"
# features.remove('is_package')
outcome_flag = so

X_train, X_test, y_train, y_test = train_test_split(train_data[features],\
                                                    train_data[outcome_flag],\
                                                    test_size=0.2,\
                                                    stratify=train_data[['is_package', 'is_booking']],\
                                                    random_state=2021
                                                   )

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="h3tqZ0Li5TGZ" executionInfo={"status": "ok", "timestamp": 1635190804982, "user_tz": -330, "elapsed": 10555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9b2ff141-61cf-41e7-85be-e6761304f9e0"
model_2 = GradientBoostingClassifier(
                    loss='deviance',
                    n_estimators=100,
                    random_state=2021
)

model_2.fit(X_train, y_train)
```

```python id="teWzlAuv-xe1" outputId="39dead05-88cb-4d3c-95db-b4734a15fd61"
pred_train_so = model_2.predict_proba(X_train)[:,1]
pred_test_so = model_2.predict_proba(X_test)[:,1]

so_train_auprc, so_train_auroc = evaluate_results(pred_train_so,\
                                                y_train)

so_val_auprc, so_val_auroc = evaluate_results(pred_test_so,\
                                                y_test)
so_train_auroc, so_val_auroc
```

```python id="2JYi59Hr-xe1" outputId="164d6464-a39f-408e-c967-712388ea1ff9"
features_ = features + [po, so]
X_train, X_test = train_test_split(train_data[features_],\
                                    test_size=0.2,\
                                    stratify=train_data[['is_package', 'is_booking']],\
                                    random_state=2021
                                   )
X_train.shape, X_test.shape
```

```python id="mqGCvgJW-xe1" outputId="240052d4-dfb3-4f37-9ce1-2feeccdd702d"
len(features)
```

```python id="SXNdTDpS-xe2"
# features.append('is_package')
pred_train_po = model_1.predict_proba(X_train[features_1])[:,1]
pred_test_po = model_1.predict_proba(X_test[features_1])[:,1]

# features.remove('is_package')
pred_train_so = model_2.predict_proba(X_train[features])[:,1]
pred_test_so = model_2.predict_proba(X_test[features])[:,1]
```

```python id="hdbjEnGE-xe2" outputId="1c52be31-39c6-4a53-f827-d7e585e6e636"
scalar_train = pd.DataFrame()

scalar_train['po_pred'] = pred_train_po
scalar_train['so_pred'] = pred_train_so

scalar_train['po_true'] = X_train[po].tolist()
scalar_train['so_true'] = X_train[so].tolist()

scalar_test = pd.DataFrame()

scalar_test['po_pred'] = pred_test_po
scalar_test['so_pred'] = pred_test_so

scalar_test['po_true'] = X_test[po].tolist()
scalar_test['so_true'] = X_test[so].tolist()

scalar_train.shape, scalar_test.shape
```

```python id="zlsANZ28-xe2" outputId="0925b2ff-ecf2-4b5a-bc65-24a5ae4f2a6d"
res = []
for wt_ in range(1, 10):
    wt = float(wt_/10)
    w1 = wt
    w2 = 1 - wt
    
    scalar_train['weighted_score'] = w1 * scalar_train['po_pred'] + \
                                     w2 * scalar_train['so_pred']

    scalar_test['weighted_score'] = w1 * scalar_test['po_pred'] + \
                                    w2 * scalar_test['so_pred']
    
    _, po_train_auroc = evaluate_results(scalar_train['weighted_score']\
                     , scalar_train['po_true'])
    _, so_train_auroc = evaluate_results(scalar_train['weighted_score']\
                     , scalar_train['so_true'])
    
    _, po_test_auroc = evaluate_results(scalar_test['weighted_score']\
                     , scalar_test['po_true'])
    _, so_test_auroc = evaluate_results(scalar_test['weighted_score']\
                     , scalar_test['so_true'])
    
    l = [w1, w2, po_train_auroc, so_train_auroc, po_test_auroc, so_test_auroc]
    
    res.append(l)
    
res_df = pd.DataFrame(res, columns=['w1', 'w2', 'po_train_auroc', \
                                    'so_train_auroc', 'po_test_auroc',\
                                    'so_test_auroc'])
res_df.shape
```

```python id="SOhV1Rgr-xe3" outputId="5c946093-fb52-4936-81a0-a5be0797754c"
scalar_train['po_true'].value_counts()
```

```python id="cM8L5GQ2-xe3" outputId="a677d226-5d2c-46f8-b3a0-367adac7cb9e"
scalar_test['po_true'].value_counts()
```

```python id="6CPUgiW--xe3" outputId="32b99c81-9980-403f-dc70-58835460af39"
scalar_train['so_true'].value_counts()
```

```python id="d4DFO1FG-xe3" outputId="c7d06553-f917-4819-9432-1baedd91aade"
scalar_test['so_true'].value_counts()
```

```python id="ZcPr64Rv-xe4" outputId="124284cd-c2f3-44ff-bda2-5fab142bab21"
res_df
```

```python id="kgJBOuzz-xe4"
po_auroc = 0.7213786201203704
so_auroc = 0.291705532936315

res_df['po_gain'] = ((res_df['po_test_auroc'] - po_auroc)/po_auroc)*100
res_df['so_gain'] = ((res_df['so_test_auroc'] - so_auroc)/so_auroc)*100
```

```python id="5v22L5z9-xe4" outputId="9ff8b9bd-9706-4b70-d06c-f7ad4ecf12c0"
res_df
```

```python id="uZL0XOnc-xe4" outputId="d2ce042d-f56b-4d87-e231-ea3aaf2de352"
res_df[['w1', 'w2', 'po_test_auroc', 'po_gain', 'so_test_auroc', 'so_gain']]
```

<!-- #region id="M1gQMPoSjREA" -->
## Takeaways
<!-- #endregion -->

<!-- #region id="cUX3Sf_MjR51" -->
1. Scalarization works surprisingly well and it should be your first choice as you head into MOO. It starts breaking down when you have more than two objectives or when the individual models start drifting in potentially opposing directions thereby invalidating the originally set λ value.
2. Moo-GBT-like methods appear to work better when the objectives are somewhat orthogonal in nature. For example, if your primary objective is improving clicks and the sub-objective is improving conversions and since all conversions require a click, the objectives are potentially highly correlated.
<!-- #endregion -->
