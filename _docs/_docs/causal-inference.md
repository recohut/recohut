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

<!-- #region id="gAzbkZMootDk" -->
# Causal Inference
> Concepts of causal inferencing with code

- toc: true
- badges: true
- comments: true
- categories: [Causal]
- image:
<!-- #endregion -->

<!-- #region id="OCY84CVipET9" -->
#### Introduction

To tackle such questions, we will introduce the key ingredient that causal analysis depends on---counterfactual reasoning---and describe the two most popular frameworks based on Bayesian graphical models and potential outcomes. Intuitively, the counterfactual framework measures causal effects by comparing measured outcomes in two almost-identical worlds---imagine two parallel universes, identical in every way up until the point where a some "treatment" occurs in one world but not the other. 

Building upon the counterfactual framework, we introduce causal graphs, which are a tool for formalizing implicit assumptions about causal mechanisms (e.g., encoding domain knowledge about causal mechanisms into an analysis); and potential outcomes methods, which are statistical tools for estimating causal effects. --- Where some intuition comes in. 

We close our introduction by presenting the randomized experiment as the simplest method for causal inference. We describe the randomized experiment in the language of the counterfactual framework, providing a causal graph and associated potential outcomes formulation, and show how this conceptually clean and simple method addresses the challenges of causal inference.




#### Conditioning-based methods

Conditioning-based methods are the workhorse of causal inference when running active experiments is not feasible. We discuss these methods by showing how each one is, in its own way, attempting to approximate the gold standard randomized experiment. **Conditioning** on key causal variables is the simplest method for isolating causal effect. **Matching and stratification** approximates conditioning at high-diemensions an continuous variable settings. **Regression** can also be used. **Doubly robust estimators** provides the best of conditioning and regression by combining propensity-based and regression-based methods. **Synthetic control method** can be used if none of the above suits, it is especially usefull when the treatment is applied to the entire population. 

#### Natural Experiments

Conditioning methods can fail if important confounders and unobserved. Here we smiply attempt to find an observed variable that acts like the randomised arm of an experiment. **Simple natural experiment** can be used, the type we see in the field, in the lab or occur as a result of some exogenous phenomena. **Instrumental variables** method ensures that we obtain the true causal effect even when there are unobserved confounders. **Regresion discontinuity** is the process to look for dicontinuity in bserved data. 

#### Sensitivity Analysis

At the end we have to see how the results change once we alter the assumptions for both observational studies and natural experiments. 


<!-- #endregion -->

<!-- #region id="rn1yRAeIt0T6" -->
Think of smoking as the confound to red-meat eating and heart attacks. 

Here we want to study the effect of treatment $T$ on the outcome $Y$, however as shown in the digraph, there is a confound $X$ that influences both $Y$ and $T$. Therefore to estimate $T\rightarrow Y$ we have to break the dependence $X\rightarrow T$ so that $T \perp X$. And $Y \perp X$ could also work but less practical. 

By using random experiment you actively assign a treatment unrelted to any confound. And by constructiona well run random experiment is $T \perp X$. The confound is equally split between the two groups as long as there is no selection bias in the sample selection procedure.

<center><img>  <img src="https://cdn.mathpix.com/snip/images/Ssb3nBeds2f8J8ulnQCQhX0N6ktxFk-wopFwExxKtrc.original.fullsize.png" height="200"/>


<!-- #endregion -->

<!-- #region id="OM9o5iX4x3Rg" -->
Lets try a new one. Our goal is to estimate the effect of excerise on cholesterol. But your age influences both your level of excercise and cholesterol. So to estimate the effect of excerices on cholesteral we are to break the dependence of age on excercies. So in a random experiment we can actively assign excercise independent of age.

If we can't actively intervene, we have to simulate random experiments through observational studies using on of six observational techniques. The first technique we will look at is **conditioning**. So clearly there is some relationship, but there is a confounder, so we can emperically define the relationship yet. Currently it looks like more stationary biking is related to higher cholesterol. In this example, older people are more likley to participate in stationary biking. We can condition on age by analysing each age group separately. 

<img> <img src="https://cdn.mathpix.com/snip/images/m6ZsRIvxmY6Cac8LTsy3kMKO-HBAn0SbrOHWVReaZvk.original.fullsize.png" />

$$
P\left(\text {Cholesterol } | \operatorname{do}\left(S_{-} B i k i n g\right)\right)=\sum_{\text {age}} P\left(\text {Cholesterol } | S_{-} \text {Biking,age}\right) P(\text {age})
$$
<!-- #endregion -->

<!-- #region id="KE4DxWrS1cXI" -->
So naturally in this example, we make a few critical assumptions. 



*   Age is the only confounder, this is called the **ignorability** or selection of obserables assumption.
*   **SUTVA assumption** says that a subject’s potential outcome is not affected by other subjects’ exposure to the treatment. There are not alternative treatment allocations that would lead to different outcomes due to network effects, such that of friends. 
*   The observations cover similar people, the common support or **overlap assumption**.
*   Don't include all age groups and the effect on excerciese, so will it **generalise** beyond observed region. 

Note it becomes very hard to know what to condition on when the dimensionality of X increases. In business X is large, it is therefore unlikley that you would make use of the conditioning techinque. 

<!-- #endregion -->

<!-- #region id="ZmrVbd_U4XuC" -->
You have to use the backdoor criterion to ensure that you have the right variables. So the casual effect is only true if the assumed graphical model is in fact correct.

<img>  <img src="https://cdn.mathpix.com/snip/images/tQvXOSUS1s-mLpBuDWQPNsWkbGZO3vskiGvRgoy5DiY.original.fullsize.png" />
<!-- #endregion -->

<!-- #region id="OaDTG3g05cO1" -->
### Now the process of matching and stratafication

![](https://cdn.mathpix.com/snip/images/4sRHsu93WmElDVquTWmRAYVH27PKQiNYZErZ5ntYeaw.original.fullsize.png)

#### And then with a bit of matching

![](https://cdn.mathpix.com/snip/images/l4Na-bCxsC1NTDQR5tEVneAsg6yUOxU2mJXhEI9KnU0.original.fullsize.png)


<!-- #endregion -->

<!-- #region id="XIUmWDx9561g" -->
Here we therefore find individual pairs of treated and untreated indviduals that are veery siilar to eachother. And this paired individuals then provide the counterfactual estimate for each other. You then simply average the difference in outcome within pairs to calucalte the average-treatment-effect (ATE) on the treated. 

The mahalanobis distance accounts for the unit differences by normalising each dimesions by the standard deviation. Here $S$ is the covariance matrix. 

$$
\text {Mahalanobis }\left(\vec{x}_{i}, \vec{x}_{j}\right)=\sqrt{(\overrightarrow{x_{i}}-\overrightarrow{x_{j}})^{T} S^{-1}(\overrightarrow{x_{i}}-\overrightarrow{x_{j}})}
$$
<!-- #endregion -->

<!-- #region id="24jVjbX07RJ7" -->
### Propensity Score

This is the individual's propensity to be treated, these scores are estimated, you can then use the propensity socre to subdivide the observational data so that $ T \perp X |score$. It therefore breaks the infleunce of confound $X$ and allows for the estimation of the true treatment effect. 

This is where a few machine learning models can be used for good measure. Here one tries to predict all the label (i.e. the treatment states) based on the observed covariates. You can use any model where the score is well calibrated so that $(100 \times p)\%$ of indviduals with score $p$ are observed to be treated. You can therefore use logistic regressions, SVMs and GAMs. 

https://onlinelibrary.wiley.com/doi/abs/10.1002/bimj.201800132


And then distance is the distance between propensity scores:

$$
\text { Distance }(\overrightarrow{x_{i}}, \overrightarrow{x_{j}})=|\hat{e}(\overrightarrow{x_{i}})-\hat{e}(\overrightarrow{x_{j}})|
$$


Propensity score matching works because individuals with similar covariates have similar scores and similar treatment likelihoods. It doesn't really matter whether or not the propensity score is accurate, the role is simply to characterise covariates and not to actually identify the treated from the untreated. If fact if the propensity score is too accurate, it means you can't disentable the covariates from the treatement status and as a reslt, any effect we observe could be due either to the treatment or to the correlated covariate, in such senario, you should not dumb down the model, instead you should redefine the problem statement and the treatment. 


<!-- #endregion -->

<!-- #region id="QvI6wwaEfUGl" -->
The code to this problem is very easy, use a machine learning model to predict the covariate status 
<!-- #endregion -->

```python id="MyZ3mcBGgpqE"
# learn propensity score model
psmodel = linear_model.LinearRegression()
psmodel.fit(covariates, treatment_status)
data['ps'] = psmodel.predict(covariates)
# find nearest neighbor matches
controlMatcher = NearestNeighbors().fit(untreated['ps’])
distances, matchIndex = controlMatch.kneighbors(treated['ps'])
# iterate over matched pairs and sum difference in outcomes
for i in range(numtreatedunits):
    treated_outcome = treated.iloc[i][outcome_name].item()
    untreated_outcome = untreated.iloc[matchIndex[i]][outcome_name].item()
    att += treated_outcome - untreated_outcome
# normalize 
att /= numtreatedunits

```

<!-- #region id="5i0MPqJchIgU" -->
With mathching we can choose to allow for replacement, it is a bias variance trade off. And if the nearest neighbour is too far away, it is advisible to use a caliper threshold to limit the acceptable distance. One should also be sure that all treated are not matched to untreated. You are more likley to used propensity score matching than Mahalanobis distance matching. Remember one central problem still remains with al these methods and it is unobserved confounds. 
<!-- #endregion -->

<!-- #region id="9j97mxGbh4YK" -->
#### Stratification

In matching it is one to one, in stratification it is many to many matching. Stratification identifies paired subpopulation with similar covariate distributions. 

We can use propensity scores to stratify populations. So we once more calculate the propensity score per individual but now instead of match we stratify them into groups. Once the groups have been established you can calculate the average treatment effect (ATE) as the weighted average of outcome differences per strata.
<!-- #endregion -->

<!-- #region id="FAIPcoPJjd3h" -->
$$
ATT =\sum_{s \in s t r a t a} \frac{1}{N_{s, T=1}}\left(\bar{Y}_{s, T=1}-\bar{Y}_{s, T=0}\right)
$$
<!-- #endregion -->

<!-- #region id="PxmGkelljujT" -->
where $
\bar{Y}_{S, T}
$ is the average outcome at strata $s$ and treatment status $T$ and $
N_{S, T=1}
$ is the number of treated individuals in strata $s$. 


<center><img><img src="https://cdn.mathpix.com/snip/images/4S4eJJY3e5RBvMQjHfm9sNMV5r8hCrXoZNEy7ZgJHfg.original.fullsize.png" height=300/></center>







<!-- #endregion -->

```python id="AtrbdzcylSBH"
# build propensity score model and assign each item a score as earlier…

# create a column 'strata' for each element that marks what strata it belongs to
data['strata'] = ((data['ps'].rank(ascending=True) / numrows) * numStrata).round(0)
data['T_y'] = data['T'] * data['outcome’]            # T_y = outcome iff treated
data['Tbar'] = 1 - data['treated’]                   # Tbar = 1 iff untreated
data['Tbar_y'] = data['Tbar'] * data['outcome']      # Tbar_y = outcome iff untreated
stratified = data.groupby('strata')
# sum weighted outcomes over all strata  (weight by treated population)
outcomes = stratified.agg({'T':['sum'],'Tbar':['sum'],'T_y':['sum'],'Tbar_y':['sum']})
# calculate per-strata effect
outcomes[‘T_y_mean'] = outcomes[‘T_y_sum'] / outcomes['T']
outcomes[‘Tbar_y_mean'] = outcomes[‘Tbar_y_sum'] / outcomes['dbar_sum']
outcomes['effect'] = outcomes[‘T_y_mean'] - outcomes[‘Tbar_y_mean’]
# weighted sum of effects over all strata
att = (outcomes['effect'] * outcomes['T']).sum() / totaltreatmentpopulation

```

<!-- #region id="qdM1YWx1lW2Y" -->
We would also have to choose how many strata to select, if you have around 100 data points, you can pick five. If you have 10k - 1mn pick 100-1000 strata. We don't wat to pick a small nuber of strata as it is a bias-variance trade-off. Matching is high variance, low bias. Often times we find that there might not be enough treated and untreated indviduals in a stratum tyically near propensity scores of 0.0 and 1.0, in which case we clip the strata from the analyis, which technically means we are now calcualting a local-average-treatment-effet. 
<!-- #endregion -->

<!-- #region id="CuHycBCjmiXi" -->
### Weighting
This is the next mechanism that is an alternative to conditioning. Is it possible ot assign weights to observations to simulate a randomised experiment. So with stratification we know that we weigh each strata's total impact by the number of treated. So weighting argues that the weighting of the treated population is similar to weighting by propensity score, so they calculate the effect by weighted sum over all individual outcomes. 

$$
A T E=\frac{1}{N_{T=1}} \sum_{i \in \text {treated}} w_{i} Y_{i}-\frac{1}{N_{T=0}} \sum_{j \in u n t r e a t e d} w_{j} Y_{j}
$$

And the Inverse Probability of Treatment Weighting (IPTW) 

$$
\begin{aligned} w_{i}=& \frac{T}{e}+\frac{1-T}{1-e} \\ \mathrm{N}_{\mathrm{T}=1}=& \sum \frac{T}{e} ; \quad N_{T=0}=\sum \frac{1-T}{1-e} \end{aligned}
$$

<!-- #endregion -->

<!-- #region id="lEktSzOKoFO1" -->
Weights on each individual act to balance the distribution of covariates in the treated and untreated groups.  (i.e., break the dependence between treatment status and covariates)

You obtain a high variance when $e$ is close to 0 or 1. Note $e$ is the propensity scores. As such it might become crucuial to clip weights. This model assumes that propensity scores are correctly specified. There are variants to this generalised weighting protocol such as the ATE on treated only. 
<!-- #endregion -->

<!-- #region id="g9Kee7F9poKi" -->
### Simple Regression
<!-- #endregion -->

<!-- #region id="1UlgDcdRpsIG" -->
Here regression and supervised learning can be used interchangably. In regression analysis, we build a model of $Y$ as a function of covariates $X$ and $T$, amnd interpret the coeffients of the covariates and treatment causally. 

$$
E(Y | X, T)=\alpha_{1} X_{1}+\alpha_{2} X_{2}+\cdots \alpha_{n} X_{n}+\alpha_{T} T
$$
for example 

$$
\text { Cholesterol }=\alpha_{\text {age}} A g e+\alpha_{\text {exercise}} \text {Exercise}
$$

The bigger the coefficient the stronger the causal relationship to the outcome, $Y$. The problem is that the causal interpretation of regressions require many assumptions. **Model Correctness** what if we use a linear model and a causal model is non-linear. **Multicollinearity** if the covariates are correlated, you can't get accurate coefficients. **Ignorability** omission of confounds will invalidate findings. This method should largely be ignored, unless you are asbolutely sure about what you are doing. 

<!-- #endregion -->

<!-- #region id="unRjuOVgvcS1" -->
### Doubly robust
<!-- #endregion -->

<!-- #region id="f55nMUinve8O" -->
This method makes use of both propensity score and regression methods. Both of which uses models that have to be correctly specified, i.e. not linear when the relationship is non-linear. With this method, if either propensity score **or** regression is correctly specified, then doubly robust is correct. As a result, it seems that the doubly robust method should always stritly be better than each method indvidually, but that is not entirely true, if both models are slightly incorrect, then doubly robust can become very biased.


$$
D R_{1}=\left\{\begin{array}{ll}{\frac{Y}{\hat{e}}-\frac{\hat{Y}_{T=1}(1-\hat{e})}{\hat{e}},} & {T=1} \\ {\hat{Y}_{T=1},} & {T=0}\end{array}\right.
$$

$$
D R_{0}=\left\{\begin{array}{ll}{\hat{Y}_{T=0},} & {T=1} \\ {\frac{Y}{1-\hat{e}}-\frac{\hat{Y}_{T=1} \hat{e}}{1-\hat{e}},} & {T=0}\end{array}\right.
$$


<!-- #endregion -->

<!-- #region id="3Bsy_EP964qA" -->
The idea is to calculate the first and second estimators for each individual and then to calucalte the mean of the two estimators over the whole study population and take the difference as the causal effect of $T$.
<!-- #endregion -->

<!-- #region id="XHyOmNb47g0-" -->
### Synthetic Controls

All the previous methods require that we observe both the treated and untreated individuals. There could be a global policy change tht would lead to eveeryone being treated. It is possible to do a pre/post comparison but it is not robust to dynamics like seasonality. The alternative is to build synthetic controls that estimate what $\bar{Y}_{T=0}$ would have been for a population were it not for the treatment. There is more to be said here, but I will leave it for later. 


<!-- #endregion -->

<!-- #region id="yjS-K5Ex_VsU" -->
## Natural Experiment

### Simple Natural Experiment

Under natural experiments we have simple natural experiments, instrumental variabls, and regression discontinuities. Natural here means, as if nature conducted an experiment. Here instead of just assuming ignorability, we find data sets that approximate an experiment. Here we have A/B testing, Lottery, Randomised policy, and an external shock to the treatment. It is never truly random, so you exploit and as-if random assignment to meature the effect of the treatment on the outcome. 
<!-- #endregion -->

<!-- #region id="13rXn3w4A0Ff" -->
### Instrumental Variables

At first you have to attempt to find an instrument variable which is anything that affects the cause but not the outcome. 

<center> <img> <img src="https://cdn.mathpix.com/snip/images/jvKsKCDIsPX1Cd2URnxOkozLtWLnIATanMYt2JdoxeU.original.fullsize.png" height=250 />
<!-- #endregion -->

<!-- #region id="6RYvkjB2B7CV" -->
As a result, a change in Y is a product of the change in Z -> X and X -> Y. And the causal effect is 

$$
\text { Causal effect }(X->Y)=\frac{Y_{Z=1}-Y_{Z=0}}{X_{Z=1}-X_{Z=0}}
$$
<!-- #endregion -->

<!-- #region id="fgn3ujSNCVBB" -->
Due to it being an external event you can look at it as-if it is random variation. Example: What is the effect of recommendations on an app store?
Instrumental Variable: External sources that drive sudden, large traffic to an app. Because IVs are not influenced by confounds, IVs’ indirect effect on outcome Y is independent of confounds too.  Because IVs do not directly influence outcome, their effect must be due to the effect of the treatment. They seem very common now. You then of course get regression discontinuities, that are quite common and can be thought of as a special case of an instrumental variable.

<!-- #endregion -->

<!-- #region id="qMk_jHzOErU5" -->
## Refutations

Causal inference is only possible with assumptions. It is critical to review your assumptions, but the question is, how can this be done. 

DoWhy focuses attention on the assumptions required for causal inference. 

Provides estimation methods such as matching and IV so that you can focus on the identifying assumptions. 

*   Models assumptions explicitly using causal graphical model. 
*   Provides an easy way to test them (if possible) or analyze sensitivity to violations. 


Unifies all methods to yield four verbs for causal inference: 
* Model
* Identify
* Estimate
* Refute

<!-- #endregion -->

<!-- #region id="QFmZtqgwFrdj" -->
Can add randomly drawn covariates into data

Rerun your analysis. 

Does the causal estimate change?  (Hint: it shouldn’t)

---

Randomize or permute the treatment.

Rerun your analysis. 

Does the causal estimate change? (Hint: it should become 0)

---
Create subsets of your data.

Rerun your analysis. 

Does the causal estimate vary across subsets? 
(Hint: it shouldn’t vary significantly)

---
Many methods (e.g., matching, stratification, weighting, regression discontinuity) depend on balancing of covariates

Can test this.

---
Question: How sensitive is your estimate to minor violations of assumptions?

E.g. How big should the effect of a confounder be so that your estimate reverses in direction?

Use simulation to add effect of unknown confounders. 

Domain knowledge helps to guide reasonable values of the simulation. 

Make comparisons to other known estimates.




<!-- #endregion -->

<!-- #region id="YU2DODDXGZFH" -->
![](https://cdn.mathpix.com/snip/images/fuJzJybm6NgRKslmS5dS1tyMzUPuYcA2eTi0DnLCqyg.original.fullsize.png)
<!-- #endregion -->

<!-- #region id="9VG57Zr_GnaH" -->
---

Always follow the four steps: Model, Identify, Estimate, Refute.
Refute is the most important step.

Aim for simplicity.
If your analysis is too complicated, it is most likely wrong. 


Try at least two methods with different assumptions. 
Higher confidence in estimate if both methods agree.

---

•Input: Observational data,  Causal graph
•Output: Causal effect between desired variables, “What-if” analysis 



<!-- #endregion -->

<!-- #region id="uNRO8Fd7N5MV" -->
--- 

When you have high dimensional problems you can use dimensionality reduction tehcniques or regularised models. 

---

And network effects complicate causal inference. An individual outcomes should not depend on another's treatment status. One can then consider partitioned sub-networks as a unit of analysis. 

---

WEIRD problem in social studies, might not  generalise to other users, platforms or cultures. So you can coroborate findings accorss. And be explicity of the potential of non-generalisability. 

---

There is a few common confounders that can lead to selection bias, theses are structured (demographics, usage patterns) and unstructures (activity, preferences). Time spent on page is not important without page, more activity can simply mean more activity at that time consider that schools are closed. 

<!-- #endregion -->

<!-- #region id="QLVQyKTmPif-" -->
### Other

Causal discovery is a harder problem than causal inference. Causal inference looks at the effects of causes, and causal discovery at the causes of effects. 

Heterogeneuos treatment effects. Average causal effect does not capture individual-level variations. Stratification is one of the simplest methods for hetrongenous treatment by strata, typical strata are demographics. You need more data when you stratify to detect statistical differences, otherwise it can purely be down to noise. For high dimensions we can use machine learning methods like random forests such as those by Susan Athey. 

Machine learning can also use causal inference methods for robust generalisable prediction. And causal inference can use ML algorithms to better model the non-linear effects of confounders. 

RL and causal inference can be used together. You can feed in A/B test into multi-armed bandits, MDPs and POMDPs to generalise a randomised experiment. So you can have two goals, one is to show the best known algorithm to the users as a recommender, and the second to keep randomising to update knowledge about competing algorithms. 

As a pratical example, you can look at contextual bandits for Yahoo News. The action is to display different news articles, this is done using an episilon greedy policy. 

<!-- #endregion -->

<!-- #region id="lPGhJMFbH5BJ" -->
## Dowhy
<!-- #endregion -->

```python id="AnoGsqc_7p0k"
# !git clone https://github.com/microsoft/dowhy.git
# !pip install -r dowhy/requirements.txt
# !python dowhy/setup.py install

!pip install dowhy
```

<!-- #region id="0ODgBVPSs4Gv" -->
### First Example
<!-- #endregion -->

```python id="f5gEq1zHOjIn" colab={"base_uri": "https://localhost:8080/"} outputId="ab8953b8-b516-40ae-f009-61140ba3df18"
import numpy as np
import pandas as pd
#from dowhy import CausalModel
from dowhy.do_why import CausalModel
import dowhy.datasets as ds
import logging

### beta 10 is what I want the cause to be. 
data = ds.linear_dataset(beta=10,
        num_common_causes=5,
        num_instruments = 2,
        num_samples=10000, 
        treatment_is_binary=True)
df = data["df"]
print(df.head())
print(data["dot_graph"])
print("\n")
print(data["gml_graph"])
```

```python id="yLTvkJGpOjFC" colab={"base_uri": "https://localhost:8080/"} outputId="229e0c27-5323-496a-b9ea-20107f1b2c7e"
# We now input a causal graph in the GML graph format (recommended). 
# You can also use the DOT format.
# We are not going to change the format
# You only need the graph when you don't specify the common causes and instruments
# If you don't add the graph not common causes or instruments, it'll be ignored as such
# With graph

model=CausalModel(
        data = df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
        instruments=data["instrument_names"],
        logging_level = logging.INFO
        )

```

```python id="xCkbkxZ6Oi--" colab={"base_uri": "https://localhost:8080/"} outputId="7cee488d-89bc-49ec-c8d4-26232540de2a"
model.view_model()
```

<!-- #region id="6q2QpLZ-uHNZ" -->
The above causal graph shows the assumptions encoded in the causal model. We can now use this graph to first identify the causal effect (go from a causal estimand to a probability expression), and then estimate the causal effect.

DoWhy philosophy: Keep identification and estimation separate

Identification can be achieved without access to the data, acccesing only the graph. This results in an expression to be computed. This expression can then be evaluated using the available data in the estimation step. It is important to understand that these are orthogonal steps.

    Identification



<!-- #endregion -->

```python id="mU_sj5lBOi00" colab={"base_uri": "https://localhost:8080/"} outputId="4a74e6d5-e0f2-404b-a4a2-2e57bd7ca6e8"
#identified_estimand = model.identify_effect()
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
```

<!-- #region id="lKvpXSenvL5s" -->
    Estimation
<!-- #endregion -->

<!-- #region id="uF_NTBly2ynC" -->
#### Regression
<!-- #endregion -->

```python id="Qsdh0CHz23ZU" colab={"base_uri": "https://localhost:8080/"} outputId="dcee43e3-241b-4798-a4ad-0e83f142a38b"
causal_estimate_reg = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)
print(causal_estimate_reg)
print("Causal Estimate is " + str(causal_estimate_reg.value))
```

<!-- #region id="vz6X6rYi3CDB" -->
#### Stratification

We will be using propensity scores to stratify units in the data.
<!-- #endregion -->

```python id="fgMP4aXB3Dzq" colab={"base_uri": "https://localhost:8080/"} outputId="1b60cbdf-a312-4a48-8225-948c769da7dd"
causal_estimate_strat = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_stratification")
print(causal_estimate_strat)
print("Causal Estimate is " + str(causal_estimate_strat.value))
```

<!-- #region id="vTIZwYTV3QnU" -->
#### Matching
<!-- #endregion -->

```python id="RgGq9nyp3Dx-" colab={"base_uri": "https://localhost:8080/"} outputId="c8171846-bfa5-478f-b3ab-55deeabfcc78"
causal_estimate_match = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching")
print(causal_estimate_match)
print("Causal Estimate is " + str(causal_estimate_match.value))
```

<!-- #region id="qyQVR5WQ3X5f" -->
#### Weighting

We will be using (inverse) propensity scores to assign weights to units in the data. DoWhy supports a few different weighting schemes:

    Vanilla Inverse Propensity Score weighting (IPS)
    Self-normalized IPS weighting (also known as the Hajek estimator)
    Stabilized IPS weighting

<!-- #endregion -->

```python id="3XWx4zO-3Duu" colab={"base_uri": "https://localhost:8080/"} outputId="24a9a8a7-398a-4a5d-f0d3-a0f1611580d6"
causal_estimate_ipw = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting", method_params={"weighting_scheme":"ips_weight"})
print(causal_estimate_ipw)
print("Causal Estimate is " + str(causal_estimate_ipw.value))
```

<!-- #region id="SQye5ATZ31U7" -->
#### Instrumental Variable

We will be using the Wald estimator for the provided instrumental variable.
<!-- #endregion -->

```python id="yOl5y5if3DrU" colab={"base_uri": "https://localhost:8080/"} outputId="bd79b0b1-dd4d-490d-989f-7fd752eafe03"
causal_estimate_iv = model.estimate_effect(identified_estimand,
        method_name="iv.instrumental_variable", method_params={'iv_instrument_name':'Z0'})
print(causal_estimate_iv)
print("Causal Estimate is " + str(causal_estimate_iv.value))
```

<!-- #region id="t4Ra5V1z4Dvl" -->
#### Regression Discontinuity

We will be internally converting this to an equivalent instrumental variables problem.
<!-- #endregion -->

```python id="b58u-1GK3DlD" colab={"base_uri": "https://localhost:8080/"} outputId="17d45b84-1ae3-463a-c6ce-6559c9fabf7b"
causal_estimate_regdist = model.estimate_effect(identified_estimand,
        method_name="iv.regression_discontinuity", 
        method_params={'rd_variable_name':'Z1',
                       'rd_threshold_value':0.5,
                       'rd_bandwidth': 0.1})
print(causal_estimate_regdist)
print("Causal Estimate is " + str(causal_estimate_regdist.value))
```

```python id="5RAWr6WUOiqk" colab={"base_uri": "https://localhost:8080/"} outputId="a97cef8c-2316-4cf5-8cfd-85138b60c196"
### Now without the graph, and instead common causes
### The point here is to ignore the entire graph, except for commone causes (ie. no instrument)
### Confounds are always assumed regardless 

# Without graph                                       
model= CausalModel(                             
        data=df,                                      
        treatment=data["treatment_name"],             
        outcome=data["outcome_name"],                 
        common_causes=data["common_causes_names"])  

model.view_model()
```

```python id="nFsisiWgOijD" colab={"base_uri": "https://localhost:8080/"} outputId="c627c36b-1b89-437e-d51b-2046da161d16"
## We get the same causal graph. Now identification and estimation is done as before.
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_stratification")         
print(estimate)
print("Causal Estimate is " + str(estimate.value))

```

<!-- #region id="cJ2Ks8q3wIZP" -->
### Now Refuting

Let us now look at ways of refuting the estimate obtained.

Adding a random common cause variable

<!-- #endregion -->

```python id="SvMdV60nOiYQ" colab={"base_uri": "https://localhost:8080/"} outputId="a558f5fa-b217-4143-9485-8542eef3590f"
res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
print(res_random)
```

<!-- #region id="-_0SAEQ7wSy2" -->
Adding an unobserved common cause variable
<!-- #endregion -->

```python id="9cyrfYrlwzdq"
res_unobserved=model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause",
                                     confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear",
                                    effect_strength_on_treatment=0.01, effect_strength_on_outcome=0.02)
print(res_unobserved)
```

<!-- #region id="d5A2cj25w9cI" -->
Replacing treatment with a random (placebo) variable
<!-- #endregion -->

```python id="-iSGV97qOiMT" colab={"base_uri": "https://localhost:8080/"} outputId="dc08e3d3-47ff-440d-e5ee-9f602a0eba35"
res_placebo=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)
## We want the effect close to zero
```

<!-- #region id="ivWokZkOw-SB" -->
Removing a random subset of the data
<!-- #endregion -->

```python id="wHCH6XDfOiF6" colab={"base_uri": "https://localhost:8080/"} outputId="2a7acb06-ec03-4580-9a0e-207825bfc62c"
res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)

```

<!-- #region id="JxkrZMOxxFXJ" -->
As you can see, the propensity score stratification estimator is reasonably robust to refutations. For reproducibility, you can add a parameter "random_seed" to any refutation method, as shown below.
<!-- #endregion -->

```python id="oeHSzCOFOiBY" colab={"base_uri": "https://localhost:8080/"} outputId="aa9d67e2-3cc2-4f2b-d373-7a520336318a"
res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", subset_fraction=0.9, random_seed = 1)
print(res_subset)
```

<!-- #region id="mPGusPCV5WMZ" -->
#### Instrumental Variable Application
<!-- #endregion -->

```python id="WaApIO0-Oh7Q"
import numpy as np
import pandas as pd
import patsy as ps # describing statistical models 

from statsmodels.sandbox.regression.gmm import IV2SLS
import os, sys
sys.path.append(os.path.abspath("../../../"))
```

```python id="M-zN_GMfOh3v"
n_points = 1000
education_abilty = 1
education_voucher = 0.5
income_abilty = 2
income_education = 4


# confounder
ability = np.random.normal(0, 3, size=n_points)

# instrument
voucher = np.random.normal(2, 1, size=n_points) 

# treatment
education = np.random.normal(5, 1, size=n_points) + education_abilty * ability +\
            education_voucher * voucher

# outcome
income = np.random.normal(10, 3, size=n_points) +\
         income_abilty * ability + income_education * education

# build dataset
data = np.stack([ability, education, income, voucher]).T
df = pd.DataFrame(data, columns = ['ability', 'education', 'income', 'voucher'])
```

```python id="_nVOtwg5Ohxn" colab={"base_uri": "https://localhost:8080/"} outputId="d720ac75-5eba-4b8e-eb2b-2695c7d7c4d3"
df.head()
```

```python id="BbH382N1Ohuw" colab={"base_uri": "https://localhost:8080/"} outputId="82fbfb83-4a83-4d9b-89d9-f186bf2a18a5"
income_vec, endog = ps.dmatrices("income ~ education", data=df)
exog = ps.dmatrix("voucher", data=df) # creates intercept

print("income")
print(income_vec[:5])
print("endog")
print(endog[:5])
print("exog")
print(exog[:5])

```

```python id="KCSRCSa0OhlE" colab={"base_uri": "https://localhost:8080/"} outputId="56ef1a1d-403b-4c27-d0fb-919f2c680613"
# outcome, treatment, instrument
m = IV2SLS(income_vec, endog, exog).fit()
m.summary()
```

<!-- #region id="fgwwnv-36-PN" -->
Now using dowhy
<!-- #endregion -->

```python id="DlwW4NKXOheQ" colab={"base_uri": "https://localhost:8080/"} outputId="fb87082e-a825-4088-d8b0-dbe57a318dcb"
model=CausalModel(
        data = df,
        treatment='education',
        outcome='income',
        common_causes=['ability'],
        instruments=['voucher']
        )

identified_estimand = model.identify_effect()

estimate = model.estimate_effect(identified_estimand,
        method_name="iv.instrumental_variable", test_significance=True
)
print(estimate)
```

<!-- #region id="lpj52elx98ck" -->
#### Different way to create graphs

<!-- #endregion -->

```python id="3HyQ5xCl-Ffl" colab={"base_uri": "https://localhost:8080/"} outputId="552ca95a-8193-49d1-fe88-53422b6e7e00"
import random
z=[i for i in range(10)]
random.shuffle(z)
df = pd.DataFrame(data = {'K': z, 'X': range(0,10), 'Y': range(0,100,10)})
df
```

```python id="UvFyKxOlOhbw" colab={"base_uri": "https://localhost:8080/"} outputId="6acdfebc-8be9-4851-b0ab-9b5f84d6809a"
# GML format 

# With GML string
model=CausalModel(
        data = df,
        treatment='X',
        outcome='Y',
        graph="""graph[directed 1 node[id "K" label "K"]  
                    node[id "X" label "X"]
                    node[id "Y" label "Y"]      
                    edge[source "K" target "X"]    
                    edge[source "K" target "Y"]     
                    edge[source "X" target "Y"]]"""
                    
        )
model.view_model()


```

```python id="3ARDIxDuOhUh"
# # With GML file
# model=CausalModel(
#         data = df,
#         treatment='X',
#         outcome='Y',
#         graph="example_graphs/simple_graph_example.gml"
#         )
# model.view_model()
```

```python id="9bbkoKbTOhPk" colab={"base_uri": "https://localhost:8080/"} outputId="0d539e2a-73e3-4569-c934-b60bc8ca0c59"
# With DOT string
model=CausalModel(
        data = df,
        treatment='X',
        outcome='Y',
        graph="digraph {Z -> X;Z -> Y;X -> Y;}"
        )
model.view_model()

```

```python id="_Kn3e-Dl-uCT"
# # With DOT file
# model=CausalModel(
#         data = df,
#         treatment='X',
#         outcome='Y',
#         graph="example_graphs/simple_graph_example.dot"
#         )
# model.view_model()
```

<!-- #region id="he6ZKH83_fQ2" -->
### Using Pandas API (do sampler)

The user should note that this is still an area of active research, so you should be careful about being too confident in bootstrapped error bars from do-samplers. Pearlian inference focuses on more fundamental quantities like the joint distribution of a set of outcomes Y, 𝑃(𝑌), which can be used to derive other statistics of interest. Inverse Probability Weighing of do-adjustment.

https://medium.com/@akelleh/introducing-the-do-sampler-for-causal-inference-a3296ea9e78d
<!-- #endregion -->

```python id="PyNwF1k4OhFN"
# the data already loaded in the previous cell. we include the import
# here you so you don't have to keep re-downloading it.

import pandas as pd
import dowhy.api

lalonde=pd.read_csv("https://raw.githubusercontent.com/sanadhis/ITT-ADA-2017/master/04%20-%20Applied%20ML/lalonde.csv")
```

<!-- #region id="jRuCGNaQA4Gq" -->
The key feature here is the do method, which produces a new dataframe replacing the treatment variable with values specified, and the outcome with a sample from the interventional distribution of the outcome. If you don't specify a value for the treatment, it leaves the treatment untouched:
<!-- #endregion -->

```python id="U4BHzbLMOhA5" colab={"base_uri": "https://localhost:8080/"} outputId="87c6126f-09e4-4a58-faf6-d8b451051e6e"
do_df = lalonde.causal.do(x='treat',
                          outcome='re78',
                          common_causes=['nodegr', 'black', 'hisp', 'age', 'educ', 'married'],
                          variable_types={'age': 'c', 'educ':'c', 'black': 'd', 'hisp': 'd', 
                                          'married': 'd', 'nodegr': 'd','re78': 'c', 'treat': 'b'})
```

<!-- #region id="fqVzVz17BBur" -->
Notice you get the usual output and prompts about identifiability. This is all dowhy under the hood!

We now have an interventional sample in do_df. It looks very similar to the original dataframe. Compare them:

<!-- #endregion -->

```python id="v3i-RPEJOg94" colab={"base_uri": "https://localhost:8080/"} outputId="a762bb93-f9bb-4391-e02f-38b810632e11"
print(lalonde.shape)
lalonde.head()
```

```python id="nCxMpTznvyF_" colab={"base_uri": "https://localhost:8080/"} outputId="cc08a6ba-9537-4352-fd12-4397a3fbc06c"
print(do_df.shape)
do_df.sort_values("id").head()
```

<!-- #region id="X2DeNN5cBbd5" -->
#### Treatment Effect Estimation
We could get a naive estimate before for a treatment effect by doing
<!-- #endregion -->

```python id="BObW0WDxBPpt" colab={"base_uri": "https://localhost:8080/"} outputId="8594ba6d-ca76-4ea2-b2e3-426bc35e7042"
print((lalonde[lalonde['treat'] == 1].mean() - lalonde[lalonde['treat'] == 0].mean())['re78'])
```

<!-- #region id="oFFP_u6WBpl6" -->
We can do the same with our new sample from the interventional distribution to get a causal effect estimate
<!-- #endregion -->

```python id="z_ngbfBUBEeE" colab={"base_uri": "https://localhost:8080/"} outputId="015968a7-b419-4fc0-a702-8ffff46a4889"
print((do_df[do_df['treat'] == 1].mean() - do_df[do_df['treat'] == 0].mean())['re78'])
```

<!-- #region id="qBbo_h3RBuG8" -->
We could get some rough error bars on the outcome using the normal approximation for a 95% confidence interval, like
<!-- #endregion -->

```python id="lCc4FdQ_Bnch" colab={"base_uri": "https://localhost:8080/"} outputId="1dec39c7-a434-4089-edec-c56e3b88e0e2"
import numpy as np
print(1.96*np.sqrt((do_df[do_df['treat'] == 1].var()/len(do_df[do_df['treat'] == 1])) + 
             (do_df[do_df['treat'] == 0].var()/len(do_df[do_df['treat'] == 0])))['re78'])
```

<!-- #region id="cv5xRVBWB6ZV" -->
but note that these DO NOT contain propensity score estimation error. For that, a bootstrapping procedure might be more appropriate.

This is just one statistic we can compute from the interventional distribution of 're78'. We can get all of the interventional moments as well, including functions of 're78'. We can leverage the full power of pandas, like

<!-- #endregion -->

```python id="rrEYCGIlB6CF" colab={"base_uri": "https://localhost:8080/"} outputId="e423531a-680a-48f8-c2e7-bbc733868764"
do_df['re78'].describe()
```

```python id="pdo4jEEdBv43" colab={"base_uri": "https://localhost:8080/"} outputId="488d2b36-8045-47db-f985-9745d6293e29"
%matplotlib inline
import seaborn as sns

sns.barplot(data=do_df, x='treat', y='re78')
```

```python id="daQcED2pCAAD" colab={"base_uri": "https://localhost:8080/"} outputId="3a735031-f260-4008-bf8a-c7d842560bec"
sns.barplot(data=lalonde, x='treat', y='re78')
```

<!-- #region id="7UP3yCQXFAf4" -->
#### Further Elaboration on Pearlian Intervention
<!-- #endregion -->

<!-- #region id="QbRkc0ffFGho" -->

The "do-sampler" is a new feature in do-why. While most potential-outcomes oriented estimators focus on estimating the specific contrast $E[Y_0 - Y_1]$, Pearlian inference focuses on more fundamental quantities like the joint distribution of a set of outcomes Y, $P(Y)$, which can be used to derive other statistics of interest.

Generally, it's hard to represent a probability distribution non-parametrically. Even if you could, you wouldn't want to gloss over finite-sample problems with you data you used to generate it. With these issues in mind, we decided to represent interventional distributions by sampling from them with an object called to "do-sampler". With these samples, we can hope to compute finite-sample statistics of our interventional data. If we bootstrap many such samples, we can even hope for good sampling distributions for these statistics. (Something that can still be tested)

The user should note that this is still an area of active research, so you should be careful about being too confident in bootstrapped error bars from do-samplers.

Note that do samplers sample from the outcome distribution, and so will vary significantly from sample to sample. To use them to compute outcomes, it's recommended to generate several such samples to get an idea of the posterior variance of your statistic of interest.

## Pearlian Interventions

Following the notion of an intervention in a Pearlian causal model, our do-samplers implement a sequence of steps:

1. Disrupt causes
2. Make Effective
3. Propagate and sample

In the first stage, we imagine cutting the in-edges to all of the variables we're intervening on. In the second stage, we set the value of those variables to their interventional quantities. In the third stage, we propagate that value forward through our model to compute interventional outcomes with a sampling procedure.

In practice, there are many ways we can implement these steps. They're most explicit when we build the model as a linear bayesian network in PyMC3, which is what underlies the MCMC do sampler. In that case, we fit one bayesian network to the data, then construct a new network representing the interventional network. The structural equations are set with the parameters fit in the initial network, and we sample from that new network to get our do sample.

In the weighting do sampler, we abstractly think of "disrupting the causes" by accounting for selection into the causal state through propensity score estimation. These scores contain the information used to block back-door paths, and so have the same statistics effect as cutting edges into the causal state. We make the treatment effective by selecting the subset of our data set with the correct value of the causal state. Finally, we generated a weighted random sample using inverse propensity weighting to get our do sample.

There are other ways you could implement these three steps, but the formula is the same. We've abstracted them out as abstract class methods which you should override if you'd like to create your own do sampler!

## Statefulness

The do sampler when accessed through the high-level pandas API is stateless by default.This makes it intuitive to work with, and you can generate different samples with repeated calls to the `pandas.DataFrame.causal.do`. It can be made stateful, which is sometimes useful. 

The 3-stage process we mentioned before is implemented by passing an internal `pandas.DataFrame` through each of the three stages, but regarding it as temporary. The internal dataframe is reset by default before returning the result.

It can be much more efficient to maintain state in the do sampler between generating samples. This is especially true when step 1 requires fitting an expensive model, as is the case with the MCMC do sampler, the kernel density sampler, and the weighting sampler. 

Instead of re-fitting the model for each sample, you'd like to fit it once, and then generate many samples from the do sampler. You can do this by setting the kwarg `stateful=True` when you call the `pandas.DataFrame.causal.do` method. To reset the state of the dataframe (deleting the model as well as the internal dataframe), you can call the `pandas.DataFrame.causal.reset` method.

Through the lower-level API, the sampler is stateful by default. The assumption is that a "power user" who is using the low-level API will want more control over the sampling process. In this case, state is carried by internal dataframe `self._df`, which is a copy of the dataframe passed on instantiation. The original dataframe is kept in `self._data`, and is used when the user resets state. 

## Integration

The do-sampler is built on top of the identification abstraction used throughout do-why. It uses a `dowhy.CausalModel` to perform identification, and builds any models it needs automatically using this identification.

## Specifying Interventions

There is a kwarg on the `dowhy.do_sampler.DoSampler` object called `keep_original_treatment`. While an intervention might be to set all units treatment values to some specific value, it's often natural to keep them set as they were, and instead remove confounding bias during effect estimation. If you'd prefer not to specify an intervention, you can set the kwarg like `keep_original_treatment=True`, and the second stage of the 3-stage process will be skipped. In that case, any intervention specified on sampling will be ignored.

If the `keep_original_treatment` flag is set to false (it is by default), then you must specify an intervention when you sample from the do sampler. For details, see the demo below!


## Demo

First, let's generate some data and a causal model. Here, Z confounds our causal state, D, with the outcome, Y.
<!-- #endregion -->

```python id="FZrWVoH-CDor" colab={"base_uri": "https://localhost:8080/"} outputId="553031ff-304b-4119-88a6-e82a5e3c6de2"
N = 5000

z = np.random.uniform(size=N)
d = np.random.binomial(1., p=1./(1. + np.exp(-5. * z)))
y = 2. * z + d + 0.1 * np.random.normal(size=N)

df = pd.DataFrame({'Z': z, 'D': d, 'Y': y})

print((df[df.D == 1].mean() - df[df.D == 0].mean())['Y'])


causes = ['D']
outcomes = ['Y']
common_causes = ['Z']

model = CausalModel(df, 
                    causes,
                    outcomes,
                    common_causes=common_causes)


## Idnetification

identification = model.identify_effect()

```

<!-- #region id="x0tkAME3FnH4" -->
Identification works! We didn't actually need to do this yet, since it will happen internally with the do sampler, but it can't hurt to check that identification works before proceeding. Now, let's build the sampler.
<!-- #endregion -->

```python id="MbA0kYpGFhnE" colab={"base_uri": "https://localhost:8080/"} outputId="6a24173b-4ba6-4e0b-ade5-312361616f73"
from dowhy.do_samplers.weighting_sampler import WeightingSampler

sampler = WeightingSampler(df,
                           causal_model=model,
                           keep_original_treatment=True,
                           variable_types={'D': 'b', 'Z': 'c', 'Y': 'c'})


```

<!-- #region id="KxOg1FSvFu2l" -->
Now, we can just sample from the interventional distribution! Since we set the keep_original_treatment flag to False, any treatment we pass here will be ignored. Here, we'll just pass None to acknowledge that we know we don't want to pass anything.

If you'd prefer to specify an intervention, you can just put the interventional value here instead as a list or numpy array.

<!-- #endregion -->

```python id="e89cYgdHFrSv" colab={"base_uri": "https://localhost:8080/"} outputId="96086491-0b30-4ead-e56f-9d5f7c626ff3"
interventional_df = sampler.do_sample(None)
print((interventional_df[interventional_df.D == 1].mean() - interventional_df[interventional_df.D == 0].mean())['Y'])

```

<!-- #region id="5dDgaa7SF4-J" -->
Now we're much closer to the true effect, which is around 1.0!
<!-- #endregion -->

<!-- #region id="WpXXNFbuIeWe" -->
## Uplift (CATE, ATE, ITE)
<!-- #endregion -->

<!-- #region id="anT5_rWQhHEk" -->
We consider a functional parameter called the conditional average treatment effect (CATE), designed to capture the heterogeneity of a treatment effect across subpopulations when the unconfoundedness assumption applies. In contrast to quantile regressions, the subpopulations of interest are defined in terms of the possible values of a set of continuous covariates rather than the quantiles of the potential outcome distributions.
<!-- #endregion -->

```python id="hl3c6hakfQIk"
similar project by microsoft

https://github.com/microsoft/EconML

```

<!-- #region id="YMjD87aIT_Ei" -->
The most famous use case of Uplift Modeling would be the 44th US president Barack Obama's 2nd presidential campaign in 2012. Obama's team used Uplift Modeling to find which voters could be persuaded to vote for him. Here are some articles.

    What is ‘Persuasion Modeling’, and how did it help Obama to win the elections?
    How Obama’s Team Used Big Data to Rally Voters
    How uplift modeling helped Obama's campaign -- and can aid marketers

<!-- #endregion -->

<!-- #region id="DF7-n5mfUQ_g" -->
Uplift Modeling estimates uplift scores (a.k.a. CATE: Conditional Average Treatment Effect or ITE: Individual Treatment Effect). Uplift score is how much the estimated conversion rate will increase by the campaign.

Suppose you are in charge of a marketing campaign to sell a product, and the estimated conversion rate (probability to buy a product) of a customer is 50 % if targeted and the estimated conversion rate is 40 % if not targeted, then the uplift score of the customer is (50–40) = +10 % points. Likewise, suppose the estimated conversion rate if targeted is 20 % and the estimated conversion rate if not targeted is 80%, the uplift score is (20–80) = -60 % points (negative value).

The range of uplift scores is between -100 and +100 % points (-1 and +1). It is recommended to target customers with high uplift scores and avoid customers with negative uplift scores to optimize the marketing campaign.
<!-- #endregion -->

<!-- #region id="7oODkZN4UUz1" -->

    CausalLift works with both A/B testing results and observational datasets.
    CausalLift can output intuitive metrics for evaluation.

<!-- #endregion -->

<!-- #region id="KnlUjS46UkM5" -->
In a word, to use for real-world business.

Existing packages for Uplift Modeling assumes the dataset is from A/B Testing (a.k.a. Randomized Controlled Trial). In real-world business, however, observational datasets in which treatment (campaign) targets were not chosen randomly are more common especially in the early stage of evidence-based decision making. CausalLift supports observational datasets using a basic methodology in Causal Inference called "Inverse Probability Weighting" based on the assumption that propensity to be treated can be inferred from the available features.

 There are 2 challenges of Uplift Modeling; explainability of the model and evaluation. CausalLift utilizes a basic methodology of Uplift Modeling called Two Models approach (training 2 models independently for treated and untreated samples to compute the CATE (Conditional Average Treatment Effects) or uplift scores) to address these challenges.

Explainability of the model - Since it is relatively simple, it is less challenging to explain how it works to stakeholders in the business.

Explainability of evaluation - To evaluate Uplift Modeling, metrics such as Qini and AUUC (Area Under the Uplift Curve) are used in research, but these metrics are difficult to explain to the stakeholders. For business, a metric that can estimate how much more profit can be earned is more practical. Since CausalLift adopted the Two-Model approach, the 2 models can be reused to simulate the outcome of following the recommendation by the Uplift Model and can estimate how much conversion rate (the proportion of people who took the desired action such as buying a product) will increase using the uplift model.

<!-- #endregion -->

<!-- #region id="RistOrwOLzIQ" -->
A meta-algorithm uses either a single base learner while having the treatment indicator as a feature (e.g. S-learner), or multiple base learners separately for each of the treatment and control groups (e.g. T-learner, X-learner and R-learner).
<!-- #endregion -->

<!-- #region id="wzZsTIDfICm1" -->
Causal ML is a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research. It provides a standard interface that allows user to estimate the Conditional Average Treatment Effect (CATE) or Individual Treatment Effect (ITE) from experimental or observational data. Essentially, it estimates the causal impact of intervention T on outcome Y for users with observed features X, without strong assumptions on the model form. Typical use cases include

* Campaign targeting optimization: An important lever to increase ROI in an advertising campaign is to target the ad to the set of customers who will have a favorable response in a given KPI such as engagement or sales. CATE identifies these customers by estimating the effect of the KPI from ad exposure at the individual level from A/B experiment or historical observational data.

* Personalized engagement: A company has multiple options to interact with its customers such as different product choices in up-sell or messaging channels for communications. One can use CATE to estimate the heterogeneous treatment effect for each customer and treatment option combination for an optimal personalized recommendation system.



The package currently supports the following methods

    Tree-based algorithms
        Uplift tree/random forests on KL divergence, Euclidean Distance, and Chi-Square
        Uplift tree/random forests on Contextual Treatment Selection
    Meta-learner algorithms
        S-learner
        T-learner
        X-learner
        R-learner

<!-- #endregion -->

<!-- #region id="laBvFe8LU2qD" -->
Table data including the following columns:

    Features
        a.k.a independent variables, explanatory variables, covariates
        e.g. customer gender, age range, etc.
        Note: Categorical variables need to be one-hot coded so propensity can be estimated using logistic regression. pandas.get_dummies can be used.
    Outcome: binary (0 or 1)
        a.k.a dependent variable, target variable, label
        e.g. whether the customer bought a product, clicked a link, etc.
    Treatment: binary (0 or 1)
        a variable you can control and want to optimize for each individual (customer)
        a.k.a intervention
        e.g. whether an advertising campaign was executed, whether a discount was offered, etc.
        Note: if you cannot find a treatment column, you may need to ask stakeholders to get the data, which might take hours to years.
    [Optional] Propensity: continuous between 0 and 1
        propensity (or probability) to be treated for observational datasets (not needed for A/B Testing results)
        If not provided, CausalLift can estimate from the features using logistic regression.

<!-- #endregion -->

```python id="w9iMELFFIROX" colab={"base_uri": "https://localhost:8080/", "height": 957} outputId="d920bf1b-248f-4e35-d9b0-fe2c06a132e3"
!pip install causalml
```

<!-- #region id="su6-oarLInDt" -->
### Average Treatment Effect Estimation with S, T, X, and R Learners
<!-- #endregion -->

<!-- #region id="rLNFPrdAT9P1" -->

<!-- #endregion -->

```python id="3-RjrtQQIjxp" colab={"base_uri": "https://localhost:8080/", "height": 689} outputId="c9b84085-5ed7-440c-d5bd-01743bfff2f8"
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from xgboost import XGBRegressor
from causalml.dataset import synthetic_data

y, X, treatment, _, _, e = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

lr = LRSRegressor()
te, lb, ub = lr.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(X, treatment, y)
print('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

nn = MLPTRegressor(hidden_layer_sizes=(10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
te, lb, ub = nn.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

## for some reason add this e
xl = BaseXRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub = xl.estimate_ate(X, e, treatment, y)
print('Average Treatment Effect (BaseXRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

rl = BaseRRegressor(learner=XGBRegressor(random_state=42))
te, lb, ub =  rl.estimate_ate(X=X, p=e, treatment=treatment, y=y)
print('Average Treatment Effect (BaseRRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))
```

<!-- #region id="GF8y7CrZJyNV" -->
### Calculate Individual Treatment Effect (ITE/CATE)
<!-- #endregion -->

<!-- #region id="wSUgipyoKBX2" -->

<!-- #endregion -->

```python id="izoAPGtjKB6-"
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *
```

<!-- #region id="LBUpjus1N4SZ" -->
**S-learner** estimates the treatment effect using a single machine learning model as follows:

**Stage 1**

Estimate the average outcomes 𝜇(𝑥)
with covariates 𝑋 and an indicator variable for treatment effect 𝑊:

$$
\mu(x)=E[Y | X=x, W=w]
$$

using a machine learning model.


**Stage 2**

Define the CATE estimate as:

$$
\hat{\tau}(x)=\hat{\mu}(x, W=1)-\hat{\mu}(x, W=0)
$$

Including the propensity score in the model can reduce bias from regularization induced confounding [hahn2017bayesian].

When the control and treatment groups are very different in covariates, a single linear model is not sufficient to encode the different relevant dimensions and smoothness of features for the control and treatment groups [alaa2018limits].


<!-- #endregion -->

```python id="RzQK-Y7NIquc" colab={"base_uri": "https://localhost:8080/", "height": 139} outputId="20b87db8-d7cb-470c-a19f-d0f9965ac55d"
# S Learner
# S-learner estimates the treatment effect using a single machine learning model
# It does nothave ITEs
learner_s = LRSRegressor()
cate_s = learner_s.fit_predict(X=X, treatment=treatment, y=y)

```

<!-- #region id="mAjitOtnPuov" -->
**T-learner** [kunzel2019metalearners] consists of two stages as follows:

**Stage 1**

Estimate the average outcomes 𝜇0(𝑥) and 𝜇1(𝑥):


$$
\begin{array}{l}{\mu_{0}(x)=E[Y(0) | X=x] \text { and }} \\ {\mu_{1}(x)=E[Y(1) | X=x]}\end{array}
$$

using machine learning models.


**Stage 2**
Define the CATE estimate as:

$$
\hat{\tau}(x)=\hat{\mu}_{1}(x)-\hat{\mu}_{0}(x)
$$

<!-- #endregion -->

```python id="dOrRzk3rPvvw"

# T Learner
# T-learner [kunzel2019metalearners] consists of two stages as follows:
learner_t = BaseTRegressor(learner=XGBRegressor())
cate_t = learner_t.fit_predict(X=X, treatment=treatment, y=y)
```

<!-- #region id="7rPD8inxQTRR" -->
**X-learner** [kunzel2019metalearners] is an extension of T-learner, and consists of three stages as follows:

**Stage 1**

Estimate the average outcomes 𝜇0(𝑥) and 𝜇1(𝑥):


$$
\begin{array}{l}{\mu_{0}(x)=E[Y(0) | X=x] \text { and }} \\ {\mu_{1}(x)=E[Y(1) | X=x]}\end{array}
$$

using machine learning models.

**Stage 2**

Impute the user level treatment effects, 𝐷1𝑖
and 𝐷0𝑗 for user 𝑖 in the treatment group based on 𝜇0(𝑥), and user 𝑗 in the control groups based on 𝜇1(𝑥):


$$
\begin{aligned} D_{i}^{1} &=Y_{i}^{1}-\hat{\mu}_{0}\left(X_{i}^{1}\right), \text { and } \\ D_{i}^{0} &=\hat{\mu}_{1}\left(X_{i}^{0}\right)-Y^{\wedge} 0i \end{aligned}
$$

Then estimate $ \tau_{0}(x)=E\left[D^{0} | X=x\right]$ and $\tau_{0}(x)=E\left[D^{0} | X=x\right]$ using machine learning models


**Stage 3**
Define the CATE estimate by a weighted average of 𝜏1(𝑥)
and 𝜏0(𝑥):

$$
\tau(x)=g(x) \tau_{0}(x)+(1-g(x)) \tau_{1}(x)
$$

where 𝑔∈[0,1]. We can use propensity scores for 𝑔(𝑥).


<!-- #endregion -->

```python id="ASao9mY8QUEy" colab={"base_uri": "https://localhost:8080/", "height": 208} outputId="414dca93-d97d-426a-9e67-07875037f4af"
# X Learner
# Extension of T learner with ability to take on additional treatments
learner_x = BaseXRegressor(learner=XGBRegressor())
cate_x = learner_x.fit_predict(X=X, p=e, treatment=treatment, y=y)

```

<!-- #region id="yNT0A_qjSQ9e" -->

**R-learner** [nie2017quasi] uses the cross-validation out-of-fold estimates of outcomes 𝑚̂^(−𝑖) times (𝑥𝑖)
and propensity scores 𝑒̂^(−𝑖) times (𝑥𝑖). It consists of two stages as follows:

**Stage 1**
Fit 𝑚̂ (𝑥)
and 𝑒̂ (𝑥) with machine learning models using cross-validation.


**Stage 2**
Estimate treatment effects by minimising the R-loss, 𝐿̂ 𝑛(𝜏(𝑥))
:

$$
\hat{L}_{n}(\tau(x))=\frac{1}{n} \sum_{i=1}^{n}\left(\left(Y_{i}-\hat{m}^{(-i)}\left(X_{i}\right)\right)-\left(W_{i}-\hat{e}^{(-i)}\left(X_{i}\right)\right) \tau\left(X_{i}\right)\right)^{2}
$$

where 𝑒^(−𝑖) times (𝑋𝑖), etc. denote the out-of-fold held-out predictions made without using the 𝑖-th training sample.

<!-- #endregion -->

```python id="fdBVQE2sPttI"

# R Learner uses propensity scores 
learner_r = BaseRRegressor(learner=XGBRegressor())
cate_r = learner_r.fit_predict(X=X, p=e, treatment=treatment, y=y)
```

```python id="-CyCEVU3J7V4" colab={"base_uri": "https://localhost:8080/", "height": 558} outputId="77b1f5ac-cdb2-4968-dbb9-c1f849d6281e"
from matplotlib import pyplot as plt

alpha=0.2
bins=30
plt.figure(figsize=(12,8))
plt.hist(cate_t, alpha=alpha, bins=bins, label='T Learner')
plt.hist(cate_x, alpha=alpha, bins=bins, label='X Learner')
plt.hist(cate_r, alpha=alpha, bins=bins, label='R Learner')
plt.vlines(cate_s[0], 0, plt.axes().get_ylim()[1], label='S Learner',
           linestyles='dotted', colors='green', linewidth=2)
plt.title('Distribution of CATE Predictions by Meta Learner')
plt.xlabel('Individual Treatment Effect (ITE/CATE)')
plt.ylabel('# of Samples')
_=plt.legend()
```

<!-- #region id="q4hGic-eKVZy" -->
### Validity of Meta-Learner

We will validate the meta-learners' performance based on the same synthetic data generation method in Part A (simulate_nuisance_and_easy_treatment).
<!-- #endregion -->

```python id="YT3GwseoKLBR" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="9c0423d1-bcde-45df-af52-655a9fa3336b"


train_summary, validation_summary = get_synthetic_summary_holdout(simulate_nuisance_and_easy_treatment,
                                                                  n=10000,
                                                                  valid_size=0.2,
                                                                  k=10)


train_summary
```

```python id="b2kdt2xUKe0M" colab={"base_uri": "https://localhost:8080/", "height": 332} outputId="da3b9e4e-05cf-4f59-c2f6-51855008df99"
train_summary
```

```python id="D9w4I0XcKqoA" colab={"base_uri": "https://localhost:8080/", "height": 332} outputId="042920bc-75cf-491a-93af-b1d0772e58bb"
validation_summary
```

```python id="FMPm2-15KuIa" colab={"base_uri": "https://localhost:8080/", "height": 558} outputId="8d7daca1-1744-4ada-8158-8689cce06c67"
scatter_plot_summary_holdout(train_summary,
                             validation_summary,
                             k=10,
                             label=['Train', 'Validation'],
                             drop_learners=[],
                             drop_cols=[])
```

```python id="0V9Fq_AlKwXv" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="bb1f3626-b2eb-4789-8e11-f347109e9807"
bar_plot_summary_holdout(train_summary,
                         validation_summary,
                         k=10,
                         drop_learners=['S Learner (LR)'],
                         drop_cols=[])
```

```python id="FJIOQ5aPK5hP" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="91a9ba18-c735-434e-a046-f94b9c5581bc"


# Single simulation
train_preds, valid_preds = get_synthetic_preds_holdout(simulate_nuisance_and_easy_treatment,
                                                       n=50000,
                                                       valid_size=0.2)


```

```python id="mJBlQAWYLLBZ" colab={"base_uri": "https://localhost:8080/", "height": 536} outputId="f707c64c-affa-4c07-e468-4bd140ac5d8c"
#distribution plot for signle simulation of Training
distr_plot_single_sim(train_preds, kind='kde', linewidth=2, bw_method=0.5,
                      drop_learners=['S Learner (LR)',' S Learner (XGB)'])
```

```python id="HOfn8groLOZ6" colab={"base_uri": "https://localhost:8080/", "height": 953} outputId="194ac952-f40e-4a66-ef66-2b924ab2e5b5"
# Scatter Plots for a Single Simulation of Training Data
scatter_plot_single_sim(train_preds)
```

<!-- #region id="8nFjqSJKbHf4" -->
### Propensity Matching and Estimation
<!-- #endregion -->

```python id="jZ5A_pooLUtr"
# Estimation
from causalml.propensity import ElasticNetPropensityModel

pm = ElasticNetPropensityModel(n_fold=5, random_state=42)
ps = pm.fit_predict(X, y)

```

```python id="G16_HPXObOQp"
# Matching

from causalml.match import NearestNeigoborMatch, create_table_one

psm = NearestNeighborMatch(replace=False,
                           ratio=1,
                           random_state=42)
matched = psm.match_by_group(data=df,
                             treatment_col=treatment_col,
                             score_col=score_col,
                             groupby_col=groupby_col)

create_table_one(data=matched,
                 treatment_col=treatment_col,
                 features=covariates)

```

<!-- #region id="mL8Z1htzcJy_" -->
#### Confidence Intervals
<!-- #endregion -->

```python id="E0Q4EkWNbTtJ" colab={"base_uri": "https://localhost:8080/", "height": 208} outputId="56dba097-6e1e-4472-ae6f-1b2c11f10a94"
# Generate synthetic data using mode 1
y, X, treatment, tau, b, e = synthetic_data(mode=1, n=10000, p=8, sigma=1.0)

treatment = np.array(['treatment_a' if val==1 else 'control' for val in treatment])

# Normal
alpha = 0.05
learner_s = BaseSRegressor(XGBRegressor(), ate_alpha=alpha, control_name='control')
ate_s, ate_s_lb, ate_s_ub = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True)
                                                   
print(np.vstack((ate_s_lb, ate_s, ate_s_ub)))

ate_s_b, ate_s_lb_b, ate_s_ub_b = learner_s.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True,
                                                         bootstrap_ci=True, n_bootstraps=100, bootstrap_size=5000)

print(np.vstack((ate_s_lb_b, ate_s_b, ate_s_ub_b)))
```

```python id="kYmHkQJfcg7a" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="1791582a-19ce-4494-abb0-3ce565307e2f"
## CATE CIs

alpha = 0.05
learner_s = BaseSRegressor(XGBRegressor(), ate_alpha=alpha, control_name='control')
cate_s, cate_s_lb, cate_s_ub = learner_s.fit_predict(X=X, treatment=treatment, y=y, return_ci=True,
                               n_bootstraps=100, bootstrap_size=5000)

cate_s

```

```python id="YgBURvLhdC6Q"
#### For work on other CIs and multiple treatments

https://github.com/uber/causalml/blob/master/examples/meta_learners_with_synthetic_data_multiple_treatment.ipynb


```

<!-- #region id="593TzsMzVquj" -->
## A/B Testing
<!-- #endregion -->

<!-- #region id="03RnmLW7mvzn" -->
Can be divided into frequentist and bayesian A/B testing. Functionally quite similar except for bayesian techniques providing us with more information around point estimates. If we stick with linear models, we could frame the problem in a bayesian way, but it feels like this approach is very similar to covariate adjustment. 
<!-- #endregion -->

<!-- #region id="ttOYp4OPnBxV" -->
For both, we would want to define the power we want and hence the required sample size. https://www.firmai.org/documents/Power%20analysis%20for%20AB%20tests/#ab-testing
<!-- #endregion -->

<!-- #region id="5x9oHc1g6GaZ" -->
### Calculate The Necessary Sample Size
<!-- #endregion -->

<!-- #region id="RLe-idoL68CS" -->
Before running an A/B test to compare a new website design (labeled the B design) to the existing design (labeled A), it is a good idea to determine how many users will be needed to evaluate if the new design performs better than the old one. The t-test is an effective statistical tool to evaulate significance once the experiment is over, and there are many online tutorials explaining how to use it. I didn’t find a comparable resource explaining the calculation of sample sizes, so I put together this notebook to demonstrate the (simple) steps.

Calculating necessary sample sizes given

* null hypothesis
* expected effect size
* false positive rate
* false negative rate. 

<!-- #endregion -->

<!-- #region id="-kbJ9TXVCJNW" -->
Now, I’ll enter some numbers to make the discussion more concrete. Imagine we have a click through rate of 5% with the original design. Call this p_a for probability(A). Suppose in addition that we decide that the click through rate must increase to at least 7% to make changing the design worthwhile. Call this p_b. Finally, we’ll calculate the average click through rate, p, assuming that our sample sizes will be equal.
<!-- #endregion -->

```python id="583oqMa16B8Q"
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import scipy.stats


p_a = .05 # assume we have a base click rate of 5% for our original design (A group)
p_b = .07 # we want to detect an increase in click rate to 7%, otherwise not worth changing the design

p = (p_a + p_b)/2.


```

<!-- #region id="TD-mtjw2CSSm" -->
n addition to these two values, we’ll need to decide on false positive and false negative rates. We can use these to look up values from the Normal distribution (results are labeled Z below). Here we chose 5% false positive rate (also called Type I error rate) and 80% power, equivalent to a 20% false negative rate (or Type II error rate). These rates are fairly standard, but completely arbitrary. These choices mean that we expect to falsely say that B is an improvement 5% of the time when actually it is no better than A, and we expect to falsely say B is not and improvement 20% of the time when actually it is better than A.
<!-- #endregion -->

```python id="dgdHFCiq6B38" colab={"base_uri": "https://localhost:8080/", "height": 54} outputId="b90eb9d6-fd78-4ef5-c484-fdc2079a42c4"
Z8 = scipy.stats.norm.ppf(.8) # we will need this to ensure 80% power (20% false negative rate)
Z95 = scipy.stats.norm.ppf(1 - .05) # we will need this for 5% false positive rate (95% confidence level), one-tailed
Z975 = scipy.stats.norm.ppf(1 - .025) # 5% false positive rate for two-tailed case

ES = abs(p_b - p_a)/np.sqrt(p*(1-p))

num_tails = 1 # presumably we are testing design b because we think it will improve the click rate...

if num_tails == 2:
    n = 2*((Z975 + Z8)/ES)**2  # two-tailed
else:
    n = 2*((Z95 + Z8)/ES)**2 # one-tailed

print('You need', round(n), ' samples in each group to get a 5% false positive and 20% false negative rate given effect size')

```

<!-- #region id="q8OTO5HtCYUe" -->
That’s it! We have the sample sizes necessary given our requirements. In this case, we need about 1743 people to experience the A design and 1743 people to experience the B design.

Let’s convince ourselves that we actually meet our specs by simulating two experimental results. In one experiment the B design results in a minimal improvement (to 7% click rate). In the other (labeled null) there is no change in the click rate.
<!-- #endregion -->

```python id="AQj4HIW26B1W" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="426ff34c-ce89-42a6-b978-fdff38922531"
n_a = int(round(n))
n_b = int(round(n))

num_experiments = 10000

conversions_a = np.random.random((n_a, num_experiments)) < p_a
conversions_b_null = np.random.random((n_b, num_experiments)) < p_a
conversions_b = np.random.random((n_b, num_experiments)) < p_b

mean_a = np.mean(conversions_a, axis=0)
mean_b_null = np.mean(conversions_b_null, axis=0)
mean_b = np.mean(conversions_b, axis=0)

#s_a = np.std(conversions_a, ddof=1)
#s_b_null = np.std(conversions_b_null, ddof=1)
#s_b = np.std(conversions_b, ddof=1)
# equivalent:
s_a = np.sqrt(np.sum((conversions_a - mean_a[np.newaxis, :])**2, axis=0)/(n_a - 1))
s_b_null = np.sqrt(np.sum((conversions_b_null - mean_b_null[np.newaxis, :])**2, axis=0)/(n_b - 1))
s_b = np.sqrt(np.sum((conversions_b - mean_b[np.newaxis, :])**2, axis=0)/(n_b - 1))

sp = np.sqrt(s_a**2/n_a + s_b**2/n_b)
sp_null = np.sqrt(s_a**2/n_a + s_b_null**2/n_b)

if num_tails == 2:
    t = abs(mean_b - mean_a) / sp # two-tailed
    t_null = abs(mean_b_null - mean_a) / sp_null # two-tailed
    results = t > Z975  # two-tailed
    results_null = t_null > Z975  # two-tailed
else:
    t = (mean_b - mean_a) / sp # one-tailed
    t_null = (mean_b_null - mean_a) / sp_null # one-tailed
    results = t > Z95 # one-tailed
    results_null = t_null > Z95 # one-tailed

false_negative_rate = 1 - np.sum(results).astype('float')/len(results)
false_positive_rate = np.sum(results_null).astype('float')/len(results_null)

print(false_negative_rate, "false negative rate, we expect it to be close to 20%")
print(false_positive_rate, "false positive rate, we expect it to be close to 5%")

```

```python id="THHD73yc6ByZ" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="442f27c0-577a-4212-86d9-bc2c4c84ffba"
fig = plt.figure()
ax = fig.add_subplot(111)

n, bins, p = ax.hist(mean_b - mean_a, np.linspace(-.04, .06, 88), color=[.8, .8, 1])
n, bins, p = ax.hist(mean_b_null - mean_a, bins, color=[1, .8, .8])

n, bins, p = ax.hist(mean_b[results==False] - mean_a[results==False], bins, color='b', alpha=.6)
n, bins, p = ax.hist(mean_b_null[results_null] - mean_a[results_null], bins, color='r', alpha=.6)

ax.text(-.02, 600, 'Null true', color='r')
ax.text(.03, 500, 'Minimum true effect size', color='b')

ax.text(.016, 300, str(round(false_negative_rate*100))+"% false negatives", color='b')
ax.text(.016, 100, str(round(false_positive_rate*100))+"% false positives", color='r')

```

<!-- #region id="Mg8qkBgmCf5x" -->
We can see that we achieve exactly the false positive and false negative rates we set out for in the two different simuluated experiments.
<!-- #endregion -->

<!-- #region id="2w8-8ZMPCzT0" -->
### Frequentist

This project looks at an A/B test run by an e-commerce website. The goal is to see if the company should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
<!-- #endregion -->

```python id="fARCReex6Bsn" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="07b149bc-3fc9-4bff-8f20-6653ca234cb9"
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
np.random.seed(42)

# import data
df = pd.read_csv('https://raw.githubusercontent.com/firmai/random-assets/master/ab_data.csv')

# show top rows
df.head()


```

<!-- #region id="etdsntWCH9Qt" -->
Mismatch Remove
<!-- #endregion -->

```python id="J0KkAneI6Bpa"

# Looking for rows where treatment/control doesn't line up with old/new pages respectively
df_t_not_n = df[(df['group'] == 'treatment') & (df['landing_page'] == 'old_page')]
df_not_t_n = df[(df['group'] == 'control') & (df['landing_page'] == 'new_page')]

# Add lengths
mismatch= len(df_t_not_n) + len(df_not_t_n)

# Create one dataframe from it
mismatch_df = pd.concat([df_t_not_n, df_not_t_n])

# Remove incriminating rows
mismatch_index = mismatch_df.index
df = df.drop(mismatch_index)


```

<!-- #region id="h5zxOhj0Jg5x" -->
Remove non-unique users 
<!-- #endregion -->

```python id="pmFtXngSIhSS" colab={"base_uri": "https://localhost:8080/", "height": 69} outputId="99ce18b6-8024-437c-e1ed-096b0a80df5b"
# Calculate number of rows in dataset and display
df_length = len(df)         
print(df_length)

# Find unique users
print("Unique users:", len(df.user_id.unique()))

# Check for not unique users
print("Non-unique users:", len(df)-len(df.user_id.unique()))

# Drop duplicated user
df = df.drop_duplicates(keep='first', subset=["user_id"])


```

<!-- #region id="blGGr-I1JrBk" -->
Probabilities
<!-- #endregion -->

```python id="N2ZVPx5P6Bmw" colab={"base_uri": "https://localhost:8080/", "height": 69} outputId="a2122ecc-d265-4cc4-a55b-75631686ac1d"
# What is the probability of an individual converting regardless of the page they receive
# Probability of user converting
print("Probability of user converting:", df.converted.mean())

#Given that an individual was in the control group, what is the probability they converted?
# Probability of control group converting
print("Probability of control group converting:", 
      df[df['group']=='control']['converted'].mean())

# Given that an individual was in the treatment group, what is the probability they converted?

# Probability of treatment group converting
print("Probability of treatment group converting:", 
      df[df['group']=='treatment']['converted'].mean())


```

<!-- #region id="3U_2AGHZJ6yU" -->
For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

* Null-hypothesis

$H_0:  p_{new} - p_{old} \leq 0$ 

*i.e.* The null hypothesis is that the difference between the population conversion rate of users given the new page and the old page will be equal to zero (the same) or lower than zero (the old page has a higher population conversion rate).

* Alternative-hypothesis

$H_1: p_{new} - p_{old} > 0$

*i.e.* The alternative hypothesis is that the difference between the population conversion rate of users given the new page and the old page will be greater than zero to zero (the new page has a higher population conversion rate).

Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the converted success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the converted rate in ab_data.csv regardless of the page.

Use a sample size for each page equal to the ones in ab_data.csv.

Perform the sampling distribution for the difference in converted between the two pages over 10,000 iterations of calculating an estimate from the null.

![alt text](https://)`a.` What is the **convert rate** for $p_{new}$ under the null? 

Given the assumption in the question, $p_{new} = p_{old}$. Hence, we should calculate the average of the real $p_{new}$ and $p_{old}$ (probability of conversion given new page and old page respectively) to calculate $p_{mean}$.

<!-- #endregion -->

```python id="-uq2hUXg6BkL" colab={"base_uri": "https://localhost:8080/", "height": 89} outputId="d4a7dbc2-0846-43b2-92e6-e380acd0cada"
## Same as before


# Calculate probability of conversion for new page
p_new = df[df['landing_page']=='new_page']['converted'].mean()

print("Probability of conversion for new page (p_new):", p_new)

# Calculate probability of conversion for old page
p_old = df[df['landing_page']=='old_page']['converted'].mean()

print("Probability of conversion for old page (p_old):", p_old)

# Calc. differences in probability of conversion for new and old page (not under H_0)
p_diff = p_new-p_old

print("Difference in probability of conversion for new and old page (not under H_0):", p_diff)
```

<!-- #region id="q7O3dt6gLvSW" -->
Hence:

$p_{new}: 0.1188$

$p_{old}: 0.1204$

The **convert rate** for $p_{new}$ under the null 

$p_{mean}=p_{old_0}=p_{new_0}: 0.1196$

The **convert rate** for $p_{old}$ under the null? 

As above $p_{new_0} - p_{old_0}= 0$

<!-- #endregion -->

```python id="1aRouuyM6Bh7" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="9d8620b6-f12c-444e-a301-d618c18f4401"
# Calculate n_new and n_old
n_new, n_old = df['landing_page'].value_counts()

print("new:", n_new, "\nold:", n_old)
```

<!-- #region id="Tq7QCHDrMgFW" -->
Hence:

Number of tests with the new page

$n_{new}: 145310$

Number of tests with the old page

$n_{old}: 145274$
<!-- #endregion -->

<!-- #region id="XYATnZ8BNSv5" -->
Now we will simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null. Store these $n_{new}$ 1's and 0's in new_page_converted.
<!-- #endregion -->

```python id="aedTDnlB6Bft" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="25fccbf6-0b20-4358-9719-0bc138997d1f"
p_mean = df["converted"].mean()

[p_mean, (1-p_mean)]
```

```python id="GH_PpoSz6BcV" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="8b6f0853-7872-4967-d8a5-3fd536bb4404"
# Simulate conversion rates under null hypothesis
# [0.11959718500778342, 0.8804028149922166] % choose [1, 0]
new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_mean, (1-p_mean)])

new_page_converted.mean()
```

<!-- #region id="33zJKv_ENwR-" -->
Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.
<!-- #endregion -->

```python id="BaXjl6Yc6BXt" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="784712b0-a24a-4965-8000-34e36d2c8222"
# Simulate conversion rates under null hypothesis
old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_mean, (1-p_mean)])

old_page_converted.mean()
```

<!-- #region id="O-ngsil9OvSX" -->
Find $p_{new}$ - $p_{old}$ for your simulated values
<!-- #endregion -->

```python id="dMAD_9oR6BVC" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="d018d1eb-1c90-41aa-9dff-cb3d80d22756"
# Calculate difference in p under the null hypothesis
new_page_converted.mean()-old_page_converted.mean()
```

<!-- #region id="CIxi4TqQPDp_" -->
Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process 
<!-- #endregion -->

```python id="9dw2qI_J6BS7" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="d5e1d90a-b135-4fdc-f958-f8f60d300d0f"
## Good, I like these simulations. 
p_diffs = []

# Re-run simulation 10,000 times
# trange creates an estimate for how long this program will take to run
for i in trange(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_mean, (1-p_mean)])
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_mean, (1-p_mean)])
    p_diff = new_page_converted.mean()-old_page_converted.mean()
    p_diffs.append(p_diff)
```

```python id="W1HYVdUX6BP6" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="73cbe3ba-fb59-4651-8563-9a561a48a163"
# Plot histogram
plt.hist(p_diffs, bins=25)
plt.title('Simulated Difference of New Page and Old Page Converted Under the Null')
plt.xlabel('Page difference')
plt.ylabel('Frequency')
plt.axvline(x=(p_new-p_old), color='r', linestyle='dashed', linewidth=1, label="Real difference")
plt.axvline(x=(np.array(p_diffs).mean()), color='g', linestyle='dashed', linewidth=1, label="Simulated difference")
plt.legend()
plt.show()

```

<!-- #region id="1iPFlqgdRXXp" -->
The simulated data creates a normal distribution (no skew) as expected due to how the data was generated. The mean of this normal distribution is 0, which which is what the data should look like under the null hypothesis.
<!-- #endregion -->

<!-- #region id="AtIDDOQDRcPl" -->
The proportion of the p_diffs are greater than the actual difference observed
<!-- #endregion -->

```python id="YKZTiDeI6BKl" colab={"base_uri": "https://localhost:8080/", "height": 69} outputId="5054a4a2-98c6-47e8-f61c-460390b3183a"
p_diff = p_new - p_old

# Find proportion of p_diffs greater than the actual difference
greater_than_diff = [i for i in p_diffs if i > p_diff]


# Calculate values
print("Actual difference:" , p_diff)

p_greater_than_diff = len(greater_than_diff)/len(p_diffs)

print('Proportion greater than actual difference:', p_greater_than_diff)

print('As a percentage: {}%'.format(p_greater_than_diff*100))

```

<!-- #region id="kazsZWIdR-m2" -->
If our sample conformed to the null hypothesis then we’d expect the proportion greater than the actual difference to be 0.5. However, we calculate that almost 90% of the population in our simulated sample lies above the real difference which does not only suggest that the new page does not do significantly better than the old page, it might even be worse!

<!-- #endregion -->

<!-- #region id="axVIMDo4SGMU" -->
We could also use a built-in to achieve similar results. Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance.
<!-- #endregion -->

```python id="vQwb-pcr6BF_" colab={"base_uri": "https://localhost:8080/", "height": 121} outputId="b96436b8-0800-4a26-b30f-6b4b5a46519c"
# Import statsmodels
import statsmodels.api as sm

# Calculate number of conversions
# Some of these values were defined ealier in this notebook: n_old and n_new

convert_old = len(df[(df['landing_page']=='old_page')&(df['converted']==1)])
convert_new = len(df[(df['landing_page']=='new_page')&(df['converted']==1)])

print("convert_old:", convert_old, 
      "\nconvert_new:", convert_new,
      "\nn_old:", n_old,
      "\nn_new:", n_new)


# Find z-score and p-value
z_score, p_value = sm.stats.proportions_ztest(count=[convert_new, convert_old], 
                                              nobs=[n_new, n_old], alternative = 'larger')

print("z-score:", z_score,
     "\np-value:", p_value)

```

```python id="KH12NHMegsGt" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="bcd90218-a892-4426-dc6c-bbe05d78a9b6"
## Similar to above, but what is happening behind the scenes. 

# null hypothesis: the samples have identical averages
# if p < 0.05, reject null hypothesis that the two samples are identical
import scipy.stats as stats

# ztest
def ztest(x1, n1, x2, n2, one_tailed=False):
    """
    One- or Two-Tailed Z-test for two samples
    
    Args:
        x1 = # of successes in Sample 1
        n1 = # of observations in Sample 1
        x2 = # of successes in Sample 2
        n2 = # of observations in Sample 2
        one_tailed = Boolean, whether or not the test should be One-Tailed
        
    Return:
        z = Z-stat
    """
    p1 = float(x1) / n1
    p2 = float(x2) / n2    
 
    p = float(x1 + x2) / (n1 + n2)
    se = p * (1. - p) * ((1. / n1) + (1. / n2))
    sse = np.sqrt(se)
    
    z = float(p1 - p2) / sse
    p = 1 - stats.norm.cdf(abs(z))
    
    if not one_tailed:
        p *= 2.
    print(z, p)
    return z, p

# Do a test with fake data:

control_observations = n_old #n1
control_successes = convert_old # x1
test_observations = n_new #n2
test_successes = convert_new #x2


## left is z-stat and right is p-value
z_stat, p_value = ztest(control_successes, control_observations, test_successes, test_observations, one_tailed=True)

def compute_standard_error_prop_two_samples(x1, n1, x2, n2, alpha=0.05):
    p1 = x1/n1
    p2 = x2/n2    
    se = p1*(1-p1)/n1 + p2*(1-p2)/n2
    return np.sqrt(se)
    
def zconf_interval_two_samples(x1, n1, x2, n2, alpha=0.05):
    p1 = x1/n1
    p2 = x2/n2    
    se = compute_standard_error_prop_two_samples(x1, n1, x2, n2)
    z_critical = stats.norm.ppf(1-0.5*alpha)
    return p2-p1-z_critical*se, p2-p1+z_critical*se

ci_low,ci_upp = zconf_interval_two_samples(control_successes, control_observations, test_successes, test_observations)
print(' 95% Confidence Interval = ( {0:.2f}% , {1:.2f}% )'
      .format(100*ci_low, 100*ci_upp))
```

<!-- #region id="Pp4Q1-9WSnXV" -->
Simply put, a z-score is the number of standard deviations from the mean a data point is. But more technically it’s a measure of how many standard deviations below or above the population mean a raw score is. Given the above definition, it would seem that the differences between the lines shown in the histogram above is -1.31 standard deviations. The p-value is roughly 10.0% which is the probability that this result is due to random chance, this is not enough evidence to reject the null hypothesis and thus we fail to do so. The p-value that we got from ab_page is 0.190 that is significant difference with p-value in A/B Testing, which is around 0.9. The reason is because there are two totally different Null Hypothesis. One inequality direction, is naturally more certain. 
<!-- #endregion -->

<!-- #region id="hejyQQ-CS93j" -->
### Regression Approach

The regression approach allows us to also add an intercept which can account for bias. It is also somewhat easier. We will use a logistic regression.
<!-- #endregion -->

<!-- #region id="7fp_ZFUbXT3s" -->
The goal is to use statsmodels to fit the regression model to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received. Add an intercept column, as well as an ab_page column, which is 1 when an individual receives the treatment and 0 if control.
<!-- #endregion -->

```python id="gaVCv0Du6BDc" colab={"base_uri": "https://localhost:8080/", "height": 398} outputId="ba5cdccd-9a59-4a6e-a153-2bd53462418c"
df3 = df # Clone dataframe in case of a mistake

df3['intercept'] = pd.Series(np.zeros(len(df3)), index=df3.index)
df3['ab_page'] = pd.Series(np.zeros(len(df3)), index=df3.index)

# Find indexes that need to be changed for treatment group
index_to_change = df3[df3['group']=='treatment'].index

# Change values
df3.set_value(index=index_to_change, col='ab_page', value=1)
df3.set_value(index=df3.index, col='intercept', value=1)

# Change datatype
df3[['intercept', 'ab_page']] = df3[['intercept', 'ab_page']].astype(int)

# Move "converted" to RHS
df3 = df3[['user_id', 'timestamp', 'group', 'landing_page', 'ab_page', 'intercept', 'converted']]

# Set up logistic regression
logit = sm.Logit(df3['converted'], df3[['ab_page', 'intercept']])

# Calculate results
result=logit.fit()

result.summary2() # result.summary() wasn't working for some reason, but this one does
```

<!-- #region id="brChlAMmXvq7" -->
Apparently the p-value associated with ab_page is 0.1899, which is slightly lower than the p-value I calculated using the z-test above. The reason why the value is lower is because I added an intercept which is meant to account for bias. This means that this value is more accurate. (As in, it’s probably closer to the true p-value)
<!-- #endregion -->

<!-- #region id="EyZy_IGoXveS" -->
Although it would seem from the outset that there is a difference between the conversion rates of new and old pages, there is just not enough evidence to reject the null hypothesis. From the histogram shown in this report, it seems that the new page does worse than the old page.
<!-- #endregion -->

<!-- #region id="wcDSWL7Oa_Yq" -->
** There is a benefit to adding additional data and that is the ability to estimate the different effects for people with different associated characteristics. **
<!-- #endregion -->

<!-- #region id="vwqReV-Gb2_P" -->
We now have an estimate of the effect size, and our uncertainty of it. Often however, we need to make a decision with this information. How exactly we make this choice should depend on the cost/benefits of the decision, but it is sometimes enough just to ask whether or not our estimated value of Δ is “significantly” different from zero. This is usually done by using the language of hypothesis testing.
<!-- #endregion -->

<!-- #region id="IIkM-yFjeFDE" -->
### Increasing the Power

You can increase the power using correlated covariates. You can use this methodology in the next section.

https://www.firmai.org/documents/variance-reduction/#generate-dataset
<!-- #endregion -->

<!-- #region id="NSBBCs8QfUIL" -->
## Bayesian
<!-- #endregion -->

```python id="bgik7cgrb2FD"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

sns.set_style('whitegrid')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

<!-- #region id="sMjp8Am3fonW" -->
The t-test, z-tests and confidence intervals are classic Frequentist test for a significant difference in means between groups. 


<!-- #endregion -->

<!-- #region id="m5c2g_pXkicg" -->
One common application of Bayesian analysis in industry is the analysis of split tests. We can use pymc3 to perform split test analysis or do the process manually by sampling from posterior distributions for the arm conversion rates.
<!-- #endregion -->

<!-- #region id="yzMC2dZLksE6" -->
The dataset below contains information on user “conversions” on a fitness app on different “arms” of a split test.

A “conversion” is jargon for whether or not a user performed a desired action or not, typically a purchase.

“Arms” are the jargon for the different versions of a product in a currently running split test. Split tests are also commonly referred to as A/B tests, where A and B denote arms in the test.

The data has 6 columns:

* arm: the version of the app this user was randomly assigned to
* gender: male/female
* age: age bins, one of 20-30, 30-40, 40-50
* day: the day (total of 21 days)
* fitness: the user's self reported fitness level from -5 to 5
* converted: 1 if the user purchased the product, 0 if not

<!-- #endregion -->

```python id="W_UMxMmZb16z"
### Each row is a unique user

data = pd.read_csv('https://raw.githubusercontent.com/firmai/random-assets/master/split_test_data.csv')
data['male'] = data.gender.map(lambda x: 1 if x == 'male' else 0)

```

<!-- #region id="N2bks0ANvKBD" -->
When a new arm is introduced into a split test, it is generally tested at a low percentage of users initially before assignment becomes balanced between the arms. This ensures that if something is terribly wrong with one of the arms it does not ruin the experience for too many potential customers. So the question is does the new arm lead the user to purchase the product.
<!-- #endregion -->

```python id="Uo6ZxXG3b14Z" colab={"base_uri": "https://localhost:8080/", "height": 104} outputId="06900661-bfb1-4f82-f560-f52a513679df"
data.groupby('arm')['converted'].agg(np.mean)

```

```python id="JkWf5DFLb12B" colab={"base_uri": "https://localhost:8080/", "height": 104} outputId="9d5316c2-b1a3-4acd-c11a-b3612bd941f6"
## You can look at overall conversion rate differences along age, gender, and fitness.

data.groupby('age')['converted'].agg(np.mean)

```

```python id="9CTOSCTQb10J" colab={"base_uri": "https://localhost:8080/", "height": 243} outputId="0ea0cd86-d115-4aa8-b783-f33670265dc0"
## Interesting relationship, the fitter you are,the more liklely you were
## to convert.
data.groupby('fitness')['converted'].agg(np.mean)

```

```python id="FNzgTq4_b1xs" colab={"base_uri": "https://localhost:8080/", "height": 86} outputId="a5e5c6a6-ab21-4a3c-8c66-e9f6c670abdd"
data.arm.value_counts()

```

<!-- #region id="BWSI_v_6xGPU" -->
Subset the data to the first 5 days. We will start by just modeling the conversion rate distributions for arms A and B through day 5. At day 5, arm C has still not been introduced yet and so there are just 2 arms.
<!-- #endregion -->

```python id="CUsmNxdUb1sU"
# import pymc3 as pm

# current = data[data.day < 5]
# print(current.shape, current.arm.unique())
# print(current.groupby('arm')['converted'].agg(np.sum))


```

<!-- #region id="XVwBDPEoxd5y" -->
Set up a pymc3 model and uniform priors for the probabilities of conversion for arms A and B. Recall that pymc3 uses the with ... syntax for defining models. The first step in setting up a new model is to define the model as the “context”. We are going to model the probability distributions for conversion rates for arms A and B. As always with Bayesian statistics, we need to define prior distributions for our belief about these probabilities/rates of conversion per arm.

Let’s say we have no belief whatsoever about rates, and so we will set an uninformative, flat priors over probabilities from 0 to 1 for both arms. This is equivalent to saying that we believe all conversion rates to be equally likely for both arms.
<!-- #endregion -->

```python id="uZaLoXVKb1pn"
with pm.Model() as day5_model:
    
    arm_A_prior = pm.Uniform('A_prior', lower=0, upper=1)
    A_p = pm.Uniform('A_prob', lower=0, upper=1)
    B_p = pm.Uniform('B_prob', lower=0, upper=1)


```

<!-- #region id="vOe6yYiNyEWP" -->
Set up pm.Bernoulli distributions to model conversions for arms A and B. We are now going to set up the “likelihood” portion of the model. This is going to model the $P(data\; 	\;\theta)$ part of Bayes theorem. Our conversions are represented by a vector of 1s and 0s denoting whether or not the user converted or not. This is known as a “Bernoulli” process and pymc3 has an approprite function to handle it:
<!-- #endregion -->

<!-- #region id="jE7XZbD4ycxV" -->
p = is set to the prior for the arm that you defined in the last section.

observed = should be set to the converted values for that arm specifically in the data.

By giving it an observed parameter, we are telling pymc3 that we want this to evaluate the likelihood of our data (the conversions) against models represented by the p= probability argument. We assign p= to be our prior belief about conversion rates for that arm because we want to update this belief (convert to posterior) based on the conversion data we have observed for that arm.
<!-- #endregion -->

```python id="vIK9hA4r6BAH"
df3 = data[data["arm"].isin(["B","A"])].reset_index(drop=True)


with day5_model:
    
    A = pm.Bernoulli('A', p=A_p, observed=df3[df3.arm == "A"].converted.values)
    B = pm.Bernoulli('B', p=B_p, observed=df3[df3.arm == "B"].converted.values) 
    
    AvB = pm.Deterministic('AvB', A_p - B_p)  ## this is the uplift
```

<!-- #region id="rwhDFEsKywBO" -->
#### Fit the model

Now that we’ve set up the prior distributions and likelihoods, we can actually fit the model.

Below is code that will run the sampling procedure to find the posteriors:
<!-- #endregion -->

```python id="352A1cPHyv1q" colab={"base_uri": "https://localhost:8080/", "height": 121} outputId="f39b59dc-793e-48be-d27c-e6a8dd2f7e96"
with day5_model:

    # construct the "trace" variable that holds samples for all of our distributions:
    trace = pm.sample(50000)
```

<!-- #region id="jt3Fs52jy6pA" -->
Again you use the context with day5_model: to run code for your model.

start = pm.find_MAP() will try to find a good starting point for the sampling process. This means that your model will converge on the “likely” area much faster (though it makes the fitting slower initially).

trace = pm.sample(50000, start=start) uses the sampling method in pymc3 to perform 50,000 sampling iterations. This will automatically assign the NUTS sampler for you. The dataset is small so the speed shouldn’t be too bad.

When this completes, the trace variable now contains the posterior samples for the distributions we specified while constructing the model.
<!-- #endregion -->

```python id="39AA1BaWyvSj"
# We defined our arm A prior distribution to be uniform and named it 'arm_A_prior'. 
# The pm.sample() procedure converted this into our posterior belief for the rate
# of conversions in arm A. You can access these posterior samples using the name
# you gave the variable when you created it:
#
# this will be a vector of values that are different potential rates of conversion
# for arm A. A histogram of these rates defines, roughly, the posterior probability
# distribution for the arm A rates after we consider the data we have collected.
```

```python id="XsyJ-ezb3Ody" colab={"base_uri": "https://localhost:8080/", "height": 298} outputId="a79dda95-7721-4c81-a400-5db65d061cef"
!pip install arviz
```

```python id="n3saxYPJyvQY" colab={"base_uri": "https://localhost:8080/", "height": 480} outputId="0fa7878c-dc07-42d2-f69e-677e1edd802e"
pm.plot_posterior(trace[5000::3], varnames=['A_prob','B_prob','AvB'],
                  ref_val=0, color='#87ceeb')

```

```python id="sCVdLogNyvOM" colab={"base_uri": "https://localhost:8080/", "height": 121} outputId="57126e0c-72eb-47d1-89f1-60592e0c1253"
# Import statsmodels
import statsmodels.api as sm

# Calculate number of conversions
# Some of these values were defined ealier in this notebook: n_old and n_new

df2 = data[data["arm"].isin(["B","A"])].reset_index(drop=True)

n_A = len(df2[(df2['arm']=='A')])
n_B = len(df2[(df2['arm']=='B')])

convert_A = len(df2[(df2['arm']=='A')&(df2['converted']==1)])
convert_B = len(df2[(df2['arm']=='B')&(df2['converted']==1)])

print("convert_A:", convert_A, 
      "\nconvert_B:", convert_B,
      "\nn_A:",n_A ,
      "\nn_B:",n_B  )

## According to this analysis new clearly performs worse.
## There is some other testing techniques we will test.

# Find z-score and p-value
z_score, p_value = sm.stats.proportions_ztest(count=[convert_B, convert_A], 
                                              nobs=[n_B, n_A])
print("z-score:", z_score,
     "\np-value:", p_value)


```

```python id="aUMqEpmp55O4" colab={"base_uri": "https://localhost:8080/", "height": 106} outputId="290cefe3-2bc0-41c6-d707-2a0251fddfed"
!pip install brewer2mpl
```

```python id="BM2n5VjWyvLI" colab={"base_uri": "https://localhost:8080/", "height": 280} outputId="f2bdccb9-85b3-4470-c67c-ee9ee5807d80"
import brewer2mpl
brewer_set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors

A = df2[df2["arm"]=="A"]["converted"]
B = df2[df2["arm"]=="B"]["converted"]

np.random.seed(1)
Bs = np.array(B)

np.random.shuffle(Bs )
fig, ax = plt.subplots(figsize=(8,4))

Bs.shape[0]/5


Bs = Bs.reshape((5,48))

plt.bar(range(1,6), [Bs[i].mean() for i in range(5)],
    color=brewer_set2[0], align='center', width=.8, label='New Design' )
plt.plot([-2, 6], [A.mean(), A.mean()], 
     color=brewer_set2[1], label='Old Design')
plt.xlim(0.5, 5.5)
plt.title('Split 5 test')
plt.legend();

```

```python id="SZ2cpOHk9h24" colab={"base_uri": "https://localhost:8080/", "height": 156} outputId="ef25256b-ce4b-41b1-e263-315ae00a937a"
scipy.stats.norm.pdf(x,mean,z_score)
```

<!-- #region id="6w7nGeqf-4j0" -->
The Z-score, or standard score, is the number of standard deviations a given data point lies above or below mean.
<!-- #endregion -->

```python id="zMLguiSzAJei" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="ad934029-a1fe-405e-80d6-e9aa1ff1d8e8"
# implementation from scratch
def ztest_proportion_two_samples(x1, n1, x2, n2, one_sided=False):
    p1 = x1/n1
    p2 = x2/n2    

    p = (x1+x2)/(n1+n2)
    se = p*(1-p)*(1/n1+1/n2)
    se = np.sqrt(se)
    
    z = (p1-p2)/se
    p = 1-stats.norm.cdf(abs(z))
    p *= 2-one_sided # if not one_sided: p *= 2
    return z, p

z,p = ztest_proportion_two_samples(convert_A, n_A, convert_B, n_B, one_sided=False)
print(' z-stat = {z} \n p-value = {p}'.format(z=z,p=p))
```

```python id="GqGdOfjU-4O6" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="5748e1bc-6562-4606-92ff-e95c1f01ed1d"
def compute_standard_error_prop_two_samples(x1, n1, x2, n2, alpha=0.05):
    p1 = x1/n1
    p2 = x2/n2    
    se = p1*(1-p1)/n1 + p2*(1-p2)/n2
    return np.sqrt(se)

compute_standard_error_prop_two_samples(convert_A, n_A, convert_B, n_B)
```

```python id="_mZi-ETgBFz2" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="b22b9c7d-941a-4bb4-828e-1ace7a1799ff"
mean
```

```python id="dklpRprMBM_K" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="108c5c56-5ed7-42e3-e8bb-d3de9957d1a1"
abs(z_score)
```

```python id="gDkd8nvMA68s" colab={"base_uri": "https://localhost:8080/", "height": 364} outputId="e04741d0-bbd0-4802-a36c-5fd7b0dc816d"
scipy.stats.norm.pdf(x,mean,abs(z_score))
```

```python id="C2wD_yVDDOTR" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="6b6adcc8-bf78-4e89-f90b-c7463f685d59"
trace['AvB']
```

```python id="qUQz-bsNDdFw"
std_err = compute_standard_error_prop_two_samples(convert_A, n_A, convert_B, n_B)
```

```python id="IDKWUo0syvHG" colab={"base_uri": "https://localhost:8080/", "height": 427} outputId="1c25fd6f-d3c3-497e-aeb5-d814d2b6c366"
import scipy

mean =  convert_A/n_A - convert_B/n_B

fig, ax = plt.subplots(figsize=(10,5))
x = np.linspace(-0.05,0.16,100)
plt.plot(x,scipy.stats.norm.pdf(x,mean,std_err), label='Frequentists: $ N(\hat{p_1}-\hat{p_2}, SE)$')
plt.hist(trace['AvB'], bins=100, normed=True, color='0.8', label='Posterior Distribution');
plt.legend()
plt.suptitle ('Baesian Posterior Distribution vs. Frequentist Standard Error', fontsize=18)
plt.title(' Binomial proportions, uniform priors' )
pass
# fig.savefig('03.03 Bayesian CrI vs CI.png', dpi=200)
# no prior information, expect similarities
```

```python id="Hxlip02Ryu-2" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="211a29e5-c94e-4d74-8c8b-8400dea6d66d"
# What is the probability that we gained less than +5% uplift in conversions?
(trace['AvB']<0.05).sum()/len(trace)

```

<!-- #region id="wLpLXLurFaM6" -->
Build a model through day 11 for an A/B/C test



Subset the data up through day 11. You will notice now that there are 3 different arms: A, B, and C.

You will need to add in the third arm into the model.

Additionally, calculate 3 “deterministic” variables that look at differences between the three arms:

    A vs. B
    A vs. C
    B vs. C

What action would you take given these results?
<!-- #endregion -->

```python id="69B7aG9jyu1z" colab={"base_uri": "https://localhost:8080/", "height": 637} outputId="4bc28505-1ff1-48d7-fd8d-370b67a4f3c8"
current = data[data["arm"].isin(["B","A","C"])].reset_index(drop=True)

with pm.Model() as day11_model:
    
    A_p = pm.Uniform('A_prob', lower=0, upper=1)
    B_p = pm.Uniform('B_prob', lower=0, upper=1)
    C_p = pm.Uniform('C_prob', lower=0, upper=1)
    
    A = pm.Bernoulli('A', p=A_p, observed=current[current.arm == 'A'].converted.values)
    B = pm.Bernoulli('B', p=B_p, observed=current[current.arm == 'B'].converted.values)
    C = pm.Bernoulli('C', p=C_p, observed=current[current.arm == 'C'].converted.values)
    
    AvB = pm.Deterministic('AvB', A_p - B_p)
    AvC = pm.Deterministic('AvC', A_p - C_p)
    BvC = pm.Deterministic('BvC', B_p - C_p)
    
    trace = pm.sample(50000)
    #trace = pm.sample(50000, step=pm.Metropolis(), start=pm.find_MAP())
    
pm.plot_posterior(trace[5000::3], varnames=['AvB','BvC','AvC'], color='#87ceeb', ref_val=0.)

```

```python id="uLv4tDj_yutj" colab={"base_uri": "https://localhost:8080/", "height": 909} outputId="9f66f85c-9a23-41df-de92-13b812b760c4"
_ = pm.traceplot(trace) ## and you can change the type plot
# https://github.com/thibalbo/bayesian-abtests-examples/blob/master/rate.ipynb
```

```python id="cB84UHCfyulx" colab={"base_uri": "https://localhost:8080/", "height": 319} outputId="8a6126dc-f497-40ba-ce3b-cb61939fe0fe"
def plot_betas(beta_traces, beta_names, colors=['steelblue','darkred','goldenrod']):
    fig, ax = plt.subplots(figsize=(9,5))
    for i, bn in enumerate(beta_names):
        ax = sns.distplot(beta_traces[i], kde=True, color=colors[i], label=bn)
    ax.legend(loc='upper right')
    plt.show()

plot_betas([trace[5000::3]['A_prob'], 
            trace[5000::3]['B_prob'],
            trace[5000::3]['C_prob']],
           ['A_prob','B_prob','C_prob'])

# We can be fairly certain that arm A has a higher conversion rate than arm B.
# There is not enough data to make a statement about arm C.
```

<!-- #region id="VhLtX3fkyuUu" -->
#### Another Approach
<!-- #endregion -->

<!-- #region id="vVuLJyGUyuLm" -->
Sample from beta distributions to evaluate the split test

Our arms are represented as Bernoulli distributed random variables (binary outcome conversion vs. failure). Our prior distributions model the probability of different rates for the arms.

    Note: a uniform distribution between 0 and 1 is equivalent to a Beta(1,1), or in other words a Beta distribution with 0 successes and 0 failures.

We know that the Beta distribution is a conjugate prior to the binomial likelihood, and therefore the posterior distributions for our arms are also beta distributions.

Create beta distributions representing the conversions vs. failures for each arm for all days.

The beta distributions will be parameterized with alpha and beta, which are equivalent to successes + 1 and failures + 1 respectively.


<!-- #endregion -->

```python id="yndeqtYQ6A7m" colab={"base_uri": "https://localhost:8080/", "height": 175} outputId="d6367bb8-4981-4ec0-d227-1ea57fee48d4"
data.groupby('arm')['converted'].agg([sum, len])

```

```python id="MnqqgpdcHfRq" colab={"base_uri": "https://localhost:8080/", "height": 319} outputId="ee45d388-eaca-4ec9-8c05-723d679bc697"
a_beta = stats.beta(67, 357)
b_beta = stats.beta(29, 241)
c_beta = stats.beta(34, 130)

#Plot the beta distributions across the 0-0.4 range of rates.

fig, ax = plt.subplots(figsize=(9,5))
rates = np.linspace(0.001, 0.40, 300)
ax.plot(rates, a_beta.pdf(rates), color='steelblue', lw=3, label='A')
ax.plot(rates, b_beta.pdf(rates), color='darkred', lw=3, label='B')
ax.plot(rates, c_beta.pdf(rates), color='goldenrod', lw=3, label='C')
ax.legend(loc='upper right')
plt.show()

```

<!-- #region id="2vTYcE8bHtn7" -->
Calculate AvB, AvC, and BvC using sampling from the beta distribution

The beta distributions for the arm are our posterior distributions for the conversion rate of each arm given the observed data.

We can calculate the distributions of differences in rates between the arms using sampling. The procedure is:

    * Set up a certain number of iterations (1000, for example)
    * For each iteration, take a random draw from each beta distribution
    * Calculate the difference between the sampled rates between the arms
    * Store the differences in lists

Then you can plot these distributions of differences.
<!-- #endregion -->

```python id="Kq7glXozHfNb" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="5f786968-39a9-4d1d-eb70-194bce84a4d7"
a_samples = a_beta.rvs(5000)
b_samples = b_beta.rvs(5000)
c_samples = c_beta.rvs(5000)

AvB = a_samples-b_samples
AvC = a_samples-c_samples
BvC = b_samples-c_samples

ax = sns.distplot(AvB)
ax.axvline(0, lw=2, ls='dashed', c='black')

```

```python id="yhWpoodLHfLd" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="3e9ca5cc-32fe-4baf-e745-bc81e2883805"
ax = sns.distplot(AvC)
ax.axvline(0, lw=2, ls='dashed', c='black')

```

```python id="F67f4xzkHfDE" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="d21bbf95-1362-4993-f002-73e376e51065"
ax = sns.distplot(BvC)
ax.axvline(0, lw=2, ls='dashed', c='black')

```

<!-- #region id="SWM4ZLp4he6i" -->
## Causal Discovery
<!-- #endregion -->

<!-- #region id="PrrlM3lflgS7" -->
#### Time Series Causal Discovery - TCDF
<!-- #endregion -->

```python id="bRwVXA8VHe_7"
## Here is a good dataset in which you can test different data
## I am happy with these, I can apply them tomorrow. 

https://github.com/sayakpaul/A-B-testing-with-Machine-Learning/blob/master/A%20B%20tests%20with%20Machine%20Learning.ipynb

```

```python id="5fvDunnaHe3v"
Causal relationship, traffic volume and weather.

https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume

Causal relationship,  capital markets
```

```python id="TAxtDoM2kfB-" colab={"base_uri": "https://localhost:8080/", "height": 139} outputId="f1881e30-4af7-4e46-83a9-19f1b7b7e2aa"
!git clone https://github.com/M-Nauta/TCDF.git
```

```python id="H2ug-hxMhadC"
import pandas as pd
```

```python id="0l3rCeMtGFhC"

```

```python id="7LJm1qO2haKK"
df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets/master/Metro_Interstate_Traffic_Volume.csv")
```

```python id="I4VEY8uqjB93" colab={"base_uri": "https://localhost:8080/", "height": 191} outputId="e6db2de5-4f47-4c9b-ebfe-8e8e3cc685db"
df.dtypes
```

```python id="5xeIdV57hZ7P" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="796b67ed-d635-457c-df0c-8889c22cdc0c"
df.head()
```

```python id="9UT7apllhZtG"
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.set_index("date_time")
```

```python id="NY0H6TBbjazd"
df = df[["traffic_volume", "temp","clouds_all","rain_1h","snow_1h"]]
```

```python id="_luzHr2FYuSp"

```

<!-- #region id="P2GOroXsYswp" -->

<!-- #endregion -->

```python id="KcNyNkRmhZpM" colab={"base_uri": "https://localhost:8080/", "height": 238} outputId="dc4cebf5-405a-4f3f-c561-8ff0628e475c"
df.head()
```

```python id="7xJG4sCzhZlP" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="bd62e089-b326-492e-d09a-c07e87bedb97"
%cd TCDF
```

```python id="04O1FIibhZOb"
df.to_csv("traffic.csv",index=False)
```

```python id="kyl1wdwdhY2o" colab={"base_uri": "https://localhost:8080/", "height": 920} outputId="08dd8313-fd85-4fa0-8dfc-bc8294220b1a"
%run -i "runTCDF.py" --help
```

```python id="t35pXXb5Hev1" colab={"base_uri": "https://localhost:8080/", "height": 784} outputId="76f1c747-b7c7-4e0d-f7ac-4eea071e1740"
"""Run TCDF"""
%matplotlib inline
%run -i "runTCDF.py" --data traffic.csv
```

```python id="NQNxgJgSnKll"
### Lets look at FRED
FRED  =pd.read_csv('https://github.com/firmai/random-assets/raw/master/capital_markets.txt')[1:] ### 130 additional series 
FRED  =FRED.set_index('sasdate')
FRED.index = pd.to_datetime(FRED.index)
FRED = FRED.ffill().bfill()
```

```python id="j8KTADw1p8Po"
FRED.to_csv("FRED.csv", index=False)
```

```python id="FSZl23zZHepS" colab={"base_uri": "https://localhost:8080/", "height": 287} outputId="cdf24845-d413-4e31-a4c6-ecd0bc9c53fb"
FRED.head()
```

```python id="LlZ-Xge3qDl8" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="5662adc3-ab53-43ea-e120-88706e72d6de"
"""Run TCDF"""
%matplotlib inline
%run -i "runTCDF.py" --data FRED.csv
```

<!-- #region id="WpbAMY1IuCaJ" -->
* AMDMNOx - Real Manufacturers' New Orders: Durable Goods
* ANDENOx - Real Value of Manufacturers' New Orders for Capital Goods:
* TOTRESNS - Total Reserves of Depository Institutions
* AMBSLREALx - Adjusted Monetary Base



<!-- #endregion -->

```python id="xBLYG-qOqDi9"
### Now lets have a look at capital markets
### ... only going to look at the last 500 records, features of which have 70% filled records
```

```python id="i7RzF0tQyAVL"
capital  =pd.read_csv('https://github.com/firmai/random-assets/raw/master/CMD.csv').iloc[-1000:,1:] ### 130 additional series 

### Lets look at FRED
capital  =capital.set_index('Date')
capital.index = pd.to_datetime(capital.index)
capital = capital.loc[:, capital.isnull().mean() < .3]
capital = capital.ffill().bfill()

```

```python id="ihJLDrwcyBR7" colab={"base_uri": "https://localhost:8080/", "height": 256} outputId="7c63e159-9ccf-4fbb-9830-7c48f48d8f03"
capital.head()
```

```python id="3ecNc9DFqDc4"

```

```python id="CD-XDqiJzIqf" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="191a74ec-dbff-45f6-ba40-ca7ef404e58e"
capital.shape
```

```python id="c5FyViNTqDYU"
capital.to_csv("capital_markets.csv", index=False)
```

```python id="eGclQKw5qDUT" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="90ffb29f-63f8-4d36-e476-b9a3f8116302"
"""Run TCDF"""
%matplotlib inline
%run -i "runTCDF.py" --data capital_markets.csv
```

<!-- #region id="jkblejJ89J-y" -->
#### Time Series Causal Discovery - Trig

<!-- #endregion -->

```python id="PjjT1i4M90je" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="1a14d23b-0798-4e90-866c-40d79739b3e1"
!git clone https://github.com/jakobrunge/tigramite.git
%cd tigramite
!python setup.py install
```

<!-- #region id="Pu360yghBgwu" -->
TIGRAMITE is a time series analysis python module. It allows to reconstruct graphical models (conditional independence graphs) from discrete or continuously-valued time series based on the PCMCI method and create high-quality plots of the results. PMCI is used to Detecting and quantifying causal associations in large nonlinear time series datasets. This tutorial explains how to use PCMCI to obtain optimal predictors.
<!-- #endregion -->

```python id="PARXSRTl5nVo" colab={"base_uri": "https://localhost:8080/", "height": 607} outputId="fbb42b46-68e7-4069-9194-c0c935de731b"
import numpy
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
numpy.random.seed(42)

data = df.values

# Data must be array of shape (time, variables)
print(data.shape)

dataframe = pp.DataFrame(data)
cond_ind_test = ParCorr()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
results = pcmci.run_pcmci(tau_max=5, pc_alpha=None)
pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                     val_matrix=results['val_matrix'],
                                     alpha_level=0.05)
```

<!-- #region id="7RscsS1E_08y" -->
What causes traffic
* Traffic and hour ago more now
* Traffic two hours ago less now
* Clouds one hour ago more traffic now
* Rain one hour ago more traffic now

What causes temperature
* Traffic one hour ago higher temperature now

What causes rain
* Traffic one hour ago more rain now
* Temperature two hours ago more rain now



<!-- #endregion -->

```python id="dzEOEdCK_iQ2"
# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline     
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

### Ignoring this for now
np.random.seed(42)
T = 150
links_coeffs = {0: [((0, -1), 0.6)],
                1: [((1, -1), 0.6), ((0, -1), 0.8)],
                2: [((2, -1), 0.5), ((1, -1), 0.7)],  # ((0, -1), c)],
                }
N = len(links_coeffs)
data, true_parents = pp.var_process(links_coeffs, T=T)


dataframe = pp.DataFrame(df.values, var_names = df.columns)
N = df.shape[1]

pred = Prediction(dataframe=dataframe,
        cond_ind_test=ParCorr(),   #CMIknn ParCorr
        prediction_model = sklearn.linear_model.LinearRegression(),
#         prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
        # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
    data_transform=sklearn.preprocessing.StandardScaler(),
    train_indices= range(int(0.8*len(df))),
    test_indices= range(int(0.8*len(df)), len(df)),
    verbosity=1
    )


```

```python id="Xe2Kar7a_dGv" colab={"base_uri": "https://localhost:8080/", "height": 984} outputId="903a6277-be33-47b4-e9c0-93d4e6997c6e"
target = 0 # I want to test what causes more traffic
tau_max = 10
steps_ahead_count = 2 
predictors = pred.get_predictors(
                  selected_targets=[target],
                  steps_ahead=steps_ahead_count,
                  tau_max=tau_max,
                  pc_alpha=None
                  )
link_matrix = np.zeros((N, N, tau_max+1), dtype='bool')
for j in [target]:
    for p in predictors[j]:
        link_matrix[p[0], j, abs(p[1])] = 1

# Plot time series graph
tp.plot_time_series_graph(
    figsize=(6, 3),
    val_matrix=np.ones(link_matrix.shape),
    link_matrix=link_matrix,
    var_names=None,
    link_colorbar_label='',
    )
```

```python id="qKy1SGb3HIcU" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="a687c17f-cd78-4d50-9275-4159be4abb4e"
pred.fit(target_predictors=predictors, 
                selected_targets=[target],
                    tau_max=tau_max)
```

```python id="gOqLtn6YO6dl" colab={"base_uri": "https://localhost:8080/", "height": 382} outputId="81ae4a16-25f4-434e-d55e-cdca5fa9f761"
predicted = pred.predict(target)
true_data = pred.get_test_array()[0]

plt.scatter(true_data, predicted)
plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean()/true_data.std()))
plt.plot(true_data, true_data, 'k-')
plt.xlabel('True test data')
plt.ylabel('Predicted test data')
```

<!-- #region id="IXeFGT6NLwRU" -->
It still underfits, if I use all the predictors it still works better:
<!-- #endregion -->

```python id="q9Xe1QU95nO3" colab={"base_uri": "https://localhost:8080/", "height": 382} outputId="d692f4d1-4f64-470f-fbe0-1b25d5e7f312"
all_predictors = {target:[(i, -tau) for i in range(N) for tau in range(1, tau_max+1)]}
pred.fit(target_predictors=all_predictors, 
                selected_targets=[target],
                    tau_max=tau_max)

# new_data = pp.DataFrame(pp.var_process(links_coeffs, T=100)[0])
predicted = pred.predict(target)
# predicted = pred.predict(target)
true_data = pred.get_test_array()[0]

plt.scatter(true_data, predicted)
plt.plot(true_data, true_data, 'k-')
plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean()/true_data.std()))
plt.xlabel('True test data')
plt.ylabel('Predicted test data')


```

```python id="pTtdRVda5nLq" colab={"base_uri": "https://localhost:8080/", "height": 424} outputId="22d7eca3-ad66-4afc-81ee-fa07dcc66b18"
tp.plot_timeseries(dataframe)
```

```python id="9SrAZj9P5nDb"
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=1)
```

```python id="zVtx3ruu5m_z" colab={"base_uri": "https://localhost:8080/", "height": 337} outputId="131859a2-9755-467e-9ee0-43586531e8e9"
correlations = pcmci.get_lagged_dependencies(tau_max=20)
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':df.columns, 
                                    'x_base':5, 'y_base':.5})
```

```python id="wpmQHA3v5m5w" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="87f7e4b0-af6f-4f30-b917-74f9ff4915d6"
pcmci.verbosity = 1
results = pcmci.run_pcmci(tau_max=10, pc_alpha=None)
```

```python id="Do2-H-AZqDOH" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="0069d9dd-b098-49a3-dcc1-59f99a4981bb"
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
pcmci.print_significant_links(
        p_matrix = results['p_matrix'], 
        q_matrix = q_matrix,
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)

link_matrix = pcmci.return_significant_parents(pq_matrix=q_matrix,
                        val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
```

```python id="ArQ9VUxmqDHX" colab={"base_uri": "https://localhost:8080/", "height": 306} outputId="8957e774-8e4b-4eb8-afa0-be037a428b4c"
tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=df.columns,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    )
## left is node colour, right is edge colout
```

```python id="kVKT6mg7TWO4" colab={"base_uri": "https://localhost:8080/", "height": 316} outputId="d3419507-5f04-4315-d978-5b170e809a4b"
# Plot time series graph
tp.plot_time_series_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=df.columns,
    link_colorbar_label='MCI',
    )


# A wider range of activities can be found here:
# https://github.com/jakobrunge/tigramite/blob/master/tutorials/tigramite_tutorial_basics.ipynb
```

<!-- #region id="lXDPPTcMXpBh" -->
#### Cross-sectional Causal Discovery
<!-- #endregion -->

```python id="DYcDewcdX6zq" colab={"base_uri": "https://localhost:8080/", "height": 645} outputId="3515676b-49cf-412a-a117-5b4e9e295638"
!pip install cdt
```

```python id="aPgzxXCLXhil"
#Import libraries
import cdt
from cdt import SETTINGS
SETTINGS.verbose=False
SETTINGS.NJOBS=16
import networkx as nx
import time
# A warning on R libraries might occur. It is for the use of the r libraries that could be imported into the framework
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


```

```python id="-GnOj3egXhgQ" colab={"base_uri": "https://localhost:8080/", "height": 455} outputId="fb59e73f-32c3-4c6d-c506-ceeddea6a858"
# Load data and graph solution
data, solution = cdt.data.load_dataset('sachs')
nx.draw_networkx(solution, font_size=8) # The plot function allows for quick visualization of the graph. 
plt.show()
print(data.shape)
data.head()


```

```python id="q3H7wKmaXhc3"
# Finding the structure of the graph
from cdt.independence.graph import FSGNN

# own data

df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets/master/Metro_Interstate_Traffic_Volume.csv")
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.set_index("date_time")
df = df[["traffic_volume", "temp","clouds_all","rain_1h","snow_1h"]]

data = df.head(2000)

Fsgnn = FSGNN(train_epochs=1000, test_epochs=500, l1=0.1, batch_size=1000)

start_time = time.time()
```

```python id="mdpibKzhXhZP" colab={"base_uri": "https://localhost:8080/", "height": 329} outputId="ac1b948e-1664-4092-f00d-add5b4fb1e37"
ugraph = Fsgnn.predict(data, threshold=1e-7)
print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))
nx.draw_networkx(ugraph, font_size=8) # The plot function allows for quick visualization of the graph.
plt.show()
# List results
pd.DataFrame(list(ugraph.edges(data='weight')), columns=['Cause', 'Effect', 'Score'])
```

```python id="jaIwM7fmXhRc"

```

```python id="TaIJAvjDXhPa"

```

```python id="smcPx4dYXhLa"

```

```python id="YtvwzHlNXhAv"

```

```python id="UJl_wtNhXgz0"

```

<!-- #region id="UKV5pOZDXgaC" -->

<!-- #endregion -->

<!-- #region id="oDbCCMOurgUA" -->
## Causal Inference

One day a team lead notices that some members of their team wear cool hats, and that these members of the team tend to be less productive. Being data drive, the Team Lead starts to record whether or not a team member wears a cool hat ($X=1$ for a cool hat, $X=0$ for no cool hat) and whether or not they are productive ($Y=1$ for productive, $Y=0$ for unproductive).

After making observations for a week, they end up with a dataset like the following:
<!-- #endregion -->

<!-- #region id="YmjUwYFmsZe6" -->

<!-- #endregion -->

```python id="nZsbRAPisalx" colab={"base_uri": "https://localhost:8080/", "height": 86} outputId="92436176-0b54-4637-8b7e-a89f7b94f56f"
!git clone https://github.com/ijmbarr/notes-on-causal-inference.git
# % to switch directory 
%cd notes-on-causal-inference
```

```python id="KTpbika8tckp"
from __future__ import division
import datagenerators as dg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")

%matplotlib inline
```

```python id="XcdvURlLVuea" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="fcee17e8-375a-4882-bf79-597159d35cd8"
observed_data_0 = dg.generate_dataset_0()
observed_data_0.head()
```

<!-- #region id="PjisCo37tKBr" -->
The first question the team lead asks is: are people wearing cool hats more likely to be productive that those who don't? This means estimating the quantity

𝑃(𝑌=1|𝑋=1)−(𝑌=1|𝑋=0)

which we can do directly from the data:

<!-- #endregion -->

```python id="2_os_W9lrxdJ" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="912c4567-cd51-4e85-f499-d75c393ad6b0"
def estimate_uplift(ds):
    """
    Estimate the difference in means between two groups.
    This is closer related to the z and t tests 
    
    Parameters
    ----------
    ds: pandas.DataFrame
        a dataframe of samples.
        
    Returns
    -------
    estimated_uplift: dict[Str: float] containing two items:
        "estimated_effect" - the difference in mean values of $y$ for treated and untreated samples.
        "standard_error" - 90% confidence intervals arround "estimated_effect"
        
        
    """
    base = ds[ds.x == 0]
    variant = ds[ds.x == 1]
    
    delta = variant.y.mean() - base.y.mean()
    delta_err = 1.96 * np.sqrt(
        variant.y.var() / variant.shape[0] + 
        base.y.var() / base.shape[0])
    
    return {"estimated_effect": delta, "standard_error": delta_err}

estimate_uplift(observed_data_0)
```

<!-- #region id="VRy2GdwZtoc3" -->
It looks like people with cool hats are less productive.To be sure, we can run a statistical test:

A chi-square test tests a null hypothesis about the relationship between two variables. For example, you could test the hypothesis that men and women are equally likely to vote "Democratic," "Republican," "Other" or "not at all." A chi-square test requires categorical variables, usually only two, but each may have any number of levels, whereas A t-test requires two variables; one must be categorical and have exactly two levels, and the other must be quantitative and be estimable by a mean.

<!-- #endregion -->

```python id="i_3gXd1FtWKt" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="17f21e5d-dc2d-45cf-c210-9a3c88f494a4"
from scipy.stats import chi2_contingency

contingency_table = (
    observed_data_0
    .assign(placeholder=1)
    .pivot_table(index="x", columns="y", values="placeholder", aggfunc="sum")
    .values
)

_, p, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")

# p-value
p
```

<!-- #region id="mGlsrEEfvdUk" -->
The p-value is low enough to accept the null hyothesis. The problem is that it has not been randomly assigned as a controlled experiment, and instead uses observational data, we can create a fake experiment to assign a hat randomly and it could be possibly show the opposite effect.
<!-- #endregion -->

```python id="RnwzPo-dv97g" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="dbf8c377-8c2a-4eec-ad98-042160598f3c"
## Dont take this too seriously, created to show how random allocation showed opposite effect
## .. the idea being that confounders have now been removed.
## This is simply running the experiment, generating the data and
## identifying the results
def run_ab_test(datagenerator, n_samples=10000, filter_=None):
    """
    Generates n_samples from datagenerator with the value of X randomized
    so that 50% of the samples recieve treatment X=1 and 50% receive X=0,
    and feeds the results into `estimate_uplift` to get an unbiased 
    estimate of the average treatment effect.
    
    Returns
    -------
    effect: dict
    """
    n_samples_a = int(n_samples / 2)
    n_samples_b = n_samples - n_samples_a
    set_X = np.concatenate([np.ones(n_samples_a), np.zeros(n_samples_b)]).astype(np.int64)
    ds = datagenerator(n_samples=n_samples, set_X=set_X)
    if filter_ != None:
        ds = ds[filter_(ds)].copy()
    return estimate_uplift(ds)

run_ab_test(dg.generate_dataset_0)
```

<!-- #region id="pCjug7GTwbLE" -->
So the opposite seems to be true. Note: In the above example, and in all following examples, I'm assuming that our samples are i.i.d., and obey the Stable unit treatment value assumption (SUTVA). Basically this means that when one person chooses, or is forced to wear a really cool hat they have no influence on the choice or effect of another person wearing a really cool hat.
<!-- #endregion -->

<!-- #region id="ft52IxotwuVN" -->
In the previous example, when we make no intervention on the system, we have an observational distribution of $Y$, conditioned on the fact we observe $X$:

$P(Y|X)$

When we force people to wear cool hats, we are making an intervention. The distribution of $Y$ is then given by the _interventional_ distribution 

$P(Y|\hbox{do}(X))$

In general these two are not the same.
<!-- #endregion -->

<!-- #region id="tGvGxDg9w2E_" -->
The question these notes will try and answer is how we can reason about the interventional distribution, when we only have access to observational data.

This is a useful question because there are lots of situations where running an A/B test to directly measure the effects of an intervention is impractical, unfeasable or unethical. In these situations we still want to be able to say something about what the effect of an intervention is - to do this we need to make some assumptions about the data generating process we are investigating.
<!-- #endregion -->

<!-- #region id="80i-nf3lzVBJ" -->
### Potential Outcomes 

One way to approach this problem is to introduce two new random variables to our system: $Y_{0}$ and $Y_{1}$, known as the Potential Outcomes. We imagine that these variables exist, and can be treated as any other random variable - the only difference is that they are never directly observed. $Y$ is defined in terms of 

 - $Y = Y_{1}$ when $X=1$
 - $Y = Y_{0}$ when $X=0$
 
This shifts the problem from one about how distributions change under the intervention, to one about data drawn i.i.d. from some underlying distribution with missing values. Under certain assumptions about why values are missing, there is well developed theory about how to estimate the missing values.

### Goals

Often we do not care about the full interventional distribution, $P(Y|\hbox{do}(X))$, and it is enough to have an estimate of the difference in means between the two groups. This is a quantity known as the Average Treatment Effect:

$\Delta = E[Y_{1} - Y_{0}]$

When we run and A/B test and compare the means of each group, this is directly the quantity we are measuring 


Two related quantities are 

 - $ATT = E[Y_{1} - Y_{0}|X=1]$, the "Average Treatment effect of the Treated"
 - $ATC = E[Y_{1} - Y_{0}|X=0]$, the "Average Treatment effect of the Control"

One way to interpret ATC is as a measure of the effect of treating only samples which wouldn't naturally be treated, and vice versa for ATT. Depending on your use case, they may be more natural measures of what you care about. The following techniques will allow us to estimate them all.


$\def\ci{\perp\!\!\!\perp}$
### Making Assumptions

When we A/B test, we randomize the assignment of $X$. This has the effect of allowing us to choose which variable of $Y_{1}$ or $Y_{0}$ is revealed to us. This makes the outcome independent of the value of $X$. We write this as

$Y_{1}, Y_{0} \ci X$

Which means that the distribution of $X, Y_{0}, Y_{1}$ factorizes as

$P(X, Y_{0}, Y_{1}) = P(X)P(Y_{0}, Y_{1})$

If this independence holds then

$E[Y_{1}|X=1] = E[Y_{1}]$

If we want to estimate the ATE using observational data, we need to use other information we have about the samples - specifically we need to **assume** that we have enough additional information to completely explain the choice of treatment each subject.

If we call the additional information the random variable $Z$, we can write this assumption as

$Y_{1}, Y_{0} \ci X \, | \, Z$

or

$P(X, Y_{0}, Y_{1}| Z) = P(X|Z)P(Y_{0}, Y_{1}|Z)$

This means that the observed treatment a sample receives, $X$, is completely explained by $Z$. This is sometimes called the "ignorability" assumption.

In our motivating example about cool hats this would mean that there is some other factor - let's call it "skill" - which impacts both the productivity of the person and whether or not they wear a cool hat. In our example above, skilled people are more likely to be productive and also less likely to were cool hats. These facts together _could_ explain why the effect of cool hats seemed to reverse when ran an A/B test. 

If we split our data on whether or not the person is skilled, we find that for each subgroup there is a positive relationship between wearing cool hats and productivity:

<!-- #endregion -->

<!-- #region id="SxIbuBOs0zLm" -->
Unfortuntly, because we never observe $Y_{0}$ and $Y_{1}$ for the same sample, we cannot test the assumption that 

$Y_{1}, Y_{0} \ci X \, | \, Z$

It is something we have to use our knownledge of the system we are investigating to evaluate.

The quality of any prediction you make depends on exactly how well this assumption holds. Simpson's Paradox is an extreme example of the fact that if $Z$ does not give contain all confounding variables, then any inference we make could be wrong. [Facebook have a good paper comparing different causal inference approaches with direct A/B test that show how effects can be overestimated when conditional independence doesn't hold](https://www.kellogg.northwestern.edu/faculty/gordon_b/files/kellogg_fb_whitepaper.pdf).

Once we have made this assumption there are a number of techniques for approaching this. I will outline a few of simpler approaches in the rest of the post, but keep in mind that this is a area of ongoing research. In human speak you can investigate the counterfactual with an additional variable, but you should trust this apprach less than interventionist studies.
<!-- #endregion -->

<!-- #region id="Gja0N8JE1MnA" -->
### Modeling the Counterfactual

From the above, it should be clear that if know $Y_{0}$ and $Y_{1}$, we can estimate the ATE. So why not just try and model them directly? Specifically we can build estimators: 
 
 - $\hat{Y}_{0}(Z) = E[Y|Z, X=0]$
 - $\hat{Y}_{1}(Z) = E[Y|Z, X=1]$. 
 
If we can model these two quantities, we can estimate the ATE as:

$\Delta = \frac{1}{N}\sum_{i}(\hat{Y}_{1}(z_{i}) - \hat{Y}_{0}(z_{i}))$

The success of this approach depends on how well we can model the potential outcomes. To see it in action, let's use the following data generating process:
<!-- #endregion -->

```python id="XNC2JRamuMoq" colab={"base_uri": "https://localhost:8080/", "height": 279} outputId="ca73c001-1708-4d85-c6fa-beba062e883f"
observed_data_1 = dg.generate_dataset_1()

observed_data_1.plot.scatter(x="z", y="y", c="x", cmap="rainbow", colorbar=False);
```

```python id="LhUL6bQm1ZPO" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="cb56ca77-7d9b-497a-8117-004b54b27dea"
sns.kdeplot(observed_data_1.loc[lambda df: df.x == 0].y, label="untreated")
sns.kdeplot(observed_data_1.loc[lambda df: df.x == 1].y, label="treated")
```

```python id="A3IRWhFM1bou" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="226c8594-0795-45a5-b59e-8cd115ec6e77"
# We can confirm this by looking at the difference in means between the two groups
print("Observed ATE: {estimated_effect:.3f} ({standard_error:.3f})".format(**estimate_uplift(observed_data_1)))
```

<!-- #region id="21fACLuM11R4" -->
However, if we look at the distribution of the covariance, 𝑍, it is clear that there is a difference between the groups. 
<!-- #endregion -->

```python id="DigtHNIS1j4G" colab={"base_uri": "https://localhost:8080/", "height": 282} outputId="79cd2e4a-80be-49e9-cdbb-392f371c3a49"
sns.kdeplot(observed_data_1.loc[lambda df: df.x == 0].z, label="untreated")
sns.kdeplot(observed_data_1.loc[lambda df: df.x == 1].z, label="treated")
```

<!-- #region id="oxyAMZCt17ej" -->
If we believe that $Z$ has some influence on the metric $Y$, this should concern us. We need some way to disentangle the effect of $X$ on $Y$ and the effect of $Z$ on $Y$.

We can check the actually ATE using our simulated A/B test and confirm that it is difference of the observed value:
<!-- #endregion -->

```python id="VZ_U1ZoK13OI" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="0ac1041e-e29f-4625-cf5e-453912b0542b"
print("Real ATE:  {estimated_effect:.3f} ({standard_error:.3f})".format(**run_ab_test(dg.generate_dataset_1)))
```

```python id="ukkNPmyf19_p"
## Interesting - https://colab.research.google.com/drive/1juY2A4SVR1-nZzLX__zwOjHy_SvLBcaD#scrollTo=vR-1V8w5rN3m
## Expansion with similar framewokr - http://www.degeneratestate.org/posts/2018/Jul/10/causal-inference-with-python-part-2-causal-graphical-models/
## https://github.com/microsoft/EconML
## You can follow through with this if you do the naming convention



```
