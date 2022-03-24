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
    language: python
    name: python3
---

<!-- #region -->
# How good is my recommender?

After having learned some possible recommender systems (RS) - [Non Personalised](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%201%20Part%20I%20-%20Non%20Personalised%20and%20Stereotyped%20Recommendation.ipynb), [Content Based](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%201%20Part%20III%20-%20Content%20Based%20Recommendation.ipynb), [User User](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%202%20Part%20I%20-%20User%20User%20Collaborative%20Filtering.ipynb) and [Item Item](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%202%20Part%20II%20-%20Item%20Item%20Collaborative%20Filtering.ipynb) Collaborative Filtering - you're going to see that we never asked one possible question: **Which one is better for me?**
  
This is not an easy question, different business can have different priorities and for sure, as in any other computer science areas, we don't have a '[one algorithm fits all](https://en.wikipedia.org/wiki/No_free_lunch_theorem)' problem.
  
In this and the next notebook, we're going to take a look at what approaches researchers and companies can take to answer if a certain RS is proper for them. To start, when talking about evaluation, we usually perform them in two main ways:

* **Offline Evaluation**: Offline evaluation is done in similar ways we evaluate machine learning models, *i.e.*, we usually have a fixed dataset, collected and immutable before the beggining of the evaluation, and then the dataset is splited into two parts, the train and test set, the RS are trained on the train and then evaluated over the test set.
* **Online Evaluation**: As the name states, the online evaluation is usually performed online, with real users interacting with different versions or algorithms of a RS and the evaluation is performed by collecting metrics associated with the user behaviour in real time.

## When do I perform one or another?

Both of these approaches have its pros and cons:

* **Offline**: 
    - **Pros** - This type of evaluation can be easier to set. By having lots of already published datasets with their respective ratings or evaluations, people can **easilly set up and evaluate** their algorithms by comparing their output with the expected output from the already published results. By having a fixed dataset and possible fixed user interactions with it (all existing ratings in the dataset) the results of an offline evaluation is also **reproducible in a easier way**, comparing to online evaluations.
    - **Cons** - There are a few discussions regarding the validity of offline evaluations. For example, the most criticized aspect of it is the overall capacity of the performance evaluation of the trained algorithm in a splited test set. The idea of a RS is to provide new recommendations that the user probably doesn't know yet. The problem of testing it in a test set is that we must have already the user's evaluations for each item/recommendation, *i.e.* we end up testing only item that we are sure the user knows. Even more, in this evaluation, if the RS recommend an item the user hadn't evaluated yet but that could be a **good recommendation, we penalise it because we don't have it in our test set**. In the end, we end up penalising the RS for doing its job.  
    
    
* **Online**:
    - **Pros** - Contrary to offline evaluations, in a online context, we have the **possibility to collect real time user interaction with the RS**, among which, reviews, clicks, preferences and etc. This can bring a whole better picture when evaluating the RS's performance. Besides, as we are evaluating real time data, instead of a static one, we're **able to provide further analysis if desired**.
    - **Cons** - Dynamic real time data also bring a negative point in the evaluation, as the **reproducibility** of the experiment can be worse, when comparing to a static script and dataset. Besides, in order to prepare (and maybe even create) the environment to test the RS, we must **expend a considerable higher amount of time to set it up**.

Below, ([Hijikata, 2014](http://soc-research.org/wp-content/uploads/2014/11/OfflineTest4RS.pdf)) provided a few useful guidelines when comparing the pros and cons of each approach:

<img src="images/notebook7_image1.png" width="500">
Source: http://soc-research.org/wp-content/uploads/2014/11/OfflineTest4RS.pdf

In this notebook, in the following sections, we are going to discuss a few different perspectives we can evaluate over RS when performing an **offline** evaluation. 

The first set of measures evaluate the test set in a pure statistical way, without considering more complex business settings:

- Accuracy of Estimated Rating:
    - MAE
    - MSE
    - RMSE
- Accuracy of Estimated Ranking:
    - Spearman's Rank
    - Kendal's Rank
    - NDMP
- Decision Making List Relevance:
    - Precision / Recall
    - ROC Curve AUC
    - Precision@N
    - Average Precision
- Accuracy Based on Ranking Position:
    - Mean Reciprocal Rank
    - Normalised Discounted Cumulative Gain (nDCG)
    
Besides the traditional ML error metrics, we're seeing here error metrics based on returned lists of outputs/recommendations. These are a little different from the traditional ML world and we're going to take a closer look at them.


# Recommender Systems Statistical Metrics

## Precision@N and Average Precision

When using a precision measure, we assume the documents we are retrieving have a binary relevance, *i.e.*, the document is important or not. When we're working with different scales, such as 1-5 star ratings, we must adapt it in order to evaluate its precision. We do this by creating a threshold defining whether the document is relevant or not. For example, in a movie RS, we could define that every rating > 3.5 could be defined as relevant, while everything smaller than 3.5 is considered irrelevant.

A precision@N metric measure how many relevant documents we've retrieved taking into account only the first N elements of a returned list of documents. Anything below the Nth document is ignored.

<img src="images/notebook7_image20.png" width="250">
Source: https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

The **average precision** is defined from the precision@N metric. Considering **only** the relevant documents that were returned from a query/program, calculate the average precision@N for each document. It can be interpreted as:

<img src="images/notebook7_image3.png" width="450">
Source: https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf


## Mean Reciprocal Rank (MRR)

Imagine a recommender system returns a list of $N$ web pages given a user query and lets make the question: "How many pages do I have to go through in order to find one that is relevant to me?" Given that each page can have a flag indicating whether the page is relevant to the user or not, we can calculate the MRR for multiples queries and obtain an average performance of the RS. For each query and returned list, its score for the MRR is going to be:

$$\frac{1}{Rank_{i}}$$

Then, for a set of lists, the final score is just the average of all individual scores:

$$MRR = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{Rank_{i}}$$

Where |Q| is the cardinality of all the queries performed.

[Wikipedia's example](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) is a good example of MRR calculation. Given a system that convert singular words to its plural form, and given the first words in the list are the ones the system gives more confidence, we can calculate the MRR as following:


|Query|	Proposed Results|	Correct response|	Rank|	Reciprocal rank|
|---|---|---|---|---|---|
|cat|	catten, cati, cats|	cats|	3|	1/3|
|tori|	torii, tori, toruses|	tori|	2|	1/2|
|virus|	viruses, virii, viri|	viruses|	1|	1|

$$MRR = AVG(\frac{1}{3} + \frac{1}{2} + 1) = 0.61$$

Why we don't just calculate the average rank of the first relevant item returned in the list? When we are checking the returned pages from a Google's query, we almost always don't go further than the first page and pages returned in the first positions have more weight than the ones in the last positions from the list. 
In a average rank calculation, if we had a list with rank 2 and other with rank 4, we would have an average of rank 3 = 1/3 = 0.66. The MRR calculation gives a much higher weight for getting a wrong item in the first position than getting a wrong position. In this case, getting the correct item in position 2 instead of position 1, gives a 0.5 penalisation ($\frac{1}{rank_1} - \frac{1}{rank_2}$), while getting a correct item in position 4 instead of position 3 gives only a 0.08 penalisation ($\frac{1}{rank_3} - \frac{1}{rank_4}$). So the MRR forces the algorithm to try to recommend relevant items always in the firsts positions.

## Normalised Discounted Cumulative Gain (nDCG)

On contrary to MRR, which measures how much do we have to go down the list in order to find the **first** relevant document, the nDCG sum the overall importance of the entire returned list. Consider for example an application where we return a number of papers related to a specific query/disease. Some papers are really important about that topic, some are neutral and some are even bad. If we attribute a score for the importance of each paper (contrary to binary relevance as in the MRR), the difinition of cumulative gain can be formalised:

$$CG_{p} = \sum_{i=1}^{p} rel_{i}$$

This is a very intuitive and simple metric. However, it doesn't consider two things:
    * It is rank independent. Changes in the rank position, doesn't affect the final score. Sometimes, we want to prioritise important documents coming first in the rank.
    * It is scale dependent. Bigger returned lists will tend to have higher CG scores.
    
In order to solve the first problem, a discounted cumulative gain score was defined, where the relevance of each document are smaller the more at the end of the rank they are. Just as a feature, the relativeness of each rank doesn't drop linearly, but logarithmically:

$$DCG_{p} = \sum_{i=1}^{p} \frac{rel_{i}}{log_{2}(i+1)}$$

Then, in order to attack the second problem, a normalised DCG was defined, by just dividing the DCG by a normalisation factor. This normalisation factor is done by calculating the *ideal DCG*, which is the DCG of the ordered by relevance result from a query. Again, an example from [Wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain):

Imagine we performed a query which returned documents that could have 3 levels of significancy: 0, 1, 2 or 3, where the bigger the number, the more important it is. One query performed by a user returned the following sequence of documents: [3, 2, 3, 0, 1, 2]. The cumulative gain is going to be:

$$CG_{p} = \sum_{i=1}^{p} rel_{i} = 3 + 2 + 3 + 0 + 1 + 2 = 11$$

We check here that changing doc3 by doc4 (values 3 and 0) we would still have the same CG. So, calculating the DCG, we'd have:

<img src="images/notebook7_image4.png", width="250">

$$DCG_{p} = \sum_{i=1}^{p} \frac{rel_{i}}{log_{2}(i+1)} = 3 + 1.262 + 1.5 + 0 + 0.387 + 0.712 = 6.861$$

Lastly, in order to normalise the DCG, we first create the ideal rank, which is just the sorted buy relevance version of the query output, calculate its DCG and then use it as a denominator from the original DCG:

|Original Rank| Ideal Rank| OriginalDCG| IdealDCG | $\frac{DCG}{IdealDCG}$|
|---|---|---|---|---|
|[3, 2, 3, 0, 1, 2]| [3, 3, 2, 2, 1, 0] | 6.861 | 8.740| 0.785|
# Business Metrics

<img src="images/notebook7_image50.png">
Source: https://medium.com/the-graph/popularity-vs-diversity-c5bc22c253ee

These metrics measure different characteristics we'd want for our RS, not just pure performance/accuracy. This goes to the same way as we do in the machine learning field, as sometimes a 100% accuracy doens't mean the model is good. A good example is a RS that provides obvious association rules between popular items:

                                        **If users takes bread, takes milk as well**

This would be probably a close to perfection RS, as the accuracy of our association rule would be close to 100%. The question is, is that useful? Lets take a look at some metrics that more closely relates to business needs or user experience:

- Coverage  
- Popularity / Novelty – Personalization
- Serendipity
- Diversity

## Coverage

Considering all the products / recommendations in a catalog, what is the percentage of it that a RS can recommend to users? Usually, companies want systems that are able to cover their entire catalog. This metric usually comes with a trade-off between it and precision, where RS usually make a balance between them, *i.e.*, RS with higher coverage can show low precision and high precision system usually covers just a small part of the catalog.

$$PredictionCoverage = \frac{|I_{p}|}{|I|}$$

Depending on the objective, the denominator $I_{p}$ can be defined differently. When comparing top-N algorithms, usualy coverage is calculated by counting the amount of items that appear in a user's top N list. Otherwise, coverage can be obtained by measuring the distinct amount of items that got recommended in the test set divided by the total number of items.

## Popularity / Novelty

Popularity can be used when companies want to optimize total sales numbers. It measures the amount of users that bought a recommended item. A RS with a high popularity metric only recommends items for people where it is really sure that people will like this.

The equation follows in the same way coverage was defined. In business domains, we can evaluate the popularity of different RS recommending different sections of a company's catalog and select the one who provides a better return in revenue, *i.e*, we can divide the number of users who got recommended **and** bought a product by the total number of users who got something recommended.


## Diversity

When recommending only popular items to a user, we can for example only recommend the best 10 super popular items from that e-commerce. This is maybe not what a company wants, as probably users don't need a RS as everyone probably already know what are these items to buy.

A Diversity metric measures how different and diverse are the items that a user gets recommended to. It is usually measured to a top N list and can be calculated by using the item's metadata, such as item category, genre, tags or keywords. If we have only one list, we can just count the distinct number of categories in a list. If we have multiple lists, *i.e.*, multiple recommendations to a set of users, we can calculate the items similarity and optimise for low similarity values.

Lastly, if we are dealing with continuous values, such as ratings lists, we can consider diversity as the spread of ratings in a RS recommendations. In this case, we don't consider a top@N list, but a sample of the RS. A RS with high *diverse* metric predict ratings with a higher standard deviation than a not diverse RS.

**Personal Opinion**: At least in my opinion, the main recommender systems nowadays, with exception Spotify, provides recommendation lists with a low index of Diversity. If I buy a book about machine learning at Amazon, I end up receiving lots of books of the same content of machine learning created by different users, and this is what I don't want want (at least for me).

## Serendipity

As we have discussed before, one flaw from the [Item Item CF](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%202%20Part%20II%20-%20Item%20Item%20Collaborative%20Filtering.ipynb) is its incapacity to provide innovative recommendations that few people know about but that could be a great recommendation to someone. This description is what can be defined as a lack of serendipity. Users will only probably get similar items as they have already bought before.

One of the areas that systems need to be really 'serendipiteous' are music streaming plataforms. Contrary to movie, items or books plataforms, music listening is subject to a diverse of factors such as long and short term preferences, contextualisation and etc.. Being able to adequate to these complex factor is really challenging to a music plataform, but at least Spotify is being able to really provide good surprises with great musics from not so famous bands and that really impress. Spotify's discover weekly is one the examples of serendipity, where people surrender to its recommendations and admit even that the algorithm knows more about their musical taste than themselves.

<img src="images/notebook7_image6.png" width="600">

Another thing about serendipity is the temporal evolution over a user's taste. [Neal Lathia](https://www.coursera.org/learn/recommender-metrics/lecture/twHNp/temporal-evaluation-of-recommenders-interview-with-neal-lathia) studied how users' satisfaction evolved when they got the same recommendation over time. As shown below, he saw that, even when users received good recommendations, if they didn't evolve over time, their satisfaction decreased over time. 

<img src="images/notebook7_image7.png" width="400">

([Kotkov, 2016](https://www.sciencedirect.com/science/article/pii/S0950705116302763)) provides a good overview and challenges on defining and calculating serendipities in RS. In it, he presents two categories of serendipity metrics, component metrics, which measures different components of serendipity, such as novelty or unexpectedness, while full metrics measure serendipity as a whole. Among full metrics, ([Murakami, 2008](https://www.researchgate.net/publication/225121950_Metrics_for_Evaluating_the_Serendipity_of_Recommendation_Lists)) created the following equation:

$$ser_{mur}(u) = \frac{1}{|R_{u}|}\sum_{i\in R_{u}}max(Pr_{u}(i) - Prim_{u}(i),0) * rel_{u}(i)$$

The difference $Pr_{u}(i) - Prim_{u}(i)$ measure the amount of serendipity for each item in the set of possible recommended items. It contains the expected rating of a new RS and it is subtracted by the expected rating from a first/traditional RS. If this difference is smaller than 0, the value is clamped at zero, as we don't want to offer serendipitous items that user probably won't like. This difference is multiplied by the item's relevance, as we just want to recommend surprising items that the we know, or we think we know, the user will like. Lastly, all these differences are averaged so we can have an idea of how much 'surprise' we can provide with this new RS algorithm.
<!-- #endregion -->
