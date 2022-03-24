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

# Recommender Systems - Introduction

Today, recommender systems (RS) are all around us, directing, or inducing, us to make our decisions on whether we buy that shirt, listen to Aerosmith or Coldplay or even suggesting past diseases diagnosis giving a patient's condition.

The main factor that brought the due attention to this was probably the internet. Due to the flood of information we suffer today on media sources and advertisements, people have a lot of struggle to find, or even define, what they want. On the other hand, this amount of data allowed scientist to create plataforms who would analyse all of this and would try to bring only the necessary information that a user would like in a short span of time. This is only a basic, defition of a RS. We can dig a litlle deeper and evaluate other possible ways we can recommend itens to a person and we will end up with the main existing fields:

* **Non-personalised and Stereotyped**: The most basic system. It doesn't evaluate other people's individual opinion, but use summary statistcs from the overall population.  
* **Content Based**: Takes into consideration what a person likes and, given the characteristics of the existing itens, it recommends the most probable itens the user would like.  
* **Collaborative**: Takes into consideration what a person likes and also what other similar people like. In this way, we can give recommendations as, as you and person P likes itens A,B and C, and person P have liked item D also, you could like item D as well. 

This notebook is going to be about the first system, Non personalised and Stereotyped recommendations.

<img src="images/notebook1_image1.jpeg">


# Non Personalised Recommendation

The most basic way to provide recommendations is a non-personalised one. Non-personalised recommendations don't take user's individual preferences nor context into consideration. 

Take for instance a newly create client at Amazon. He wouldn't have bough any item on the marketplace, so Amazon doesn't know what the particular tastes of this new person are, so the best way to start with any possible recommendation that the new customer could like is what other clients, regardless of any their individual tastes, had also bought.  

## Stereotyped Recommendation

One little improvement we can make still on the domain of non-personalised recommendations is to do crude sterotype divisions on the metrics. Basic ratings per sex, city or economical status are some examples of categories in can easily create and can improve the recommendation quality if we believe there are really distinct products who are directed for each of these segments.

<img src="images/notebook1_image2.jpg" width="400">


# Small data analysis
In order to proper understand, let's work wih a table from [Coursera's Recommender System Course 1](https://drive.google.com/file/d/0BxANCLmMqAyIeDJlYWU0SG5YREE/view?usp=sharing) and take a look at one movie matrix and their respective user's ratings. Each row is a user and each column is a movie. Movies that a specific user didn't rate is shown as *Nan*.

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
```

```python
reviews = pd.read_csv('data/non_personalised_stereotyped_rec.csv')
print('Nbr Rows/Users: ' + str(reviews.shape[0]) + ' - Nbr Columns/Movies: ' + str(reviews.shape[1]))
reviews.head()
```

## Non Personalised Statistics

In non-personalised and stereotyped statistics, we can take a look at look at:
  
* Mean Rating: In average, what was the mean value of ratings for that specific item?
* Rating Count: How many people rated that item? - Popularity.
* % Good Rating: Given a threshold, *e.g.* 4, what is the % of people whi gave a good rating for that item?
* Association Rate: Given all people who rated an item A, what % of those also rated other item B? - Association.
* Pearson Correlation: Given the rating a person gave to item A, does it correlates to a rating this same person gave to item B? - Correlation
* Average Rating by Sex: Same as mean rating, but segmented by sex
* % Good Rating by Sex: Same as % good rating, but segmented by sex.

**Lets take a look at each of these statistics and see if they could provide some meaningful recommendations.**

First, lets create a function that receives all the metrics, and return the index of the *n* best statistics:

```python
def return_best_n(statistics, n):
    # statistics: array of size review.shape[1] containing one statistic calculated from the dataset
    # n: number of indices to be returned
    # returns: array of size *n* containing the indices of the best scored statistics
    statistics = pd.DataFrame({'statistic':statistics})
    return statistics.sort_values('statistic', ascending = False).iloc[:n]
```

## Mean Rating

This statistic is intuitive. People tend to review Shawshank Redemption with higher scores, **even though we don't know how many people rated it**

```python
means = reviews.iloc[:,2:].apply(np.mean)
return_best_n(means,3)
```

## Rating Count

Index of popularity, this shows that the movie people most evaluated was Toy Story 3. As an extra, can we have any input of popularity for the high rated movies above?

```python
count = reviews.iloc[:,2:].apply(lambda col : np.sum(~np.isnan(col)))
return_best_n(count,3)
```

### Extra: Popularity Evaluation for high rated movies from 3.2

Considering the size of our small database, the amount of ratings for the popular movies was on a decent amount, *i.e.*, there was't any movie with 1 or 2 ratings only.

```python
movies = ['318: Shawshank Redemption, The (1994)', 
          '260: Star Wars: Episode IV - A New Hope (1977)',
         '541: Blade Runner (1982)']
count.loc[count.index.isin(movies)]
```

## % of Good Ratings (>= 4)

We have two movies that were highly reviewed AND they appear as higher reviews as well. Shawshank Redemption seems to be the golden movie here until now :)

```python
good_count = reviews.iloc[:,2:].apply(lambda col : np.sum(col >=  4)/np.sum(~np.isnan(col)))
return_best_n(good_count,3)
```

## Association Rate (in relation to Toy Story):

In this context, the idea of the association rate is ask the question: "**How likely a person who saw Movie M also ended up watching Toy Story?**". This non-personalised metric can serve a great purpose on the famous "**Because you watched M, here is N (Toy Story in this case)**."

At the end of the function we just remove the first element, because of course Toy Story is the first place with association value of 1.

So it seems in this database people who watch Star Wars also tend to watch Toy Story, weird

```python
def coocurrenceWithToyStory(col, toyStoryRatings):
    x = np.sum((~np.isnan(col)) & (~np.isnan(toyStoryRatings)))/np.sum(~np.isnan(toyStoryRatings))
    return x

toyStoryCol = reviews['1: Toy Story (1995)']

coocurenceToyStory = reviews.iloc[:,2:].apply(lambda col : coocurrenceWithToyStory(col, toyStoryCol))
return_best_n(coocurenceToyStory,4)[1:4]
```

## Pearson Correlation:

The correlation analysis evaluate if the amount of ratings a person gives to one of the movies can provide good hints on what could be the rating on ToyStory **and** the other way around (what Toy Story ratings can indicate on what is going to be the rating on the others movies).

The correlation close to 1 on Shawshank Redemption indicate that people tend to give almost the same rankings they give to Toy Story. A good recommendation for that is "Because you **liked** X you might want to see Y"

```python
def pearson(col, toyStory):
    validRows = np.logical_and(~np.isnan(col),~np.isnan(toyStory))
    return pearsonr(col[validRows], toyStory[validRows])[0]

pearson_corr = reviews.iloc[:,2:].apply(lambda col : pearson(col, toyStoryCol))
return_best_n(pearson_corr,4)[1:4]
```

## Average Rating by Sex:

We already see some trends by checking what womans and mens tend to watch (Pulp Fiction?). But as these are different movies, lets check which movies have the biggest difference on average ratings.

```python
means_w = reviews.loc[reviews['Gender (1 =F, 0=M)'] == 1].iloc[:,2:].apply(np.mean)
print('Average rating by woman')
return_best_n(means_w,3)
```

```python
means_m = reviews.loc[reviews['Gender (1 =F, 0=M)'] == 0].iloc[:,2:].apply(np.mean)
print('Average rating by man')
return_best_n(means_m,3)
```

### Difference in average rating

```python
means_w2 = return_best_n(means_w,len(means_w)).sort_index()
means_m2 = return_best_n(means_m,len(means_m)).sort_index()

means_w2['means_m'] = means_m2.statistic
means_w2['w-m'] = means_w2['statistic'] - means_w2['means_m']
print('Biggest differences in average score Woman - Man')
means_w2.sort_values('w-m',ascending=False)['w-m'][0:3]
```

```python
means_w2['m-w'] = means_w2['means_m'] - means_w2['statistic']
print('Biggest differences in average score Man - Woman')
means_w2.sort_values('m-w',ascending=False)['m-w'][0:3]
```

## % Good Rating by Sex (>= 4):



```python
good_count_w = reviews[reviews['Gender (1 =F, 0=M)'] == 1].iloc[:,2:].apply(lambda col : np.sum(col >=  4)/np.sum(~np.isnan(col)))
return_best_n(good_count_w,3)
```

```python
good_count_m = reviews[reviews['Gender (1 =F, 0=M)'] == 0].iloc[:,2:].apply(lambda col : np.sum(col >=  4)/np.sum(~np.isnan(col)))
return_best_n(good_count_m,3)
```

# Conclusion

As we're going to check next notebooks, non-personalised and stereotyped recommendations are simpler than other techniques and bring some pros and cons:

**Pros:**
- We don't need past data from user and neither his or her taste on particular products
- Statistics are simple and easy to explain

**Cons:**
- In order to provide confident statistics, products should have a reasonable amount of ratings and this also implies having a reasonable amount of users
- Incapable of reaching more fine grained groups. Stereotyped recommendations will only work if there is are explicit products made for each of the segments you've created.
