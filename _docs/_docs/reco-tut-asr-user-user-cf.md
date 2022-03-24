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

# Collaborative Filtering (CF)



<!-- #region -->
In month 1, we learn about some commom techniques to recommend items to a user.  


[The 1st notebook](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%201%20Part%20I%20-%20Non%20Personalised%20and%20Stereotyped%20Recommendation.ipynb) presented non-personalised and stereotyped recommendations, which only took averages from the population's avaliations (ratings) in order to predict and present the most popular items.


[The 2nd notebook](https://github.com/caiomiyashiro/RecommenderSystemsNotebooks/blob/master/Month%201%20Part%20III%20-%20Content%20Based%20Recommendation.ipynb) introduced a little of personalisation, where we created a user's taste vector and used it to 'match' the user array with other documents.
    
This notebook introduce the concept of **collaborative filtering**, a recommendation strategy to find and match similar entities. I say entities because we have two different variants on collaborative filtering: 


* User User CF: First CF technique created, the User User CF only takes into consideration only the user's past behaviour, *i.e.*, its ratings, and nothing about the items's characteristics. The ideia is pretty simple: If two users $U_{1}$ and $U_{2}$ have liked items $I_{a}$ and $I_{b}$, but user $U_{2}$ liked an item $I_{c}$ that $U_{1}$ hasn't seen yet. We infer that item $I_{c}$ would be a good recommendation for $U_{1}$. The following picture gives a good representation about it.

<img src="images/notebook4_image1.png" width="600">

* Item Item CF: The User User CF has some drawbacks, which we are going to talk about later. Because of these drawbacks, a more efficient approach was created, the Item Item CF. This technique doesn't take into consideration the users' similarities but only on item similarities. With this, new item predictions for a user $U$ can be easily calculated taking into account the ratings the user gave for similar items. This approach is going to be presented in the next notebook.

# Example Dataset

For the next explanations in Nearest Neighboors for CF we're going to use the [dataset](https://drive.google.com/file/d/0BxANCLmMqAyIQ0ZWSy1KNUI4RWc/view?usp=sharing) provided from the Coursera Specialisation in Recommender Systems, specifically the data from the assignment on User User CF in [course 2](https://www.coursera.org/learn/collaborative-filtering) from the specialisation: 

The dataset is a matrix with size 100 movies x 25 users and each cell $c_{m,u}$ contains the rating user $u$ gave to movie $m$. If user $u$ didn't rate movie $m$, the cell is empty. As the float values were stored with commas and consequently were being casted as strings, I had to process it a little bit to replace the commas for dots and then convert the column to floats


<!-- #endregion -->

```python
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
```

```python
df = pd.read_csv('data/User-User Collaborative Filtering - movie-row.csv', dtype=object, index_col=0)

# replace commas for dots and convert previous string column into float
def processCol(col):
    return col.astype(str).apply(lambda val: val.replace(',','.')).astype(float)
df = df.apply(processCol)

print('Dataset shape: ' + str(df.shape))
df.head()
```

# Nearest Neighboors for CF

The approach for doing CF with nearest neighboors is to compare what you want to be matched with other similiar entities. With this, we have to define two things: 
  
* One, in order to bring the most similar items or other customers with similar tastes, we must limit the amount of entities we compare it with.
* Second, when doing predictions for an unseen data, we must match it with neighboors who have already rated the data we want.
  
With these two constraints, we see we have a trade off when deciding the amount of neighboors. If the number of neighboors is set to a too low value, the chances is that we end up with a lot of entities not having reviewed the same thing, and we end up not being able to provide confident predictions for our objective. If we set the bar too high, we will include too many different neighboors in our comparison, with different tastes than the user we want predict recommendations to.

([Herlocker et all, 2002](https://grouplens.org/site-content/uploads/evaluating-TOIS-20041.pdf)) made a feel experiments with different configurations for User User CF and discovered that, for most commercial applications used nowadays, an optimal number of neighboors to consider is between 20 and 30.
  
In short, we have the following steps in a Nearest Neighboor CF:
    - Starting from the User x Movie matrix, calculate the similarity between all users.
    - For each user, filter the K most similar neighboors
    - Predict new rating for user based on its nearest neighboors

## Similarity Function

If we want to compare the similarity in terms of ratings between two users $u_{1}$ and $u_{2}$, we have as input to the similarty function, two arrays, containing all reviews that each user made to each item, and blank values when the user didn't rate that specific item.  

When we want to compare the similarity between two vectors, we have a few options, such as:

* [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)  
* Mean centered euclidean distance  
* [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* Mean centered cosine similarity  
  
In the next notebook, I go through the interpretation and subtle differences of using each one of these metrics, but for now, lets take the studies already perfomed by ([Herlocker et all, 2002](https://grouplens.org/site-content/uploads/evaluating-TOIS-20041.pdf)) with similarity metrics and start with the pearson correlation, as it has performed better than the other metrics (apart from the mean centered cosine similarity) in terms of finding good user neighboors to get data for predictions.

## Calculating User User Similarity with Pearson:

```python
df_corr = df.corr(method = 'pearson')
df_corr.head()
```

## Select K Nearest Neighboors for each User

Here, before searching for the nearest neighboors, we filter out the correlation of the user by itself, which is always going to be 1.

```python
def findKNearestUsers(userCorrCol, k = 5):
    return userCorrCol[userCorrCol.index != userCorrCol.name].nlargest(n = k).index.tolist()

kNeighboors = df_corr.apply(lambda col: findKNearestUsers(col))
kNeighboors
```

## Predicting New Rating

Now that we have all our most similar users, how can we predict unseen items to a user, *i.e.* predict the rating value for a item an user hasn't evaluated yet?

One way to predict it is to select all nearest neighboors $subset(N_{u})$ that have rated a specific item $i$ of interest and average them out to have a prediction. Of course, we want to consider one thing:

As we select by the K nearest neighboors, we have different levels of similarity between users. As so, we want to users with bigger similarity to have more impact, *i.e.*, weight, in the average calculation. In one extreme, if there were only 5 users in the system sharing two product reviews and one of the them is completely unrelated to our user of interest, even though he is a 'neighboor', we want him to have a minimum weight in our average calculation.
  
The way to do this is just an weighted average:

$$\frac{\sum_{n=1}^{k} r_{n}w_{n}}{\sum_{n=1}^{k} w_{n}}$$
  
The $r$ are the ratings of the neighboors **rated the item of interest** and $w$ are they similarity with the user of interest. The traditional arithmetic average is the weighted average with all $w$ = 1.  

Finally, just to make things easier, lets calculate the rating prediction for all possible movies. Of course, in a real scenario, you wouldn't calculate it as it would be a waste of processing power.


```python
%%time
def calculatePredictionsUser(kNeighboors, user_correlations, df):
    
    def calculatePredictionsUserMovie(kNeighboors, user_correlations, movieRowRatings): 
        hasRatedMovie = ~np.isnan(movieRowRatings)
        if(np.sum(hasRatedMovie) != 0): # only return value if there was at least one neighboor who also rated that movie
            return np.dot(movieRowRatings.loc[hasRatedMovie], user_correlations.loc[hasRatedMovie])/np.sum(user_correlations[hasRatedMovie])
        else:
            return np.nan
        
    # looking at one user, apply function for each row = movie and predict rating for that movie
    return df.apply(lambda movieRow: calculatePredictionsUserMovie(kNeighboors, user_correlations, movieRow[kNeighboors]), axis=1)
    

####### Starting process point
# call function sending user's neighboors, neighboors similarity and movie ratings df     
moviePredictions = df.apply(lambda userRatings: calculatePredictionsUser(kNeighboors[userRatings.name], 
                                                      df_corr[userRatings.name][kNeighboors[userRatings.name]],
                                                      df))
print("Taking a look at an example user's predictions and 10 best movies recommended by highest score")
moviePredictions['3867'].sort_values(ascending=False).head(10)

```

## Mean Normalised Weighted Average

The pearson correlation evaluate **how** linear dependent the two users are, and **not how much**. This implies that pearson rating between users $U_{1}$ = [3,3,3,3,3] and $U_{2}$ = [4,4,4,4,4] and between users $U_{3}$ = [2,2,2,2,2] and $U_{4}$ = [5,5,5,5,5] would be the same. In short, we can't average between users because the Pearson correlation doesn't take into account the scale variability between users, *i.e.* users who vote 3 for a movie he though it was good against another user who votes 5 for the same criteria.

In order to account for this variability, we can improve our previous weighted average and consider how many points each neighboor deviated **from the average** when calculating our weighted average. Finally, as we are making the weighted average of how much each user deviated from the average, we must input this value to the user of interest own average value:

$$\bar{r_{u}} + \frac{\sum_{n=1}^{k} (r_{n} - \bar{r_{n}})w_{n}}{\sum_{n=1}^{k} w_{n}}$$

We took the same function as above, but added two extra parameters:
- $userMeanRating$: mean average ratings for a specific user
- $neighboorsMeanRating$: mean average rating for all the nearest neighboors for a specific user

```python
def calculatePredictionsUserNorm(kNeighboors, user_correlations, userMeanRating, neighboorsMeanRating, df):
    
    def calculatePredictionsUserMovieNorm(kNeighboors, user_correlations, userMeanRating, neighboorsMeanRating, movieRowRatings): 
        hasRatedMovie = ~np.isnan(movieRowRatings)
        if(np.sum(hasRatedMovie) != 0): # only return value if there was at least one neighboor who also rated that movie
            userRatingDeviation = movieRowRatings.loc[hasRatedMovie] - neighboorsMeanRating.loc[hasRatedMovie]
            numerator = np.dot(userRatingDeviation, user_correlations.loc[hasRatedMovie])
            return userMeanRating + numerator/np.sum(user_correlations[hasRatedMovie])
        else:
            return np.nan
        
    # looking at one user, apply function for each row = movie and predict rating for that movieprint
    return df.apply(lambda movieRow: calculatePredictionsUserMovieNorm(kNeighboors, 
                                                                       user_correlations,
                                                                       userMeanRating,
                                                                       neighboorsMeanRating,
                                                                       movieRow[kNeighboors]), axis=1)
    

####### Starting process point

meanRatingPerUser = df.apply(np.mean)

# call function sending user's neighboors, neighboors similarity and movie ratings df     
moviePredictionsNorm = df.apply(lambda userRatings: 
                                          calculatePredictionsUserNorm(kNeighboors[userRatings.name], 
                                                      df_corr[userRatings.name][kNeighboors[userRatings.name]],
                                                      np.mean(userRatings),                 
                                                      meanRatingPerUser[kNeighboors[userRatings.name]],
                                                      df))
print("Taking a look at an example user's predictions and 10 best movies recommended by highest score")
moviePredictionsNorm['3867'].sort_values(ascending=False).head(10)

```

## Comparison Between Approaches

Lets compare both approaches and see any possible difference:

```python
finalMovie = pd.DataFrame()

finalMovie['TitleNotNorm'] = moviePredictions['3867'].sort_values(ascending=False).head(10).index
finalMovie['withoutNormalisation'] = moviePredictions['3867'].sort_values(ascending=False).head(10).values
finalMovie['TitleNorm'] = moviePredictionsNorm['3867'].sort_values(ascending=False).head(10).index
finalMovie['normalised'] = moviePredictionsNorm['3867'].sort_values(ascending=False).head(10).values
finalMovie
```

### First weird result - We had a normalised score > 5 for the first place:

In terms of normalised score, this can happen as the user we are evaluating already rates movies with quite high average value and, when we add the average deviation from the mean from other users, that might or might not be in the same scale, we might end up surpassing the conceptual threshold of 5 stars, let's confirm it quickly:

```python
print('Average score for user 3867: ' + str(df[['3867']].apply(np.mean).values[0]))

#########
neighboors = kNeighboors['3867']
weights = df_corr[['3867']].loc[neighboors]
means = df[neighboors].apply(np.mean)
ratings = df.loc[['1891: Star Wars: Episode V - The Empire Strikes Back (1980)']][neighboors]
existingRatings = list(~(ratings.apply(np.isnan).values[0]))

# weighted average deviation
denominator = np.dot(ratings.loc[:,existingRatings] - means[existingRatings], weights[existingRatings]).tolist()[0]
avgWeightedDeviation = (denominator/np.sum(weights[existingRatings])).values[0]

print('How much from the mean the nearest neighboors deviated from their mean in ' +
      'Star Wars: Episode V - The Empire Strike: ' + str(avgWeightedDeviation))
```

So, user 3687 didn't have a really high mean, but it got neighboors that had smaller average review score and Star wars got scores way above their traditional average, bumping the predicted score for user 3687 really high, even higher than the allowed 5 start points.

### Where is Fargo in the Non-Normalised Scores?

In the normalised scores, Fargo appears in the 4th place, but it didn't make it to the top 10 at the non normalised scores. What could've happened?

```python
print('Average score for user 3867: ' + str(df[['3867']].apply(np.mean).values[0]))

#########
neighboors = kNeighboors['3867']
weights = df_corr[['3867']].loc[neighboors]
means = df[neighboors].apply(np.mean)
ratings = df.loc[['275: Fargo (1996)']][neighboors]
existingRatings = list(~(ratings.apply(np.isnan).values[0]))

print('How many neighboors have rated this movie: ' + str(np.sum(existingRatings)))
print('My neighboors ratings: ' + str(ratings.loc[:,existingRatings].values[0][0]))

weightedAvg = float((ratings.loc[:,existingRatings].values * weights[existingRatings]).iloc[:,0].values[0]/np.sum(weights[existingRatings]))
print('--- Final score for normal weighted average: ' + str(weightedAvg))
# weighted average deviation
denominator = np.dot(ratings.loc[:,existingRatings] - means[existingRatings], weights[existingRatings]).tolist()[0]
avgWeightedDeviation = (denominator/np.sum(weights[existingRatings])).values[0]

print('\nHow much from the mean the nearest neighboors deviated from their mean in ' +
      'Fargo (1996): ' + str(avgWeightedDeviation))
print('--- Final score for Normalised weighted average: ' + str(df[['3867']].apply(np.mean).values[0] + avgWeightedDeviation))

```

This was a bit harder to calculate, as we wanted to compare why non normalised and normalised calculations created so different scores for Fargo. As we see, on the non normalised score, we had just 1 neighboor who had seen Fargo **and** this neighboor reviewed it more than 1 point above its average. So the score was good but it didn't show as good at the non normalised score because it was a good score only for the neighboor, not in the total 5 point scale. 

# Final Considerations on User User CF

User User CF brings an step forward from the non personalised and content based recommenders. With it, we bring personalised recommendations but without having the challenge on how to characterise and maintain the item set in your inventory. However, User User CF still have some problems:

- User User CF does **not** scale. This is the main problem for the User User CF and it is caused by two factors.
        1. Calculating the User User similarity matrix: For a 100 x 25 matrix, my notebook (I7 with 16 GB RAM) already took a few seconds to process it. When we scale this size by million times, as in a real e-commerce store, this become unfeasible. Even worse when new users are registering and reviewing products every day.
        2. User similarity doesn't hold for a long time. Users's taste is a research area on its own but in short we can summarise that the actual users' taste for online products can change quite quick. If the service wants to account for the costumer's short time interests, he should recalculate the entire User User matrix.
        
As we are going to see in the next notebook, Item Item CF adjust (a little) for these disadvantages above. Considering that Item Item similarity are more long term stable than a User User matrix, the Item Item CF allow us to calculate the similarity matrix offline and less often than User User.
