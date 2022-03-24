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
# Content Based Recommendation (CBR)

When we use Non personalised and stereotyped recommendations, we have the benefit that we don't need to know about our products neither user preferences (apart from explicit metrics). However, in order to provide confident recommendations, we have to have a big number of useful reviews, and account for all the risks when using a evaluation such as 5 starts (see Part I).

CBR are the other way around. In this approach, we don't need a big amount of recommending users, but we need to keep track on the items description and a user profile, which can be used to match with determined items. This difference in approach comes with some pros and cons:
* **Pros:**
* As stated before, we don't need a big number of recommending users in order to provide confident recommendations.
* With the feature above, this means items can be readilly recommended, given we extract the items' characteristics.  


* **Cons:**
* Item descriptions can be a tricky subject. Being able to automatically process and extract these descriptions goes into the fields on Natural Language Processing or even maybe Computer Vision. Besides, very often we rely on subjective qualifiers. When going to a restaurant or hotel, we usually search for tag indications on confort or taste, and this can be very individual to each person. 

One very simple approach for CBR is make use of key words and integrate it with past user review on determined domains. On this notebook we exemplify this approach and some next more 'smart' approaches, such as the TD-IDF statistics, which match a user's taste with documents that contains the key words but that are not commom in all the other documents. Lets go!

<img src="images/notebook3_image1.jpg">

<!-- #endregion -->

# Small data analysis II

Lets work with a small dataset as we did with notebook I - dataset [here](https://d396qusza40orc.cloudfront.net/flex-umntestsite/on-demand_files/Assignment%202.xls).

The main table represent a set a documents and each column contains a possible keywork/characteristic with which we could classify the document. The terms vary from sports to economics and are marked as 1 with the specific documento contains this topic. 

Besides the main table, we also load a review vector for 2 users, which show which document the user marked as 'liked'. These vectors are going to be combined with the document feature vectors in order to create a proper 'Taste vector', *i.e.*, what are the features a user liked and with what weight. In order to simplify the math, the number of stars was reduced to liked the movie (liked = 1, didn't like = 0, didn't review = Nan)

```python
import pandas as pd
import numpy as np
```

```python
reviewsDS = pd.read_csv('data/content_based_filtering.csv')
docsTopics = reviewsDS.iloc[:20,:11]
docsTopics.index = docsTopics.iloc[:,0]
docsTopics.drop('Unnamed: 0', axis = 1, inplace=True)
print('Nbr Rows/Users: ' + str(reviewsDS.shape[0]) + ' - Nbr Columns/Movies: ' + str(reviewsDS.shape[1]))

userReviews = reviewsDS.iloc[:20,[0,14,15]]
userReviews.index = userReviews.iloc[:,0]
userReviews.drop('Unnamed: 0', axis = 1, inplace=True)
docsTopics.head()
```

```python
userReviews.head()
```

## User Profiles

Given what documents did each user like, we can establish ways of how to create a user profile, *i.e.* identify which features is the user more prone to like. 

Each time a user 'liked' a document, we can say he also liked the topics that are contained in the document. By summing up all topics for all the documents the user liked, we can have an idea of what are the user's prefered topics and with what intensity.

```python
# makes the dot product between user reviews and doc topics
def getTasteVector(userCol, docsTopics):
    return docsTopics.apply(lambda docCol : np.dot(userCol, docCol))

userTastes = userReviews.apply(lambda col : getTasteVector(col, docsTopics))
userTastes
```

## Document Score Predictions

Now that we have the users' taste vector, we can make a dot product of it by each document and have and idea of which document would the user like more.

```python
def getDocScores(userTaste, docsTopics):
    return docsTopics.apply(lambda docTopic: np.dot(userTaste, docTopic),axis=1)

docScores = userTastes.apply(lambda userTaste: getDocScores(userTaste, docsTopics))
docScores.head()
```

```python
print('Predicted prefered docs for User 1 by content')
docScores.sort_values('User 1', ascending=False)['User 1'].head()
```

```python
print('Predicted prefered docs for User 2 by content')
docScores.sort_values('User 2', ascending=False)['User 2'].head()
```

## Normalised Weights

In these calculations, an article who had marked as having All topics would probably be ranked first, as it would fit for almost users, regardless of their taste.

In order to reduce this impact, we must normalise these scores, by considering how many terms this document covers. So, if a user's taste match with a document having only 1 topic, whis would probably be more important than matching with a document with all 10 topics present.

One different normalisation we're going to do here is to divide the boolean topic indicator from the documents by $\sqrt{\sum topics = True}$, *i.e.*, instead of dividing the boolean flag (value 1) of a document by $\frac{1}{\sum topics = True}$, we divide it only by $\frac{1}{\sqrt{\sum topics = True}}$. This different normalisation still punish documents with higher number of covered topics, but punish higher number of topics everytime less than if using a normal linear normalisation. We can see on the plot below what is going to be the normalisation factor by the number of total covered topics in each document.

```python
import matplotlib.pyplot as plt
plt.plot(np.arange(1,100,1), np.sqrt(np.arange(1,100,1)))
plt.title('Total number of topics versus Normalisation factor')
plt.xlabel('Number of total topics')
plt.ylabel('Normalisation factor');
```

Lets create a new normalised topic-document score by this new rule:

```python
norm_docsTopics = docsTopics.apply(lambda doc: doc/np.sqrt(np.sum(doc)),axis=1)
norm_docsTopics.head()
```

Lets calculate now the new users' taste vectors and document score predictions:

```python
norm_userTastes = userReviews.apply(lambda col : getTasteVector(col, norm_docsTopics))
print("New normalised users' taste vectors")
norm_userTastes
```

```python
norm_docScores = norm_userTastes.apply(lambda norm_userTastes: getDocScores(norm_userTastes, norm_docsTopics))
norm_docScores.head()
```

```python
print('Predicted prefered docs for User 1 by content')
norm_docScores.sort_values('User 1', ascending=False)['User 1'].head()
```

```python
print('Predicted prefered docs for User 2 by content')
norm_docScores.sort_values('User 2', ascending=False)['User 2'].head()
```

An interesting change in this normalisation change is on the second position of User 1 list. Before, doc1 and doc12 were tied on 2nd place but now doc1 went to 5th position. If we take a look on the number of total terms in documents 1 and 12: 

```python
docsTopics.iloc[[0,11],:].sum(axis=1)
```

We see that doc1 has more terms than doc12, making it less valuable when calculation the match between them and the user's taste vector. In the end, doc6 got before doc1 and doc12 when comparing the ranking before and after normalisation. Lets take a look a his number of terms:

```python
docsTopics.iloc[[5],:].sum(axis=1)
```

So we see that doc6 had only 2 terms and probably these two terms matched also with good numbers on the user's taste vector:

```python
print(norm_docsTopics.iloc[5,:])
print(norm_userTastes)
```

## TD-IDF

A more modern approach that has given good results in different recommender systems, and in Information Retrieval in general, is to take into account not only the document relevance, *i.e.*, if it covers too much or few topics, but also consider the topic relevance by itself.

By using this, not only it considers if the document *x* covers lots of topics, but also if it covers important topics, *i.e.* only few other documents also cover this topic in specific.

The final score for this doc-user matching is going to be:   

$$docVector * IDF * tasteVector$$
  
with $docVector$ being the equivalent of the TF part of the TF-IDF formula and it's finally multiplied by the taste vector in order to consider the user's taste in the process.

We already have the normalised doc-topics weights, so lets calculate the IDF of each term. We're going to divide each value from each topic by in how many other documents that same topic appears.

```python
IDF = norm_docsTopics.apply(lambda col: 1/np.sum(col != 0))
IDF
```

Lets now calculate the final document-user match scores

```python
def dot_product3(par1, par2, par3):
    result = 0
    for i in range(len(par1)):
        result += par1[i] * par2[i] * par3[i]
    return result

def TF_IDF_scores(norm_docsTopics, IDF, userTasteVector):   
    return norm_docsTopics.apply(lambda row: dot_product3(row, IDF, userTasteVector), axis=1)

finalScores = norm_userTastes.apply(lambda userVector: TF_IDF_scores(norm_docsTopics, IDF, userVector))
finalScores.head()
```

```python
print('Predicted prefered docs for User 1 by content')
finalScores.sort_values('User 1', ascending=False)['User 1'].head()
```

```python
print('Predicted prefered docs for User 2 by content')
finalScores.sort_values('User 2', ascending=False)['User 2'].head()
```

## Final Comparison

Lets compare all 3 types of content based filtering and their resulting ranks:

```python
print('Final Comparison for User 1')
m1 = docScores.sort_values('User 1', ascending=False)['User 1'].head(10)
m2 = norm_docScores.sort_values('User 1', ascending=False)['User 1'].head(10)
m3 = finalScores.sort_values('User 1', ascending=False)['User 1'].head(10)
finalDFUser1 = pd.DataFrame({'Normal rank':m1.index, 'Normal score':m1.values,
                            'Normalised rank':m2.index, 'Normalised score':m2.values,
                            'IDF rank':m3.index, 'IDF score':m3.values})
finalDFUser1 = finalDFUser1[['Normal rank','Normal score',
                            'Normalised rank', 'Normalised score',
                            'IDF rank', 'IDF score']]
finalDFUser1
```

```python
print('Final Comparison for User 2')
m1 = docScores.sort_values('User 2', ascending=False)['User 2'].head(10)
m2 = norm_docScores.sort_values('User 2', ascending=False)['User 2'].head(10)
m3 = finalScores.sort_values('User 2', ascending=False)['User 2'].head(10)
finalDFUser1 = pd.DataFrame({'Normal rank':m1.index, 'Normal score':m1.values,
                            'Normalised rank':m2.index, 'Normalised score':m2.values,
                            'IDF rank':m3.index, 'IDF score':m3.values})
finalDFUser1 = finalDFUser1[['Normal rank','Normal score',
                            'Normalised rank', 'Normalised score',
                            'IDF rank', 'IDF score']]
finalDFUser1
```

On user 1, we can see two documents who changed place in terms of importance after we used the different algorithms. Doc 6 was in 5th place using the basic (first) algorithm but it changed to a 2nd place after considering document and term importances.  The interpretation here is that probably the document covers only a few topics, but besides matching well with the user's taste, it also covers very specific topics, that few other documents cover.

On the inverted direction, doc1 had a good position using the first filter, but it wen down after we normalised and used IDF in the next filterings. This situation shows a document with a reasonable number of topics, which artifically create a 'match' with the user's taste vector in the beggining but, after accounting for document specificity and term importance, lots of the values for the topics inside this documents probably went down, bringing it to a lower rank in overall.
