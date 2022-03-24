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

<!-- #region id="x3wreg5wRPZx" -->
# Double domain recommendations on ML-1m
<!-- #endregion -->

<!-- #region id="F5_C2X-YMGgJ" -->
## Introduction
<!-- #endregion -->

<!-- #region id="XqPMKCEDNKCA" -->
In this tutorial, we are training DCDCSR model on MovieLens 1m dataset. Data has been divided into four parts D1,D2,D3 and D4. D1 and D2 have users common. D1 and D3 have items common. D1 and D4 have no user and no item in common. We test the model on the testing part of the dataset i.e., testing set from 10% dataset of D1 and calculated the MAE, RMSD, Precision and Recall values. Same is repeated with every dataset. Case 1 :- cross domain recommendation D1 is Target Domain and D4 is Source Domain. Case 2 :- cross domain recommendation D2 is Target Domain and D3 is Source Domain. Case 3 :- cross domain recommendation D3 is Target Domain and D2 is Source Domain. Case 4 :- cross domain recommendation D4 is Target Domain and D1 is Source Domain.
<!-- #endregion -->

<!-- #region id="dhMqCRtpNIDC" -->
### Model Architecture
<!-- #endregion -->

<!-- #region id="1GcY2Sv-zX_j" -->
<p><center><img src='_images/T490340_1.png'></center></p>
<!-- #endregion -->

<!-- #region id="VMkXjPglL_qt" -->
### Training Procedure
<!-- #endregion -->

<!-- #region id="i5CCN7M0za7c" -->
<p><center><img src='_images/T490340_2.png'></center></p>
<!-- #endregion -->

<!-- #region id="2OdUJNWyME7C" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Uvx4xNcmMokj" executionInfo={"status": "ok", "timestamp": 1635748444256, "user_tz": -330, "elapsed": 2284, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="16ce426c-7992-4c11-ae8a-afd89ca167e1"
!git clone https://github.com/Worm4047/crossDomainRecommenderSystem.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="rtH9RTdUNGmF" executionInfo={"status": "ok", "timestamp": 1635748446763, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="819bfc37-b705-47c5-ea88-b1d7ccb26df7"
%cd crossDomainRecommenderSystem
```

<!-- #region id="QnEYZL5YPdj9" -->
## Simple recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wweh3EoUNH7F" executionInfo={"status": "ok", "timestamp": 1635749439566, "user_tz": -330, "elapsed": 25116, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="39aea72c-78de-4afd-863b-129b9209d7bb"
from math import *


def sim_distance(prefs,person1,person2):
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: si[item]=1
    if len(si)==0: return 0
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])

    return 1/(1+sum_of_squares)


def sim_pearson(prefs,p1,p2):
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: si[item]=1
    if len(si)==0: return 0

    # Sum calculations
    n=len(si)

    # Sums of all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])

    # Sums of the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])	

    # Sum of the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])

    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0

    r=num/den

    return r


# Returns the best matches for person from the prefs dictionary. 
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
        scores=[(similarity(prefs,person,other),other) 
                        for other in prefs if other!=person]
        scores.sort()
        scores.reverse()
        return scores[0:n]

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        # don't compare me to myself
        if other==person: continue
        sim=similarity(prefs,person,other)

        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim

    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
        
        # Flip item and person
        result[item][person]=prefs[person][item]
    return result


def calculateSimilarItems(prefs,n=10):
    # Create a dictionary of items showing which other items they
    # are most similar to.
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
        result[item]=scores
    return result


def getRecommendedItems(prefs,itemMatch,user):
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):

        # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity

    # Divide each total score by total weighting to get an average
    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]

    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings


def loadMovieLens(path='movielens',file='/u1.base'):
    # Get movie titles
    movies={}
    for line in open(path+'/u.item', encoding="ISO-8859-1"):
        (id,title)=line.split('|')[0:2]
        movies[id]=title

    # Load data
    prefs={}
    for line in open(path+file, encoding="ISO-8859-1"):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movieid]=float(rating)
    return prefs


if __name__=='__main__':
    trainPrefs = loadMovieLens()
    testPrefs = loadMovieLens(file='/u1.test')
    movies={}
    for line in open('movielens/u.item', encoding="ISO-8859-1"):
        (id,title)=line.split('|')[0:2]
        movies[id]=title
    for user in testPrefs:
        pred = getRecommendations(trainPrefs,user)
        count=-1
        preds={}
        for rating,item in pred:
            preds[item]=rating
            # print(movies[item],rating,item)
        accuracies=[]
        for movie in testPrefs[user]:
            if not movie in preds:continue 
            actualRating = testPrefs[user][movie]
            predcitedRating = preds[movie]
            diff = fabs(fabs(predcitedRating) - fabs(actualRating))
            # print(predcitedRating,actualRating,diff)
            accu = float(diff)/actualRating
            if accu > 1:
                continue
            accuracies.append(1 - accu)
        print((sum(accuracies)/len(accuracies))*100)
```

<!-- #region id="_V9gipZRN3jT" -->
## Double-domain recommendations
<!-- #endregion -->

```python id="hnc-kgJvP_kh"
from math import *


def sim_distance(prefs,person1,person2):
  si={}
  for item in prefs[person1]: 
    if item in prefs[person2]: si[item]=1
  if len(si)==0: return 0
  sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in prefs[person1] if item in prefs[person2]])
  return 1/(1+sum_of_squares)

  
def sim_pearson(prefs,p1,p2):
  si={}
  for item in prefs[p1]: 
    if item in prefs[p2]: si[item]=1
  if len(si)==0: return 0

  # Sum calculations
  n=len(si)
  
  # Sums of all the preferences
  sum1=sum([prefs[p1][it] for it in si])
  sum2=sum([prefs[p2][it] for it in si])
  
  # Sums of the squares
  sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
  sum2Sq=sum([pow(prefs[p2][it],2) for it in si])	
  
  # Sum of the products
  pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
  
  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  if den==0: return 0

  r=num/den

  return r


# Returns the best matches for person from the prefs dictionary. 
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
  scores=[(similarity(prefs,person,other),other) 
                  for other in prefs if other!=person]
  scores.sort()
  scores.reverse()
  return scores[0:n]


# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(domain1,domain2,person,similarity=sim_pearson):
  totals={}
  simSums={}
  for other in domain2:
    # don't compare me to myself
    if other==person: continue
    sim=similarity(domain2,person,other)

    # ignore scores of zero or lower
    if sim<=0: continue
    # print other,sim
    for item in domain1[other]:
	    
      # only score movies I haven't seen yet
      if item not in domain1[person] or domain1[person][item]==0:
        # Similarity * Score
        totals.setdefault(item,0)
        totals[item]+=domain1[other][item]*sim
        # Sum of similarities
        simSums.setdefault(item,0)
        simSums[item]+=sim

  # Create the normalized list
  rankings=[(total/simSums[item],item) for item,total in totals.items()]

  # Return the sorted list
  rankings.sort()
  rankings.reverse()
  return rankings


def loadMovieLens(path='movielens',file='/u1.base'):
  # Get movie titles
  movies={}
  for line in open(path+'/u.item', encoding="ISO-8859-1"):
    (id,title)=line.split('|')[0:2]
    movies[id]=title
  
  # Load data
  prefs={}
  for line in open(path+file, encoding="ISO-8859-1"):
    (user,movieid,rating,ts)=line.split('\t')
    prefs.setdefault(user,{})
    prefs[user][movieid]=float(rating)
  return prefs
```

```python colab={"base_uri": "https://localhost:8080/"} id="hms2WFvnPhvA" executionInfo={"status": "ok", "timestamp": 1635749403890, "user_tz": -330, "elapsed": 28054, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="937b83b5-9125-46ee-adaf-bb54bd3fd85b"
import csv
from sklearn.model_selection import train_test_split


if __name__=='__main__':

    file = open('movielens/u.data', encoding="ISO-8859-1")
    data = csv.reader(file, delimiter='\t')

    t = [row for row in data]

    fDomainDict,sDomainDict={},{}
    fDomainDict=loadMovieLens('movielens','/u1.base')
    sDomainDict=loadMovieLens('movielens','/u2.base')
    fDomainTest =loadMovieLens('movielens','/u1.test')

    sumAccuracy=0
    lenCount=0
    for user in fDomainTest:
        pred = getRecommendations(fDomainDict,sDomainDict,user)
        count=-1
        preds={}
        for rating,item in pred:
            preds[item]=rating
            # print(movies[item],rating,item)
        accuracies=[]
        for movie in fDomainTest[user]:
            if not movie in preds:continue 
            actualRating = fDomainTest[user][movie]
            predcitedRating = preds[movie]
            accu = fabs((predcitedRating - actualRating)/actualRating)
            if accu > 1:
                continue
            accuracies.append(1 - accu)
        lenCount+=1
        print((sum(accuracies)/len(accuracies))*100)
        sumAccuracy+=(sum(accuracies)/len(accuracies))*100
        
    print('Average Accuracy')
    print(float(sumAccuracy)/lenCount)
```

<!-- #region id="Lcrj3zyTLOcV" -->
## Citations

A Deep Framework for Cross-Domain and Cross-System Recommendations. Feng Zhu, Yan Wang, Chaochao Chen, Guanfeng Liu, Mehmet Orgun, Jia Wu. 2018. IJCAI. [https://www.ijcai.org/proceedings/2018/0516.pdf](https://www.ijcai.org/proceedings/2018/0516.pdf)
<!-- #endregion -->
