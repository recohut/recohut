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

<!-- #region id="4s6T6G1a45Jr" -->
# Netflix Movie Recommendation System
<!-- #endregion -->

<!-- #region id="t1vW29Np45KO" -->
## Business Problem

<p>Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.</p>

<p>Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.</p>

<p>Credits: https://www.netflixprize.com/rules.html</p>

## Problem Statement
<p>Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is 10% better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)</p>

## Sources
* https://www.netflixprize.com/rules.html
* https://www.kaggle.com/netflix-inc/netflix-prize-data
* Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 (very nice blog)
* surprise library: http://surpriselib.com/ (we use many models from this library)
* surprise library doc: http://surprise.readthedocs.io/en/stable/getting_started.html (we use many models from this library)
* installing surprise: https://github.com/NicolasHug/Surprise#installation
* Research paper: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf (most of our work was inspired by this paper)
* SVD Decomposition : https://www.youtube.com/watch?v=P5mlg91as1c

<p><b>Real world/Business Objectives and constraints</b></p> 

<p><b>Objectives:</b></p>
1.	Predict the rating that a user would give to a movie that he has not yet rated.<br>
2.	Minimize the difference between predicted and actual rating (RMSE and MAPE).

<p><b>Constraints:</b></p>
1.	Some form of interpretability.
2.	There is no low latency requirement as the recommended movies can be precomputed earlier.

<p><b>Type of Data:</b></p>
* There are 17770 unique movie IDs.
* There are 480189 unique user IDs.
* There are ratings. Ratings are on a five star (integral) scale from 1 to 5.
<!-- #endregion -->

<!-- #region id="gkjcGNJU45KS" -->
<p><b>Data Overview</b></p>
<b>Data files :</b><br>

1. combined_data_1.txt
2. combined_data_2.txt
3. combined_data_3.txt
4. combined_data_4.txt
5. movie_titles.csv
  
The first line of each file [combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt] contains the movie id followed by a colon. Each subsequent line in the file corresponds to a customerID, rating from a customer and its date.
<!-- #endregion -->

<!-- #region id="vgWLbzH445KT" -->
<p style = "font-size: 22px"><b>Example Data Point</b></p>
<pre>
1:
1488844,3,2005-09-06
822109,5,2005-05-13
885013,4,2005-10-19
30878,4,2005-12-26
823519,3,2004-05-03
893988,3,2005-11-17
124105,4,2004-08-05
1248029,3,2004-04-22
1842128,4,2004-05-09
2238063,3,2005-05-11
1503895,4,2005-05-19
2207774,5,2005-06-06
2590061,3,2004-08-12
2442,3,2004-04-14
543865,4,2004-05-28
1209119,4,2004-03-23
804919,4,2004-06-10
1086807,3,2004-12-28
1711859,4,2005-05-08
372233,5,2005-11-23
1080361,3,2005-03-28
1245640,3,2005-12-19
558634,4,2004-12-14
2165002,4,2004-04-06
1181550,3,2004-02-01
1227322,4,2004-02-06
427928,4,2004-02-26
814701,5,2005-09-29
808731,4,2005-10-31
662870,5,2005-08-24
337541,5,2005-03-23
786312,3,2004-11-16
1133214,4,2004-03-07
1537427,4,2004-03-29
1209954,5,2005-05-09
2381599,3,2005-09-12
525356,2,2004-07-11
1910569,4,2004-04-12
2263586,4,2004-08-20
2421815,2,2004-02-26
1009622,1,2005-01-19
1481961,2,2005-05-24
401047,4,2005-06-03
2179073,3,2004-08-29
1434636,3,2004-05-01
93986,5,2005-10-06
1308744,5,2005-10-29
2647871,4,2005-12-30
1905581,5,2005-08-16
2508819,3,2004-05-18
1578279,1,2005-05-19
1159695,4,2005-02-15
2588432,3,2005-03-31
2423091,3,2005-09-12
470232,4,2004-04-08
2148699,2,2004-06-05
1342007,3,2004-07-16
466135,4,2004-07-13
2472440,3,2005-08-13
1283744,3,2004-04-17
1927580,4,2004-11-08
716874,5,2005-05-06
4326,4,2005-10-29
</pre>
<!-- #endregion -->

<!-- #region id="R33dJ2KJ45KU" -->
## Mapping the real world problem to a Machine Learning Problem
<!-- #endregion -->

<!-- #region id="oYaS3lQP45KV" -->
<p><b>Type of Machine Learning Problem</b></p>
<p>
For a given movie and user we need to predict the rating would be given by him/her to the movie. 
The given problem is a Recommendation problem 
It can also seen as a Regression problem 
</p>
<!-- #endregion -->

<!-- #region id="pnP8uLFQ45KV" -->
<p><b>Performance metric</b></p>
1. Mean Absolute Percentage Error
2. Root Mean Square Error

<!-- #endregion -->

<!-- #region id="2hI_tE_c45KW" -->
<p><b>Machine Learning Objective and Constraints</b></p>
1. Try to Minimize RMSE
2. Provide some form of interpretability
<!-- #endregion -->

```python id="PoXzuH1S45KW"
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


import xgboost as xgb
from surprise import Reader, Dataset
from surprise import BaselineOnly
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV
```

<!-- #region id="elqpzipO45KY" -->
## 1. Reading and Storing Data
<!-- #endregion -->

<!-- #region id="KKeB3p8Q45KZ" -->
### Data Pre-processing
<!-- #endregion -->

```python id="a4LDSeWB45KZ" outputId="8c6f7440-2159-435a-8c4d-6315d2327d78"
if not os.path.isfile("../Data/NetflixRatings.csv"): 
#This line: "os.path.isfile("../Data/NetflixRatings.csv")" simply checks that is there a file with the name "NetflixRatings.csv" in the 
#in the folder "/Data/". If the file is present then it return true else false
    startTime = datetime.now()
    data = open("../Data/NetflixRatings.csv", mode = "w") #this line simply creates the file with the name "NetflixRatings.csv" in 
    #write mode in the folder "Data".
#     files = ['../Data/combined_data_1.txt','../Data/combined_data_2.txt', '../Data/combined_data_3.txt', '../Data/combined_data_4.txt']
    files = ['../Data/combined_data_2.txt', '../Data/combined_data_4.txt']
    for file in files:
        print("Reading from file: "+str(file)+"...")
        with open(file) as f:  #you can think of this command "with open(file) as f" as similar to 'if' statement or a sort of 
            #loop statement. This command says that as long as this file is opened, perform the underneath operation.
            for line in f:
                line = line.strip() #line.strip() clears all the leading and trailing spaces from the string, as here each line
                #that we are reading from a file is a string.
                #Note first line consist of a movie id followed by a semi-colon, then second line contains custID,rating,date
                #then third line agains contains custID,rating,date which belong to that movie ID and so on. The format of data
                #is exactly same as shown above with the heading "Example Data Point". Check out above.
                if line.endswith(":"):
                    movieID = line.replace(":", "") #this will remove the trailing semi-colon and return us the leading movie ID.
                else:
                    #here, in the below code we have first created an empty list with the name "row "so that we can insert movie ID 
                    #at the first position and rest customerID, rating and date in second position. After that we have separated all 
                    #four namely movieID, custID, rating and date with comma and converted a single string by joining them with comma.
                    #then finally written them to our output ".csv" file.
                    row = [] 
                    row = [x for x in line.split(",")] #custID, rating and date are separated by comma
                    row.insert(0, movieID)
                    data.write(",".join(row))
                    data.write("\n")
        print("Reading of file: "+str(file)+" is completed\n")
    data.close()
    print("Total time taken for execution of this code = "+str(datetime.now() - startTime))
```

```python id="X4ia7JpY45Ke" outputId="4a99b123-4251-4a59-82cc-e6c90fdd1f49"
# creating data frame from our output csv file.
if not os.path.isfile("../Data/NetflixData.pkl"):
    startTime = datetime.now()
    Final_Data = pd.read_csv("../Data/NetflixRatings.csv", sep=",", names = ["MovieID","CustID", "Ratings", "Date"])
    Final_Data["Date"] = pd.to_datetime(Final_Data["Date"])
    Final_Data.sort_values(by = "Date", inplace = True)
    print("Time taken for execution of above code = "+str(datetime.now() - startTime))
```

```python id="p6NkiJU145Kh"
# storing pandas dataframe as a picklefile for later use
if not os.path.isfile("../Data/NetflixData.pkl"):
    Final_Data.to_pickle("../Data/NetflixData.pkl")
else:
    Final_Data = pd.read_pickle("../Data/NetflixData.pkl")
```

```python id="HSK9ih_m45Kj" outputId="f0f405b8-25ac-4ae6-d598-b96cdb09b7ed"
Final_Data.head()
```

```python id="TXwJZUXK45Kl" outputId="87a9a858-8c61-482c-f295-b84e71198dca"
Final_Data.describe()["Ratings"]
```

<!-- #region id="blRAVge845Km" -->
### Checking for NaN
<!-- #endregion -->

```python id="GzJ_FXH045Km" outputId="199b1da3-38a1-46a1-93ca-3740d42841c8"
print("Number of NaN values = "+str(Final_Data.isnull().sum()))
```

<!-- #region id="vv4Zg_hk45Kn" -->
### Removing Duplicates
<!-- #endregion -->

```python id="8O5lB65z45Ko" outputId="bcdc39d3-a8f5-4f7c-ce1b-4c1358c88825"
duplicates = Final_Data.duplicated(["MovieID","CustID", "Ratings"])
print("Number of duplicate rows = "+str(duplicates.sum()))
```

<!-- #region id="wDmxhA-L45Kp" -->
### Basic Statistics
<!-- #endregion -->

```python id="m2uicpcT45Kp" outputId="7930f9ce-fa1a-41fb-b75d-55568e58ee0e"
print("Total Data:")
print("Total number of movie ratings = "+str(Final_Data.shape[0]))
print("Number of unique users = "+str(len(np.unique(Final_Data["CustID"]))))
print("Number of unique movies = "+str(len(np.unique(Final_Data["MovieID"]))))
```

<!-- #region id="01O5uFKk45Kq" -->
### Spliting data into Train and Test(80:20)
<!-- #endregion -->

```python id="UXS03m-W45Kr"
if not os.path.isfile("../Data/TrainData.pkl"):
    Final_Data.iloc[:int(Final_Data.shape[0]*0.80)].to_pickle("../Data/TrainData.pkl")
    Train_Data = pd.read_pickle("../Data/TrainData.pkl")
    Train_Data.reset_index(drop = True, inplace = True)
else:
    Train_Data = pd.read_pickle("../Data/TrainData.pkl")
    Train_Data.reset_index(drop = True, inplace = True)

if not os.path.isfile("../Data/TestData.pkl"):
    Final_Data.iloc[int(Final_Data.shape[0]*0.80):].to_pickle("../Data/TestData.pkl")
    Test_Data = pd.read_pickle("../Data/TestData.pkl")
    Test_Data.reset_index(drop = True, inplace = True)
else:
    Test_Data = pd.read_pickle("../Data/TestData.pkl")
    Test_Data.reset_index(drop = True, inplace = True)
```

<!-- #region id="gkE6-YpJ45Kr" -->
### Basic Statistics in Train data
<!-- #endregion -->

```python id="D7GZ8MCE45Ks" outputId="84f3a3a7-7cf0-459e-b654-f52dfd5d9c10"
Train_Data.head()
```

```python id="f7646pye45Ks" outputId="03ec87e3-82d4-4345-f6d3-83f628de7aad"
print("Total Train Data:")
print("Total number of movie ratings in train data = "+str(Train_Data.shape[0]))
print("Number of unique users in train data = "+str(len(np.unique(Train_Data["CustID"]))))
print("Number of unique movies in train data = "+str(len(np.unique(Train_Data["MovieID"]))))
print("Highest value of a User ID = "+str(max(Train_Data["CustID"].values)))
print("Highest value of a Movie ID = "+str(max(Train_Data["MovieID"].values)))
```

<!-- #region id="X6jIz4gZ45Kt" -->
### Basic Statistics in Test data
<!-- #endregion -->

```python id="Nhd9od2Z45Ku" outputId="b43eb750-e044-42bc-923d-19a1656d2f60"
Test_Data.head()
```

```python id="AhDhmgOF45Kv" outputId="44a2262b-bc47-40ea-d545-f64fd271ff77"
print("Total Test Data:")
print("Total number of movie ratings in Test data = "+str(Test_Data.shape[0]))
print("Number of unique users in Test data = "+str(len(np.unique(Test_Data["CustID"]))))
print("Number of unique movies in Test data = "+str(len(np.unique(Test_Data["MovieID"]))))
print("Highest value of a User ID = "+str(max(Test_Data["CustID"].values)))
print("Highest value of a Movie ID = "+str(max(Test_Data["MovieID"].values)))
```

<!-- #region id="bxc8j-je45Kv" -->
## 2. Exploratory Data Analysis on Train Data
<!-- #endregion -->

```python id="V8_2RD6e45Kw"
def changingLabels(number):
    return str(number/10**6) + "M"
```

```python id="DCJKuCBL45Kw" outputId="7863613c-b482-46e5-82ab-b91e90d6512f"
plt.figure(figsize = (12, 8))
ax = sns.countplot(x="Ratings", data=Train_Data)

ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])

plt.tick_params(labelsize = 15)
plt.title("Distribution of Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings(Millions)", fontsize = 20)
plt.show()
```

```python id="9F1D4cak45Kx"
Train_Data["DayOfWeek"] = Train_Data.Date.dt.weekday_name
```

```python id="N30AewE345Ky" outputId="7ca420a9-507f-4ced-e867-8c95a5a48081"
Train_Data.tail()
```

<!-- #region id="PND1mCX845Kz" -->
### Number of Ratings per month
<!-- #endregion -->

```python id="mETyhxP745K1" outputId="c9b910af-2acb-454d-83a8-8fec8dda80f5"
plt.figure(figsize = (10,8))
ax = Train_Data.resample("M", on = "Date")["Ratings"].count().plot()
#this above resample() function is a sort of group-by operation.Resample() function can work with dates. It can take months,
#days and years values independently. Here, in parameter we have given "M" which means it will group all the rows Monthly using 
#"Date" which is already present in the DataFrame. Now after grouping the rows month wise, we have just counted the ratings 
#which are grouped by months and plotted them. So, below plot shows that how many ratings are there per month. 
#In resample(), we can also give "6M" for grouping the rows every 6-Monthly, we can also give "Y" for grouping
#the rows yearly, we can also give "D" for grouping the rows by day.
#Resample() is a function which is designed to work with time and dates.
#This "Train_Data.resample("M", on = "Date")["Ratings"].count()" returns a pandas series where keys are Dates and values are 
#counts of ratings grouped by months.You can even check it and print it. Then we are plotting it, where it automatically takes
#Dates--which are keys on--x-axis and counts--which are values on--y-axis.
ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
ax.set_title("Number of Ratings per month", fontsize = 20)
ax.set_xlabel("Date", fontsize = 20)
ax.set_ylabel("Number of Ratings Per Month(Millions)", fontsize = 20)
plt.tick_params(labelsize = 15)
plt.show()
```

```python id="DO6bFWQX45K2"
#Train_Data.resample("M", on = "Date")["Ratings"].count()
```

<!-- #region id="Am92KJsx45K2" -->
### Analysis of Ratings given by user
<!-- #endregion -->

```python id="mHBlsBhl45K3"
no_of_rated_movies_per_user = Train_Data.groupby(by = "CustID")["Ratings"].count().sort_values(ascending = False)
```

```python id="Pew2dsO745K3" outputId="368aa0cc-8646-4595-ec23-9bb22d9ba8b7"
no_of_rated_movies_per_user.head()
```

```python id="Fbb5Ask245K4" outputId="43ef2c99-4be1-495a-9a3d-f8f32fb3d8ea"
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(14,7))

sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, ax = axes[0])
axes[0].set_title("PDF", fontsize = 18)
axes[0].set_xlabel("Number of Ratings by user", fontsize = 18)
axes[0].tick_params(labelsize = 15)

sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, cumulative = True, ax = axes[1])
axes[1].set_title("CDF", fontsize = 18)
axes[1].set_xlabel("Number of Ratings by user", fontsize = 18)
axes[1].tick_params(labelsize = 15)

fig.subplots_adjust(wspace=2)
plt.tight_layout()
plt.show()
```

<!-- #region id="BxUO4IjS45K5" -->
* Above PDF graph shows that almost all of the users give very few ratings. There are very few users who's ratings count is high.
* Similarly, above CDF graph shows that almost 99% of users give very few ratings.
<!-- #endregion -->

```python id="H4vutnet45K5" outputId="81a0212b-8f1a-435a-d990-ea83aac1bbd4"
print("Information about movie ratings grouped by users:")
no_of_rated_movies_per_user.describe()
```

```python id="r2VkngtC45K5"
# no_of_rated_movies_per_user.describe()["75%"]
```

```python id="0rNP2nBW45K6"
quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01))
```

```python id="4UgaFOMg45K6" outputId="c2a335d6-6798-4e43-bf1f-e3296268bac4"
fig = plt.figure(figsize = (10, 6))

axes = fig.add_axes([0.1,0.1,1,1])
axes.set_title("Quantile values of Ratings Per User", fontsize = 20)
axes.set_xlabel("Quantiles", fontsize = 20)
axes.set_ylabel("Ratings Per User", fontsize = 20)
axes.plot(quantiles)

plt.scatter(x = quantiles.index[::5], y = quantiles.values[::5], c = "blue", s = 70, label="quantiles with 0.05 intervals")
plt.scatter(x = quantiles.index[::25], y = quantiles.values[::25], c = "red", s = 70, label="quantiles with 0.25 intervals")
plt.legend(loc='upper left', fontsize = 20)

for x, y in zip(quantiles.index[::25], quantiles.values[::25]):
    plt.annotate(s = '({},{})'.format(x, y), xy = (x, y), fontweight='bold', fontsize = 16, xytext=(x-0.05, y+180))
    
axes.tick_params(labelsize = 15)
```

```python id="nPyo7fu245K7" outputId="d0f7ec52-2ffe-4fd8-8fd2-357b3c001758"
quantiles[::5]
```

```python id="KiCtQ4xl45K7" outputId="b5349418-90cf-49aa-fa93-662cc993706c"
print("Total number of ratings below 75th percentile = "+str(sum(no_of_rated_movies_per_user.values<=133)))
print("Total number of ratings above 75th percentile = "+str(sum(no_of_rated_movies_per_user.values>133)))
```

<!-- #region id="pmxZJ5B545K8" -->
### Analysis of Ratings Per Movie
<!-- #endregion -->

```python id="oTzdYlpc45K8"
no_of_ratings_per_movie = Train_Data.groupby(by = "MovieID")["Ratings"].count().sort_values(ascending = False)
```

```python id="pO_Ea2OX45K8" outputId="7ebe542d-2ba7-4141-b989-b2c96b513bff"
fig = plt.figure(figsize = (12, 6))
axes = fig.add_axes([0.1,0.1,1,1])
plt.title("Number of Ratings Per Movie", fontsize = 20)
plt.xlabel("Movie", fontsize = 20)
plt.ylabel("Count of Ratings", fontsize = 20)
plt.plot(no_of_ratings_per_movie.values)
plt.tick_params(labelsize = 15)
axes.set_xticklabels([])
plt.show()
```

<!-- #region id="gCIqzeW645K9" -->
<b>It is very skewed</b>
<p>It clearly shows that there are some movies which are very popular and were rated by many users as comapared to other movies</p>
<!-- #endregion -->

<!-- #region id="4PP6XJYx45K9" -->
### Analysis of Movie Ratings on Day of Week
<!-- #endregion -->

```python id="_WjTfTWy45K9" outputId="6994e6d9-da21-4667-a5fb-30bdd3941354"
fig = plt.figure(figsize = (12, 8))

axes = sns.countplot(x = "DayOfWeek", data = Train_Data)
axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
axes.set_xlabel("Day of Week", fontsize = 20)
axes.set_ylabel("Number of Ratings", fontsize = 20)
axes.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
axes.tick_params(labelsize = 15)

plt.show()
```

```python id="KMe0D_oD45K-" outputId="506566c4-f907-417e-c0b6-c8aa42d6f77c"
fig = plt.figure(figsize = (12, 8))

axes = sns.boxplot(x = "DayOfWeek", y = "Ratings", data = Train_Data)
axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
axes.set_xlabel("Day of Week", fontsize = 20)
axes.set_ylabel("Number of Ratings", fontsize = 20)
axes.tick_params(labelsize = 15)

plt.show()
```

```python id="UX-XlHFO45K-" outputId="0acdf32d-77da-40b1-a444-1d3b56acc9a6"
average_ratings_dayofweek = Train_Data.groupby(by = "DayOfWeek")["Ratings"].mean()
print("Average Ratings on Day of Weeks")
print(average_ratings_dayofweek)
```

<!-- #region id="uve4tzqa45K_" -->
## 3. Creating USER-ITEM sparse matrix from data frame
<!-- #endregion -->

```python id="KswAqlfE45K_" outputId="b4f82a42-11e7-4368-809e-bc583292c996"
startTime = datetime.now()
print("Creating USER_ITEM sparse matrix for train Data")
if os.path.isfile("../Data/TrainUISparseData.npz"):
    print("Sparse Data is already present in your disk, no need to create further. Loading Sparse Matrix")
    TrainUISparseData = sparse.load_npz("../Data/TrainUISparseData.npz")
    print("Shape of Train Sparse matrix = "+str(TrainUISparseData.shape))
    
else:
    print("We are creating sparse data")
    TrainUISparseData = sparse.csr_matrix((Train_Data.Ratings, (Train_Data.CustID, Train_Data.MovieID)))
    print("Creation done. Shape of sparse matrix = "+str(TrainUISparseData.shape))
    print("Saving it into disk for furthur usage.")
    sparse.save_npz("../Data/TrainUISparseData.npz", TrainUISparseData)
    print("Done\n")

print(datetime.now() - startTime)
```

```python id="b7-HS1aM45LA" outputId="f0682d73-a49b-491d-d991-6a5489462222"
startTime = datetime.now()
print("Creating USER_ITEM sparse matrix for test Data")
if os.path.isfile("../Data/TestUISparseData.npz"):
    print("Sparse Data is already present in your disk, no need to create further. Loading Sparse Matrix")
    TestUISparseData = sparse.load_npz("../Data/TestUISparseData.npz")
    print("Shape of Test Sparse Matrix = "+str(TestUISparseData.shape))
else:
    print("We are creating sparse data")
    TestUISparseData = sparse.csr_matrix((Test_Data.Ratings, (Test_Data.CustID, Test_Data.MovieID)))
    print("Creation done. Shape of sparse matrix = "+str(TestUISparseData.shape))
    print("Saving it into disk for furthur usage.")
    sparse.save_npz("../Data/TestUISparseData.npz", TestUISparseData)
    print("Done\n")

print(datetime.now() - startTime)
```

```python id="ZyqbmjJN45LB"
#If you can see above that the shape of both train and test sparse matrices are same, furthermore, how come this shape of sparse
#matrix has arrived:
#Shape of sparse matrix depends on highest value of User ID and highest value of Movie ID. 
#Now the user whose user ID is highest is present in both train data and test data. Similarly, the movie whose movie ID is
#highest is present in both train data and test data. Hence, shape of both train and test sparse matrices are same.
```

```python id="4DlNLCyk45LB" outputId="21be4fc2-8e0a-4e65-aacd-ab86e341f8e5"
rows,cols = TrainUISparseData.shape
presentElements = TrainUISparseData.count_nonzero()

print("Sparsity Of Train matrix : {}% ".format((1-(presentElements/(rows*cols)))*100))
```

```python id="n97Mw7mf45LC" outputId="fd83aee4-deaf-40d1-8965-da2d8cd50ff3"
rows,cols = TestUISparseData.shape
presentElements = TestUISparseData.count_nonzero()

print("Sparsity Of Test matrix : {}% ".format((1-(presentElements/(rows*cols)))*100))
```

<!-- #region id="1zH5Ue9v45LD" -->
### Finding Global average of all movie ratings, Average rating per user, and Average rating per movie
<!-- #endregion -->

```python id="C59RSoUp45LH"
def getAverageRatings(sparseMatrix, if_user):
    ax = 1 if if_user else 0
    #axis = 1 means rows and axis = 0 means columns 
    sumOfRatings = sparseMatrix.sum(axis = ax).A1  #this will give an array of sum of all the ratings of user if axis = 1 else 
    #sum of all the ratings of movies if axis = 0
    noOfRatings = (sparseMatrix!=0).sum(axis = ax).A1  #this will give a boolean True or False array, and True means 1 and False 
    #means 0, and further we are summing it to get the count of all the non-zero cells means length of non-zero cells
    rows, cols = sparseMatrix.shape
    averageRatings = {i: sumOfRatings[i]/noOfRatings[i] for i in range(rows if if_user else cols) if noOfRatings[i]!=0}
    return averageRatings
```

<!-- #region id="nzbbc1AY45LH" -->
### Global Average Rating
<!-- #endregion -->

```python id="bbhq9Ysw45LI" outputId="eedcff4d-4ba9-4ca8-aef9-fcc1e16f6946"
Global_Average_Rating = TrainUISparseData.sum()/TrainUISparseData.count_nonzero()
print("Global Average Rating {}".format(Global_Average_Rating))
```

<!-- #region id="B1E5SHkV45LI" -->
### Average Rating Per User
<!-- #endregion -->

```python id="FnlV6Dwo45LJ"
AvgRatingUser = getAverageRatings(TrainUISparseData, True)
```

```python id="wKujCckn45LJ" outputId="66648f62-194c-4507-dbb0-0cc57c609477"
print("Average rating of user 25 = {}".format(AvgRatingUser[25]))
```

<!-- #region id="AZKPaP3245LK" -->
### Average Rating Per Movie
<!-- #endregion -->

```python id="MwcCtvLH45LK"
AvgRatingMovie = getAverageRatings(TrainUISparseData, False)
```

```python id="9wNTA3l945LK" outputId="cc8bae90-16a6-4de9-9e1d-fe23c749cd5b"
print("Average rating of movie 4500 = {}".format(AvgRatingMovie[4500]))
```

<!-- #region id="GmgeCZ1t45LL" -->
### PDF and CDF of Average Ratings of Users and Movies
<!-- #endregion -->

```python id="XwKXeM0_45LL" outputId="829847b2-27f8-41a3-c131-022c0e5d9f57"
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
fig.suptitle('Avg Ratings per User and per Movie', fontsize=25)

user_average = [rats for rats in AvgRatingUser.values()]
sns.distplot(user_average, hist = False, ax = axes[0], label = "PDF")
sns.kdeplot(user_average, cumulative = True, ax = axes[0], label = "CDF")
axes[0].set_title("Average Rating Per User", fontsize=20)
axes[0].tick_params(labelsize = 15)
axes[0].legend(loc='upper left', fontsize = 17)

movie_average = [ratm for ratm in AvgRatingMovie.values()]
sns.distplot(movie_average, hist = False, ax = axes[1], label = "PDF")
sns.kdeplot(movie_average, cumulative = True, ax = axes[1], label = "CDF")
axes[1].set_title("Average Rating Per Movie", fontsize=20)
axes[1].tick_params(labelsize = 15)
axes[1].legend(loc='upper left', fontsize = 17)

plt.subplots_adjust(wspace=0.2, top=0.85)
plt.show()
```

<!-- #region id="B_V8J82045LL" -->
### Cold Start Problem
<!-- #endregion -->

<!-- #region id="q_Q2iesg45LM" -->
#### Cold Start Problem with Users
<!-- #endregion -->

```python id="IjogzT4j45LM" outputId="e05bd85f-7a4f-4e65-981c-56d0940537f3"
total_users = len(np.unique(Final_Data["CustID"]))
train_users = len(AvgRatingUser)
uncommonUsers = total_users - train_users
                  
print("Total number of Users = {}".format(total_users))
print("Number of Users in train data= {}".format(train_users))
print("Number of Users not present in train data = {}({}%)".format(uncommonUsers, np.round((uncommonUsers/total_users)*100), 2))
```

<!-- #region id="5trHn3D045LM" -->
#### Cold Start Problem with Movies
<!-- #endregion -->

```python id="k7T77LnZ45LN" outputId="8db6a466-f902-4516-c922-3d6cf8765b67"
total_movies = len(np.unique(Final_Data["MovieID"]))
train_movies = len(AvgRatingMovie)
uncommonMovies = total_movies - train_movies
                  
print("Total number of Movies = {}".format(total_movies))
print("Number of Movies in train data= {}".format(train_movies))
print("Number of Movies not present in train data = {}({}%)".format(uncommonMovies, np.round((uncommonMovies/total_movies)*100), 2))
```

<!-- #region id="_1IvL64w45LN" -->
## 4. Computing Similarity Matrices
<!-- #endregion -->

<!-- #region id="PBQQB0y745LO" -->
### Computing User-User Similarity Matrix
<!-- #endregion -->

<!-- #region id="4_qWIUp945LO" -->
Calculating User User Similarity_Matrix is __not very easy__(_unless you have huge Computing Power and lots of time_)
<!-- #endregion -->

```python id="6MOKTXmx45LP" outputId="3db3e391-cacb-4027-de31-7572cd51d9e3"
row_index, col_index = TrainUISparseData.nonzero()
rows = np.unique(row_index)
for i in rows[:100]:
    print(i)
```

```python id="LkWvzzIT45LQ"
#Here, we are calculating user-user similarity matrix only for first 100 users in our sparse matrix. And we are calculating 
#top 100 most similar users with them.
def getUser_UserSimilarity(sparseMatrix, top = 100):
    startTimestamp20 = datetime.now()  
    
    row_index, col_index = sparseMatrix.nonzero()  #this will give indices of rows in "row_index" and indices of columns in 
    #"col_index" where there is a non-zero value exist.
    rows = np.unique(row_index)
    similarMatrix = np.zeros(61700).reshape(617,100)    # 617*100 = 61700. As we are building similarity matrix only 
    #for top 100 most similar users.
    timeTaken = []
    howManyDone = 0
    for row in rows[:top]:
        howManyDone += 1
        startTimestamp = datetime.now().timestamp()  #it will give seconds elapsed
        sim = cosine_similarity(sparseMatrix.getrow(row), sparseMatrix).ravel()
        top100_similar_indices = sim.argsort()[-top:]
        top100_similar = sim[top100_similar_indices]
        similarMatrix[row] = top100_similar
        timeforOne = datetime.now().timestamp() - startTimestamp
        timeTaken.append(timeforOne)
        if howManyDone % 20 == 0:
            print("Time elapsed for {} users = {}sec".format(howManyDone, (datetime.now() - startTimestamp20)))
    print("Average Time taken to compute similarity matrix for 1 user = "+str(sum(timeTaken)/len(timeTaken))+"seconds")
    
    fig = plt.figure(figsize = (12,8))
    plt.plot(timeTaken, label = 'Time Taken For Each User')
    plt.plot(np.cumsum(timeTaken), label='Cumulative Time')
    plt.legend(loc='upper left', fontsize = 15)
    plt.xlabel('Users', fontsize = 20)
    plt.ylabel('Time(Seconds)', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.show()
    
    return similarMatrix
```

```python id="MdqjgRnK45LR" outputId="84551b5c-08f4-4fd1-ae8d-b14949db9d84"
simMatrix = getUser_UserSimilarity(TrainUISparseData, 100)
```

<!-- #region id="pcN_h8jU45LS" -->
<p>We have __401901 Users__ in our training data.<br><br>Average time taken to compute similarity matrix for one user is __3.635 sec.__<br><br>For 401901 users:<br><br>_401901*3.635 == 1460910.135sec == 405.808hours == 17Days_<br><br>Computation of user-user similarity matrix is impossible if computational power is limited. On the other hand, if we try to reduce the dimension say by truncated SVD then it would take even more time because truncated SVD creates dense matrix and amount of multiplication for creation of user-user similarity matrix would increase dramatically.<br><br>__Is there any other way to compute user-user similarity???__<br><br>We maintain a binary Vector for users, which tells us whether we already computed similarity for this user or not..<br><br>
__OR__<br><br>Compute top (let's just say, 1000) most similar users for this given user, and add this to our datastructure, so that we can just access it(similar users) without recomputing it again. <br><br>__If it is already computed__<br><br>Just get it directly from our datastructure, which has that information. In production time, We might have to recompute similarities, if it is computed a long time ago. Because user preferences changes over time. If we could maintain some kind of Timer, which when expires, we have to update it ( recompute it ). <br><br>Which datastructure to use:<br><br>It is purely implementation dependant.<br><br>
One simple method is to maintain a **Dictionary Of Dictionaries**.<br><br>

key    :    userid<br>
value  :    Again a dictionary<br>
            key   : _Similar User<br>
            value: Similarity Value>
<!-- #endregion -->

<!-- #region id="bXzxxDq845LV" -->
### Computing Movie-Movie Similarity Matrix
<!-- #endregion -->

```python id="wIwKu3pL45LW" outputId="24cc61b5-d181-4a37-da60-ac2ddb8c06a4"
start = datetime.now()

if not os.path.isfile("../Data/m_m_similarity.npz"):
    print("Movie-Movie Similarity file does not exist in your disk. Creating Movie-Movie Similarity Matrix...")
    
    m_m_similarity = cosine_similarity(TrainUISparseData.T, dense_output = False)
    print("Done")
    print("Dimension of Matrix = {}".format(m_m_similarity.shape))
    print("Storing the Movie Similarity matrix on disk for further usage")
    sparse.save_npz("../Data/m_m_similarity.npz", m_m_similarity)
else:
    print("File exists in the disk. Loading the file...")
    m_m_similarity = sparse.load_npz("../Data/m_m_similarity.npz")
    print("Dimension of Matrix = {}".format(m_m_similarity.shape))
    
print(datetime.now() - start)
```

<!-- #region id="YEht-MsK45LW" -->
### Does Movie-Movie Similarity Works?
### Let's pick random movie and check it's top 10 most similar movies.
<!-- #endregion -->

```python id="9QVJ-nDp45LX"
movie_ids = np.unique(m_m_similarity.nonzero())
```

```python id="uHc-4f4C45LX"
similar_movies_dict = dict()
for movie in movie_ids:
    smlr = np.argsort(-m_m_similarity[movie].toarray().ravel())[1:100]
    similar_movies_dict[movie] = smlr
```

```python id="RsUAI4uC45LX"
movie_titles_df = pd.read_csv("../Data/movie_titles.csv",sep = ",", header = None, names=['MovieID', 'Year_of_Release', 'Movie_Title'], index_col = "MovieID", encoding = "iso8859_2")
```

```python id="GOUCGELu45LX" outputId="586e8299-8627-42df-be03-64a9552852a9"
movie_titles_df.head()
```

<!-- #region id="eJRjnOL-45LY" -->
### Similar Movies to: __Godzilla's Revenge__
<!-- #endregion -->

```python id="gEKr6iOP45LZ" outputId="20de81af-a8f3-491c-b3dd-28a6dbabf676"
movieID_GR = 17765

print("Name of the movie -------> "+str(movie_titles_df.loc[movieID_GR][1]))

print("Number of ratings by users for movie {} is {}".format(movie_titles_df.loc[movieID_GR][1], TrainUISparseData[:,movieID_GR].getnnz()))

print("Number of similar movies to {} is {}".format(movie_titles_df.loc[movieID_GR][1], m_m_similarity[movieID_GR].count_nonzero()))
```

```python id="Bx0zCQi445LZ"
# Meaning of "[:,17765]" means get all the values of column "17765".
# "getnnz()" give count of explicitly-stored values (nonzeros).
```

```python id="_OKbHrLE45La"
all_similar = sorted(m_m_similarity[movieID_GR].toarray().ravel(), reverse = True)[1:]

similar_100 = all_similar[:101]
```

```python id="Bvp4aJJe45La" outputId="5f5bae96-3116-4a75-cf80-c27634a43b88"
plt.figure(figsize = (10, 8))
plt.plot(all_similar, label = "All Similar")
plt.plot(similar_100, label = "Top 100 Similar Movies")
plt.title("Similar Movies to Godzilla's Revenge", fontsize = 25)
plt.ylabel("Cosine Similarity Values", fontsize = 20)
plt.tick_params(labelsize = 15)
plt.legend(fontsize = 20)
plt.show()
```

<!-- #region id="RtOgoi7x45Lb" -->
### Top 10 Similar Movies to: __Godzilla's Revenge__
<!-- #endregion -->

```python id="_0LuM53U45Lc" outputId="92715bf8-a542-4bc5-c308-8dc12d8b51b0"
movie_titles_df.loc[similar_movies_dict[movieID_GR][:10]]
```

<!-- #region id="CaAYfibk45Lc" -->
<p>__It seems that Movie-Movie similarity is working perfectly.__</p>
<!-- #endregion -->

<!-- #region id="3ZXKZ4d_45Ld" -->
## 5. Machine Learning Models
<!-- #endregion -->

```python id="0JleVx5V45Ld"
def get_sample_sparse_matrix(sparseMatrix, n_users, n_movies):
    startTime = datetime.now()
    users, movies, ratings = sparse.find(sparseMatrix)
    uniq_users = np.unique(users)
    uniq_movies = np.unique(movies)
    np.random.seed(15)   #this will give same random number everytime, without replacement
    userS = np.random.choice(uniq_users, n_users, replace = False)
    movieS = np.random.choice(uniq_movies, n_movies, replace = False)
    mask = np.logical_and(np.isin(users, userS), np.isin(movies, movieS))
    sparse_sample = sparse.csr_matrix((ratings[mask], (users[mask], movies[mask])), 
                                                     shape = (max(userS)+1, max(movieS)+1))
    print("Sparse Matrix creation done. Saving it for later use.")
    sparse.save_npz(path, sparse_sample)
    print("Done")
    print("Shape of Sparse Sampled Matrix = "+str(sparse_sample.shape))
    
    print(datetime.now() - start)
    return sparse_sample
```

<!-- #region id="lE6Wnv1d45Le" -->
### Creating Sample Sparse Matrix for Train Data
<!-- #endregion -->

```python id="BDUaqqk445Le" outputId="dbec9c15-6d51-4787-d79f-0f8d06a0a741"
path = "../Data/TrainUISparseData_Sample.npz"
if not os.path.isfile(path):
    print("Sample sparse matrix is not present in the disk. We are creating it...")
    train_sample_sparse = get_sample_sparse_matrix(TrainUISparseData, 4000, 400)
else:
    print("File is already present in the disk. Loading the file...")
    train_sample_sparse = sparse.load_npz(path)
    print("File loading done.")
    print("Shape of Train Sample Sparse Matrix = "+str(train_sample_sparse.shape))
```

<!-- #region id="_P4zZsmO45Lf" -->
### Creating Sample Sparse Matrix for Test Data
<!-- #endregion -->

```python id="H_p5hiOh45Lg" outputId="b0e9b7f0-b91d-4e68-8547-cc269bac0ed3"
path = "../Data/TestUISparseData_Sample.npz"
if not os.path.isfile(path):
    print("Sample sparse matrix is not present in the disk. We are creating it...")
    test_sample_sparse = get_sample_sparse_matrix(TestUISparseData, 2000, 200)
else:
    print("File is already present in the disk. Loading the file...")
    test_sample_sparse = sparse.load_npz(path)
    print("File loading done.")
    print("Shape of Test Sample Sparse Matrix = "+str(test_sample_sparse.shape))
```

<!-- #region id="Tdlh3Eaq45Lg" -->
### Finding Global Average of all movie ratings, Average rating per User, and Average rating per Movie (from sampled train)
<!-- #endregion -->

```python id="Ojspj8wr45Lh" outputId="5b029abb-ebb3-450c-d6d8-91a27936288f"
print("Global average of all movies ratings in Train Sample Sparse is {}".format(np.round((train_sample_sparse.sum()/train_sample_sparse.count_nonzero()), 2)))
```

<!-- #region id="x8KFgBCX45Lh" -->
### Finding Average of all movie ratings
<!-- #endregion -->

```python id="4fjNShIK45Lk" outputId="8408b381-0766-49eb-af75-09756dc75fd7"
globalAvgMovies = getAverageRatings(train_sample_sparse, False)
print("Average move rating for movie 14890 is {}".format(globalAvgMovies[14890]))
```

<!-- #region id="uH2BE1ZJ45Ll" -->
### Finding Average rating per User
<!-- #endregion -->

```python id="v9YXjp4l45Ll" outputId="772d4d6b-fb6c-46ec-9114-778e1d190b08"
globalAvgUsers = getAverageRatings(train_sample_sparse, True)
print("Average user rating for user 16879 is {}".format(globalAvgMovies[16879]))
```

<!-- #region id="weSJOq9a45Lm" -->
### Featurizing data
<!-- #endregion -->

```python id="99vtnR6v45Lm" outputId="a3635a09-efb2-4877-db5f-8c006bd6ea04"
print("No of ratings in Our Sampled train matrix is : {}".format(train_sample_sparse.count_nonzero()))
print("No of ratings in Our Sampled test matrix is : {}".format(test_sample_sparse.count_nonzero()))
```

<!-- #region id="chKwaK2s45Lm" -->
### Featurizing data for regression problem
<!-- #endregion -->

<!-- #region id="2V3Q-3xw45Ln" -->
### Featurizing Train Data
<!-- #endregion -->

```python id="0zpouh_D45Ln"
sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(train_sample_sparse)
```

```python id="58mY6G7645Ln" outputId="1c67cc2a-badc-48d3-d1c9-48a932fccb9d"
if os.path.isfile("../Data/Train_Regression.csv"):
    print("File is already present in your disk. You do not have to prepare it again.")
else:
    startTime = datetime.now()
    print("Preparing Train csv file for {} rows".format(len(sample_train_ratings)))
    with open("../Data/Train_Regression.csv", mode = "w") as data:
        count = 0
        for user, movie, rating in zip(sample_train_users, sample_train_movies, sample_train_ratings):
            row = list()
            row.append(user)  #appending user ID
            row.append(movie) #appending movie ID
            row.append(train_sample_sparse.sum()/train_sample_sparse.count_nonzero()) #appending global average rating

#----------------------------------Ratings given to "movie" by top 5 similar users with "user"--------------------#
            similar_users = cosine_similarity(train_sample_sparse[user], train_sample_sparse).ravel()
            similar_users_indices = np.argsort(-similar_users)[1:]
            similar_users_ratings = train_sample_sparse[similar_users_indices, movie].toarray().ravel()
            top_similar_user_ratings = list(similar_users_ratings[similar_users_ratings != 0][:5])
            top_similar_user_ratings.extend([globalAvgMovies[movie]]*(5-len(top_similar_user_ratings)))
            #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "movie" average
            #rating. Let say only 3 out of 5 ratings are available then rest 2 will be "movie" average rating.
            row.extend(top_similar_user_ratings)
            
 #----------------------------------Ratings given by "user" to top 5 similar movies with "movie"------------------#
            similar_movies = cosine_similarity(train_sample_sparse[:,movie].T, train_sample_sparse.T).ravel()
            similar_movies_indices = np.argsort(-similar_movies)[1:]
            similar_movies_ratings = train_sample_sparse[user, similar_movies_indices].toarray().ravel()
            top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
            top_similar_movie_ratings.extend([globalAvgUsers[user]]*(5-len(top_similar_movie_ratings)))
            #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "user" average
            #rating. Let say only 3 out of 5 ratings are available then rest 2 will be "user" average rating.
            row.extend(top_similar_movie_ratings)
            
 #----------------------------------Appending "user" average, "movie" average & rating of "user""movie"-----------#
            row.append(globalAvgUsers[user])
            row.append(globalAvgMovies[movie])
            row.append(rating)
            
#-----------------------------------Converting rows and appending them as comma separated values to csv file------#
            data.write(",".join(map(str, row)))
            data.write("\n")
    
            count += 1
            if count % 2000 == 0:
                print("Done for {}. Time elapsed: {}".format(count, (datetime.now() - startTime)))
                
    print("Total Time for {} rows = {}".format(len(sample_train_ratings), (datetime.now() - startTime)))
```

```python id="ZWK0eud345Lo" outputId="e4d1da7c-980b-4eb1-a330-3f9ae4df0cf6"
Train_Reg = pd.read_csv("../Data/Train_Regression.csv", names = ["User_ID", "Movie_ID", "Global_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "User_Average", "Movie_Average", "Rating"])
Train_Reg.head()
```

```python id="TaTTLQiw45Lo" outputId="08ad42c8-ba12-42bb-96a3-24d14d6e1c6f"
print("Number of nan Values = "+str(Train_Reg.isnull().sum().sum()))
```

<!-- #region id="Ics3wWFE45Lp" -->
<p><b>User_ID:</b> ID of a this User</p>

<p><b>Movie_ID:</b> ID of a this Movie</p>

<p><b>Global_Average:</b> Global Average Rating</p>

<p><b>Ratings given to this Movie by top 5 similar users with this User:</b> (SUR1, SUR2, SUR3, SUR4, SUR5)</p>
   
<p><b>Ratings given by this User to top 5 similar movies with this Movie:</b> (SMR1, SMR2, SMR3, SMR4, SMR5)</p>

<p><b>User_Average:</b> Average Rating of this User</p>

<p><b>Movie_Average:</b> Average Rating of this Movie</p>

<p><b>Rating:</b> Rating given by this User to this Movie</p>
<!-- #endregion -->

```python id="l65-w6On45Lp" outputId="ca2708dd-75e7-4618-8c41-1524da3c64c7"
print("Shape of Train DataFrame = {}".format(Train_Reg.shape))
```

<!-- #region id="wZtKfgMo45Lq" -->
### Featurizing Test Data
<!-- #endregion -->

```python id="WPKw5U3945Lq"
sample_test_users, sample_test_movies, sample_test_ratings = sparse.find(test_sample_sparse)
```

```python id="kVO2ILIl45Lr" outputId="4514cee1-9dec-4e3b-e5c4-e169dce7ee4e"
if os.path.isfile("../Data/Test_Regression.csv"):
    print("File is already present in your disk. You do not have to prepare it again.")
else:
    startTime = datetime.now()
    print("Preparing Test csv file for {} rows".format(len(sample_test_ratings)))
    with open("../Data/Test_Regression.csv", mode = "w") as data:
        count = 0
        for user, movie, rating in zip(sample_test_users, sample_test_movies, sample_test_ratings):
            row = list()
            row.append(user)  #appending user ID
            row.append(movie) #appending movie ID
            row.append(train_sample_sparse.sum()/train_sample_sparse.count_nonzero()) #appending global average rating

#-----------------------------Ratings given to "movie" by top 5 similar users with "user"-------------------------#
            try:
                similar_users = cosine_similarity(train_sample_sparse[user], train_sample_sparse).ravel()
                similar_users_indices = np.argsort(-similar_users)[1:]
                similar_users_ratings = train_sample_sparse[similar_users_indices, movie].toarray().ravel()
                top_similar_user_ratings = list(similar_users_ratings[similar_users_ratings != 0][:5])
                top_similar_user_ratings.extend([globalAvgMovies[movie]]*(5-len(top_similar_user_ratings)))
                #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "movie" 
                #average rating. Let say only 3 out of 5 ratings are available then rest 2 will be "movie" average rating.
                row.extend(top_similar_user_ratings)
            #########Cold Start Problem, for a new user or a new movie#########    
            except(IndexError, KeyError):
                global_average_train_rating = [train_sample_sparse.sum()/train_sample_sparse.count_nonzero()]*5
                row.extend(global_average_train_rating)
            except:
                raise
                
 #-----------------------------Ratings given by "user" to top 5 similar movies with "movie"-----------------------#
            try:
                similar_movies = cosine_similarity(train_sample_sparse[:,movie].T, train_sample_sparse.T).ravel()
                similar_movies_indices = np.argsort(-similar_movies)[1:]
                similar_movies_ratings = train_sample_sparse[user, similar_movies_indices].toarray().ravel()
                top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
                top_similar_movie_ratings.extend([globalAvgUsers[user]]*(5-len(top_similar_movie_ratings)))
                #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "user" 
                #average rating. Let say only 3 out of 5 ratings are available then rest 2 will be "user" average rating.
                row.extend(top_similar_movie_ratings)
            #########Cold Start Problem, for a new user or a new movie#########
            except(IndexError, KeyError):
                global_average_train_rating = [train_sample_sparse.sum()/train_sample_sparse.count_nonzero()]*5
                row.extend(global_average_train_rating)
            except:
                raise
                
 #-----------------------------Appending "user" average, "movie" average & rating of "user""movie"----------------#
            try:        
                row.append(globalAvgUsers[user])
            except (KeyError):
                global_average_train_rating = train_sample_sparse.sum()/train_sample_sparse.count_nonzero()
                row.append(global_average_train_rating)
            except:
                raise
                
            try:
                row.append(globalAvgMovies[movie])
            except(KeyError):
                global_average_train_rating = train_sample_sparse.sum()/train_sample_sparse.count_nonzero()
                row.append(global_average_train_rating)
            except:
                raise
                
            row.append(rating)
            
#------------------------------Converting rows and appending them as comma separated values to csv file-----------#
            data.write(",".join(map(str, row)))
            data.write("\n")
    
            count += 1
            if count % 100 == 0:
                print("Done for {}. Time elapsed: {}".format(count, (datetime.now() - startTime)))
                
    print("Total Time for {} rows = {}".format(len(sample_test_ratings), (datetime.now() - startTime)))
```

```python id="tf2Rvkot45Lr" outputId="8acf8c38-3335-4180-8bb6-1ac85a6328b5"
Test_Reg = pd.read_csv("../Data/Test_Regression.csv", names = ["User_ID", "Movie_ID", "Global_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "User_Average", "Movie_Average", "Rating"])
Test_Reg.head()
```

```python id="uLXzD4Zi45Ls" outputId="e1ffc793-60f5-4947-e776-e6104ef6f853"
print("Number of nan Values = "+str(Test_Reg.isnull().sum().sum()))
```

<!-- #region id="8hPX-8nf45Lt" -->
<p><b>User_ID:</b> ID of a this User</p>



<p><b>Movie_ID:</b> ID of a this Movie</p>



<p><b>Global_Average:</b> Global Average Rating</p>



<p><b>Ratings given to this Movie by top 5 similar users with this User:</b> (SUR1, SUR2, SUR3, SUR4, SUR5)</p>

    
    
<p><b>Ratings given by this User to top 5 similar movies with this Movie:</b> (SMR1, SMR2, SMR3, SMR4, SMR5)</p>


<p><b>User_Average:</b> Average Rating of this User</p>


<p><b>Movie_Average:</b> Average Rating of this Movie</p>


<p><b>Rating:</b> Rating given by this User to this Movie</p>
<!-- #endregion -->

```python id="6tQm41Qd45Lt" outputId="388b8013-9bf6-4b84-9efe-05bbfef51138"
print("Shape of Test DataFrame = {}".format(Test_Reg.shape))
```

<!-- #region id="MhemvWIM45Lu" -->
### Transforming Data for Surprise Models
<!-- #endregion -->

<!-- #region id="ZJRRvH7W45Lu" -->
#### Transforming Train Data
<!-- #endregion -->

<!-- #region id="cVjsyk_G45Lv" -->
- We can't give raw data (movie, user, rating) to train the model in Surprise library.


- They have a separate format for TRAIN and TEST data, which will be useful for training the models like SVD, KNNBaseLineOnly....etc..,in Surprise.


- We can form the trainset from a file, or from a Pandas  DataFrame. 
http://surprise.readthedocs.io/en/stable/getting_started.html#load-dom-dataframe-py 
<!-- #endregion -->

```python id="Jp75MZue45Lv" outputId="ce50adc7-5470-4168-af00-aa6bc27f4af2"
Train_Reg[['User_ID', 'Movie_ID', 'Rating']].head(5)
```

```python id="h0Pa69wV45Lw"
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(Train_Reg[['User_ID', 'Movie_ID', 'Rating']], reader)

trainset = data.build_full_trainset()
```

<!-- #region id="DFxMig2845Lw" -->
#### Transforming Test Data

- For test data we just have to define a tuple (user, item, rating).
- You can check out this link: https://github.com/NicolasHug/Surprise/commit/86cf44529ca0bbb97759b81d1716ff547b950812
- Above link is a github of surprise library. Check methods "def all_ratings(self)" and "def build_testset(self)" from line
  177 to 201(If they modify the file then line number may differ, but you can always check aforementioned two methods).
- "def build_testset(self)" method returns a list of tuples of (user, item, rating).
<!-- #endregion -->

```python id="5VIVWjnA45Lw"
testset = list(zip(Test_Reg["User_ID"].values, Test_Reg["Movie_ID"].values, Test_Reg["Rating"].values))
```

```python id="ofXZ8p7a45Lx" outputId="9411db64-8863-4f62-dc53-7c39b8f41956"
testset[:5]
```

<!-- #region id="c_CbXAg945Lx" -->
### Applying Machine Learning Models
<!-- #endregion -->

<!-- #region id="yVXQvjFF45Lx" -->
<p>We have two Error Metrics.</p>
<p><b>->   RMSE: Root Mean Square Error: </b>RMSE is the error of each point which is squared. Then mean is calculated. Finally root of that mean is taken as final value.</p>
<p><b>->   MAPE: Mean Absolute Percentage Error: </b>The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of prediction accuracy of a forecasting method.</p>
<p>where At is the actual value and Ft is the forecast value.</p>
<p>
The difference between At and Ft is divided by the actual value At again. The absolute value in this calculation is summed for every forecasted point in time and divided by the number of fitted points n. Multiplying by 100% makes it a percentage error.</p>
<!-- #endregion -->

<!-- #region id="D_DWoSPu45Ly" -->
<b>We can also use other regression models. But  we are using exclusively XGBoost as it is typically fairly powerful in practice.</b>
<!-- #endregion -->

```python id="jFPUFb-u45Ly"
error_table = pd.DataFrame(columns = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"])
model_train_evaluation = dict()
model_test_evaluation = dict()
```

```python id="UEJQaxR845Lz"
def make_table(model_name, rmse_train, mape_train, rmse_test, mape_test):
    global error_table
    #All variable assignments in a function store the value in the local symbol table; whereas variable references first look 
    #in the local symbol table, then in the global symbol table, and then in the table of built-in names. Thus, global variables 
    #cannot be directly assigned a value within a function (unless named in a global statement), 
    #although they may be referenced.
    error_table = error_table.append(pd.DataFrame([[model_name, rmse_train, mape_train, rmse_test, mape_test]], columns = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"]))
    error_table.reset_index(drop = True, inplace = True)
```

<!-- #region id="eBwyLnOw45Lz" -->
### Utility Functions for Regression Models
<!-- #endregion -->

```python id="neABGRnx45Lz"
def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(abs((y_true - y_pred)/y_true))*100
    return rmse, mape
```

```python id="4c8covGc45L0"
def train_test_xgboost(x_train, x_test, y_train, y_test, model_name):
    startTime = datetime.now()
    train_result = dict()
    test_result = dict()
    
    clf = xgb.XGBRegressor(n_estimators = 100, silent = False, n_jobs  = 10)
    clf.fit(x_train, y_train)
    
    print("-"*50)
    print("TRAIN DATA")
    y_pred_train = clf.predict(x_train)
    rmse_train, mape_train = error_metrics(y_train, y_pred_train)
    print("RMSE = {}".format(rmse_train))
    print("MAPE = {}".format(mape_train))
    print("-"*50)
    train_result = {"RMSE": rmse_train, "MAPE": mape_train, "Prediction": y_pred_train}
    
    print("TEST DATA")
    y_pred_test = clf.predict(x_test)
    rmse_test, mape_test = error_metrics(y_test, y_pred_test)
    print("RMSE = {}".format(rmse_test))
    print("MAPE = {}".format(mape_test))
    print("-"*50)
    test_result = {"RMSE": rmse_test, "MAPE": mape_test, "Prediction": y_pred_test}
        
    print("Time Taken = "+str(datetime.now() - startTime))
    
    plot_importance(xgb, clf)
    
    make_table(model_name, rmse_train, mape_train, rmse_test, mape_test)
    
    return train_result, test_result
```

```python id="acW_dWp445L0"
def plot_importance(model, clf):
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_axes([0,0,1,1])
    model.plot_importance(clf, ax = ax, height = 0.3)
    plt.xlabel("F Score", fontsize = 20)
    plt.ylabel("Features", fontsize = 20)
    plt.title("Feature Importance", fontsize = 20)
    plt.tick_params(labelsize = 15)
    
    plt.show()
```

<!-- #region id="r6mkP2zK45L0" -->
### Utility Functions for Surprise Models
<!-- #endregion -->

```python id="B0V12Pf945L3"
def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    predicted = np.array([pred.est for pred in predictions])
    return actual, predicted
#in surprise prediction of every data point is returned as dictionary like this:
#"user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}"
#In this dictionary, "r_ui" is a key for actual rating and "est" is a key for predicted rating 
```

```python id="y1Tnpsah45L4"
def get_error(predictions):
    actual, predicted = get_ratings(predictions)
    rmse = np.sqrt(mean_squared_error(actual, predicted)) 
    mape = np.mean(abs((actual - predicted)/actual))*100
    return rmse, mape
```

```python id="a37ZRNxf45L4"
my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

def run_surprise(algo, trainset, testset, model_name):
    startTime = datetime.now()
    
    train = dict()
    test = dict()
    
    algo.fit(trainset)
    #You can check out above function at "https://surprise.readthedocs.io/en/stable/getting_started.html" in 
    #"Train-test split and the fit() method" section
    
#-----------------Evaluating Train Data------------------#
    print("-"*50)
    print("TRAIN DATA")
    train_pred = algo.test(trainset.build_testset())
    #You can check out "algo.test()" function at "https://surprise.readthedocs.io/en/stable/getting_started.html" in 
    #"Train-test split and the fit() method" section
    #You can check out "trainset.build_testset()" function at "https://surprise.readthedocs.io/en/stable/FAQ.html#can-i-use-my-own-dataset-with-surprise-and-can-it-be-a-pandas-dataframe" in 
    #"How to get accuracy measures on the training set" section
    train_actual, train_predicted = get_ratings(train_pred)
    train_rmse, train_mape = get_error(train_pred)
    print("RMSE = {}".format(train_rmse))
    print("MAPE = {}".format(train_mape))
    print("-"*50)
    train = {"RMSE": train_rmse, "MAPE": train_mape, "Prediction": train_predicted}
    
#-----------------Evaluating Test Data------------------#
    print("TEST DATA")
    test_pred = algo.test(testset)
    #You can check out "algo.test()" function at "https://surprise.readthedocs.io/en/stable/getting_started.html" in 
    #"Train-test split and the fit() method" section
    test_actual, test_predicted = get_ratings(test_pred)
    test_rmse, test_mape = get_error(test_pred)
    print("RMSE = {}".format(test_rmse))
    print("MAPE = {}".format(test_mape))
    print("-"*50)
    test = {"RMSE": test_rmse, "MAPE": test_mape, "Prediction": test_predicted}
    
    print("Time Taken = "+str(datetime.now() - startTime))
    
    make_table(model_name, train_rmse, train_mape, test_rmse, test_mape)
    
    return train, test
```

<!-- #region id="_HOtQTDR45L4" -->
## 1. XGBoost 13 Features
<!-- #endregion -->

```python id="GXeBgC3v45L5" outputId="c13b90ce-aecb-4d86-e1bb-2270c0cab156"
x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

y_train = Train_Reg["Rating"]

y_test = Test_Reg["Rating"]

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGBoost_13")

model_train_evaluation["XGBoost_13"] = train_result
model_test_evaluation["XGBoost_13"] = test_result
```

<!-- #region id="inBpOGJX45L5" -->
## 2. Surprise BaselineOnly Model
<!-- #endregion -->

<!-- #region id="7258RKNq45L6" -->
### Predicted Rating
> $\large\hat{r}_{ui} = \mu + b_u + b_i$<br><br>

- $\mu$: Average Global Ratings in training data<br>
- $b_u$: User-Bias<br>
- $b_i$: Item-Bias

### Optimization Function
> $\large \sum_{r_ui \in R_{Train}} \left(r_{ui} - (\mu + b_u + b_i)\right)^2 + \lambda \left(b_u^2 + b_i^2 \right). \left[minimize\; b_u, b_i \right]$
<!-- #endregion -->

```python id="DaJpyMn-45L6" outputId="254fedc6-7b9d-4b1a-cf17-bbe0d318b630"
bsl_options = {"method":"sgd", "learning_rate":0.01, "n_epochs":25}

algo = BaselineOnly(bsl_options=bsl_options)
#You can check the docs of above used functions at:https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#baseline-estimates-configuration
#at section "Baselines estimates configuration".

train_result, test_result = run_surprise(algo, trainset, testset, "BaselineOnly")

model_train_evaluation["BaselineOnly"] = train_result
model_test_evaluation["BaselineOnly"] = test_result
```

<!-- #region id="4JF1CuaQ45L7" -->
## 3. XGBoost 13 Features + Surprise BaselineOnly Model
<!-- #endregion -->

<!-- #region id="PpRT_K6145L7" -->
### Adding predicted ratings from Surprise BaselineOnly model to our Train and Test Dataframe
<!-- #endregion -->

```python id="FCzYS4E245L7"
Train_Reg["BaselineOnly"] = model_train_evaluation["BaselineOnly"]["Prediction"]
```

```python id="Et9vWUN145L8" outputId="b7232e73-bdc0-45d1-a642-5d5741b488b0"
Train_Reg.head()
```

```python id="_ncC3YAP45L8" outputId="d8992438-c681-4154-c942-61934f4db3c3"
print("Number of nan values = "+str(Train_Reg.isnull().sum().sum()))
```

```python id="w6wbdxRB45L9" outputId="6b61bfef-9ec9-4a0c-df6f-41eceac0cd1e"
Test_Reg["BaselineOnly"] = model_test_evaluation["BaselineOnly"]["Prediction"]
Test_Reg.head()
```

```python id="fVBj9q2545L9" outputId="dd239e68-b0c9-49b3-b8df-3d1ab1440cc3"
print("Number of nan values = "+str(Test_Reg.isnull().sum().sum()))
```

```python id="S-LGm66845L-" outputId="04be6bb1-ad59-4500-adb9-db58eb7a86d9"
x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

y_train = Train_Reg["Rating"]

y_test = Test_Reg["Rating"]

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL")

model_train_evaluation["XGB_BSL"] = train_result
model_test_evaluation["XGB_BSL"] = test_result
```

<!-- #region id="fw_houM445L-" -->
## 4. Surprise KNN-Baseline with User-User and Item-Item Similarity
<!-- #endregion -->

<!-- #region id="i61v8jog45L-" -->
### Prediction $\hat{r}_{ui}$ in case of user-user similarity

$\large \hat{r}_{ui} = b_{ui} + \frac{ \sum\limits_{v \in N^k_i(u)}
\text{sim}(u, v) \cdot (r_{vi} - b_{vi})} {\sum\limits_{v
\in N^k_i(u)} \text{sim}(u, v)}$

- $\pmb{b_{ui}}$ - Baseline prediction_ of (user,movie) rating which is "$b_{ui} = \mu + b_u + b_i$".

- $ \pmb {N_i^k (u)}$ - Set of __K similar__ users (neighbours) of __user (u)__ who rated __movie(i)__  

- _sim (u, v)_ - Similarity between users __u and v__ who also rated movie 'i'. This is exactly same as our hand-crafted features 'SUR'- 'Similar User Rating'. Means here we have taken 'k' such similar users 'v' with user 'u' who also rated movie 'i'. $r_{vi}$ is the rating which user 'v' gives on item 'i'. $b_{vi}$ is the predicted baseline model rating of user 'v' on item 'i'.
    - Generally, it will be cosine similarity or Pearson correlation coefficient. 
    - But we use __shrunk Pearson-baseline correlation coefficient__, which is based on the pearsonBaseline similarity ( we take     - base line predictions instead of mean rating of user/item)<br><br><br><br>  

### Prediction $\hat{r}_{ui}$ in case of item-item similarity

$\large \hat{r}_{ui} = b_{ui} + \frac{ \sum\limits_{j \in N^k_u(i)}
\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\sum\limits_{j \in
N^k_u(j)} \text{sim}(i, j)}$

- __Notation is same as of user-user similarity__<br><br><br>


#### Documentation you can check at:
KNN BASELINE: https://surprise.readthedocs.io/en/stable/knn_inspired.html

PEARSON_BASELINE SIMILARITY: http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.pearson_baseline

SHRINKAGE: Neighborhood Models in http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf
<!-- #endregion -->

<!-- #region id="UD-GemUQ45L_" -->
### 4.1 Surprise KNN-Baseline with User-User.
<!-- #endregion -->

<!-- #region id="Zex7rmnF45L_" -->
#### Cross- Validation
<!-- #endregion -->

```python id="xwtver1e45L_" outputId="bd8753f4-4115-4348-9850-cd400634fea2"
param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [True], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}

gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
```

<!-- #region id="_6fD2gJi45MA" -->
### Applying KNNBaseline User-User with best parameters
<!-- #endregion -->

```python id="hOVhi28m45MA" outputId="360f4225-6012-4d43-e746-ae09e4c6082d"
sim_options = {'name':'pearson_baseline', 'user_based':True, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}

bsl_options = {'method': 'sgd'} 

algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)

train_result, test_result = run_surprise(algo, trainset, testset, "KNNBaseline_User")

model_train_evaluation["KNNBaseline_User"] = train_result
model_test_evaluation["KNNBaseline_User"] = test_result
```

<!-- #region id="Q5Fe3rO345MB" -->
### 4.2 Surprise KNN-Baseline with Item-Item.
<!-- #endregion -->

<!-- #region id="Q4jPbtvT45MB" -->
#### Cross- Validation
<!-- #endregion -->

```python id="L_yXzOue45MB" outputId="8b610cf3-6665-44ca-c6ac-c8295a6c7814"
param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [False], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}

gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
```

<!-- #region id="JXvHoRcB45MC" -->
#### Applying KNNBaseline Item-Item with best parameters
<!-- #endregion -->

```python id="itD_G8MO45MC" outputId="c322be05-bcbd-434a-9e1c-6841ac9f468b"
sim_options = {'name':'pearson_baseline', 'user_based':False, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}

bsl_options = {'method': 'sgd'} 

algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)

train_result, test_result = run_surprise(algo, trainset, testset, "KNNBaseline_Item")

model_train_evaluation["KNNBaseline_Item"] = train_result
model_test_evaluation["KNNBaseline_Item"] = test_result
```

<!-- #region id="9unKQkR445MD" -->
## 5. XGBoost 13 Features + Surprise BaselineOnly + Surprise KNN Baseline
<!-- #endregion -->

<!-- #region id="tFp0r1rD45ME" -->
### Adding predicted ratings from Surprise KNN Baseline model to our Train and Test Dataframe
<!-- #endregion -->

```python id="gXVwUXV045ME"
Train_Reg["KNNBaseline_User"] = model_train_evaluation["KNNBaseline_User"]["Prediction"]
Train_Reg["KNNBaseline_Item"] = model_train_evaluation["KNNBaseline_Item"]["Prediction"]

Test_Reg["KNNBaseline_User"] = model_test_evaluation["KNNBaseline_User"]["Prediction"]
Test_Reg["KNNBaseline_Item"] = model_test_evaluation["KNNBaseline_Item"]["Prediction"]
```

```python id="QvbGhXWK45MF" outputId="e47d51ff-35a4-493a-e4c5-e95f66f585ab"
Train_Reg.head()
```

```python id="WvSaC1wy45MG" outputId="cf450c19-187b-479f-c393-e996989e7662"
print("Number of nan values in Train Data "+str(Train_Reg.isnull().sum().sum()))
```

```python id="FmOH7lPV45MH" outputId="c664aff7-0515-46a1-9a68-0b23e7f35974"
Test_Reg.head()
```

```python id="Rosoi2x745MI" outputId="437c2cc5-7334-4f7b-a919-358ef22dd6db"
print("Number of nan values in Test Data "+str(Test_Reg.isnull().sum().sum()))
```

```python id="1k-6dZch45MI" outputId="bb778c76-dc8f-4838-9fcf-ae69361ec18f"
x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

y_train = Train_Reg["Rating"]

y_test = Test_Reg["Rating"]

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN")

model_train_evaluation["XGB_BSL_KNN"] = train_result
model_test_evaluation["XGB_BSL_KNN"] = test_result
```

<!-- #region id="tjcUC3bh45MJ" -->
## 6. Matrix Factorization SVD 
<!-- #endregion -->

<!-- #region id="C7QShkKJ45MJ" -->
#### Prediction $\hat{r}_{ui}$ is set as:<br>

$\large \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u$
- $\pmb q_i$ - Representation of item(movie) in latent factor space
        
- $\pmb p_u$ - Representation of user in new latent factor space<br>

__If user u is unknown, then the bias $b_u$ and the factors $p_u$ are assumed to be zero. The same applies for item i with $b_i$ and $q_i$.__<br><br><br>


#### Optimization Problem<br>

$\large \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +
\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\right) \left[minimize\; b_u, b_i, q_i, p_u \right]$
<br><br><br>

SVD Documentation: https://surprise.readthedocs.io/en/stable/matrix_factorization.html
<!-- #endregion -->

<!-- #region id="8IO4FjBS45MJ" -->
#### Cross- Validation
<!-- #endregion -->

```python id="rUIBJxt445MK" outputId="f836b0f9-7df0-4096-cdf7-368940b2f5da"
param_grid  = {'n_factors': [5,7,10,15,20,25,35,50,70,90]}   #here, n_factors is the equivalent to dimension 'd' when matrix 'A'
#is broken into 'b' and 'c'. So, matrix 'A' will be of dimension n*m. So, matrices 'b' and 'c' will be of dimension n*d and m*d.

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
```

<!-- #region id="hhAckA1w45MN" -->
### Applying SVD with best parameters
<!-- #endregion -->

```python id="mQdU4M6V45MN" outputId="b8729ae6-4363-4d4e-9f55-c42245d03fac"
algo = SVD(n_factors = gs.best_params['rmse']['n_factors'], biased=True, verbose=True)

train_result, test_result = run_surprise(algo, trainset, testset, "SVD")

model_train_evaluation["SVD"] = train_result
model_test_evaluation["SVD"] = test_result
```

<!-- #region id="I7fK72s345MO" -->
## 7. Matrix Factorization SVDpp with implicit feedback
<!-- #endregion -->

<!-- #region id="8Bx_q3sj45MP" -->
#### Prediction $\hat{r}_{ui}$ is set as:<br>
$\large \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\left(p_u +
|I_u|^{-\frac{1}{2}} \sum_{j \in I_u}y_j\right)$<br><br>

 - $ \pmb{I_u}$ --- the set of all items rated by user u. $|I_u|$ is a length of that set.<br>

-  $\pmb{y_j}$ --- Our new set of item factors that capture implicit ratings. Here, an implicit rating describes the fact that a user u rated an item j, regardless of the rating value. $y_i$ is an item vector. For every item j, there is an item vector $y_j$ which is an implicit feedback. Implicit feedback indirectly reflects opinion by observing user behavior including purchase history, browsing history, search patterns, or even mouse movements. Implicit feedback usually denotes the presence or absence of an event. For example, there is a movie 10 where user has just checked the details of the movie and spend some time there, will contribute to implicit rating. Now, since here Netflix has not provided us the details that for how long a user has spend time on the movie, so here we are considering the fact that even if a user has rated some movie then it means that he has spend some time on that movie which contributes to implicit rating.<br><br>

__If user u is unknown, then the bias $b_u$ and the factors $p_u$ are assumed to be zero. The same applies for item i with $b_i$, $q_i$ and $y_i$.__<br><br><br>

#### Optimization Problem

$\large \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +
\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2 + ||y_j||^2\right).\left[minimize\; b_u, b_i, q_i, p_u, y_j \right]$<br><br>

SVDpp Documentation: https://surprise.readthedocs.io/en/stable/matrix_factorization.html
<!-- #endregion -->

<!-- #region id="fKP7iCHI45MP" -->
#### Cross- Validation
<!-- #endregion -->

```python id="c2AFzHnn45MQ" outputId="aae84fd1-58fa-4f32-fbb4-8d05d4c77287"
param_grid = {'n_factors': [10, 30, 50, 80, 100], 'lr_all': [0.002, 0.006, 0.018, 0.054, 0.10]}

gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
```

<!-- #region id="PhukKhW945MQ" -->
#### Applying SVDpp with best parameters
<!-- #endregion -->

```python id="8NE3UUll45MR" outputId="daf0e254-cc48-4329-f0b9-fe1c8aa768ce"
algo = SVDpp(n_factors = gs.best_params['rmse']['n_factors'], lr_all = gs.best_params['rmse']["lr_all"], verbose=True)

train_result, test_result = run_surprise(algo, trainset, testset, "SVDpp")

model_train_evaluation["SVDpp"] = train_result
model_test_evaluation["SVDpp"] = test_result
```

<!-- #region id="rvo4JpRM45MR" -->
## 8. XGBoost 13 Features + Surprise BaselineOnly + Surprise KNN Baseline + SVD + SVDpp
<!-- #endregion -->

```python id="Amq5Evxk45MR"
Train_Reg["SVD"] = model_train_evaluation["SVD"]["Prediction"]
Train_Reg["SVDpp"] = model_train_evaluation["SVDpp"]["Prediction"]

Test_Reg["SVD"] = model_test_evaluation["SVD"]["Prediction"]
Test_Reg["SVDpp"] = model_test_evaluation["SVDpp"]["Prediction"]
```

```python id="QLz4wO-045MS" outputId="35ee2e40-7eb4-4be2-bb19-9c7f2dda09cf"
Train_Reg.head()
```

```python id="Eu45-ONO45MT" outputId="92689da9-e5ec-40d3-cdcb-9fade2b21049"
print("Number of nan values in Train Data "+str(Train_Reg.isnull().sum().sum()))
```

```python id="vqN7E6LS45MU" outputId="c999829a-cd78-4bec-a3f9-0c6ac8c79d11"
Test_Reg.head()
```

```python id="CzMcV05Y45MV" outputId="7ea03800-42e0-47fe-87b1-7b5085634171"
print("Number of nan values in Test Data "+str(Test_Reg.isnull().sum().sum()))
```

```python id="9PPQ0FBD45MY" outputId="9d8cbad6-032c-450e-f79b-0caf7aa20be5"
x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

y_train = Train_Reg["Rating"]

y_test = Test_Reg["Rating"]

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN_MF")

model_train_evaluation["XGB_BSL_KNN_MF"] = train_result
model_test_evaluation["XGB_BSL_KNN_MF"] = test_result
```

<!-- #region id="UoREbMfw45MY" -->
## 9. Surprise KNN Baseline + SVD + SVDpp
<!-- #endregion -->

```python id="juC_E1wQ45MZ" outputId="10f98811-2166-4a3d-b627-e924e2bd0e75"
x_train = Train_Reg[["KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp"]]

x_test = Test_Reg[["KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp"]]

y_train = Train_Reg["Rating"]

y_test = Test_Reg["Rating"]

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_KNN_MF")

model_train_evaluation["XGB_KNN_MF"] = train_result
model_test_evaluation["XGB_KNN_MF"] = test_result
```

<!-- #region id="4plghZvL45MZ" -->
## Summary
<!-- #endregion -->

```python id="r5t5hHZd45MZ"
error_table2 = error_table.drop(["Train MAPE", "Test MAPE"], axis = 1)
```

```python id="W26L_-4m45Mc" outputId="254dc0d1-a462-4f35-9d7e-35ea894532bc"
error_table2.plot(x = "Model", kind = "bar", figsize = (14, 8), grid = True, fontsize = 15)
plt.title("Train and Test RMSE and MAPE of all Models", fontsize = 20)
plt.ylabel("Error Values", fontsize = 20)
plt.legend(bbox_to_anchor=(1, 1), fontsize = 20)
plt.show()
```

```python id="C7Er2p6X45Md" outputId="b3f7e648-0398-47e9-f25a-eeabe8c5532c"
error_table.drop(["Train MAPE", "Test MAPE"], axis = 1).style.highlight_min(axis=0)
```

<!-- #region id="_4qUuU9z45Me" -->
So, far our best model is SVDpp with Test RMSE of 1.067583
<!-- #endregion -->

<!-- #region id="zMzH7FiV49iR" -->
## README

## Check out my blog on building a Recommender System at the following link:
https://medium.com/@gauravsharma2656/how-to-built-a-recommender-system-rs-616c988d64b2
## Business Problem 
Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.
Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.
Credits: https://www.netflixprize.com/rules.html
## Problem Statement
Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is 10% better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)
## Sources 
* https://www.netflixprize.com/rules.html
* https://www.kaggle.com/netflix-inc/netflix-prize-data
* Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 (very nice blog)
* surprise library: http://surpriselib.com/ (we use many models from this library)
* surprise library doc: http://surprise.readthedocs.io/en/stable/getting_started.html (we use many models from this library)
* installing surprise: https://github.com/NicolasHug/Surprise#installation
* Research paper: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf (most of our work was inspired by this paper)
* SVD Decomposition : https://www.youtube.com/watch?v=P5mlg91as1c
## Real world/Business Objectives and constraints 
### Objectives:
1. Predict the rating that a user would give to a movie that he has not yet rated.
2. Minimize the difference between predicted and actual rating (RMSE and MAPE) 
### Constraints:
1. Some form of interpretability.
2. There is no low latency requirement as the recommended movies can be precomputed earlier.

## Type of Data:
* There are 17770 unique movie IDs.
* There are 480189 unique user IDs.
* There are ratings. Ratings are on a five star (integral) scale from 1 to 5.
* There is a date on which the movie is watched by the user in the format YYYY-MM-DD.
## Getting Started
Start by downloading the project and run "NetflixMoviesRecommendation.ipynb" file in ipython-notebook.

## Prerequisites
You need to have installed following softwares and libraries in your machine before running this project.
1. Python 3
2. Anaconda: It will install ipython notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy, scipy.
3. XGBoost
4. Surprise

## Installing
1. Python 3: https://www.python.org/downloads/
2. Anaconda: https://www.anaconda.com/download/
3. XGBoost: conda install -c conda-forge xgboost
4. Surprise: pip install surprise


## Built With
*	ipython-notebook - Python Text Editor
*	sklearn - Machine learning library
*	seaborn, matplotlib.pyplot, - Visualization libraries
*	numpy, scipy- number python library
*	pandas - data handling library
* XGBoost - Used for making regression models
*	Surprise - used for making recommendation system models

## Authors
*	Gaurav Sharma - Complete work  

## Acknowledgments
*	Applied AI Course
<!-- #endregion -->
