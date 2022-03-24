# Data Science Bookcamp

## Key points

1. It is based on the book [*Data Science Bookcamp*](https://learning.oreilly.com/library/view/data-science-bookcamp/9781617296253), authored by Leonard Apeltsin.
2. The topic includes probability, statistics, graph theory, text analysis, clustering, and geographic analysis.
3. Code is provided in the form of Jupyter notebooks (with Colab support). It is also available on github [here](https://github.com/sparsh-ai/general-recsys/tree/T426474).
4. For convenience, here are the direct links to Jupyter notebooks:
    1. Tutorial - [Probability theory and the sample space analysis in python](https://nbviewer.org/gist/sparsh-ai/0818e19d4a51c41dd14adb97ce7f5e58)
    2. Tutorial - [Finding the winning strategy in a sum-21 card game](https://nbviewer.org/gist/sparsh-ai/c0372e84f4177b762bcd196f1c04092f)
    3. Tutorial - [Statistical analysis in python](https://nbviewer.org/gist/sparsh-ai/74d263fe9472f3aa3d850027c85f8083)
    4. Tutorial - [Clustering analysis in python](https://nbviewer.org/gist/sparsh-ai/b6bf62384f006dbe48edeff0684546ee)
    5. Tutorial - [Geography location visualization and analysis in python](https://nbviewer.org/gist/sparsh-ai/fc74f531f3a99c7838db48d76d1b2b09)
    6. Tutorial - [Measuring text similarities in python](https://nbviewer.org/gist/sparsh-ai/8939c5a39748a1fe7a5df0a5f024492e)
    7. Tutorial - [Using basic graph theory to rank websites by popularity](https://nbviewer.org/gist/sparsh-ai/326d2ad84d0965a4270048d64c5f4493)
    8. Tutorial - [Utilizing undirected graphs to optimize the travel-time between towns](https://nbviewer.org/gist/sparsh-ai/02ebeec31311ee862a1261da56224a81)
    9. Case study - [Finding the winning strategy in a red-color-win card game](https://nbviewer.org/gist/sparsh-ai/7dc86ca2ba5ef176332a0c780d251308)
    10. Case study - [Assessing online ad-clicks for significance](https://nbviewer.org/gist/sparsh-ai/d2f1cb590c7f6a37c46a51fb7fa3265e)
    11. Case study - [Tracking disease outbreaks using news headlines](https://nbviewer.org/gist/sparsh-ai/5593bd827a8d9b5704177215c09310a6)
    12. Case study - [Using online job postings to improve your data science resume](https://nbviewer.org/gist/sparsh-ai/d595527e0496c3466a3e34ef04c9b748)
    13. Case study - [Predicting future friendships from social network data](https://nbviewer.org/gist/sparsh-ai/fa15c92c2570396a3b25780f089a2dee)

## 1. Probability

Few things in life are certain; most things are driven by chance. Whenever we cheer for our favorite sports team, or purchase a lottery ticket, or make an investment in the stock market, we hope for some particular outcome, but that outcome cannot ever be guaranteed. Randomness permeates our day-to-day experiences. Fortunately, that randomness can still be mitigated and controlled. We know that some unpredictable events occur more rarely than others and that certain decisions carry less uncertainty than other much-riskier choices. Driving to work in a car is safer than riding a motorcycle. Investing part of your savings in a retirement account is safer than betting it all on a single hand of blackjack. We can intrinsically sense these trade-offs in certainty because even the most unpredictable systems still show some predictable behaviors. These behaviors have been rigorously studied using *probability theory*. Probability theory is an inherently complex branch of math. However, aspects of the theory can be understood without knowing the mathematical underpinnings. In fact, difficult probability problems can be solved in Python without needing to know a single math equation. Such an equation-free approach to probability requires a baseline understanding of what mathematicians call a *sample space*.

### 1.1 - Probability theory and the sample space analysis in python

If we flip an unbiased coin, what is the probability that a head will come? It is 50% obviously.

Now flip a coin again, but this time, it is biased, shows 4 heads for each tail. What is the probability now of coming a head? It is 80%. Still tractable, right?

Let's take another question, what is the probability of having exactly 2 daughters in a family of 4 children? We might need a paper this time but we can still solve it. It will be 6/16 = 37.5%.

Let's take another one, what is the probability of sum to be 21 in rolling of 6 fair dice? We can solve this on paper but it will take lots of our precious time. So, we take help of computers to do the calculations on our behalf. And the major advantage is that the code that we write for a simple case can be reused for a complex case also.

In [this](https://nbviewer.org/gist/sparsh-ai/0818e19d4a51c41dd14adb97ce7f5e58) notebook, we are learning all these fundamentals concepts of probability by solving simple puzzles like flipping coins, shuffling cards, rolling dice etc.

### 1.2 - Finding the winning strategy in a sum-21 card game

Now, we will check if we learned the probability concepts or just assumed that we learned. Imagine we have a deck of cards. We will win the game if the sum of card is equal to 21. Our goal is to find the strategy that will help us winning this game. In [this](https://nbviewer.org/gist/sparsh-ai/c0372e84f4177b762bcd196f1c04092f) notebook, we are building this strategy.

### 1.3 - Finding the winning strategy in a red-color-win card game

Would you like to win a bit of money? Let’s wager on a card game for minor stakes. In front of you is a shuffled deck of cards. All 52 cards lie face down. Half the cards are red, and half are black. I will proceed to flip over the cards one by one. If the last card I flip over is red, you’ll win a dollar. Otherwise, you’ll lose a dollar.

Here’s the twist: you can ask me to halt the game at any time. Once you say “Halt,” I will flip over the next card and end the game. That next card will serve as the final card. You will win a dollar if it’s red.

We can play the game as many times as you like. The deck will be reshuffled every time. After each round, we’ll exchange money. What is your best approach to winning this game?

:::note

To address the problem at hand, we will need to know how to 1) Compute the probabilities of observable events using sample space analysis, 2) Plot the probabilities of events across a range of interval values, 3) Simulate random processes, such as coin flips and card shuffling, using Python, 4) Evaluate our confidence in decisions drawn from simulations using confidence interval analysis.

:::

Our goal is to discover a strategy that best predicts a red card in the deck. We will do so by

1. Developing multiple strategies for predicting red cards in a randomly shuffled deck.
2. Applying each strategy across multiple simulations to compute its probability of success within a high confidence interval. If these computations prove to be intractable, we will instead focus on those strategies that perform best across a 10-card sample space.
3. Returning the simplest strategy associated with the highest probability of success.

[Here](https://nbviewer.org/gist/sparsh-ai/7dc86ca2ba5ef176332a0c780d251308) is the solution notebook.

## 2. Statistics

Statistics is a branch of mathematics dealing with the collection and interpretation of numeric data. It is the precursor of all modern data science. The term *statistic* originally signified “the science of the state” because statistical methods were first developed to analyze the data of state governments. Since ancient times, government agencies have gathered data pertaining to their populace. That data would be used to levy taxes and organize large military campaigns. Hence, critical state decisions depended on the quality of data. Poor record keeping could lead to potentially disastrous results. That is why state bureaucrats were very concerned by any random fluctuations in their records. Probability theory eventually tamed these fluctuations, making the randomness interpretable. Ever since then, statistics and probability theory have been closely intertwined.

Statistics and probability theory are closely related, but in some ways, they are very different. Probability theory studies random processes over a potentially infinite number of measurements. It is not bound by real-world limitations. This allows us to model the behavior of a coin by imagining millions of coin flips. In real life, flipping a coin millions of times is a pointlessly time-consuming endeavor. Surely we can sacrifice some data instead of flipping coins all day and night. Statisticians acknowledge these constraints placed on us by the data-gathering process. Real-world data collection is costly and time consuming. Every data point carries a price. We cannot survey a country’s population without employing government officials. We cannot test our online ads without paying for every ad that’s clicked. Thus, the size of our final dataset usually depends on the size of our initial budget. If the budget is constrained, then the data will also be constrained. This trade-off between data and resourcing lies at the heart of modern statistics. Statistics help us understand exactly how much data is sufficient to draw insights and make impactful decisions. The purpose of statistics is to find meaning in data even when that data is limited in size.

### 2.1 - Statistical analysis in python

What is the probability of seeing 16 heads in 20 flips? You are correct if you answered `0.0046`. Would you believe if I tell you that this can be calculated simply by `stats.binom.pmf(16, 20, 0.5)`. Now, what is the probability of seeing 240 tails in 840 flips with a biased coin of 3 head to 1 tail ratio? It sounds simple to calculate now, isn't it? It will be `stats.binom.pmf(240, 840, .25)`. In same way, we can plot the distributions of probability for different flips like this:

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled.png)

The plotted distributions grow more dispersed around their central positions as these central positions relocate right. but why? To answer this, we need to get familiar with statistical concepts. [This](https://nbviewer.org/gist/sparsh-ai/74d263fe9472f3aa3d850027c85f8083) tutorial will help you to achieve this familiarity.

### 2.2 - Assessing online ad-clicks for significance

Fred is a loyal friend, and he needs your help. Fred just launched a burger bistro in the city of Brisbane. The bistro is open for business, but business is slow. Fred wants to entice new customers to come and try his tasty burgers. To do this, Fred will run an online advertising campaign directed at Brisbane residents. Every weekday, between 11:00 a.m. and 1:00 p.m., Fred will purchase 3,000 ads aimed at hungry locals. Every ad will be viewed by a single Brisbane resident. The text of every ad will read, “Hungry? Try the Best Burger in Brisbane. Come to Fred’s.” Clicking the text will take potential customers to Fred’s site. Each displayed ad will cost our friend one cent, but Fred believes the investment will be worth it.

Fred is getting ready to execute his ad campaign. However, he runs into a problem. Fred previews his ad, and its text is blue. Fred believes that blue is a boring color. He feels that other colors could yield more clicks. Fortunately, Fred’s advertising software allows him to choose from 30 different colors. Is there a text color that will bring more clicks than blue? Fred decides to find out.

Fred instigates an experiment. Every weekday for a month, Fred purchases 3,000 online ads. The text of every ad is assigned to one of 30 possible colors. The advertisements are distributed evenly by color. Thus, 100 ads with the same color are viewed by 100 people every day. For example, 100 people view a blue ad, and another 100 people view a green ad. These numbers add up to 3,000 views that are distributed across the 30 colors. Fred’s advertising software automatically tracks all daily views. It also records the daily clicks associated with each of the 30 colors. The software stores this data in a table. That table holds the clicks per day and views per day for every specified color. Each table row maps a color to the views and clicks for all analyzed days.

Fred has carried out his experiment. He obtained ad-click data for all 20 weekdays of the month. That data is organized by color. Now, Fred wants to know if there is a color that draws significantly more ad clicks than blue. Unfortunately, Fred doesn’t know how to properly interpret the results. He’s not sure which clicks are meaningful and which clicks have occurred purely randomly. Fred is brilliant at broiling burgers but has no training in data analysis. This is why Fred has turned to you for help. Fred asks you to analyze his table and to compare the counts of daily clicks. He’s searching for a color that draws significantly more ad clicks than blue. Are you willing to help Fred? If so, he’s promised you free burgers for a year!

:::note

To address the problem at hand, we need to know how to do the following: 1) Measure the centrality and dispersion of sampled data, 2) Interpret the significance of two diverging means through p-value calculation, 3) Minimize mistakes associated with misleading p-value measurements, 4) Load and manipulate data stored in tables using Python.

:::

Our aim is to discover an ad color that generates significantly more clicks than blue. We will do so by following these steps:

1. Load and clean our advertising data using Pandas.
2. Run a permutation test between blue and the other recorded colors.
3. Check the computed p-values for statistical significance using a properly determined significance level.

[This](https://nbviewer.org/gist/sparsh-ai/d2f1cb590c7f6a37c46a51fb7fa3265e) is the solution notebook.

## 3. Clustering

*Clustering* is the process of organizing data points into conceptually meaningful groups. What makes a given group “conceptually meaningful”? There is no easy answer to that question. The usefulness of any clustered output is dependent on the task we’ve been assigned.

Imagine that we’re asked to cluster a collection of pet photos. Do we cluster fish and lizards in one group and fluffy pets (such as hamsters, cats, and dogs) in another? Or should hamsters, cats, and dogs be assigned three separate clusters of their own? If so, perhaps we should consider clustering pets by breed. Thus, Chihuahuas and Great Danes fall into diverging clusters. Differentiating between dog breeds will not be easy. However, we can easily distinguish between Chihuahuas and Great Danes based on breed size. Maybe we should compromise: we’ll cluster on both fluffiness and size, thus bypassing the distinction between the Cairn Terrier and the similar-looking Norwich Terrier.

Is the compromise worth it? It depends on our data science task. Suppose we work for a pet food company, and our aim is to estimate demand for dog food, cat food, and lizard food. Under these conditions, we must distinguish between fluffy dogs, fluffy cats, and scaly lizards. However, we won’t need to resolve differences between separate dog breeds. Alternatively, imagine an analyst at a vet’s office who’s trying to group pet patients by their breed. This second task requires a much more granular level of group resolution.

Different situations require different clustering techniques. As data scientists, we must choose the correct clustering solution. Over the course of our careers, we will cluster thousands (if not tens of thousands) of datasets using a variety of clustering techniques. The most commonly used algorithms rely on some notion of centrality to distinguish between clusters.

### 3.1 - Clustering analysis in python

Suppose a bull’s-eye is located at a coordinate of `[0, 0]`. And another at `[6,0]`.

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-1.png)

Suppose we just have these points, and want to know the bull's eye coordinates. We can use K-means clustering algorithm to find these centroids.

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-2.png)

Our clustering model has located the centroids in the data. Now, we can reuse these centroids to analyze new data-points that the model has not seen before.

Let's take a little complex case now. Suppose that an astronomer discovers a new planet at the far-flung edges of the solar system. The plant, much like our Saturn, has multiple rings spinning in constant orbit around its center. Each ring is formed from thousands of rocks. We'll model these rocks as individual points, defined by x and y coordinates.

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-3.png)

Three ring-groups are clearly present in the plot. Lets search for these 3 clusters using K-means.

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-4.png)

The output is an utter failure! We need to design an algorithm that will cluster data within dense regions of space. To find the correct clusters in these type of cases, we can exploy density-based clustering algorithm DBSCAN. Running this algorithm on our data gives the following results:

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-5.png)

DBSCAN has successfully identified the 3 rock rings. The algorithm succeeded where K-means had failed.

In [this](https://nbviewer.org/gist/sparsh-ai/b6bf62384f006dbe48edeff0684546ee) notebook, we will learn these types of clustering fundamentals.

## 4. Geographic and Cartography Analysis

People have relied on location information since before the dawn of recorded history. Cave dwellers once carved maps of hunting routes into mammoth tusks. Such maps evolved as civilizations flourished. The ancient Babylonians fully mapped the borders of their vast empire. Much later, in 3000 BC, Greek scholars improved cartography using mathematical innovations. The Greeks discovered that the Earth was round and accurately computed the planet’s circumference. Greek mathematicians laid the groundwork for measuring distances across the Earth’s curved surface. Such measurements required the creation of a geographic coordinate system: a rudimentary system based on latitude and longitude was introduced in 2000 BC.

Combining cartography with latitude and longitude helped revolutionize maritime navigation. Sailors could more freely travel the seas by checking their positions on a map. Roughly speaking, maritime navigation protocols followed these three steps:

Data observation—A sailor recorded a series of observations including wind direction, the position of the stars, and (after approximately AD 1300) the northward direction of a compass.

Mathematical and algorithmic analysis of data—A navigator analyzed all of the data to estimate the ship’s position. Sometimes the analysis required trigonometric calculations. More commonly, the navigator consulted a series of rule-based measurement charts. By algorithmically adhering to the rules in the charts, the navigator could figure out the ship’s coordinates.

Visualizing and decision making—The captain examined the computed location on a map relative to the expected destination. Then the captain would give orders to adjust the ship’s orientation based on the visualized results.

This navigation paradigm perfectly encapsulates the standard data science process. As data scientists, we are offered raw observations. We algorithmically analyze that data. Then, we visualize the results to make critical decisions. Thus, data science and location analysis are linked. That link has only grown stronger through the centuries. Today, countless corporations analyze locations in ways the ancient Greeks could never have imagined. Hedge funds study satellite photos of farmlands to make bets on the global soybean market. Transport-service providers analyze vast traffic patterns to efficiently route fleets of cars. Epidemiologists process newspaper data to monitor the global spread of disease.

### 4.1 - Geography location visualization and analysis in python

Whether it is about plotting a city on the global map, or measuring distances, python is at our disposal to support in our analysis. In [this](https://nbviewer.org/gist/sparsh-ai/fc74f531f3a99c7838db48d76d1b2b09) notebook, we will learn all the fundamentals related to geography and cartography analysis.

### 4.2 - Tracking disease outbreaks using news headlines

Congratulations! You have just been hired by the American Institute of Health. The Institute monitors disease epidemics in both foreign and domestic lands. A critical component of the monitoring process is analyzing published news data. Each day, the Institute receives hundreds of news headlines describing disease outbreaks in various locations. The news headlines are too numerous to be analyzed by hand.

Your first assignment is as follows: You will process the daily quota of news headlines and extract locations that are mentioned You will then cluster the headlines based on their geographic distribution. Finally, you will review the largest clusters within and outside the United States. Any interesting findings should be reported to your immediate superior.

The file `headlines.txt` contains the hundreds of headlines that you must analyze. Each headline appears on a separate line in the file.

:::note

To address the problem at hand, we need to know how to do the following: 1) Cluster datasets using multiple techniques and distance measures. 2) Measure distances between locations on a spherical globe. 3) Visualize locations on a map. 4) Extract location coordinates from headline text.

:::

Our goal is to extract locations from disease-related headlines to uncover the largest active epidemics within and outside of the United States. We will do as follows:

1. Load the data.
2. Extract locations from the text using regular expressions and the GeoNamesCache library.
3. Check the location matches for errors.
4. Cluster the locations based on geographic distance.
5. Visualize the clusters on a map, and remove any errors.
6. Output representative locations from the largest clusters to draw interesting conclusions.

[This](https://nbviewer.org/gist/sparsh-ai/5593bd827a8d9b5704177215c09310a6) is the solution notebook.

## 5. Text Analysis

Rapid text analysis can save lives. Let’s consider a real-world incident when US soldiers stormed a terrorist compound. In the compound, they discovered a computer containing terabytes of archived data. The data included documents, text messages, and emails pertaining to terrorist activities. The documents were too numerous to be read by any single human being. Fortunately, the soldiers were equipped with special software that could perform very fast text analysis. The software allowed the soldiers to process all of the text data without even having to leave the compound. The onsite analysis immediately revealed an active terrorist plot in a nearby neighborhood. The soldiers instantly responded to the plot and prevented a terrorist attack.

This swift defensive response would not have been possible without *natural language processing* (NLP) techniques. NLP is a branch of data science that focuses on speedy text analysis. Typically, NLP is applied to very large text datasets. NLP use cases are numerous and diverse and include the following:

- Corporate monitoring of social media posts to measure the public’s sentiment toward a company’s brand
- Analyzing transcribed call center conversations to monitor common customer complaints
- Matching people on dating sites based on written descriptions of shared interests
- Processing written doctors’ notes to ensure proper patient diagnosis

These use cases depend on fast analysis. Delayed signal extraction could be costly. Unfortunately, the direct handling of text is an inherently slow process. Most computational techniques are optimized for numbers, not text. Consequently, NLP methods depend on a conversion from pure text to a numeric representation. Once all words and sentences have been replaced with numbers, the data can be analyzed very rapidly.

### 5.1 - Measuring text similarities in python

Consider these 3 sentences: 

```
text1 = 'She sells seashells by the seashore.'
text2 = 'Seashells! The seashells are on sale! By the seashore.'
text3 = 'She sells 3 seashells to John, who lives by the lake.'
```

These 3 looks similar but how can we quantify this similarity?

Well, to start with, we can create a binary matrix like this:

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-6.png)

And use distance measures like Euclidean, Jaccard or even Tanimoto to quantify the similarity of these binary vectors.

A better way is to also consider the frequency of terms in the sentence:

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-7.png)

Understanding concepts like these will cover the basics of your text analysis skill and put you in the right NLP track. In [this](https://nbviewer.org/gist/sparsh-ai/8939c5a39748a1fe7a5df0a5f024492e) notebook, 

### 5.2 - Using online job postings to improve your data science resume

We’re ready to expand our data science career. Six months from now, we’ll apply for a new job. In preparation, we begin to draft our resume. The early draft is rough and incomplete. It doesn’t yet cover our career goals or education.

Our resume draft is far from perfect. It’s possible that certain vital data science skills are not yet represented. If so, what are those missing skills? We decide to find out analytically. After all, we are data scientists! We fill in gaps in knowledge using rigorous analysis, so why shouldn’t we apply that rigorous analysis to ourselves?

First we need some data. We go online and visit a popular job-search site. The website offers millions of searchable job listings, posted by understaffed employers. A built-in search engine allows us to filter the jobs by keyword, such as *analyst* or *data scientist*. Additionally, the search engine can match jobs to uploaded documents. This feature is intended to search postings based on resume content. Unfortunately, our resume is still a work in progress. So instead, we search on the table of contents of a book! We copy and paste the first 15 listed sections of the table of contents into a text file.

Next, we upload the file to the job-search site. Material is compared against millions of job listings, and thousands of job postings are returned. Some of these postings may be more relevant than others; we can’t vouch for the search engine’s overall quality, but the data is appreciated. We download the HTML from every posting.

Our goal is to extract common data science skills from the downloaded data. We’ll then compare these skills to our resume to determine which skills are missing. To reach our goal, we’ll proceed like this:

1. Parse out all the text from the downloaded HTML files.
2. Explore the parsed output to learn how job skills are commonly described in online postings. Perhaps specific HTML tags are more commonly used to underscore job skills.
3. Try to filter out any irrelevant job postings from our dataset. The search engine isn’t perfect. Perhaps some irrelevant postings were erroneously downloaded. We can evaluate relevance by comparing the postings with our resume and the table of contents.
4. Cluster the job skills within the relevant postings, and visualize the clusters.
5. Compare the clustered skills to our resume content. We’ll then make plans to update our resume with any missing data science skills.

:::note

To address the problem at hand, we need to know how to do the following: 1) Measure similarity between texts. 2) Efficiently cluster large text datasets. 3) Visually display multiple text clusters. 4) Parse HTML files for text content.

:::

Our rough draft of the resume is stored in the file resume.txt. The full text of that draft is as follows:

```
Experience
1. Developed probability simulations using NumPy
2. Assessed online ad clicks for statistical significance using permutation testing
3. Analyzed disease outbreaks using common clustering algorithms
Additional Skills
1. Data visualization using Matplotlib
2. Statistical analysis using SciPy
3. Processing structured tables using Pandas
4. Executing K-means clustering and DBSCAN clustering using scikit-learn
5. Extracting locations from text using GeoNamesCache
6. Location analysis and visualization using GeoNamesCache and Cartopy
7. Dimensionality reduction with PCA and SVD using scikit-learn
8. NLP analysis and text topic detection using scikit-learn
```

Our preliminary draft is short and incomplete. To compensate for any missing material, we also use the partial table of contents of the book, which is stored in the file table_of_contents.txt. It covers the first 15 sections of the book, as well as all the top-level subsection headers. The table of contents file has been utilized to search for thousands of relevant job postings that were downloaded and stored in a job_postings directory. Each file in the directory is an HTML file associated with an individual posting. These files can be viewed locally in a web browser.

[Here](https://nbviewer.org/gist/sparsh-ai/d595527e0496c3466a3e34ef04c9b748) is the solution notebook.

## 6. Graph Networks

The study of connections can potentially yield billions of dollars. In the 1990s, two graduate students analyzed the properties of interconnected web pages. Their insights led them to found Google. In the early 2000s, an undergraduate began to digitally track connections between people. He went on to launch Facebook. Connection analysis can lead to untold riches, but it can also save countless lives. Tracking the connections between proteins in cancer cells can generate drug targets that will wipe out that cancer. Analyzing connections between suspected terrorists can uncover and prevent devious plots. These seemingly disparate scenarios have one thing in common: they can be studied using a branch of mathematics called *network theory* by some and *graph theory* by others.

*Network theory* is the study of connections between objects. These objects can be anything: people connected by relationships, web pages connected by web links, or cities connected by roads. A collection of objects and their dispersed connections is called either a *network* or a *graph*, depending on whom you ask. Engineers prefer to use the term *network*, while mathematicians prefer *graph*. For our intents and purposes, we’ll use the two terms interchangeably. Graphs are simple abstractions that capture the complexity of our entangled, interconnected world. Properties of graphs remain surprisingly consistent across systems in society and nature. Graph theory is a framework for mathematically tracking these consistencies. It combines ideas from diverse branches of mathematics, including probability theory and matrix analysis. These ideas can be used to gain useful real-world insights ranging from search engine page rankings to social circle clustering, so some knowledge of graph theory is indispensable to doing good data science.

In real life, clouds are constantly in motion, and so are many networks. Most networks worth studying are perpetually buzzing with dynamic activity. Cars race across networks of roads, causing traffic congestion near popular towns. In that same vein, web traffic flows across the internet as billions of users explore the many web links. Our social networks are also flowing with activity as gossip, rumors, and cultural memes spread across tight circles of close friends. Understanding this dynamic flow can help uncover friend groups in an automated manner. Understanding the flow can also help us identify the most heavily trafficked web pages on the internet. Such modeling of dynamic network activity is critical to the function of many large tech organizations. In fact, one of the modeling methods presented in this section led to the founding of a trillion-dollar company.

### 6.1 - Using basic graph theory to rank websites by popularity

There are many data science websites on the internet. Some sites are more popular than others. Suppose you wish to estimate the most popular data science website using data that is publicly available. This precludes privately tracked traffic data. What should you do? Network theory offers us a simple way of ranking websites based on their public links.

In [this](https://nbviewer.org/gist/sparsh-ai/326d2ad84d0965a4270048d64c5f4493) notebook, we will understand basics of graph theory like adjacency matrix, creating and plotting graphs using NetworkX library, in-degree calculation etc.

### 6.2 - Utilizing undirected graphs to optimize the travel-time between towns

In business logistics, product delivery time can impact certain critical decisions. Consider the following scenario, in which you’ve opened your own kombucha brewery. Your plan is to deliver batches of the delicious fermented tea to all the towns within a reasonable driving radius. More specifically, you’ll only deliver to a town if it’s within a two-hour driving distance of the brewery; otherwise, the gas costs won’t justify the revenue from that delivery. A grocery store in a neighboring county is interested in regular deliveries. What is the fastest driving time between your brewery and that store?

Normally, you could obtain the answer by searching for directions on a smartphone, but we’ll assume that existing tech solutions are not available (perhaps the area is remote and the local maps have not been scanned into an online database). In other words, you need to replicate the travel time computations carried out by existing smartphone tools. To do this, you consult a printed map of the local area. On the map, roads zigzag between towns, and some towns connect directly via a road. Conveniently, the travel times between connected towns are illustrated clearly on the map. We can model these connections using undirected graphs.

In real-life, many routes can exist between localized towns. Let's build a graph containing more than a dozen towns. These cities will be spread across multiple counties. Within our graph model, the travel-time between towns will increase when cities are in different counties. We'll assume that:

A. Our towns cover six different counties.

B. Each county contains 3-10 towns.

C. 90% of towns within a single county are directly connected by road. The average travel-time on a county road is 20 minutes.

D. 5% of cities across different counties are directly connected by a road. The average travel-time across an intra-county road is 45 minutes.

Here is how we implement this in python:

```python
# Modeling random intra-county roads
def add_random_edge(G, node1, node2, prob_road=0.9, 
                    mean_drive_time=20):
    if np.random.binomial(1, prob_road):
        drive_time = np.random.normal(mean_drive_time)
        G.add_edge(node1, node2, travel_time=round(drive_time, 2))

nodes = list(G.nodes())
for node1 in nodes[:-1]:
    for node2 in nodes[node1 + 1:]:
        add_random_edge(G, node1, node2)

# Modeling a second random county
def random_county(county_id):
    num_towns = np.random.randint(3, 10)
    G = nx.Graph()
    nodes = [(node_id, {'county_id': county_id})
            for node_id in range(num_towns)]
    G.add_nodes_from(nodes)
    for node1, _ in nodes[:-1]:
        for node2, _ in nodes[node1 + 1:]:
            add_random_edge(G, node1, node2)
    return G

# Merging two separate graphs
G = nx.disjoint_union(G, G2)

# Adding random inter-county roads
def add_intracounty_edges(G):
    nodes = list(G.nodes(data=True))
    for node1, attributes1 in nodes[:-1]:
        county1 = attributes1['county_id']
        for node2, attributes2 in nodes[node1:]:
            if county1 != attributes2['county_id']:
                add_random_edge(G, node1, node2,
                                prob_road=0.05, mean_drive_time=45)
    return G

# Simulating six interconnected counties
G = random_county(0)
for county_id in range(1, 6):
    G2 = random_county(county_id)
    G = nx.disjoint_union(G, G2)
G = add_intracounty_edges(G)

# Coloring nodes by county
county_colors = ['salmon', 'khaki', 'pink', 'beige', 'cyan', 'lavender']
county_ids = [G.nodes[n]['county_id'] 
              for n in G.nodes]
node_colors = [county_colors[id_] 
               for id_ in county_ids]
nx.draw(G, with_labels=True, node_color=node_colors)
```

![Untitled](/img/content-tutorials-raw-data-science-bookcamp-untitled-8.png)

Isn't it beautiful!

Suppose we want to know the fastest travel-time between *Town 0* and *Town 30*. In the process, we’ll need to compute the fastest travel-time between *Town 0* and every other town. We can use networkx for this:

`nx.shortest_path(G, weight='travel_time', source=0)[30]` -> [0, 13, 28, 30] is the shortest path.

Like this, there are lots of concepts in [this](https://nbviewer.org/gist/sparsh-ai/02ebeec31311ee862a1261da56224a81) notebook, that will help you strengthen your network analysis skills.

### 6.3 - Predicting future friendships from social network data

Welcome to FriendHook, Silicon Valley’s hottest new startup. FriendHook is a social networking app for college undergrads. To join, an undergrad must scan their college ID to prove their affiliation. After approval, undergrads can create a FriendHook profile, which lists their dorm name and scholastic interests. Once a profile is created, an undergrad can send *friend requests* to other students at their college. A student who receives a friend request can either approve or reject it. When a friend request is approved, the pair of students are officially *FriendHook friends*. Using their new digital connection, FriendHook friends can share photographs, collaborate on coursework, and keep each other up to date on the latest campus gossip.

The FriendHook app is a hit. It’s utilized on hundreds of college campuses worldwide. The user base is growing, and so is the company. You are FriendHook’s first data science hire! Your first challenging task will be to work on FriendHook’s friend recommendation algorithm.

Sometimes FriendHook users have trouble finding their real-life friends on the digital app. To facilitate more connections, the engineering team has implemented a simple friend-recommendation engine. Once a week, all users receive an email recommending a new friend who is not yet in their network. The users can ignore the email, or they can send a friend request. That request is then either accepted or rejected/ignored.

Currently, the recommendation engine follows a simple algorithm called the *friend-of-a-friend recommendation algorithm*. The algorithm works like this. Suppose we want to recommend a new friend for student A. We pick a random student B who is already friends with student A. We then pick a random student C who is friends with student B but not student A. Student C is then selected as the recommended friend for student A.

![The friend-of-a-friend recommendation algorithm in action. Mary has two friends: Fred and Bob. One of these friends (Bob) is randomly selected. Bob has two additional friends: Marty and Alice. Neither Alice nor Marty are friends with Mary. A friend of a friend (Marty) is randomly selected. Mary receives an email suggesting that she should send a friend request to Marty.](/img/content-tutorials-raw-data-science-bookcamp-untitled-9.png)

The friend-of-a-friend recommendation algorithm in action. Mary has two friends: Fred and Bob. One of these friends (Bob) is randomly selected. Bob has two additional friends: Marty and Alice. Neither Alice nor Marty are friends with Mary. A friend of a friend (Marty) is randomly selected. Mary receives an email suggesting that she should send a friend request to Marty.

Essentially, the algorithm assumes that a friend of your friend is also likely to be your friend. This assumption is reasonable but also a bit simplistic. How well does this assumption hold? Nobody knows! However, as the company’s first data scientist, it’s your job to find out. You have been tasked with building a model that predicts student behavior in response to the recommendation algorithm.

Your task is to build a model that predicts user behavior based on user profiles and social network data. The model must generalize to other colleges and universities. This generalizability is very important—a model that cannot be utilized at other colleges is worthless to the product team. Consider, for example, a model that accurately predicts behavior in one or two of the dorms at the sampled university. In other words, it requires specific dorm names to make accurate predictions. Such a model is not useful because other universities will have different dormitory names. Ideally, the model should generalize to all dormitories across all universities worldwide.

Once you’ve built the generalized model, you should explore its inner workings. Your goal is to gain insights into how university life facilitates new FriendHook connections.

The project goals are ambitious but also very doable. You can complete them by carrying out the following tasks:

1. Load the three datasets pertaining to user behavior, user profiles, and the user friendship network. Explore each dataset, and clean it as required.
2. Build and evaluate a model that predicts user behavior based on user profiles and established friendship connections. You can optionally split this task into two subtasks: build a model using just the friendship network, and then add the profile information and test whether this improves the model’s performance.
3. Determine whether the model generalizes well to other universities.
4. Explore the inner workings of the model to gain better insights into student behavior.

Our data contains three files stored in a friendhook directory. These files are CSV tables and are named Profiles.csv, Observations.csv, and Friendships.csv. Let’s discuss each table individually.

**The Profiles table**

Profiles.csv contains profile information for all the students at the chosen university. This information is distributed across six columns: `Profile_ID`, `Sex`, `Relationship_ Status`, `Major`, `Dorm`, and `Year`. Maintaining student privacy is very important to the FriendHook team, so all the profile information has been carefully encrypted. FriendHook’s encryption algorithm takes in descriptive text and returns a unique, scrambled 12-character code known as a *hash code*. Suppose, for example, that a student lists their major as physics. The word *physics* is then scrambled and replaced with a hash code such as *b90a1221d2bc*. If another student lists their major as art history, a different hash code is returned (for example, *983a9b1dc2ef*). In this manner, we can check whether two students share the same major without necessarily knowing the identity of that major. All six profile columns have been encrypted as a precautionary measure. Let’s discuss the separate columns in detail:

- `Profile_ID`—A unique identifier used to track each student. The identifier can be linked to the user behaviors in the `Observations` table. It can also be linked to FriendHook connections in the `Friendships` table.
- `Sex`—This optional field describes the sex of a student as `Male` or `Female`. Students who don’t wish to specify a gender can leave the `Sex` field blank. Blank inputs are stored as empty values in the table.
- `Relationship_Status`—This optional field specifies the relationship status of the student. Each student has three relationship categories to choose from: `Single`, `In a Relationship`, or `It’s Complicated`. All students have a fourth option of leaving this field blank. Blank inputs are stored as empty values in the table.
- `Major`—The chosen area of study for the student, such as physics, history, economics, etc. This field is required to activate a FriendHook account. Students who have not yet picked their major can select `Undecided` from among the options.
- `Dorm`—The name of the dormitory where the student resides. This field is required to activate a FriendHook account. Students who reside in off-campus housing can select `Off-Campus Housing` from among the options.
- `Year`—The undergraduate student’s year. This field must be set to one of four options: `Freshman`, `Sophomore`, `Junior`, or `Senior`.

**The Observations table**

Observations.csv contains the observed user behavior in response to the emailed friend recommendation. It includes the following five fields:

- `Profile_ID`—The ID of the user who received a friend recommendation. The ID corresponds to the profile ID in the `Profiles` table.
- `Selected_Friend`—An existing friend of the user in the `Profile_ID` column.
- `Selected_Friend_of_Friend`—A randomly chosen friend of `Selected_Friend` who is not yet a friend of `Profile_ID`. This random friend of a friend is emailed as a friend recommendation for the user.
- `Friend_Requent_Sent`—A Boolean column that is `True` if a user sends a friend request to the suggested friend of a friend or `False` otherwise.
- `Friend_Request_Accepted`—A Boolean column that is `True` only if a user sends a friend request and that request is accepted.

This table stores all the observed user behaviors in response to the weekly recommendation email. Our goal is to predict the Boolean outputs of the final two table columns based on the profile and social networking data.

**The Friendships table**

Friendships.csv contains the FriendHook friendship network corresponding to the selected university. This network was used as input into the friend-of-a-friend recommendation algorithm. The `Friendships` table has just two columns: `Friend A` and `Friend B`. These columns contain profile IDs that map to the `Profile_ID` columns of the `Profiles` and `Observations` tables. Each row corresponds to a pair of FriendHook friends. For instance, the first row contains IDs `b8bc075e54b9` and `49194b3720b6`. From these IDs, we can infer that the associated students have an established FriendHook connection. Using the IDs, we can look up the profile of each student. The profiles then allow us to explore whether the friends share the same major or reside together in the same dorm.

To address the problem at hand, we need to know how to do the following:

- Analyze network data using Python
- Discover friendship clusters in social networks
- Train and evaluate supervised machine learning models
- Probe the inner workings of trained models to draw insights from our data

FriendHook is a popular social networking app designed for college campuses. Students can connect as friends in the FriendHook network. A recommendation engine emails users weekly with new friend suggestions based on their existing connections; students can ignore these recommendations, or they can send out friend requests. We have been provided with one week’s worth of data pertaining to friend recommendations and student responses. That data is stored in the friendhook/Observations.csv file. We’re provided with two additional files: friendhook/Profiles.csv and friendhook/Friendships.csv, containing user profile information and the friendship graph, respectively. The user profiles have been encrypted to protect student privacy. Our goal is to build a model that predicts user behavior in response to the friend recommendations. We will do so by following these steps:

1. Load the three datasets containing the observations, user profiles, and friendship connections.
2. Train and evaluate a supervised model that predicts behavior based on network features and profile features. We can optionally split this task into two subtasks: training a model using network features, and then adding profile features and evaluating the shift in model performance.
3. Check to ensure that the model generalizes well to other universities.
4. Explore the inner workings of our model to gain better insights into student behavior.

[This](https://nbviewer.org/gist/sparsh-ai/fa15c92c2570396a3b25780f089a2dee) is the solution notebook.

## Conclusion

We learned some very basic concepts of data science and machine learning via tutorials and case studies. Feel free to experiment and tinker with the code and if you have suggestion, please put it under `git > issues` tab. For convenience, here is the list of Jupyter notebooks:

1. Tutorial - [Probability theory and the sample space analysis in python](https://nbviewer.org/gist/sparsh-ai/0818e19d4a51c41dd14adb97ce7f5e58)
2. Tutorial - [Finding the winning strategy in a sum-21 card game](https://nbviewer.org/gist/sparsh-ai/c0372e84f4177b762bcd196f1c04092f)
3. Tutorial - [Statistical analysis in python](https://nbviewer.org/gist/sparsh-ai/74d263fe9472f3aa3d850027c85f8083)
4. Tutorial - [Clustering analysis in python](https://nbviewer.org/gist/sparsh-ai/b6bf62384f006dbe48edeff0684546ee)
5. Tutorial - [Geography location visualization and analysis in python](https://nbviewer.org/gist/sparsh-ai/fc74f531f3a99c7838db48d76d1b2b09)
6. Tutorial - [Measuring text similarities in python](https://nbviewer.org/gist/sparsh-ai/8939c5a39748a1fe7a5df0a5f024492e)
7. Tutorial - [Using basic graph theory to rank websites by popularity](https://nbviewer.org/gist/sparsh-ai/326d2ad84d0965a4270048d64c5f4493)
8. Tutorial - [Utilizing undirected graphs to optimize the travel-time between towns](https://nbviewer.org/gist/sparsh-ai/02ebeec31311ee862a1261da56224a81)
9. Case study - [Finding the winning strategy in a red-color-win card game](https://nbviewer.org/gist/sparsh-ai/7dc86ca2ba5ef176332a0c780d251308)
10. Case study - [Assessing online ad-clicks for significance](https://nbviewer.org/gist/sparsh-ai/d2f1cb590c7f6a37c46a51fb7fa3265e)
11. Case study - [Tracking disease outbreaks using news headlines](https://nbviewer.org/gist/sparsh-ai/5593bd827a8d9b5704177215c09310a6)
12. Case study - [Using online job postings to improve your data science resume](https://nbviewer.org/gist/sparsh-ai/d595527e0496c3466a3e34ef04c9b748)
13. Case study - [Predicting future friendships from social network data](https://nbviewer.org/gist/sparsh-ai/fa15c92c2570396a3b25780f089a2dee)