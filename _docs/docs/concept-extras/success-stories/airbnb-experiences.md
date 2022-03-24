# Airbnb Experiences

[Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled.png)

Airbnb Experiences is a one-of-a-kind offering by Airbnb where people get to experience the local culture of their hosts. These are handcrafted activities designed and led by expert hosts that offer a unique taste of the local scene and culture. Each experience is vetted for quality by a team of editors before it makes its way onto the platform. As the number of Experiences grew, Search & Discoverability, as well as Personalization, have become very important factors for the growth and success of the marketplace. But when Airbnb started the Experiences offering, they were not sure how to rank them.

In this story, we describe the stages of Experience Ranking development using machine learning at different growth phases of the marketplace, from small to mid-size and large. We will learn how Airbnb built and iterated on a machine learning Search Ranking platform for a new two-sided marketplace and how they helped it grow. 

<!---------------------------->

## Launch

Duration: 2

Airbnb Experiences was launched in November 2016 with 500 Experiences in 12 cities worldwide.

During 2017, Airbnb grew the business to 5,000 Experiences in 60 cities.

In 2018, the rapid growth continued, and Airbnb managed to bring Experiences to more than 1,000 destinations, including unique places like Easter Island, Tasmania, and Iceland. Airbnb finished the year strong with more than 20,000 active Experiences.

<!---------------------------->

## Need for Personalization

Duration: 2

As the number of Experiences grew, Search & Discoverability, as well as Personalization, have become very important factors for the growth and success of the marketplace. But when Airbnb started the Experiences offering, they were not sure how to rank them.

In the beginning, the amount of Experiences that needed to be ranked in Search was small, and Airbnb just started collecting data on user interactions with Experiences (impressions, clicks, and bookings). At that moment, the best choice was to just randomly re-rank Experiences daily, until a small dataset is collected for development of the Stage 1 ML model.

<!---------------------------->

## Stage 1: MVP

Duration: 5

### Data

To train the first machine learning model for ranking Experiences, Airbnb collected search logs (i.e. clicks) of users who ended up making bookings.

### Labels

When labeling training data, Airbnb was mainly interested in two labels: experiences that were booked (which were treated as positive labels) and experiences that were clicked but not booked (which were treated as negative labels). In this manner, Airbnb collected a training dataset of 50,000 examples.

### Features

They decided to rank solely based on *Experience Features.* In total they built 25 features, some of which were:

- **Experience duration** (e.g. 1h, 2h, 3h, etc.)
- **Price** and **Price-per-hour**
- **Category** (e.g. cooking class, music, surfing, etc.)
- **Reviews** (rating, number of reviews)
- **Number of bookings** (last 7 days, last 30 days)
- **Occupancy of past and future instances** (e.g. 60%)
- **Maximum number of seats** (e.g. max 5 people can attend)
- **Click-through rate**

### Training

Given the training data, labels, and features, Airbnb used the Gradient Boosted Decision Tree (GBDT) model. At this point, they treated the problem as binary classification with a log-loss loss function.

### Offline evaluation

*AUC and NDCG*: They re-ranked the Experiences based on model scores (probabilities of booking) and tested where the booked Experience would rank among all Experiences the user clicked (the higher the better).

*Partial dependency plot*: In addition, to get a sense of what a trained model learned, they plotted partial dependency plots for several most important Experience features.

- Experiences with more bookings per 1k viewers will rank higher
- Experiences with higher average review rating will rank higher
- Experiences with lower prices will rank higher

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-1.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-1.png)

### Online evaluation

Since offline testing often has too many assumptions, e.g. in Airbnb's case it was limited to re-ranking what users clicked and not the entire inventory, they conducted an online experiment, i.e. A/B test, as the next step. They compared the Stage 1 ML model to the rule-based random ranking in terms of the number of bookings. The results were very encouraging as they were able to improve bookings by +13% with the Stage 1 ML ranking model.

### MLOps

The entire ranking pipeline, including training and scoring, was implemented offline and ran daily in Airflow. The output was just a complete ordering of all Experiences, i.e. an ordered list, which was uploaded to production machines and used every time a search was conducted to rank a subset of Experiences that satisfied the search criteria.

<!---------------------------->

## Stage 2: Personalization

Duration: 10

The stage 1 model was limited to using only Experience Features, and as a result, the ranking of Experiences was the same for all users. In addition, all query parameters (number of guests, dates, location, etc.) served only as filters for retrieval (e.g. fetch Paris Experiences available next week for 2 guests), and the ranking of Experiences did not change based on those inputs.

Two randomly chosen Experiences are likely to be very different, e.g. a Cooking Class vs. a Surf Lesson. At the same time, guests may have different interests and ideas of what they want to do on their trip, and it is our goal to capture that interest fast and serve the right content higher in search results. So, the next step was to add the Personalization capability to the ML ranking model.

### Features based on Airbnb's home booking data

- Booked Home location
- Trip dates
- Trip length
- Number of guests
- Trip price (Below/Above Market)
- Type of trip: Family, Business
- First trip or returning to location
- Domestic / International trip
- Lead days

Let's understand and validate the intuition behind 2 engineered features: Distance between Booked Home and Experience, and Experience available during Booked Trip.

**Distance between Booked Home and Experience.** Knowing Booked Home location (latitude and longitude) as well as Experience meeting location, we can compute their distance in miles. Data shows that users like convenience, i.e. large fraction of booked Airbnb Experiences are near booked Airbnb Home.

**Experience available during Booked Trip.** Given Home check-in and check-out dates, we have an idea on which dates the guest is looking to book Experiences and can mark Experiences as available or not during those dates.

These two features (in addition to others) were used when training the new ML ranking model. Below we show their partial dependency plots.

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-2.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-2.png)

The plots confirmed that features behavior matches what we intuitively expected the model will learn, i.e. Experiences that are closer to Booked Home will rank higher (have higher scores), and Experiences that are available for Booked Trip dates will rank higher (which is very convenient because even in dateless search we can leverage trip dates).

### Features based on clickstream data

Given the user’s short-term search history, we can infer useful information that can help us personalize future searches:

- **Infer user interest in certain categories:** For example, if the user is mostly clicking on *Music* Experiences we can infer that the user’s interest is in *Music.*
- **Infer the user’s time-of-day availability:** For example, if the user is mostly clicking on *Evening* Experiences we can infer that the user is available at that time of day.
- **Category Intensity:** Weighted sum of user clicks on Experiences that have that particular category, where the sum is over the last 15 days (d_0 to d_now) and A is the number of actions (in this case clicks) on certain category on day d.
- **Category Recency:** Number of days that passed since the user last clicked on an Experience in that category.
    
    Let's validate the intuition for the 2 engineered features - intensity and recency:
    
    ![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-3.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-3.png)
    
    Airbnb built the same types of features (intensity & recency) for several different user actions, including wishlisting and booking a certain category.
    
- **Time of Day Personalization:** Different Experiences are held at different times of day (e.g. early morning, evening, etc.). Similar to how we track clicks on different categories, we can also track clicks on different times of day and compute Time of Day Fit between the user’s time-of-day percentages and the Experience’s time of day, as described below:
    
    ![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-4.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-4.png)
    
    As it can be observed, the model learned to use this feature in a way that ranks Experiences held at time of day that the user prefers higher.
    

### Preprocessing

- Generated the training data that contains those features by reconstructing the past based on search logs.
- Computed the personalization features only if the user interacted with more than one Experience and category (to avoid cases where the user clicked only one Experience / category, e.g. Surfing, and ended up booking that category).

### Training

As search traffic contains searches by both logged-in and logged-out users, Airbnb found it more appropriate to train two models, one with personalization features for logged-in users and one without personalization features that will serve log-out traffic. The main reason was that the logged-in model trained with personalization features relies too much on the presence of those features, and as such is not appropriate for usage on logged-out traffic.

### Online evaluation

Results showed that Personalization matters as they were able to improve bookings by +7.9% compared to the Stage 1 model.

### MLOps

- They created a look-up table keyed off of user id that contained personalized ranking of all Experiences for that user, and use key 0 for ranking for logged-out users.
- This required daily offline computation of all these rankings in Airflow by computing the latest features and scoring the two models to produce rankings. Because of the high cost involved with pre-computing personalized rankings for all users (O(NM), where N is the number of users and M is the number of Experiences), they limited N to only 1 million most active users.
- The personalization features at this point were computed only daily, which means that they have up to one day latency (also a factor that can be greatly improved with more investment in infrastructure).

Stage 2 implementation was a temporary solution used to validate personalization gains before investing more resources in building an Online Scoring Infrastructure in Stage 3, which was needed as both N and M are expected to grow much more.

<!---------------------------->

## Stage 3: Move to Online Scoring

Duration: 10

After Airbnb demonstrated significant booking gains from iterating on their ML ranking model and after inventory and training data grew to the extent where training a more complex model is possible, they were ready to invest more engineering resources to build an Online Scoring Infrastructure and target more booking gains.

### Features

Moving to Online Scoring unlocks a whole new set of features that can be used: Query Feature

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-5.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-5.png)

This means that we would be able to use the entered location, number of guests, and dates to engineer more features. For example,

- we can use the entered location, such as city, neighborhood, or place of interest, to compute Distance between Experience and Entered Location. This feature helps us rank those Experiences closer to entered location higher.
- we can use the entered number of guests (singles, couple, large group) to calculate how it relates to the number of guests in an average booking of Experience that needs to be ranked. This feature helps us rank better fit Experiences higher.

In the online setting, Airbnb was also able to leverage the user’s browser language setting to do language personalization on the fly. Below is an example of Stage 3 ML model ranking Experiences offered in Russian higher when browser language is Russian.

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-6.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-6.png)

Finally, in the online setting we also know the Country which the user is searching from. We can use the country information to personalize the Experience ranking based on Categories preferred by users from those countries. Airbnb used this information to engineer several personalization features at the Origin — Destination level.

### Training

To train the model with *Query Features* Airbnb first added them to their historical training data. The inventory at that moment was 16,000 Experiences and they had more than 2 million labeled examples to be used in training with a total of 90 ranking features. As mentioned before, they trained two GBDT models:

- **Model for logged-in users***,* which **uses *Experience Features*, *Query Features,* and *User (Personalization) Features*
- **Model for logged-out traffic**, which **uses *Experience & Query Features,* trained using data (clicks & bookings) of logged-in users but not considering *Personalization Features*

The advantage of having an online scoring infrastructure is that we can use logged-in model for far more uses than before, because there is no need to pre-compute personalized rankings as we did in Stage 2. Airbnb used the logged-in model whenever personalization signals were available for a particular user id, else the model fall back to using logged-out model.

### Online evaluation

Airbnb conducted an A/B test to compare the Stage 3 models to Stage 2 models. Once again, they were able to grow the bookings, this time by +5.1%.

### Implementation

To implement online scoring of thousands of listings in real time, Airbnb have built their own ML infra in the context of their search service. There are mainly three parts of the infrastructure, 1) getting model input from various places in real time, 2) model deployment to production, and 3) model scoring.

The model requires three types of signals to conduct scoring: Experience Features, Query Features, and User Features. Different signals were stored differently depending on their size, update frequency, etc. Specifically, due to their large size (hundreds of millions of user keys), the User Features were stored in an online key-value store and search server boxes can look them up when a user does the search. The Experience Features are on the other hand not that large (tens of thousands of Experiences), and as such can be stored in memory of the search server boxes and read directly from there. Finally, the Query Features are not stored at all, and they are just read as they come in from the front end.

Experience and User Features are both updated daily as the Airflow pipeline feature generation job finishes. Airbnb is also working on transitioning some of the features to the online world, by using a key-value store that has both read and write capabilities which would allow them to update the features instantly as more data comes in (e.g. new experience reviews, new user clicks, etc.).

In the model deployment process, they transform the GBDT model file, which originally came from their training pipeline in JSON format, to an internal Java GBDT structure, and load it within the search service application when it starts.

During the scoring stage, they first pull in all the features (User, Experience, and Query Features) from their respective locations and concatenate them in a vector used as input to the model. Next, depending on if User Features are empty or not they decide which model to use, i.e. logged-out or logged-in model, respectively. Finally, they return the model scores for all Experiences and rank them on the page in descending order of the scores.

<!---------------------------->

## Stage 4: Handle Business Rules

Duration: 5

Up to this point the ranking model’s objective was to grow bookings. However, a marketplace such as Airbnb Experiences may have several other secondary objectives, as we call them Business Rules, that can be achieved through machine learning.

One such important Business Rule is to Promote Quality. From the beginning they believed that if guests have a really good experience they will come back and book Experiences again in the near future. For that reason, they started collecting feedback from users in terms of 1) star ratings, ranging from 1 to 5, and 2) additional structured multiple-response feedback on whether the Experience was unique, better than expected, engaging, etc.

As more and more data became available to support their rebooking hypothesis, the trend became more clear. As it can be observed on the left in the figure below, guests who have had a great experience (leave a 5-star rating) are 1.5x more likely to rebook another Experience in the next 90 days compared to guests who have had a less good time (leave 4 star rating or lower).

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-7.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-7.png)

This motivated them to experiment with their objective function, where they changed their binary classification (+1 = booked, -1 = click & not booked) to introduce weights in training data for different quality tiers (e.g. highest for very high quality bookings, lowest for very low quality bookings). The quality tiers were defined by their Quality Team via data analysis. For example:

- *very high quality* Experience is one with >50 reviews, >4.95 review rating and >55% guests saying the Experience was *unique* and *better than expected.*
- *very low quality* Experience **is one with **>10 reviews, <4.7 review rating.

When testing the model trained in such a way the A/B test results (on the right in the figure above) showed that they can leverage machine learning ranking to get more of v*ery high quality* bookings and less of v*ery low quality bookings*, while keeping the overall bookings neutral.

In a similar way, they successfully tackled several other secondary objectives:

- **Discovering and promoting potential *new* *hits* early** using cold-start signals and promoting them in ranking (this led to **+14% booking gain for *new hits*** and **neutral overall bookings)
- **Enforcing diversity** in the **top 8 results** such that they can show the diverse set of categories, which is especially important for low-intent traffic (this led to **+2.3% overall booking gain**).
- **Optimize Search without Location for Clickability** For *Low Intent* users that land on their webpage but do not search with specified location they think a different objective should be used. Their first attempt was to choose the *Top 18* from *all locations* based on their ranking model scores and then **re-rank** based on **Click-through-rate** (this led to **+2.2% overall booking gain** compared to scenario where they do not re-rank based on CTR).

<!---------------------------->

## Monitoring and Explaining

Duration: 10

For any two-sided marketplace it is very important to be able to explain why certain items rank the way they do.

In Airbnb's case it is valuable because they can:

- Give hosts concrete feedback on what factors lead to improvement in the ranking and what factors lead to decline.
- Keep track of the general trends that the ranking algorithm is enforcing to make sure it is the behavior we want to have in our marketplace.

To build out this capability they used [Apache Superset](https://superset.incubator.apache.org/) and [Airflow](https://airflow.apache.org/) to create two dashboards:

- Dashboard that tracks rankings of specific Experiences in their market over time, as well as values of feature used by the ML model.
- Dashboard that shows overall ranking trends for different groups of Experiences (e.g. how 5-star Experiences rank in their market).

In the figure below, there is an example of an Experience whose ranking (left panel) improved from position 30 to position 1 (top ranked). To explain why, we can look at the plots that track various statistics of that Experience (right panel), which are either directly used as features in the ML model or used to derive features.

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-8.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-8.png)

It can clearly be observed that the ranking improved because number of reviews grew from 0 to 60, while maintaining a 5.0 review rating and >60% users said the Experience was better than expected. In addition, the host lowered the price, which may also have lead to the improvement of ranking.\

In the next figure, there is an example of an Experience whose ranking degraded from position 4 to position 94. Once more, the signals which the ranking model uses as input can tell the story.

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-9.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-9.png)

The Experience started getting bad reviews (avg. rating reduced from 4.87 to 4.82), host increased the price by $20, and overall booking numbers decreased. In addition, in that market the time of day that Experience is held at (early morning) became less and less popular (slight seasonality effect). All these factors combined lead to the ranking decline.

In addition, to be able to track what kind of ranking behavior they are enforcing it was useful to look at how certain groups of Experiences rank in their markets. In the figure below, there is a snapshot of dashboard where we can track average ranking (lower is better) of different dimensions.

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-10.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-10.png)

For example, the left plot shows that Experiences with >50 reviews rank much better than experiences with 10–30 reviews. Looking at the other two plots we can see that on average Experiences with review rating of >4.9 rank the best (much better than ones with lower average rating) and that Experiences for which >55% users say their experience was unique rank much better than non-unique Experiences.

This type of dashboard is useful for making business decisions and modifications to the ranking algorithm to enforce better behaviors. For example, based on the figure that shows ranking trends of different price range groups (shown below) they noticed that very low price Experiences have too big of an advantage in ranking. They decided to try to reduce that advantage by removing price as one of the signals used by the ranking model.

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-11.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-11.png)

The result of removing the price and retraining the model was that the difference in ranking reduced (after July 1st), without hurting overall bookings. This demonstrates the usefulness of reporting and how ranking can be manipulated using machine learning to achieve desired ranking behavior.

<!---------------------------->

## Conclusion

Duration: 2

### Summary of Business Impact so far

![/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-12.png](/img/content-concepts-case-studies-raw-case-studies-airbnb-experiences-untitled-12.png)

The main take-away is: “Don’t wait until you have big data, you can do quite a bit with small data to help grow and improve your business.”

### Next steps

- **Training data construction** (by logging the feature values at the time of scoring instead of reconstructing them based on best guess for that day)
- **Loss function** (e.g. by using pairwise loss, where they compare booked Experience with Experience that was ranked higher but not booked, a setup that is far more appropriate for ranking)
- **Training labels** (e.g. by using utilities instead of binary labels, i.e. assigning different values to different user actions, such as: 0 for impression, 0.1 for click, 0.2 for click with selected date & time, 1.0 for booking, 1.2 for high quality booking)
- **Adding more real-time signals** (e.g. being able to personalize based on immediate user activities, i.e. clicks that happened 10 minutes ago instead of 1 day ago)
- **Explicitly asking users about types of activities they wish to do on their trip** (so they can personalize based on declared interest in addition to inferred ones)
- **Tackling position bias** that is present in the data they use in training
- **Optimizing for additional secondary objectives**, such as helping hosts who host less often than others (e.g. 1–2 a month) and hosts who go on vacation and come back
- **Testing different models** beyond GBDT
- **Finding more types of supply that work well in certain markets** by leveraging predictions of the ranking model.
- **Explore/Exploit framework**
- **Test human-in-the-loop approach** (e.g. Staff picks)

### Links and References

1. [Originally published on medium by Mihajlo Grbovic, Senior Machine Learning Scientist @ Airbnb.](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)
2. Airbnb Experiences Ranking Algorithm Explained - [Part I](https://youtu.be/35HfN_C7-2c), [Part II](https://youtu.be/emxPgRbtTFg), and [Part III](https://youtu.be/Ox1BIXSdlYo).
3. [https://arxiv.org/pdf/1911.05887.pdf](https://arxiv.org/pdf/1911.05887.pdf)
4. [https://digital.hbs.edu/platform-rctom/submission/airbnb-customizing-recommendations-for-every-trip/](https://digital.hbs.edu/platform-rctom/submission/airbnb-customizing-recommendations-for-every-trip/)
5. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) `Airbnb`
6. [Applying Deep Learning To Airbnb Search](https://arxiv.org/abs/1810.09591) ([Paper](https://arxiv.org/pdf/1810.09591.pdf)) `Airbnb`
7. [Managing Diversity in Airbnb Search](https://arxiv.org/abs/2004.02621) ([Paper](https://arxiv.org/pdf/2004.02621.pdf)) `Airbnb`
8. [Improving Deep Learning for Airbnb Search](https://arxiv.org/abs/2002.05515) ([Paper](https://arxiv.org/pdf/2002.05515.pdf)) `Airbnb`

### Have a Question?

- [Fill out this form](https://form.jotform.com/211377288388469)
- [Raise issue on Github](https://github.com/recohut/reco-step/issues)