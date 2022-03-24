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
# Discussions on Non-personalised Recommendations Metrics - 5 Star Evaluation

The most basic way to provide recommendations is a non-personalised one. Non-personalised recommendations don't take user's individual preferences nor context into consideration. 
Take for instance a newly create client at Amazon. He wouldn't have bough any item on the marketplace, so Amazon doesn't know what are the particular tastes of this new person, so the best way to start with any possible recommendation which the new customer would like are what other clients, regardless of any their individual tastes, are buying.  

The recommendations are always based on a score, which can be calculated from basic metrics, such as:  

* Explicit metrics: Evaluations given directly by the customer, *i.e.*, ratings, overall score, likes, etc.
* Implicit metrics: Implict user behaviour, such as clicks, view, time spent, etc.  
  
Each of these types of metrics comes with pros and cons. **Explicit metrics** are good because we directly ask the user about what he thinks. However, some questions/problems arise:  

* Question: Do de users know how to really reflect their tastes on a one number final metric?
* Problem: Users ratings varies between each other. Some people rate a movie 5 stars when it was like he expected, others have this bar at 3 stars and only give a 5 stars rating when the movie surpass all his expectations.
* Problem: Users ratings varies over time. If a user watch the same movie months after he watched for the first time, his evaluation can change.  


When we are talking about **Implicit Metrics**, we assume that the user's actions speak more than its own words. Besides, collecting behaviour data is faster and without the human judgment that can affect the explicit metrics. But it also comes with some subtleties that we have to concern about:  

* Dealing with a one number summary is easier to work than with multiple variables. If you're tracking multiple data, such as number of clicks, views and time, we have to come up with a way to average them, giving the proper weights, and this weights of course change between domains.  
* Given facebook on the last days and the [disaster with Cambridge Analytics](https://www.cnbc.com/2018/04/10/facebook-cambridge-analytica-a-timeline-of-the-data-hijacking-scandal.html) there are lots of spaces to analyse where people might not be confortable to make it available to you.

# Problems on the 5 star ratings

<img src="images/notebook2_image1.jpg">
<!-- #endregion -->

One of the main types of ratings used today is the 5 stars ratings. Used by companies such as Amazon, previously by Netflix and TripAdvisor, these companies use these evaluations in order to trace a profile between people and their products, using non-personalised and personalised recommendations.

There is a lot of discussions on whether this type of evaluation if it is effective or not. The following observations and references written below were extracted from the first course on [Coursera's Recommender System Specialisation](https://www.coursera.org/learn/recommender-systems-introduction/home/info), more specifically on the last week of the first course, an interview with Martijn Willeman on [Psychology of Preference & Rating](https://www.coursera.org/learn/recommender-systems-introduction/lecture/ch4oV/psychology-of-preference-rating-interview-with-martijn-willemsen)

## What is a Star Rating? 

That is the first question the interview began with. and "**What does it mean to give something a 1 start? Or a 5 star?**"

Ranking by symbols date from the 19th century, where [Mariana Starke](https://en.wikipedia.org/wiki/Star_(classification)#History), in her 1820's guidebook, used exclamations points to rate works of art of specific value. **My** guess is, by some marketing play, people made an effort to translate the symbols to stars, an image that transmits more 'richness' by its shiny interpretation and its abstract relatioship to diamonds.  

After the readoption of the 5 start ratings, popularised again by Netflix and Uber, lots of companies started to use this form of quality evaluation. The difficulties start when the scale is not properly set. Of course we have boundaries for it (5 points in a whole start review and 10 points in a half star scale review), but we aren't usually provided concrete references on which to base ourselves next time we rate something. This scenario brings instability to ratings distributions and it is made worse because of our human perception, that tends to review it based on instantenous (on the fly) evaluations.

## Ratings on different scales - contextual and temporal evaluation

People's taste can vary a lot depending on the context. If you just came back from work on a friday at 20:00, you'd probably don't like a book on how to be effective on your work. But you would totally accept this book during the weekend, after a long rest and maybe thinking about your future. If this book had been offered at this person in these two different moments and the person had to evaluate to quality of the recommendation, it'd probably receive very different evaluations in these two moments. 

Reviews variance don't even have to relate to the product. As we're going to see in the next sections, people can rate a taxi driver with a one star, just because he didn't make to your appointment in time, considering that you were the person who actually left home too late since the beggining.

In these context, a summary of the ratings (averages, max, min) could be calculated over totally different situations, giving a unreliable and unstable value to the end users.

Secondly, it has been studied by [Fischhoff, 1991](https://www.cmu.edu/dietrich/sds/docs/fischhoff/ValueElictationAnythngThere.pdf)how much a person takes into consideration at the moment of reviewing something. The paper shows that people, when making this sort of decision, pratically does it spontaneously, without actually considering complex and historic factors, such as personal past taste or match with similar past itens. It is basically a momentanly "*Yes, I liked it* / *No I didn't like it*".

The problem of this is that people can actually differ on their ratings for a product they've seen before after testing it again. This intersects a little with the contextualisation the rating is done but it also adds even more noise on what would be defined a person's taste or a product review.

## Human's susceptibility to be influenciated

Another interesting study conducted by [*Cosley et. al., 2003*](https://pdfs.semanticscholar.org/d7d5/47012091d11ba0b0bf4a6630c5689789c22e.pdf) intended to show how people are suscetible to previous feedbacks on the way the way they do an evaluation / review. This study basically adapted the original psychology study from [Asch](https://en.wikipedia.org/wiki/Asch_conformity_experiments) where people had to state which line on a paper was was equal as one reference line. Right after a person started to evaluate, a team from the study started to give opinions about the problem, deviating the person to reach and individual conclusion. In the study, even with it having an obvious result, a significant amount of participants actually ended up deciding the wrong answer.

The adaptation of the original study was applied to eletronic product reviews. They brought a set of people to the study and also a list of products that these people had already bought in the near past, in order to avoid temporal discrepancies in the evaluation. They splited the people in two groups. Both had to reevaluate the reviews given to the products they had bough before, but one group would get a 'recommendation' of what was the predicted evaluation for that person. After the tests, it was shown that, among people who had been shown the predicted recommendation score, they actually ended up following the given value, whether that was a lower or a higher score than the score the same person had given before.

This indication of people's suscetibility for feedbacks in their evaluation creates a sort of loop effect in the main recommendation systems front-ends, where products with already good reviews end up receiving good reviews just because, it's got good reviews, and the same goes to bad reviewed product, who can end up on loop of bad reviews and be never or hardly able to leave this situation.

## The harshness on 5 star ratings thresholds made by driver's companies

As a bias from working actually on a mobility company (Mytaxi - 19.04.2018), I want also to give some opinions about the way the 5 star rating is used by the main companies in transportation, whether it is with private drivers (Uber, Lync) or Federated Taxi Companies (Mytaxi, X).

These companies today work with ratings in a way to gamify the quality of the drivers using their product. By allowing  people to evaluate how their last trip was, in terms of driver and sometimes car, the idea is that drivers would try to provide the best service as possible, as a few bad ratings could put his job at risk. 

The first company to start using that in a global scale was Uber, which began with a big anti-taxi flag in the beginning. People were actually tired of acommodated taxists providing bad services and not being able to do anything, besides paying a relativelly high price. Uber took this in their hands and tried to give the power back to the users, making them responsible whether a driver would continue to work for Uber or not.

For some time this approach worked! Taxists, by seeing someone providing better experience by a lower price, had to reinvent themselves and it didn't take long to see taxi drivers with newer cars offering water, candy and other facilities to their passenger. **Side note**: I'm not going to enter in the regulation and taxes fields, as this would be an entire other story. I just wanted to opinate on the impact of putting the power to keep a driver's job on the population. Lets continue.

Has the rating system rating worked? We could say so. Is it fair? That's where our math and rating system problems knowledge goes in and question the validity of this.

<img src="images/notebook2_image2.jpg" width="350">

The picture above show some extreme example. As the decision to keep a driver in the plataform in solely dependent on passengers ratings, this opens door to some frauds against the drivers as well.

"Oh but this a too extreme example, people don't do this". Unfortunately, because of the lack of knowledge on how the driver's rating works in terms of deciding if he works or not for a company, together with the scale variability on context, people end up (**prejudicando**) the driver without actually know it.

Normally, these companies works with some hard defined threshold, *i.e.*, if a driver gets an average rating below this value, they have the risk of being excluded. Also, on a less hard threshold, drivers with higher rates can be prioritised to receive a 'drive passenger' offer. Coming from context and, in this case, personal variability on ratings, we see that people also rate drivers differently. Some gives a driver a five star if it was they way they expected, *i.e.* they were brought from A to B with no problems. Others only give a 5 star if the driver did something crazily unexpected. Half expected time to get to destination, chocolate and wifi on board, also chose charmander as a starting pokemon, it can be **whatever** he or she thinks is spectacular. If it was a normal ride, they would give it a traditional 3 stars rating. Coming back to the hard-defined threshold value used by the companies, they all expect the passenger scale stays on a average rating = 5 instead of other possible, and usually set the high bar as well. Uber and Lync were using a threshold of 4.6 average over 100 last trips a time ago, but I don't know how it is now. So, what happens when the company set a high scale but people, unconsciously, are rating in a different lower scale / criteria? Yeah, problems.

**The weight of the averages**: 

Another bad feature of this threshold calculation for the drivers has its fundaments on basic math. Given the high average value the drivers have to have in the last 100 trips, a low value can have a higher impact on the driver's score and, even more unfortunately, make the driver compensate even 9x more to get the older score back. Let's see one example.

Suppose you're a driver and have a great average on 4.8. You pick up a passenger but, because he had a really bad day, he also didn't go with your face and evaluated you with a 1 star. Your new rating will be: $$4.8 - \frac{4.8 - 1}{100} = 4.762$$  

What would be the new rating if instead of 1 it was a great 5 stars?  

$$4.8 - \frac{4.8 - 5}{100} = 4.802$$

Wow, that was a miserable increase! The problem of the average is that it gives a higher weight when a given score is farther than the actual average. As the driver's average **must** stay close to 5, all 5 star ratings would have a way smaller impact than a 1 star in the final average.

**Getting back the lost stars**

The bad part hasn't ended. As we saw, new values close to the actual average makes the number variates very little. So, when we received our 1 star, we would have to have many more 5 star ratings to recover what 1 booking has done. Lets check how many rides would a rider have to do in order to at least return to the old value, before the 1 star.

```python
actual_rating = 4.762
count = 0
while(actual_rating <= 4.8):
    actual_rating = actual_rating - ((actual_rating - 5)/100)
    print('New Score: ' + str(actual_rating))
    count += 1
print('It was needed ' + str(count) + ' "5 stars ratings" to go back to original value')
```

So, at a 4.8 initial ranking, it would be needed 18 "5 stars rankings" **just** to go back to the previous 4.8, and this IF the driver doesn't get any other 1 star rating during his quest on going back to the beloved score he had 1 day ago.

## Alternatives?

Finally, we have seen the 5 star rating schema suffers because of people and their lack of context while evaluating, either content and quality, and also suffer in its math, by severely punishing drivers in bad ratings more than rewarding them in good evaluations, in a "Did no more than it was expected" style. So, regarding only the drivers and passengers relationship, this is something we could start doing to avoid this inefficiences:

First of all (my opinion), educate passengers and make sure they know what is the impact on rating a driver if it is a lower value than the expected from the company. Also help them understand what exactly they should evaluate when riding. For example, we could have a lovely driver with a perfect car AND services, but got bad instincts for driving. What should passengers evaluate? Of course we should take into account usability issues, but here we can use as a good reference TripAdvisor and its different ratings segment:

<img src="images/notebook2_image3.png" width="500">

Second, as provided by the references above, people don't cope well with scales. So we might think in two different approaches when calculating the scores:

* Keeping the rating score **BUT** consider also the variability of users' reviewing process
* Make use of of binary choice for evaluation (positive or negative) and sometimes neutral.

**The First Approach** would work on the same way as the existent evaluation performs. The exception would be that the final score given by a passenger to a driver would how much the passenger deviated from his average evaluation. Given the score $Sc_{D}$ the score from driver before the rating update, $AVG(R_{p})$ the average rating from passenger P and $R_{p,d}$ the score from passenger P to driver D, the score update would be: 

$$Sc_{D} = Sc_{D} + \frac{AVG(R_{p}) - R_{p,d}}{100}$$

Note that this apprach only fix the User's different rating scale, but does **not** fix the type of penalty that the average gives to small scores given from people who in average rate drivers with high scores. (**Suggestion?**)

**Binary Choice**: A way to complete get rid of the failures of a 5 star rating system is to actually change to a different ratings schema. The biggest example on this was [Netflix's change](https://www.businessinsider.de/why-netflix-replaced-its-5-star-rating-system-2017-4?r=US&IR=T) to thumbs up and thumbs down system. Of course Netflix has its own purposes and business motives ([increasing user engagement](https://www.appcues.com/blog/rating-system-ux-star-thumbs) in this case), but at least we see they also wanted to remove this indecision of a proper 5 star ratings. A binary rating system brings some advantage such as:
- Critics Syndrome: This was intersting to Netflix, as people used to incorporate a critic personality when using the star schema. The discrepancy would show up as someone who watched ALL movies from Adam Sandler but rated all of them with 2 stars, and rated all movie classics and documentaries as 5 stars, without even have watched it.
- Defined scale. 'Liked', or 'Not Liked'. Pretty simple right? That means a probable returning on number of feedbacks, as the criteria to vote/review is much more simplified as a 5 star rating.

**Disadvantage**:

- Being a more complex metric, 5 star ratings brings a more complete view of complex recommendations, such as expensive product purchases. If we buying expensive shoes, furnitures or something more, stars, star distribution and reviews can come better as simple Thumbs Up/Down percentages.


