# Doordash Contextual Bandit

[Read here on Doordash's blog](https://doordash.engineering/2020/01/27/personalized-cuisine-filter/)

**Doordash also adopted a contextual bandit approach for [cuisine recommendations](https://doordash.engineering/2020/01/27/personalized-cuisine-filter/), with the addition of multiple geolocation levels.** The bandit explores by suggesting new cuisine types to customers to gauge their interest, and exploits to recommend customers their most preferred cuisines.

To model the “average” cuisine preference in each location, they introduced multiple levels in their bandit. As an example, they shared how levels could go from the lowest level of district, through submarket, market, and region.

![/img/content-concepts-case-studies-raw-case-studies-doordash-contextual-bandit-untitled.png](/img/content-concepts-case-studies-raw-case-studies-doordash-contextual-bandit-untitled.png)

Each geolocation level provides prior knowledge so that cold-start customers can be represented by the prior of the location, until the bandit collects enough data about them for personalization. The geolocation priors also allows Doordash to balance the customer’s existing preferences with the hot favorites of each geolocation. A sushi-lover ordering food from a new geolocation may be exposed to the hot favorite in the area (e.g., fried chicken), balancing his preferences with local popularity.