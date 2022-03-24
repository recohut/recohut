"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3715],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return m}});var o=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function r(e,t){if(null==e)return{};var n,o,a=function(e,t){if(null==e)return{};var n,o,a={},i=Object.keys(e);for(o=0;o<i.length;o++)n=i[o],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(o=0;o<i.length;o++)n=i[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=o.createContext({}),d=function(e){var t=o.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},c=function(e){var t=d(e.components);return o.createElement(l.Provider,{value:t},e.children)},h={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},u=o.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,l=e.parentName,c=r(e,["components","mdxType","originalType","parentName"]),u=d(n),m=a,p=u["".concat(l,".").concat(m)]||u[m]||h[m]||i;return n?o.createElement(p,s(s({ref:t},c),{},{components:n})):o.createElement(p,s({ref:t},c))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,s=new Array(i);s[0]=u;var r={};for(var l in t)hasOwnProperty.call(t,l)&&(r[l]=t[l]);r.originalType=e,r.mdxType="string"==typeof e?e:a,s[1]=r;for(var d=2;d<i;d++)s[d]=n[d];return o.createElement.apply(null,s)}return o.createElement.apply(null,n)}u.displayName="MDXCreateElement"},17264:function(e,t,n){n.r(t),n.d(t,{assets:function(){return c},contentTitle:function(){return l},default:function(){return m},frontMatter:function(){return r},metadata:function(){return d},toc:function(){return h}});var o=n(87462),a=n(63366),i=(n(67294),n(3905)),s=["components"],r={},l="MarketCloud Real-time",d={unversionedId:"concept-extras/success-stories/marketcloud-real-time",id:"concept-extras/success-stories/marketcloud-real-time",title:"MarketCloud Real-time",description:"Model Training and Deployment",source:"@site/docs/concept-extras/success-stories/marketcloud-real-time.md",sourceDirName:"concept-extras/success-stories",slug:"/concept-extras/success-stories/marketcloud-real-time",permalink:"/ai/docs/concept-extras/success-stories/marketcloud-real-time",editUrl:"https://github.com/sparsh-ai/ai/docs/concept-extras/success-stories/marketcloud-real-time.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"LinkedIn GLMix",permalink:"/ai/docs/concept-extras/success-stories/linkedin-glmix"},next:{title:"Netflix Personalize Images",permalink:"/ai/docs/concept-extras/success-stories/netflix-personalize-images"}},c={},h=[{value:"<strong>Model Training and Deployment</strong>",id:"model-training-and-deployment",level:3},{value:"<strong>Scalability and Customizability</strong>",id:"scalability-and-customizability",level:3},{value:"<strong>Monitoring and Retraining Strategy</strong>",id:"monitoring-and-retraining-strategy",level:3},{value:"<strong>Real-Time Scoring</strong>",id:"real-time-scoring",level:3},{value:"<strong>Ability to Turn Recommendations On and Off</strong>",id:"ability-to-turn-recommendations-on-and-off",level:3},{value:"<strong>Pipeline Structure and Deployment Strategy</strong>",id:"pipeline-structure-and-deployment-strategy",level:3},{value:"<strong>Monitoring and Feedback</strong>",id:"monitoring-and-feedback",level:3},{value:"<strong>Retraining Models</strong>",id:"retraining-models",level:3},{value:"<strong>Updating Models</strong>",id:"updating-models",level:3},{value:"<strong>Runs Overnight, Sleeps During Daytime</strong>",id:"runs-overnight-sleeps-during-daytime",level:3},{value:"<strong>Option to Manually Control Models</strong>",id:"option-to-manually-control-models",level:3},{value:"<strong>Option to Automatically Control Models</strong>",id:"option-to-automatically-control-models",level:3},{value:"<strong>Monitoring Performance</strong>",id:"monitoring-performance",level:3},{value:"<strong>Closing Thoughts</strong>",id:"closing-thoughts",level:3}],u={toc:h};function m(e){var t=e.components,r=(0,a.Z)(e,s);return(0,i.kt)("wrapper",(0,o.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"marketcloud-real-time"},"MarketCloud Real-time"),(0,i.kt)("h3",{id:"model-training-and-deployment"},(0,i.kt)("strong",{parentName:"h3"},"Model Training and Deployment")),(0,i.kt)("p",null,"To better\xa0illustrate the MLOps process\xa0for this type of use case, the following sections focus on the specific example of a hypothetical company\xa0deploying an automated pipeline to train and deploy recommendation engines. The company is a global software company (let\u2019s call it MarketCloud) headquartered in Silicon Valley."),(0,i.kt)("p",null,"One of MarketCloud\u2019s products is a software-as-a-service (SaaS) platform called SalesCore. SalesCore is a B2B product that allows its users (businesses) to drive sales to customers in a simple manner by keeping track of deals, clearing tedious\xa0administration\xa0tasks off their desks, and creating customized product offers for each customer."),(0,i.kt)("p",null,"From time to time, MarketCloud\u2019s clients use the cloud-based SalesCore while on a call with their customers, adjusting their sales strategy by looking at past interactions with the customers as well as at the product offers and discounts suggested by\xa0SalesCore."),(0,i.kt)("p",null,"MarketCloud is a mid-sized company with an annual revenue of around $200 million and a few thousand employees. From salespeople at a brewing company to those in a telecommunication entity, MarketCloud\u2019s clients represent a range of businesses."),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-concepts-case-studies-raw-case-studies-marketcloud-real-time-untitled.png",src:n(35331).Z,width:"867",height:"419"})),(0,i.kt)("p",null,"MarketCloud would like to automatically display product suggestions on SalesCore to the salespeople trying to sell products to the customers. Suggestions would be made based on customers\u2019 information and their past interaction records with the salesperson; suggestions would therefore be customized for each customer. In other words, SalesCore is based on a recommendation engine used in a pull (inbound calls) or push (outbound calls) context. Salespeople would be able to incorporate in their sales strategy the suggested products while on a call with their customers."),(0,i.kt)("p",null,"To implement this idea, MarketCloud needs to\xa0build a recommendation engine and embed it into SalesCore\u2019s platform, which, from a model training and deployment standpoint, presents several challenges. We\u2019ll present these challenges in this section, and in the next section we\u2019ll show MLOps strategies that allow the company to handle each of them."),(0,i.kt)("h3",{id:"scalability-and-customizability"},(0,i.kt)("strong",{parentName:"h3"},"Scalability and Customizability")),(0,i.kt)("p",null,"MarketCloud\u2019s business model (selling software for client companies to help them sell their own products) presents an interesting situation.\xa0Each client company has its own dataset, mainly about its products and customers, and it doesn\u2019t wish to share the data with other companies."),(0,i.kt)("p",null,"If MarketCloud has around four thousand clients\xa0using SalesCore, that means instead of having a universal recommender system for all the clients, it would need to create four thousand different systems. MarketCloud needs to come up with a way to build four thousand recommendation systems as efficiently as possible since there is no way it can handcraft that many systems one by one."),(0,i.kt)("h3",{id:"monitoring-and-retraining-strategy"},(0,i.kt)("strong",{parentName:"h3"},"Monitoring and Retraining Strategy")),(0,i.kt)("p",null,"Each of the four thousand recommendation engines\xa0would be trained on the customer data of the corresponding client.\xa0Therefore, each of them would be a different model, yielding a different performance result and making it nearly impossible for the company to manually keep an eye on all four thousand. For example, the recommendation engine for client A in the beverage industry might consistently give good product suggestions, while the engine for client B in the telecommunication sector might seldom provide good suggestions. MarketCloud needed to come up with a way to automate the monitoring and the subsequent model retraining strategy in case the performance degraded."),(0,i.kt)("h3",{id:"real-time-scoring"},(0,i.kt)("strong",{parentName:"h3"},"Real-Time Scoring")),(0,i.kt)("p",null,"In many situations, MarketCloud\u2019s clients use SalesCore when they are talking to their customers on the phone.\xa0The sales negotiation evolves every single minute during the call, and the salesperson needs to adjust the strategy during the interaction with the customer, so the recommendation engine has to be responsive to real-time requests."),(0,i.kt)("p",null,"For example, imagine you as a salesperson is on a call with your customer to sell telecommunication devices. The customer tells you what his office looks like, the existing infrastructure at the office such as optic fiber, the type of WiFi network, and so forth. Upon entering this information in SalesCore, you want the platform to give you a suggestion for the products that your customer could feasibly purchase. This response from the platform needs to be in real-time, not 10 minutes later, after the call, or on the following day."),(0,i.kt)("h3",{id:"ability-to-turn-recommendations-on-and-off"},(0,i.kt)("strong",{parentName:"h3"},"Ability to Turn Recommendations On and Off")),(0,i.kt)("p",null,"Responsible AI principles acknowledge\xa0that retaining\xa0human involvement is important.\xa0This can be done through a human-in-command design,",(0,i.kt)("a",{parentName:"p",href:"https://learning.oreilly.com/library/view/introducing-mlops/9781492083283/ch10.html#ch01fn8"},"1"),"\xa0by which it should be possible\xa0",(0,i.kt)("em",{parentName:"p"},"not"),"\xa0to use the AI. In addition, adoption is likely to be low if users cannot override AI recommendations. Some clients value using their own intuition about which products to recommend to their customers. For this reason, MarketCloud wants to give its clients full control to turn the recommendation engine on and off so\xa0that the clients can use\xa0the recommendations\xa0when they want."),(0,i.kt)("h3",{id:"pipeline-structure-and-deployment-strategy"},(0,i.kt)("strong",{parentName:"h3"},"Pipeline Structure and Deployment Strategy")),(0,i.kt)("p",null,"To efficiently\xa0build four thousand recommendation engines, MarketCloud decided to make one data pipeline as a prototype and duplicate it four thousand times.\xa0This prototype pipeline consists of the necessary data preprocessing steps and a single recommendation engine, built on an example dataset. The algorithms used in the recommendation engines will be the same across all four thousand pipelines, but they will be trained with the specific data associated with each client."),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-concepts-case-studies-raw-case-studies-marketcloud-real-time-untitled-1.png",src:n(37403).Z,width:"1437",height:"533"})),(0,i.kt)("p",null,"In this way, MarketCloud can efficiently launch four thousand recommendation systems. The users will still retain some room for customization, because the engine is trained with their own data, and each algorithm will work with different parameters\u2014i.e., it\u2019s adopted to the customer and product information of each client."),(0,i.kt)("p",null,"What makes it possible to scale up a single pipeline to four thousand pipelines is the universal schema of the dataset. If a dataset from client A has 100 columns whereas client B has 50, or if the column \u201cnumber of purchased items\u201d from client A is an integer whereas the same column from client B is a string, they would need to go through different preprocessing pipelines."),(0,i.kt)("p",null,"Although each client has different customer and product data, at the point that this data is registered on SalesCore, it acquires the same number of columns and the same data types for each column. This makes things easier, as MarketCloud simply needs to copy a single pipeline four thousand times."),(0,i.kt)("p",null,"Each recommendation system embedded in the four thousand pipelines will have different API endpoints.\xa0On the surface, it looks like when a user clicks the \u201cshow product recommendations\u201d button, SalesCore displays a list of suggested products. But in the background, what is happening is that by clicking the button, the user is hitting the specific API endpoint associated with the ranked product lists for the specific\xa0customer."),(0,i.kt)("h3",{id:"monitoring-and-feedback"},(0,i.kt)("strong",{parentName:"h3"},"Monitoring and Feedback")),(0,i.kt)("p",null,"Maintaining four thousand recommendation systems is not an easy task, and while there have already\xa0been many MLOps considerations\xa0until this point, this is maybe the most complex part. Each system\u2019s performance needs to be monitored and updated as needed. To implement this monitoring strategy at a large scale, MarketCloud can automate the scenario for retraining and updating the models."),(0,i.kt)("h3",{id:"retraining-models"},(0,i.kt)("strong",{parentName:"h3"},"Retraining Models")),(0,i.kt)("p",null,"Clients obtain\xa0\xa0new customers, some\xa0of the customers churn, every once in a while new products are added to or dropped from their catalogs; the bottom line is that customer and product data are constantly changing, and recommendation systems have to reflect the latest data. It\u2019s the only way they can maintain the quality of the recommendation, and, more importantly, avoid a situation such as recommending a WiFi router that is outdated and no longer supported."),(0,i.kt)("p",null,"To reflect the latest data, the team could program a scenario to automatically update the database with the newest customer and product data, retraining the model with the latest datasets every day at midnight. This automation scenario could then be implemented in all four thousand data pipelines."),(0,i.kt)("p",null,"The retraining frequency can differ depending on the use case. Thanks to the high degree of automation, retraining every night in this case is possible. In other contexts, retraining could be triggered by various signals (e.g., signification volume of new information or drift in customer behavior, be it aperiodic or seasonal)."),(0,i.kt)("p",null,"In addition, the delay between the recommendation and the point in time at which its effect is evaluated has to be taken into account. If the impact is only known with a delay of several months, it is unlikely that retraining every day is adequate. Indeed, if the behavior changes so fast that retraining it every day is needed, it is likely that the model is completely outdated when it is used to make recommendations several months after the most recent ones in the training data."),(0,i.kt)("h3",{id:"updating-models"},(0,i.kt)("strong",{parentName:"h3"},"Updating Models")),(0,i.kt)("p",null,"Updating models is\xa0also one of the\xa0key features of automation strategies at scale. In this case, for each of the four thousand pipelines, retrained models must be compared to the existing models.\xa0Their performances can be compared using metrics such as RMSE (root-mean-square error), and only when the performance of the retrained model beats the prior one does the retrained model get deployed to SalesCore."),(0,i.kt)("h3",{id:"runs-overnight-sleeps-during-daytime"},(0,i.kt)("strong",{parentName:"h3"},"Runs Overnight, Sleeps During Daytime")),(0,i.kt)("p",null,"Although the model is retrained every day, users do not interact directly with the model.\xa0Using the updated model, the platform actually finishes calculating the ranked list of products for all the customers during the night. On the following day, when a user hits the \u201cshow product recommendations\u201d button, the platform simply looks at the customer ID and returns the ranked list of products for the specific customer."),(0,i.kt)("p",null,"To the user, it looks as if the recommendation engine is running in real time. In reality, however, everything is already prepared overnight, and the engine is sleeping during daytime. This makes it possible to get the recommendation instantly without any downtime."),(0,i.kt)("h3",{id:"option-to-manually-control-models"},(0,i.kt)("strong",{parentName:"h3"},"Option to Manually Control Models")),(0,i.kt)("p",null,"Although the monitoring, retraining, and\xa0updating of\xa0the models is fully automated, MarketCloud still leaves room for its clients to turn the models on and off. More precisely, MarketCloud allows the users to choose from three options to interact with the models:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Turn on to get the recommendation based on the most updated dataset"),(0,i.kt)("li",{parentName:"ul"},"Freeze to stop retraining with the new data, but keep using the same model"),(0,i.kt)("li",{parentName:"ul"},"Turn off to completely stop using the recommendation functionality of SalesCore")),(0,i.kt)("p",null,"Machine learning algorithms attempt to convert practical knowledge into meaningful algorithms to automate processing tasks. However, it is still good practice to leave room for users to rely on their domain knowledge, as they are presumed to be far more capable of identifying, articulating, and demonstrating day-to-day process problems in business."),(0,i.kt)("p",null,"The second option is important because it allows users to stay in the current quality of the recommendation without having the recommendation engines updated with the newer data. Whether the current model is replaced with a retrained one depends on the mathematical evaluation based on metrics such as the RMSE. However, if users feel that the product recommendations on SalesCore are already working well for pushing sales, they have the choice not to risk changing the recommendation quality."),(0,i.kt)("h3",{id:"option-to-automatically-control-models"},(0,i.kt)("strong",{parentName:"h3"},"Option to Automatically Control Models")),(0,i.kt)("p",null,"For those\xa0that don\u2019t want to manually handle\xa0the models, the platform could also propose A/B testing so that the impact of new versions is tested before fully switching to them. Multi-armed bandit algorithms (an algorithm that allows for maximization of the revenue of a user facing multiple slot machines, each with a different probability to win and a different proportion of the money given back on average) are used for this purpose."),(0,i.kt)("p",null,"Let\u2019s assume that several model versions are available. The goal is to use the most efficient one, but to do that, the algorithm obviously has to first learn which is the most efficient. Therefore, it balances these two objectives: sometimes, it tries algorithms that may not be the most efficient to learn if they are efficient (exploration), and sometimes it uses the version that is likely to be the most efficient to maximize the revenue (exploitation). In addition, it forgets past information, as the algorithm knows the most efficient today may not be the most efficient tomorrow."),(0,i.kt)("p",null,"The most advanced option consists in training different models for different KPIs (click, buy, expected revenue, etc.).\xa0A method inspired from ensemble models would then allow for the solving of conflicts between models."),(0,i.kt)("h3",{id:"monitoring-performance"},(0,i.kt)("strong",{parentName:"h3"},"Monitoring Performance")),(0,i.kt)("p",null,"When a salesperson\xa0suggests a customer buy\xa0the products recommended by SalesCore, the interaction of the customer with the recommended products as well as whether the customer bought them or not is recorded. This record can then be used to keep track of the performance of the recommender system, overwriting the customer and product dataset with this record to feed the most updated information to the model when it is retrained."),(0,i.kt)("p",null,"Thanks to this ground truth recording process, dashboards showing model performance can be presented to the user, including performance comparison from A/B testing.\xa0Because the ground truth is obtained quickly, data drift monitoring is secondary. A version of the model is trained every night, but, thanks to the freeze mechanism, the user can choose the active version based on the quantitative information. It is customary to keep the human in the loop on these high-impact decisions where the performance metrics have a hard time capturing the full context around the decision."),(0,i.kt)("p",null,"In the case of A/B testing, it is important that only one experiment be done at a time on a group of customers; the impact of combined strategies cannot be simply added. With such considerations in mind, it is possible to build a sound baseline to perform a counterfactual analysis and derive the increased revenue and/or the decreased churn linked to a new strategy."),(0,i.kt)("p",null,"Apart from this, MarketCloud can also monitor the algorithm performance at a macro level, by checking how many clients froze or turned off the recommender systems. If many clients turned off the recommender\xa0systems, that\u2019s a strong indicator that they are not satisfied with the recommendation quality."),(0,i.kt)("h3",{id:"closing-thoughts"},(0,i.kt)("strong",{parentName:"h3"},"Closing Thoughts")),(0,i.kt)("p",null,"This use case is peculiar in the sense that MarketCloud built a sales platform that many other companies use to sell products, where the ownership of the data belongs to each company, and the data cannot be shared across companies. This brings a challenging situation where MarketCloud must create different recommender systems for each of the users instead of pooling all the data to create a universal recommendation engine."),(0,i.kt)("p",null,"MarketCloud can overcome this obstacle by creating a single pipeline into which data from many different companies can be fed. By having the data go through an automated recommendation engine training scenario, MarketCloud created many recommendation engines trained on different datasets. Good MLOps processes are what allow the company to do this at scale."),(0,i.kt)("p",null,"It\u2019s worth noting that though this use case is fictionalized, it is based on reality. The real-life team that tackled a similar project took around three months to finish. The team used a data science and machine learning platform to orchestrate the duplication of a single pipeline to four thousand copies and to automate the processes to feed corresponding datasets to each pipeline and train the models. Of necessity, they accepted trade-offs between the recommendation quality and scalability to efficiently launch the product. If the team had carefully crafted a custom recommendation engine for each of the four thousand pipelines by, for example, choosing the best algorithm for each client, the recommendation engines would have been of a higher quality, but they would have never been able to complete the project with such a small team in such a short period of time."))}m.isMDXComponent=!0},37403:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-case-studies-raw-case-studies-marketcloud-real-time-untitled-1-2d3efb79eeaf71258068e02ad3c57caf.png"},35331:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-case-studies-raw-case-studies-marketcloud-real-time-untitled-cfb95eb6c1b1f010f241fef79368e975.png"}}]);