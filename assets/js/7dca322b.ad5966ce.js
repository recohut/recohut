"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4432],{3905:function(e,t,r){r.d(t,{Zo:function(){return p},kt:function(){return d}});var n=r(67294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var s=n.createContext({}),c=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},p=function(e){var t=c(e.components);return n.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,i=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),m=c(r),d=a,f=m["".concat(s,".").concat(d)]||m[d]||u[d]||i;return r?n.createElement(f,o(o({ref:t},p),{},{components:r})):n.createElement(f,o({ref:t},p))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=r.length,o=new Array(i);o[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,o[1]=l;for(var c=2;c<i;c++)o[c]=r[c];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},89159:function(e,t,r){r.r(t),r.d(t,{assets:function(){return p},contentTitle:function(){return s},default:function(){return d},frontMatter:function(){return l},metadata:function(){return c},toc:function(){return u}});var n=r(87462),a=r(63366),i=(r(67294),r(3905)),o=["components"],l={title:"Predicting Electronics Resale Price",authors:"sparsh",tags:["regression"]},s="Problem Statement",c={permalink:"/recohut/blog/2021/10/01/predicting-electronics-resale-price",editUrl:"https://github.com/sparsh-ai/recohut/blog/blog/2021-10-01-predicting-electronics-resale-price.mdx",source:"@site/blog/2021-10-01-predicting-electronics-resale-price.mdx",title:"Predicting Electronics Resale Price",description:"/img/content-blog-raw-blog-predicting-electronics-resale-price-untitled.png",date:"2021-10-01T00:00:00.000Z",formattedDate:"October 1, 2021",tags:[{label:"regression",permalink:"/recohut/blog/tags/regression"}],readingTime:1.875,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],frontMatter:{title:"Predicting Electronics Resale Price",authors:"sparsh",tags:["regression"]},prevItem:{title:"Personalized Unexpectedness in  Recommender Systems",permalink:"/recohut/blog/2021/10/01/personalized-unexpectedness-in-recommender-systems"},nextItem:{title:"Real-time news personalization with Flink",permalink:"/recohut/blog/2021/10/01/real-time-news-personalization-with-flink"}},p={authorsImageUrls:[void 0]},u=[{value:"Framework",id:"framework",level:3},{value:"List of Variables",id:"list-of-variables",level:3}],m={toc:u};function d(e){var t=e.components,l=(0,a.Z)(e,o);return(0,i.kt)("wrapper",(0,n.Z)({},m,l,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-predicting-electronics-resale-price-untitled.png",src:r(49769).Z,width:"592",height:"146"})),(0,i.kt)("h1",{id:"objective"},"Objective"),(0,i.kt)("p",null,"Predict the resale price based on brand, part id and purchase quantity"),(0,i.kt)("h1",{id:"milestones"},"Milestones"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Data analysis and discovery - What is the acceptable variance the model needs to meet in terms of similar part number and quantity?"),(0,i.kt)("li",{parentName:"ul"},"Model research and validation - Does the model meet the variance requirement? (Variance of the model should meet or be below the variance of the sales history)"),(0,i.kt)("li",{parentName:"ul"},"Model deployment - Traffic will increase 10 fold. So, model needs to be containerized or dockerized"),(0,i.kt)("li",{parentName:"ul"},"Training - Model needs to be trainable on new sales data. Methodology to accept or reject the variance of the newly trained model documented.")),(0,i.kt)("h1",{id:"deliverables"},"Deliverables"),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"Data Analysis and Discovery (identify target variance for pricing model in terms of similar part numbers and quantities). Analysis should be done on the 12 following quantity ranges: 1-4, 5-9, 10-24, 25-49, 50-99, 100-249, 250-499, 500-999, 1000-2499, 2500-4999, 5000-9999, 10000+.")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"ModelA Training (Resale Value Estimation ","[$]"," (Brand+PartNo.+Quantity)")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"ModelA Validation (variance analysis and comparison with sales history variance in terms of similar part numbers and quantities)")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"ModelA Containerization")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"ModelA re-training based on new sales data")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"ScriptA to calculate variance for new sales data (feedback for training results)")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"Documentation for re-training")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("p",{parentName:"li"},"ModelA deployment and API"))),(0,i.kt)("h1",{id:"modeling-approach"},"Modeling Approach"),(0,i.kt)("h3",{id:"framework"},"Framework"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Fully connected regression neural network"),(0,i.kt)("li",{parentName:"ul"},"NLP feature extraction from part id"),(0,i.kt)("li",{parentName:"ul"},"Batch generator to feed large data in batches"),(0,i.kt)("li",{parentName:"ul"},"Hyperparameter tuning to find the best model fit")),(0,i.kt)("h3",{id:"list-of-variables"},"List of Variables"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"2 years of sales history"),(0,i.kt)("li",{parentName:"ul"},"PRC"),(0,i.kt)("li",{parentName:"ul"},"PARTNO"),(0,i.kt)("li",{parentName:"ul"},"ORDER_NUMBER"),(0,i.kt)("li",{parentName:"ul"},"ORIG_ORDER_QTY"),(0,i.kt)("li",{parentName:"ul"},"UNIT_COST"),(0,i.kt)("li",{parentName:"ul"},"UNIT_REASLE"),(0,i.kt)("li",{parentName:"ul"},"UOM (UNIT OF MEASUREMENT)")),(0,i.kt)("h1",{id:"bucket-of-ideas"},"Bucket of Ideas"),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},"Increase n-gram range; e.g. in part_id ABC-123-23, these are 4-grams: ABC-, BC-1, C-12, -123, 123-, 23-2, 3-23; Idea is to see if increasing this range further will increase the model's performance"),(0,i.kt)("li",{parentName:"ol"},"Employ Char-level LSTM to capture sequence information; e.g. in same part_id ABC-123-23, currently we are not maintaining sequence of grams, we don't know if 3-23 is coming at first or last; here, the idea is to see if lstm model can be employed to capture this sequence information to improve model's performance"),(0,i.kt)("li",{parentName:"ol"},"New Loss function - including cost based loss")))}d.isMDXComponent=!0},49769:function(e,t,r){t.Z=r.p+"assets/images/content-blog-raw-blog-predicting-electronics-resale-price-untitled-3fed3bed15da063ddd1654593580a934.png"}}]);