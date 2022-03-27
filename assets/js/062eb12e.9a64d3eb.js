"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5095],{3905:function(e,t,r){r.d(t,{Zo:function(){return c},kt:function(){return u}});var n=r(67294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var d=n.createContext({}),l=function(e){var t=n.useContext(d),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},c=function(e){var t=l(e.components);return n.createElement(d.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,d=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),m=l(r),u=a,h=m["".concat(d,".").concat(u)]||m[u]||p[u]||o;return r?n.createElement(h,i(i({ref:t},c),{},{components:r})):n.createElement(h,i({ref:t},c))}));function u(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=m;var s={};for(var d in t)hasOwnProperty.call(t,d)&&(s[d]=t[d]);s.originalType=e,s.mdxType="string"==typeof e?e:a,i[1]=s;for(var l=2;l<o;l++)i[l]=r[l];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},53541:function(e,t,r){r.r(t),r.d(t,{assets:function(){return c},contentTitle:function(){return d},default:function(){return u},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return p}});var n=r(87462),a=r(63366),o=(r(67294),r(3905)),i=["components"],s={},d="Wide and Deep",l={unversionedId:"models/wide-and-deep",id:"models/wide-and-deep",title:"Wide and Deep",description:"Wide and Deep Learning Model, proposed by Google, 2016, is a DNN-Linear mixed model, which combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.",source:"@site/docs/models/wide-and-deep.md",sourceDirName:"models",slug:"/models/wide-and-deep",permalink:"/recohut/docs/models/wide-and-deep",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/wide-and-deep.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"VSKNN",permalink:"/recohut/docs/models/vsknn"},next:{title:"Word2vec",permalink:"/recohut/docs/models/word2vec"}},c={},p=[{value:"Architecture",id:"architecture",level:2},{value:"Links",id:"links",level:2}],m={toc:p};function u(e){var t=e.components,s=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,n.Z)({},m,s,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"wide-and-deep"},"Wide and Deep"),(0,o.kt)("p",null,"Wide and Deep Learning Model, proposed by Google, 2016, is a DNN-Linear mixed model, which combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation."),(0,o.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"research paper")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1606.07792"},"Cheng et. al., ",(0,o.kt)("em",{parentName:"a"},"Wide & Deep Learning for Recommender Systems"),". RecSys, 2016.")),(0,o.kt)("blockquote",{parentName:"div"},(0,o.kt)("p",{parentName:"blockquote"},"Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.")))),(0,o.kt)("h2",{id:"architecture"},"Architecture"),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:r(28400).Z,width:"1394",height:"284"})),(0,o.kt)("p",null,"To understand the concept of deep & wide recommendations, it\u2019s best to think of it as two separate, but collaborating, engines. The wide model, often referred to in the literature as the linear model, memorizes users and their past product choices. Its inputs may consist simply of a user identifier and a product identifier, though other attributes relevant to the pattern (such as time of day) may also be incorporated."),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"/img/content-models-raw-mp1-wide-and-deep-untitled-1.png",src:r(22566).Z,width:"1024",height:"503"})),(0,o.kt)("p",null,"The deep portion of the model, so named as it is a deep neural network, examines the generalizable attributes of a user and their product choices. From these, the model learns the broader characteristics that tend to favor users\u2019 product selections."),(0,o.kt)("p",null,"Together, the wide and deep submodels are trained on historical product selections by individual users to predict future product selections. The end result is a single model capable of calculating the probability with which a user will purchase a given item, given both memorized past choices and generalizations about a user\u2019s preferences. These probabilities form the basis for user-specific product rankings, which can be used for making recommendations."),(0,o.kt)("p",null,"The goal with wide and deep recommenders is to provide the same level of customer intimacy that, for example, our favorite barista does. This model uses explicit and implicit feedback to expand the considerations set for customers. Wide and deep recommenders go beyond simple weighted averaging of customer feedback found in some collaborative filters to balance what is understood about the individual with what is known about similar customers. If done properly, the recommendations make the customer feel understood and this should translate into greater value for both the customer and the business."),(0,o.kt)("iframe",{width:"727",height:"409",src:"https://www.youtube.com/embed/Xmw9SWJ0L50",title:"YouTube video player",frameborder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowfullscreen:!0}),(0,o.kt)("p",null,"The intuitive logic of the wide-and-deep recommender belies the complexity of its actual construction. Inputs must be defined separately for each of the wide and deep portions of the model and each must be trained in a coordinated manner to arrive at a single output, but tuned using optimizers specific to the nature of each submodel. Thankfully, the\xa0",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("a",{parentName:"strong",href:"https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier"},"Tensorflow DNNLinearCombinedClassifier estimator")),"\xa0provides a pre-packaged architecture, greatly simplifying the assembly of the overall model."),(0,o.kt)("h2",{id:"links"},"Links"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep"},"https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://dl.acm.org/doi/pdf/10.1145/2988450.2988454"},"https://dl.acm.org/doi/pdf/10.1145/2988450.2988454")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.widedeep.html"},"https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.widedeep.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.wdl.html"},"https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.wdl.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.wdl.html"},"https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.wdl.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/wide_deep"},"https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/wide_deep")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://docs.databricks.com/applications/machine-learning/reference-solutions/recommender-wide-n-deep.html"},"https://docs.databricks.com/applications/machine-learning/reference-solutions/recommender-wide-n-deep.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://medium.com/analytics-vidhya/wide-deep-learning-for-recommender-systems-dc99094fc291"},"https://medium.com/analytics-vidhya/wide-deep-learning-for-recommender-systems-dc99094fc291")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html"},"https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html"))))}u.isMDXComponent=!0},22566:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp1-wide-and-deep-untitled-1-a33765ffc43537e2143e2cc14bd6fa59.png"},28400:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp1-wide-and-deep-untitled-78e8caab74bd5160d7143fea0676b8bb.png"}}]);