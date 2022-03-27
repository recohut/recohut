"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6790],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return m}});var a=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=a.createContext({}),d=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},c=function(e){var t=d(e.components);return a.createElement(s.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),u=d(n),m=r,g=u["".concat(s,".").concat(m)]||u[m]||p[m]||i;return n?a.createElement(g,o(o({ref:t},c),{},{components:n})):a.createElement(g,o({ref:t},c))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,o=new Array(i);o[0]=u;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:r,o[1]=l;for(var d=2;d<i;d++)o[d]=n[d];return a.createElement.apply(null,o)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},45360:function(e,t,n){n.d(t,{Z:function(){return k}});var a=n(67294),r=n(39960),i=n(63725),o=n(86010),l="cardContainer_woeA",s="cardContainerLink_iOSj",d="cardTitle_pNjP",c="cardDescription_qC_k",p=n(13919),u=n(95999);function m(e){var t=e.href,n=e.children,i=(0,o.Z)("card margin-bottom--lg padding--lg",l,t&&s);return t?a.createElement(r.Z,{href:t,className:i},n):a.createElement("div",{className:i},n)}function g(e){var t=e.href,n=e.icon,r=e.title,i=e.description;return a.createElement(m,{href:t},a.createElement("h2",{className:(0,o.Z)("text--truncate",d),title:r},n," ",r),a.createElement("div",{className:(0,o.Z)("text--truncate",c),title:i},i))}function h(e){var t=e.item,n=(0,i.Wl)(t);return a.createElement(g,{href:n,icon:"\ud83d\uddc3\ufe0f",title:t.label,description:(0,u.I)({message:"{count} items",id:"theme.docs.DocCard.categoryDescription",description:"The default description for a category card in the generated index about how many items this category includes"},{count:t.items.length})})}function N(e){var t,n=e.item,r=(0,p.Z)(n.href)?"\ud83d\udcc4\ufe0f":"\ud83d\udd17",o=(0,i.xz)(null!=(t=n.docId)?t:void 0);return a.createElement(g,{href:n.href,icon:r,title:n.label,description:null==o?void 0:o.description})}function f(e){var t=e.item;switch(t.type){case"link":return a.createElement(N,{item:t});case"category":return a.createElement(h,{item:t});default:throw new Error("unknown item type "+JSON.stringify(t))}}function k(e){var t=e.items;return a.createElement("div",{className:"row"},t.map((function(e,t){return a.createElement("article",{key:t,className:"col col--6"},a.createElement(f,{item:e}))})))}},30114:function(e,t,n){n.r(t),n.d(t,{assets:function(){return u},contentTitle:function(){return c},default:function(){return h},frontMatter:function(){return d},metadata:function(){return p},toc:function(){return m}});var a=n(87462),r=n(63366),i=(n(67294),n(3905)),o=n(45360),l=n(63725),s=["components"],d={},c="Models",p={unversionedId:"models/models",id:"models/models",title:"Models",description:"In the following sections, we will more systematically introduce the following models:",source:"@site/docs/models/models.mdx",sourceDirName:"models",slug:"/models/",permalink:"/recohut/docs/models/",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/models.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Video Action Recognition",permalink:"/recohut/docs/concept-extras/vision/video-action-recognition"},next:{title:"A3C",permalink:"/recohut/docs/models/a3c"}},u={},m=[{value:"CTR Prediction Models",id:"ctr-prediction-models",level:2}],g={toc:m};function h(e){var t=e.components,n=(0,r.Z)(e,s);return(0,i.kt)("wrapper",(0,a.Z)({},g,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"models"},"Models"),(0,i.kt)("p",null,"In the following sections, we will more systematically introduce the following models:"),(0,i.kt)(o.Z,{items:(0,l.jA)().items,mdxType:"DocCardList"}),(0,i.kt)("h2",{id:"ctr-prediction-models"},"CTR Prediction Models"),(0,i.kt)("p",null,"In the early stage of recommendation systems, people spent much time on tedious and onerous feature engineering. At that time, the dimensions of the raw features are relatively small, which makes it possible to implement different combinations of raw features. The newly created features are then fed into a shallow model, such as Logistic Regression (LR) and Gradient Boosting Decision Trees (GBDT) are widely used in CTR prediction. Then, Factorization Machine (FM) transforms the learning of users and items features into a small, shared vector shape. Based on the above method, Field-aware and Field-weighted FM (FFM) further consider the different impact of the fields that a feature belongs to in order to improve the performance of CTR prediction. Along this line, Attentional Factorization Machines (AFM) are proposed to automatically learn weights of deep cross-features and Neural Factorization Machines (NFM) enhances FMs by modelling higher-order and non-linear feature interactions."),(0,i.kt)("p",null,"Recently, the success of deep neural networks (DNN) in natural language processing and computer vision brings a new direction to the recommendation system. Among them, the Wide & Deep learning introduces deep neural networks to the CTR prediction. It jointly trains a deep neural network along with the traditional wide linear model. The deep neural networks liberated people from feature engineering while generalizing better combinations of the features. Lots of variants of the Wide & Deep learning have been proposed since it revolutionizes the development of the CTR prediction. Deep & Cross network (DCN) replaces the wide linear part with cross-network, which generates explicit feature crossing among low and high level layers. DeepFM combines the power of DNN and factorization machines for feature representation in the recommendation system. Furthermore, xDeepFM extends the work of DNN by proposing a Compressed Interaction Network to enumerate and compress all feature interactions. Overall, the deep models mentioned above all construct a similar model structure by combing the low-order and high-order features, which greatly reduce the effort of feature engineering and improve the performance of CTR prediction."),(0,i.kt)("p",null,"However, these aforementioned shallow or deep models take statistical and categorical features as input, while discarding the sequential behavior information of users. For example, users may search items at an e-commerce system, then view some items of interest, and these items are likely to be clicked or purchased next time. Since the historical behaviors explicitly indicate the preference of the users, it has gained much more attention in the recommendation systems. Among them, DIN proposes a local activation unit that learns the dynamic user interests from the sequential behavior features. DIEN designs an interest evolving layer an attentional update gate to model the dependency between sequential behaviors. The researches above realized the importance of user\u2019s historical behaviors. Unfortunately, they just project other information (i.e., user-specific and context) into one vector and did not pay equal attention to the interactions between the candidate item and fine-grained information, while modeling this interaction has shown extensively progress in many tasks, such as search recommendations and knowledge distillation."),(0,i.kt)("p",null,"Different from all previous methods, MIAN can explore the sequential behavior and other fine-grained information simultaneously. Specifically, compared to shallow and deep models, MIAN has a remarkable ability to encode user\u2019s preference through sequential behavior. Compared to sequential models, MIAN can model fine-grained feature interactions better when historical behavior is not enough or representative."),(0,i.kt)("table",null,(0,i.kt)("thead",{parentName:"table"},(0,i.kt)("tr",{parentName:"thead"},(0,i.kt)("th",{parentName:"tr",align:null},"Model"),(0,i.kt)("th",{parentName:"tr",align:null},"Paper"),(0,i.kt)("th",{parentName:"tr",align:null},"Publication"),(0,i.kt)("th",{parentName:"tr",align:null},"Type"),(0,i.kt)("th",{parentName:"tr",align:null},"Description"))),(0,i.kt)("tbody",{parentName:"table"},(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"LR"),(0,i.kt)("td",{parentName:"tr",align:null},"Predicting Clicks: Estimating the Click-Through Rate for New Ads"),(0,i.kt)("td",{parentName:"tr",align:null},"WWW'07"),(0,i.kt)("td",{parentName:"tr",align:null},"Shallow"),(0,i.kt)("td",{parentName:"tr",align:null},"Logistic regression (LR) is a simple baseline model for CTR prediction. With the online learning algorithm, FTRL, proposed by Google, LR has been widely adopted in industry. It\u2019s a widely used baseline and applies linear transformation to model the relationship of all the features.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FM"),(0,i.kt)("td",{parentName:"tr",align:null},"Factorization Machines"),(0,i.kt)("td",{parentName:"tr",align:null},"ICDM'10"),(0,i.kt)("td",{parentName:"tr",align:null},"Shallow"),(0,i.kt)("td",{parentName:"tr",align:null},"While LR fails to capture non-linear feature interactions, Rendle et al. propose factorization machine (FM) that embeds features into dense vectors and models pairwise feature interactions as inner products of the corresponding embedding vectors. Notably, FM also has a linear time complexity in terms of the number of features.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"CCPM"),(0,i.kt)("td",{parentName:"tr",align:null},"A Convolutional Click Prediction Model"),(0,i.kt)("td",{parentName:"tr",align:null},"CIKM'15"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"CCPM reports the first attempt to use convolution for CTR prediction, where feature embeddings are aggregated hierarchically through convolution networks.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FFM"),(0,i.kt)("td",{parentName:"tr",align:null},"Field-aware Factorization Machines for CTR Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"RecSys'16"),(0,i.kt)("td",{parentName:"tr",align:null},"Shallow"),(0,i.kt)("td",{parentName:"tr",align:null},"Field-aware factorization machine (FFM) is an extension of FM that considers field information for feature interactions. It was a winner model in several Kaggle contests on CTR prediction.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"YoutubeDNN"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep Neural Networks for YouTube Recommendations"),(0,i.kt)("td",{parentName:"tr",align:null},"RecSys'16"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"DNN is a straightforward deep model, which applies a fully-connected network (termed DNN) after the concatenation of feature embeddings for CTR prediction.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"Wide&Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"Wide & Deep Learning for Recommender Systems"),(0,i.kt)("td",{parentName:"tr",align:null},"DLRS'16"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"Wide&Deep is a general learning framework proposed by Google that combines a wide (or shallow) network and deep network to achieve the advantages of both. It jointly trains a linear model and a deep MLP model to the CTR prediction.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"IPNN"),(0,i.kt)("td",{parentName:"tr",align:null},"Product-based Neural Networks for User Response Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"ICDM'16"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"PNN is a product-based network that feeds the inner (or outer) products of features embeddings as the input of DNN. Due to the huge memory requirement of pairwise outer products, we use the inner product version, IPNN.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"DeepCross"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features"),(0,i.kt)("td",{parentName:"tr",align:null},"KDD'16"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"Inspired by residual networks, deep crossing to add residual connections between layers of DNNs. DCN is proposed to handle a set of sparse and dense features, and learn cross high-order features jointly with traditional deep MLP.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"HOFM"),(0,i.kt)("td",{parentName:"tr",align:null},"Higher-Order Factorization Machines"),(0,i.kt)("td",{parentName:"tr",align:null},"NIPS'16"),(0,i.kt)("td",{parentName:"tr",align:null},"Shallow"),(0,i.kt)("td",{parentName:"tr",align:null},"Since FM only captures second-order feature interactions, HOFM aims to extend FM to higher-order factorization machines. However, it results in exponential feature combinations that consume huge memory and take a long running time.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"DeepFM"),(0,i.kt)("td",{parentName:"tr",align:null},"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"IJCAI'17"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"DeepFM is an extension of Wide&Deep that substitutes LR with FM to explicitly model second-order feature interactions. It combines the explicit high-order interaction module with deep MLP module and traditional FM module, and requires no manual feature engineering.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"NFM"),(0,i.kt)("td",{parentName:"tr",align:null},"Neural Factorization Machines for Sparse Predictive Analytics"),(0,i.kt)("td",{parentName:"tr",align:null},"SIGIR'17"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"Similar to PNN, NFM proposes a Bi-interaction layer that pools the pairwise feature interactions to a vector and then feed it to a DNN for CTR prediction.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"AFM"),(0,i.kt)("td",{parentName:"tr",align:null},"Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks"),(0,i.kt)("td",{parentName:"tr",align:null},"IJCAI'17"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"Instead of treating all feature interactions equally as in FM, AFM learns the weights of feature interactions via attentional networks. Different from FwFM, AFM adjusts the weights dynamically according to the input data sample.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"DCN"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep & Cross Network for Ad Click Predictions"),(0,i.kt)("td",{parentName:"tr",align:null},"ADKDD'17"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"In DCN, a cross network is proposed to perform high-order feature interactions in an explicit way. In addition, it also integrates a DNN network following the Wide&Deep framework.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FwFM"),(0,i.kt)("td",{parentName:"tr",align:null},"Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising"),(0,i.kt)("td",{parentName:"tr",align:null},"WWW'18"),(0,i.kt)("td",{parentName:"tr",align:null},"Shallow"),(0,i.kt)("td",{parentName:"tr",align:null},"It considers field-wise weights of features interactions. Compared with FFM, it reports comparable performance but uses much fewer model parameters.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"xDeepFM"),(0,i.kt)("td",{parentName:"tr",align:null},"xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"),(0,i.kt)("td",{parentName:"tr",align:null},"KDD'18"),(0,i.kt)("td",{parentName:"tr",align:null},"xDeepFM"),(0,i.kt)("td",{parentName:"tr",align:null},"While high-order feature interactions modeled by DCN are bit-wise, xDeepFM proposes to capture high-order feature interactions in a vector-wise way via a compressed interaction network (CIN). It uses Compressed Interaction Network to enumerate and compress all feature interactions, for modeling an explicit order of interactions.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"DIN"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep Interest Network for Click-Through Rate Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"KDD'18"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"It\u2019s an early work exploits users\u2019 historical behaviors and uses the attention mechanism to activate user behaviors in which the user be interested in different items.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FiGNN"),(0,i.kt)("td",{parentName:"tr",align:null},"FiGNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"CIKM'19"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"FiGNN leverages the message passing mechanism of graph neural networks to learn high-order features interactions.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"AutoInt/AutoInt+"),(0,i.kt)("td",{parentName:"tr",align:null},"AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"),(0,i.kt)("td",{parentName:"tr",align:null},"CIKM'19"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"AutoInt leverages self-attention networks to learn high-order features interactions. AutoInt+ integrates AutoInt with a DNN network.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FiBiNET"),(0,i.kt)("td",{parentName:"tr",align:null},"FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"RecSys'19"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"FiBiNET leverages squeeze-excitation network to capture important features, and proposes bilinear interactions to enhance feature interactions.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FGCNN"),(0,i.kt)("td",{parentName:"tr",align:null},"Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"WWW'19"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"FGCNN applies convolution networks and recombination layers to generate additional combinatorial features to enrich existing feature representations.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"HFM/HFM+"),(0,i.kt)("td",{parentName:"tr",align:null},"Holographic Factorization Machines for Recommendation"),(0,i.kt)("td",{parentName:"tr",align:null},"AAAI'19"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"HFM proposes holographic representation and computes compressed outer products via circular convolution to model pairwise feature interactions. HFM+ further integrates a DNN network with HFM.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"ONN"),(0,i.kt)("td",{parentName:"tr",align:null},"Operation-aware Neural Networks for User Response Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"Neural Networks'20"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"ONN (a.k.a., NFFM) is a model built on FFM. It feeds the interaction outputs from FFM to a DNN network for CTR prediction.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"AFN/AFN+"),(0,i.kt)("td",{parentName:"tr",align:null},"Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions"),(0,i.kt)("td",{parentName:"tr",align:null},"AAAI'20"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"AFN applies logarithmic transformation layers to learn adaptive-order feature interactions. AFN+ further integrates AFN with a DNN network.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"LorentzFM"),(0,i.kt)("td",{parentName:"tr",align:null},"Learning Feature Interactions with Lorentzian Factorization"),(0,i.kt)("td",{parentName:"tr",align:null},"AAAI'20"),(0,i.kt)("td",{parentName:"tr",align:null},"Shallow"),(0,i.kt)("td",{parentName:"tr",align:null},"LorentzFM embed features into a hyperbolic space and model feature interactions via triangle inequality of Lorentz distance.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"InterHAt"),(0,i.kt)("td",{parentName:"tr",align:null},"Interpretable Click-through Rate Prediction through Hierarchical Attention"),(0,i.kt)("td",{parentName:"tr",align:null},"WSDM'20"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"InterHAt employs hierarchical attention networks to model high-order feature interactions in an efficient manner.")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FLEN"),(0,i.kt)("td",{parentName:"tr",align:null},"FLEN: Leveraging Field for Scalable CTR Prediction"),(0,i.kt)("td",{parentName:"tr",align:null},"DLP-KDD'20"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"FLEN: Leveraging Field for Scalable CTR Prediction")),(0,i.kt)("tr",{parentName:"tbody"},(0,i.kt)("td",{parentName:"tr",align:null},"FmFM"),(0,i.kt)("td",{parentName:"tr",align:null},"FM^2: Field-matrixed Factorization Machines for Recommender Systems"),(0,i.kt)("td",{parentName:"tr",align:null},"WWW'21"),(0,i.kt)("td",{parentName:"tr",align:null},"Deep"),(0,i.kt)("td",{parentName:"tr",align:null},"FM^2: Field-matrixed Factorization Machines for Recommender Systems")))))}h.isMDXComponent=!0}}]);