"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2806],{3905:function(e,t,r){r.d(t,{Zo:function(){return d},kt:function(){return u}});var n=r(67294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var l=n.createContext({}),c=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},d=function(e){var t=c(e.components);return n.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),m=c(r),u=a,h=m["".concat(l,".").concat(u)]||m[u]||p[u]||o;return r?n.createElement(h,i(i({ref:t},d),{},{components:r})):n.createElement(h,i({ref:t},d))}));function u(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=m;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:a,i[1]=s;for(var c=2;c<o;c++)i[c]=r[c];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},74859:function(e,t,r){r.r(t),r.d(t,{assets:function(){return d},contentTitle:function(){return l},default:function(){return u},frontMatter:function(){return s},metadata:function(){return c},toc:function(){return p}});var n=r(87462),a=r(63366),o=(r(67294),r(3905)),i=["components"],s={},l="DCN",c={unversionedId:"models/dcn",id:"models/dcn",title:"DCN",description:"DCN stands for Deep and Cross Network. Manual explicit feature crossing process is very laborious and inefficient. On the other hand, automatic implicit feature crossing methods like MLPs cannot efficiently approximate even 2nd or 3rd-order feature crosses. Deep-cross networks provides a solution to this problem. DCN was designed to learn explicit and bounded-degree cross features more effectively. It starts with an input layer (typically an embedding layer), followed by a cross network containing multiple cross layers that models explicit feature interactions, and then combines with a deep network that models implicit feature interactions.",source:"@site/docs/models/dcn.md",sourceDirName:"models",slug:"/models/dcn",permalink:"/ai/docs/models/dcn",editUrl:"https://github.com/sparsh-ai/ai/docs/models/dcn.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"CoKE",permalink:"/ai/docs/models/coke"},next:{title:"DDPG",permalink:"/ai/docs/models/ddpg"}},d={},p=[{value:"Architecture",id:"architecture",level:2},{value:"<strong>Feature Cross</strong>",id:"feature-cross",level:3},{value:"<strong>Cross Network</strong>",id:"cross-network",level:3},{value:"<strong>Deep &amp; Cross Network Architecture</strong>",id:"deep--cross-network-architecture",level:3},{value:"<strong>Low-rank DCN</strong>",id:"low-rank-dcn",level:3},{value:"List of experiments",id:"list-of-experiments",level:2},{value:"Links",id:"links",level:2}],m={toc:p};function u(e){var t=e.components,s=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,n.Z)({},m,s,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"dcn"},"DCN"),(0,o.kt)("p",null,"DCN stands for Deep and Cross Network. Manual explicit feature crossing process is very laborious and inefficient. On the other hand, automatic implicit feature crossing methods like MLPs cannot efficiently approximate even 2nd or 3rd-order feature crosses. Deep-cross networks provides a solution to this problem. DCN was designed to learn explicit and bounded-degree cross features more effectively. It starts with an input layer (typically an embedding layer), followed by a cross network containing multiple cross layers that models explicit feature interactions, and then combines with a deep network that models implicit feature interactions."),(0,o.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"research paper")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1708.05123"},"Ruoxi Wang, Bin Fu, Gang Fu and Mingliang Wang, \u201c",(0,o.kt)("em",{parentName:"a"},"Deep & Cross Network for Ad Click Predictions"),"\u201d. KDD, 2017.")),(0,o.kt)("blockquote",{parentName:"div"},(0,o.kt)("p",{parentName:"blockquote"},"Feature engineering has been the key to the success of many prediction models. However, the process is non-trivial and often requires manual feature engineering or exhaustive searching. DNNs are able to automatically learn feature interactions; however, they generate all the interactions implicitly, and are not necessarily efficient in learning all types of cross features. In this paper, we propose the Deep & Cross Network (DCN) which keeps the benefits of a DNN model, and beyond that, it introduces a novel cross network that is more efficient in learning certain bounded-degree feature interactions. In particular, DCN explicitly applies feature crossing at each layer, requires no manual feature engineering, and adds negligible extra complexity to the DNN model. Our experimental results have demonstrated its superiority over the state-of-art algorithms on the CTR prediction dataset and dense classification dataset, in terms of both model accuracy and memory usage.")))),(0,o.kt)("iframe",{width:"727",height:"409",src:"[https://www.youtube.com/embed/kUuvRStz7CU](https://www.youtube.com/embed/kUuvRStz7CU)",title:"YouTube video player",frameborder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowfullscreen:!0}),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Deep and cross network, short for DCN"),", came out of Google Research, and is designed to learn explicit and bounded-degree cross features effectively:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"large and sparse feature space is extremely hard to train."),(0,o.kt)("li",{parentName:"ul"},"Oftentimes, we needed to do a lot of manual feature engineering, including designing cross features, which is very challenging and less effective."),(0,o.kt)("li",{parentName:"ul"},"Whilst possible to use additional neural networks under such circumstances, it's not the most efficient approach.")),(0,o.kt)("p",null,(0,o.kt)("em",{parentName:"p"},"Deep and cross network (DCN) is specifically designed to tackle all above challenges.")),(0,o.kt)("h2",{id:"architecture"},"Architecture"),(0,o.kt)("h3",{id:"feature-cross"},(0,o.kt)("strong",{parentName:"h3"},"Feature Cross")),(0,o.kt)("p",null,"Let's say we're building a recommender system to sell a blender to customers. Then our customers' past purchase history, such as purchased bananas and purchased cooking books, or geographic features are single features. If one has purchased both bananas and cooking books, then this customer will be more likely to click on the recommended blender. The combination of purchased bananas and the purchased cooking books is referred to as feature cross, which provides additional interaction information beyond the individual features. You can keep adding more cross features to even higher degrees:"),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:r(54895).Z,width:"1717",height:"719"})),(0,o.kt)("h3",{id:"cross-network"},(0,o.kt)("strong",{parentName:"h3"},"Cross Network")),(0,o.kt)("p",null,"In real world recommendation systems, we often have large and sparse feature space. So identifying effective feature processes in this setting would often require manual feature engineering or exhaustive search, which is highly inefficient. To tackle this issue,\xa0",(0,o.kt)("strong",{parentName:"p"},(0,o.kt)("em",{parentName:"strong"},"Google Research team has proposed Deep and Cross Network, DCN."))),(0,o.kt)("p",null,"It starts with an input layer, typically an embedding layer, followed by a cross network containing multiple cross layers that models explicitly feature interactions, and then combines with a deep network that models implicit feature interactions. The deep network is just a traditional multilayer construction. But the core of DCN is really the cross network. It explicitly applies feature crossing at each layer. And the highest polynomial degree increases with layer depth. The figure here shows the deep and cross layer in the mathematical form."),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:r(82343).Z,width:"1251",height:"435"})),(0,o.kt)("h3",{id:"deep--cross-network-architecture"},(0,o.kt)("strong",{parentName:"h3"},"Deep & Cross Network Architecture")),(0,o.kt)("p",null,"There are a couple of ways to combine the cross network and the deep network:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Stack the deep network on top of the cross network."),(0,o.kt)("li",{parentName:"ul"},"Place deep & cross networks in parallel.")),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:r(40106).Z,width:"1342",height:"538"})),(0,o.kt)("h3",{id:"low-rank-dcn"},(0,o.kt)("strong",{parentName:"h3"},"Low-rank DCN")),(0,o.kt)("p",null,"To reduce the training and serving cost, we leverage low-rank techniques to approximate the DCN weight matrices. The rank is passed in through argument projection_dim; a smaller projection_dim results in a lower cost. Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost. In practice, we've observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN."),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:r(12421).Z,width:"609",height:"236"})),(0,o.kt)("h2",{id:"list-of-experiments"},"List of experiments"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://www.tensorflow.org/recommenders/examples/dcn"},"TFRS | Notebook")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://medium.com/analytics-vidhya/deep-cross-network-dcn-for-deep-learning-recommendation-systems-8923d6544686"},"Blog")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://keras.io/examples/structured_data/wide_deep_cross_networks/"},"Keras Blog | Notebook")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://paperswithcode.com/paper/dcn-m-improved-deep-cross-network-for-feature"},"Paper | Code")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://www.arxiv-vanity.com/papers/1708.05123/"},"Paper")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/jyfeather/Tensorflow-DCN"},"Code")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/Nirvanada/Deep-and-Cross-Keras"},"Code | Keras")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-2-deep-learning-for-recommender-systems/"},"Blog | Nvidia")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://youtu.be/28bl_UcsvCY"},"Official Video")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://nbviewer.jupyter.org/github/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_1_CombineCategories.ipynb"},"RapidsAI Notebook"))),(0,o.kt)("h2",{id:"links"},"Links"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://www.tensorflow.org/recommenders/examples/dcn"},"https://www.tensorflow.org/recommenders/examples/dcn")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/shenweichen/DeepCTR-Torch"},"https://github.com/shenweichen/DeepCTR-Torch")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.dcn.html"},"https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.dcn.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.dcn.html"},"https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.dcn.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.dcn.html"},"https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.dcn.html")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://medium.com/@SeoJaeDuk/deep-cross-network-for-ad-click-predictions-1714321f739a"},"https://medium.com/@SeoJaeDuk/deep-cross-network-for-ad-click-predictions-1714321f739a"))))}u.isMDXComponent=!0},82343:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp2-dcn-untitled-1-325c5104e54f664f27dae3a7f6f2a986.png"},40106:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp2-dcn-untitled-2-0a7c84496181e60de42959ae42654edc.png"},12421:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp2-dcn-untitled-3-e6da89f71e3602458d019813e0c8885d.png"},54895:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp2-dcn-untitled-2c85807ebece66eec429f38f500c815c.png"}}]);