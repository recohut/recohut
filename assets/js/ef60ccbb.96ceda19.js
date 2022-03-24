"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[227],{3905:function(e,t,n){n.d(t,{Zo:function(){return d},kt:function(){return p}});var r=n(67294);function s(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){s(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function a(e,t){if(null==e)return{};var n,r,s=function(e,t){if(null==e)return{};var n,r,s={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(s[n]=e[n]);return s}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(s[n]=e[n])}return s}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},d=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},h=r.forwardRef((function(e,t){var n=e.components,s=e.mdxType,o=e.originalType,c=e.parentName,d=a(e,["components","mdxType","originalType","parentName"]),h=l(n),p=s,f=h["".concat(c,".").concat(p)]||h[p]||u[p]||o;return n?r.createElement(f,i(i({ref:t},d),{},{components:n})):r.createElement(f,i({ref:t},d))}));function p(e,t){var n=arguments,s=t&&t.mdxType;if("string"==typeof e||s){var o=n.length,i=new Array(o);i[0]=h;var a={};for(var c in t)hasOwnProperty.call(t,c)&&(a[c]=t[c]);a.originalType=e,a.mdxType="string"==typeof e?e:s,i[1]=a;for(var l=2;l<o;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}h.displayName="MDXCreateElement"},21522:function(e,t,n){n.r(t),n.d(t,{assets:function(){return d},contentTitle:function(){return c},default:function(){return p},frontMatter:function(){return a},metadata:function(){return l},toc:function(){return u}});var r=n(87462),s=n(63366),o=(n(67294),n(3905)),i=["components"],a={},c="GRU4Rec",l={unversionedId:"models/gru4rec",id:"models/gru4rec",title:"GRU4Rec",description:"It uses session-parallel mini-batch approach where we first create an order for the sessions and then, we use the first event of the first X sessions to form the input of the first mini-batch (the desired output is the second events of our active sessions). The second mini-batch is formed from the second events and so on. If any of the sessions end, the next available session is put in its place. Sessions are assumed to be independent, thus we reset the appropriate hidden state when this switch occurs.",source:"@site/docs/models/gru4rec.md",sourceDirName:"models",slug:"/models/gru4rec",permalink:"/ai/docs/models/gru4rec",editUrl:"https://github.com/sparsh-ai/ai/docs/models/gru4rec.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"GLMix",permalink:"/ai/docs/models/glmix"},next:{title:"HMLET",permalink:"/ai/docs/models/hmlet"}},d={},u=[{value:"Links",id:"links",level:2}],h={toc:u};function p(e){var t=e.components,a=(0,s.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},h,a,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"gru4rec"},"GRU4Rec"),(0,o.kt)("p",null,"It uses session-parallel mini-batch approach where we first create an order for the sessions and then, we use the first event of the first X sessions to form the input of the first mini-batch (the desired output is the second events of our active sessions). The second mini-batch is formed from the second events and so on. If any of the sessions end, the next available session is put in its place. Sessions are assumed to be independent, thus we reset the appropriate hidden state when this switch occurs."),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"/img/content-models-raw-mp2-gru4rec-untitled.png",src:n(49690).Z,width:"742",height:"651"})),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"/img/content-models-raw-mp2-gru4rec-untitled-1.png",src:n(93074).Z,width:"742",height:"268"})),(0,o.kt)("p",null,"One of the first successful approaches for using RNNs in the recommendation domain is the GRU4Rec network (Hidasi, Karatzoglou, Baltrunas, and Tikk, 2015). A RNN with GRUs was used for the session-based recommendation. A novel training mechanism called session-parallel mini-batches is used in GRU4Rec, as shown in Figure 3. Each position in a mini-batch belongs to a particular session in the training data. The network finds a hidden state for each position in the batch separately, but this hidden state is kept and used in the next iteration at the positions when the same session continues with the next batch. However, it is erased at the positions of new sessions coming up with the start of the next batch. The network is always updated with the session beginning and used to predict the subsequent events. GRU4Rec architecture is composed of an embedding layer followed by multiple optional GRU layers, a feed-forward network, and a softmax layer for output score predictions for candidate items. The session items are one-hot-encoded in a vector representing all items\u2019 space to be fed into the network as input. On the other hand, a similar output vector is obtained from the softmax layer to represent the predicted ranking of items. Additionally, the authors designed two new loss functions, namely, Bayesian personalized ranking (BPR) loss and regularized approximation of the relative rank of the relevant item (TOP1) loss. BPR uses a pairwise ranking loss function by averaging the target item\u2019s score with several sampled negative ones in the loss value. TOP1 is the regularized approximation of the relative rank of the relevant item loss. Later, Hidasi and Karatzoglou (2018) extended their work by modifying the two-loss functions introduced previously by solving the issues of vanishing gradient faced by TOP1 and BPR when the negative samples have very low predicted likelihood that approaches zero. The newly proposed losses merge between the knowledge from the deep learning and the literature of learning to rank. The evaluation of the new extended version shows a clear superiority over the older version of the network. Thus, we have included the extended version of the GRU4Rec network, denoted by GRU4Rec+, in our evaluation study."),(0,o.kt)("h2",{id:"links"},"Links"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://recbole.io/docs/user_guide/model/sequential/gru4rec.html"},"https://recbole.io/docs/user_guide/model/sequential/gru4rec.html"))))}p.isMDXComponent=!0},93074:function(e,t,n){t.Z=n.p+"assets/images/content-models-raw-mp2-gru4rec-untitled-1-eba35697029090991c0b3d7e408ecca0.png"},49690:function(e,t,n){t.Z=n.p+"assets/images/content-models-raw-mp2-gru4rec-untitled-8655ccfa589331e6781e8d889c03ad48.png"}}]);