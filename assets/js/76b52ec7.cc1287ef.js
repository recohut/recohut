"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1701],{3905:function(e,t,r){r.d(t,{Zo:function(){return u},kt:function(){return d}});var n=r(67294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var c=n.createContext({}),l=function(e){var t=n.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},u=function(e){var t=l(e.components);return n.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),m=l(r),d=o,h=m["".concat(c,".").concat(d)]||m[d]||p[d]||a;return r?n.createElement(h,i(i({ref:t},u),{},{components:r})):n.createElement(h,i({ref:t},u))}));function d(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=m;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,i[1]=s;for(var l=2;l<a;l++)i[l]=r[l];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},27878:function(e,t,r){r.r(t),r.d(t,{assets:function(){return u},contentTitle:function(){return c},default:function(){return d},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return p}});var n=r(87462),o=r(63366),a=(r(67294),r(3905)),i=["components"],s={},c="SVAE",l={unversionedId:"models/svae",id:"models/svae",title:"SVAE",description:"SVAE stands for Sequential Variational Autoencoder.",source:"@site/docs/models/svae.md",sourceDirName:"models",slug:"/models/svae",permalink:"/recohut/docs/models/svae",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/svae.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Struc2Vec",permalink:"/recohut/docs/models/struc2vec"},next:{title:"TAaMR",permalink:"/recohut/docs/models/taamr"}},u={},p=[{value:"Links",id:"links",level:2}],m={toc:p};function d(e){var t=e.components,r=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,n.Z)({},m,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"svae"},"SVAE"),(0,a.kt)("p",null,"SVAE stands for Sequential Variational Autoencoder. "),(0,a.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,a.kt)("div",{parentName:"div",className:"admonition-heading"},(0,a.kt)("h5",{parentName:"div"},(0,a.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,a.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,a.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"research paper")),(0,a.kt)("div",{parentName:"div",className:"admonition-content"},(0,a.kt)("p",{parentName:"div"},(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1811.09975"},"Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco and Vikram Pudi, \u201c",(0,a.kt)("em",{parentName:"a"},"Sequential Variational Autoencoders for Collaborative Filtering"),"\u201d. WSDM, 2019.")),(0,a.kt)("blockquote",{parentName:"div"},(0,a.kt)("p",{parentName:"blockquote"},"Variational autoencoders were proven successful in domains such as computer vision and speech processing. Their adoption for modeling user preferences is still unexplored, although recently it is starting to gain attention in the current literature. In this work, we propose a model which extends variational autoencoders by exploiting the rich information present in the past preference history. We introduce a recurrent version of the VAE, where instead of passing a subset of the whole history regardless of temporal dependencies, we rather pass the consumption sequence subset through a recurrent neural network. At each time-step of the RNN, the sequence is fed through a series of fully-connected layers, the output of which models the probability distribution of the most likely future preferences. We show that handling temporal information is crucial for improving the accuracy of the VAE: In fact, our model beats the current state-of-the-art by valuable margins because of its ability to capture temporal dependencies among the user-consumption sequence using the recurrent encoder still keeping the fundamentals of variational autoencoders intact.")))),(0,a.kt)("h2",{id:"links"},"Links"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/SVAE-PyTorch"},"https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/SVAE-PyTorch")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Sequential-Variational-Autoencoders-for-Collaborative-Filtering.pdf"},"https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Sequential-Variational-Autoencoders-for-Collaborative-Filtering.pdf"))))}d.isMDXComponent=!0}}]);