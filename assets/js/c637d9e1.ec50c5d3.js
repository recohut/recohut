"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3147],{3905:function(t,e,r){r.d(e,{Zo:function(){return b},kt:function(){return p}});var a=r(67294);function n(t,e,r){return e in t?Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}):t[e]=r,t}function o(t,e){var r=Object.keys(t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(t);e&&(a=a.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),r.push.apply(r,a)}return r}function l(t){for(var e=1;e<arguments.length;e++){var r=null!=arguments[e]?arguments[e]:{};e%2?o(Object(r),!0).forEach((function(e){n(t,e,r[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(r,e))}))}return t}function i(t,e){if(null==t)return{};var r,a,n=function(t,e){if(null==t)return{};var r,a,n={},o=Object.keys(t);for(a=0;a<o.length;a++)r=o[a],e.indexOf(r)>=0||(n[r]=t[r]);return n}(t,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);for(a=0;a<o.length;a++)r=o[a],e.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(t,r)&&(n[r]=t[r])}return n}var s=a.createContext({}),c=function(t){var e=a.useContext(s),r=e;return t&&(r="function"==typeof t?t(e):l(l({},e),t)),r},b=function(t){var e=c(t.components);return a.createElement(s.Provider,{value:e},t.children)},g={inlineCode:"code",wrapper:function(t){var e=t.children;return a.createElement(a.Fragment,{},e)}},m=a.forwardRef((function(t,e){var r=t.components,n=t.mdxType,o=t.originalType,s=t.parentName,b=i(t,["components","mdxType","originalType","parentName"]),m=c(r),p=n,u=m["".concat(s,".").concat(p)]||m[p]||g[p]||o;return r?a.createElement(u,l(l({ref:e},b),{},{components:r})):a.createElement(u,l({ref:e},b))}));function p(t,e){var r=arguments,n=e&&e.mdxType;if("string"==typeof t||n){var o=r.length,l=new Array(o);l[0]=m;var i={};for(var s in e)hasOwnProperty.call(e,s)&&(i[s]=e[s]);i.originalType=t,i.mdxType="string"==typeof t?t:n,l[1]=i;for(var c=2;c<o;c++)l[c]=r[c];return a.createElement.apply(null,l)}return a.createElement.apply(null,r)}m.displayName="MDXCreateElement"},45381:function(t,e,r){r.r(e),r.d(e,{assets:function(){return b},contentTitle:function(){return s},default:function(){return p},frontMatter:function(){return i},metadata:function(){return c},toc:function(){return g}});var a=r(87462),n=r(63366),o=(r(67294),r(3905)),l=["components"],i={},s="Matrix Factorizations",c={unversionedId:"tutorials/matrix-factorizations",id:"tutorials/matrix-factorizations",title:"Matrix Factorizations",description:"Neural Matrix Factorization (NMF)",source:"@site/docs/tutorials/matrix-factorizations.mdx",sourceDirName:"tutorials",slug:"/tutorials/matrix-factorizations",permalink:"/recohut/docs/tutorials/matrix-factorizations",editUrl:"https://github.com/sparsh-ai/recohut/docs/tutorials/matrix-factorizations.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Mathematics",permalink:"/recohut/docs/tutorials/mathematics"},next:{title:"MLOps",permalink:"/recohut/docs/tutorials/mlops"}},b={},g=[{value:"Neural Matrix Factorization (NMF)",id:"neural-matrix-factorization-nmf",level:2},{value:"Neural Collaborative Filtering (NCF)",id:"neural-collaborative-filtering-ncf",level:2}],m={toc:g};function p(t){var e=t.components,r=(0,n.Z)(t,l);return(0,o.kt)("wrapper",(0,a.Z)({},m,r,{components:e,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"matrix-factorizations"},"Matrix Factorizations"),(0,o.kt)("h2",{id:"neural-matrix-factorization-nmf"},"Neural Matrix Factorization (NMF)"),(0,o.kt)("table",null,(0,o.kt)("thead",{parentName:"table"},(0,o.kt)("tr",{parentName:"thead"},(0,o.kt)("th",{parentName:"tr",align:null},"Description"),(0,o.kt)("th",{parentName:"tr",align:null},"Notebook"))),(0,o.kt)("tbody",{parentName:"table"},(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Movielens NMF"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-06-11-recostep-movielens-neural-mf.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"}))," ",(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/T517223%20%7C%20NeuMF%20on%20ML-1m.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Bookcross NMF"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-01-book-crossing-surprise-svd-nmf.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Movielens MLP MF"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-04-mf-mlp-movielens-in-pytorch.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Movielens MF Pytorch"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-05-mf-movielens-pytorch.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"}))," ",(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-17-movie-mf-torch.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Yelp GMF"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-11-gmf-yelp.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"GMF"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-18-gmf.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Neural Matrix Factorization from scratch in PyTorch"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-04-21-rec-algo-ncf-pytorch-pyy0715.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Various models"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/T499733%20%7C%20Training%20various%20Matrix%20Factorization%20models%20on%20ML-100k%20in%20PyTorch%20Lightning%20Framework.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))))),(0,o.kt)("h2",{id:"neural-collaborative-filtering-ncf"},"Neural Collaborative Filtering (NCF)"),(0,o.kt)("table",null,(0,o.kt)("thead",{parentName:"table"},(0,o.kt)("tr",{parentName:"thead"},(0,o.kt)("th",{parentName:"tr",align:null},"Description"),(0,o.kt)("th",{parentName:"tr",align:null},"Notebook"))),(0,o.kt)("tbody",{parentName:"table"},(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"NCF on Movielens using Analytics Zoo library"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-06-27-analytics-zoo-ncf-movielens.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"NCF on Goodreads using Analytics Zoo library"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-06-27-analytics-zoo-ncf-goodreads.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"NCF from scratch in pytorch"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-03-ncf-from-scratch-pytorch.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"NCF from scratch in Tensorflow"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-14-ncf-movielens-tensorflow.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"Neural Collaborative Filtering Recommenders"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-07-ncf.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"NCF Tensorflow"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-31-ncf-tf.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"}))," ",(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/T416527%20%7C%20NCF%20on%20ML-1m%20in%20TF%202x.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},"NCF Torch"),(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-31-ncf-torch-2.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"}))," ",(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2022-01-31-ncf-torch.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"}))," ",(0,o.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2021-07-05-movie-recommender-retrieval-pytorch-lightning.ipynb",alt:""}," ",(0,o.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})))))))}p.isMDXComponent=!0}}]);