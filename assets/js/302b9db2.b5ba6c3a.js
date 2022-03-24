"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6062],{3905:function(a,e,t){t.d(e,{Zo:function(){return N},kt:function(){return k}});var s=t(67294);function n(a,e,t){return e in a?Object.defineProperty(a,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):a[e]=t,a}function m(a,e){var t=Object.keys(a);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(a);e&&(s=s.filter((function(e){return Object.getOwnPropertyDescriptor(a,e).enumerable}))),t.push.apply(t,s)}return t}function p(a){for(var e=1;e<arguments.length;e++){var t=null!=arguments[e]?arguments[e]:{};e%2?m(Object(t),!0).forEach((function(e){n(a,e,t[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(a,Object.getOwnPropertyDescriptors(t)):m(Object(t)).forEach((function(e){Object.defineProperty(a,e,Object.getOwnPropertyDescriptor(t,e))}))}return a}function r(a,e){if(null==a)return{};var t,s,n=function(a,e){if(null==a)return{};var t,s,n={},m=Object.keys(a);for(s=0;s<m.length;s++)t=m[s],e.indexOf(t)>=0||(n[t]=a[t]);return n}(a,e);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(a);for(s=0;s<m.length;s++)t=m[s],e.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(a,t)&&(n[t]=a[t])}return n}var i=s.createContext({}),l=function(a){var e=s.useContext(i),t=e;return a&&(t="function"==typeof a?a(e):p(p({},e),a)),t},N=function(a){var e=l(a.components);return s.createElement(i.Provider,{value:e},a.children)},o={inlineCode:"code",wrapper:function(a){var e=a.children;return s.createElement(s.Fragment,{},e)}},c=s.forwardRef((function(a,e){var t=a.components,n=a.mdxType,m=a.originalType,i=a.parentName,N=r(a,["components","mdxType","originalType","parentName"]),c=l(t),k=n,h=c["".concat(i,".").concat(k)]||c[k]||o[k]||m;return t?s.createElement(h,p(p({ref:e},N),{},{components:t})):s.createElement(h,p({ref:e},N))}));function k(a,e){var t=arguments,n=e&&e.mdxType;if("string"==typeof a||n){var m=t.length,p=new Array(m);p[0]=c;var r={};for(var i in e)hasOwnProperty.call(e,i)&&(r[i]=e[i]);r.originalType=a,r.mdxType="string"==typeof a?a:n,p[1]=r;for(var l=2;l<m;l++)p[l]=t[l];return s.createElement.apply(null,p)}return s.createElement.apply(null,t)}c.displayName="MDXCreateElement"},83489:function(a,e,t){t.r(e),t.d(e,{assets:function(){return N},contentTitle:function(){return i},default:function(){return k},frontMatter:function(){return r},metadata:function(){return l},toc:function(){return o}});var s=t(87462),n=t(63366),m=(t(67294),t(3905)),p=["components"],r={},i="SASRec",l={unversionedId:"models/sasrec",id:"models/sasrec",title:"SASRec",description:"SASRec stands for Self-Attentive Sequential Recommendation. It relies on the sequence modeling capabilities of self-attentive neural networks to predict the occurence of the next item in a user\u2019s consumption sequence. To be precise, given a user \ud835\udc62 and their time-ordered consumption history $S^\ud835\udc62 = (S1^u, S2^u, \\dots, S_{|S^u|}^\ud835\udc62),$ SASRec first applies self-attention on $S^\ud835\udc62$ followed by a series of non-linear feed-forward layers to finally obtain the next item likelihood.",source:"@site/docs/models/sasrec.md",sourceDirName:"models",slug:"/models/sasrec",permalink:"/ai/docs/models/sasrec",editUrl:"https://github.com/sparsh-ai/ai/docs/models/sasrec.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"SARSA",permalink:"/ai/docs/models/sarsa"},next:{title:"SDNE",permalink:"/ai/docs/models/sdne"}},N={},o=[{value:"Architecture",id:"architecture",level:2}],c={toc:o};function k(a){var e=a.components,r=(0,n.Z)(a,p);return(0,m.kt)("wrapper",(0,s.Z)({},c,r,{components:e,mdxType:"MDXLayout"}),(0,m.kt)("h1",{id:"sasrec"},"SASRec"),(0,m.kt)("p",null,"SASRec stands for Self-Attentive Sequential Recommendation. It relies on the sequence modeling capabilities of self-attentive neural networks to predict the occurence of the next item in a user\u2019s consumption sequence. To be precise, given a user \ud835\udc62 and their time-ordered consumption history ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msup"},"S"),(0,m.kt)("mi",{parentName:"msup"},"\ud835\udc62")),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("msubsup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msubsup"},"S"),(0,m.kt)("mn",{parentName:"msubsup"},"1"),(0,m.kt)("mi",{parentName:"msubsup"},"u")),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("msubsup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msubsup"},"S"),(0,m.kt)("mn",{parentName:"msubsup"},"2"),(0,m.kt)("mi",{parentName:"msubsup"},"u")),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mo",{parentName:"mrow"},"\u2026"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("msubsup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msubsup"},"S"),(0,m.kt)("mrow",{parentName:"msubsup"},(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"\u2223"),(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msup"},"S"),(0,m.kt)("mi",{parentName:"msup"},"u")),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"\u2223")),(0,m.kt)("mi",{parentName:"msubsup"},"\ud835\udc62")),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},",")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"S^\ud835\udc62 = (S_1^u, S_2^u, \\dots, S_{|S^u|}^\ud835\udc62),")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6833em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.6644em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u")))))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1.247em",verticalAlign:"-0.497em"}}),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.6644em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.4519em",marginLeft:"-0.0576em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"1"))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2481em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.6644em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.4519em",marginLeft:"-0.0576em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"2"))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2481em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"minner"},"\u2026"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.6644em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.378em",marginLeft:"-0.0576em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"\u2223"),(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.5935em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.786em",marginRight:"0.0714em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.5em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size3 size1 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u")))))))),(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"\u2223")))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.497em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mclose"},")"),(0,m.kt)("span",{parentName:"span",className:"mpunct"},",")))))," SASRec first applies self-attention on ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msup"},"S"),(0,m.kt)("mi",{parentName:"msup"},"\ud835\udc62"))),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"S^\ud835\udc62")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6833em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.6644em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u"))))))))))))," followed by a series of non-linear feed-forward layers to finally obtain the next item likelihood."),(0,m.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,m.kt)("div",{parentName:"div",className:"admonition-heading"},(0,m.kt)("h5",{parentName:"div"},(0,m.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,m.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,m.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"research paper")),(0,m.kt)("div",{parentName:"div",className:"admonition-content"},(0,m.kt)("p",{parentName:"div"},(0,m.kt)("a",{parentName:"p",href:"https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf"},"Wang-Cheng Kang and Julian McAuley, \u201c",(0,m.kt)("em",{parentName:"a"},"Self-Attentive Sequential Recommendation"),"\u201d. ICDM, 2018.")),(0,m.kt)("blockquote",{parentName:"div"},(0,m.kt)("p",{parentName:"blockquote"},"Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the \u2018context\u2019 of users\u2019 activities on the basis of actions they have performed recently. To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). Markov Chains assume that a user\u2019s next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. The goal of our work is to balance these two goals, by proposing a self-attention based sequential model (SASRec) that allows us to capture long-term semantics (like an RNN), but, using an attention mechanism, makes its predictions based on relatively few actions (like an MC). At each time step, SASRec seeks to identify which items are \u2018relevant\u2019 from a user\u2019s action history, and use them to predict the next item. Extensive empirical studies show that our method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. Moreover, the model is an order of magnitude more efficient than comparable CNN/RNN-based models. Visualizations on attention weights also show how our model adaptively handles datasets with various density, and uncovers meaningful patterns in activity sequences.")))),(0,m.kt)("h2",{id:"architecture"},"Architecture"),(0,m.kt)("p",null,"Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the \u2018context\u2019 of users\u2019 activities on the basis of actions they have performed recently. To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). Markov Chains assume that a user\u2019s next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. SASRec captures the long-term semantics (like an RNN), but, using an attention mechanism, makes its predictions based on relatively few actions (like an MC)."),(0,m.kt)("p",null,(0,m.kt)("img",{loading:"lazy",alt:"US512148 _ General Recommenders-L186674 _ SASRec Model.drawio.png",src:t(72295).Z,width:"832",height:"592"})),(0,m.kt)("p",null,"At each time step, SASRec seeks to identify which items are \u2018relevant\u2019 from a user\u2019s action history, and use them to predict the next item. Extensive empirical studies show that this method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. Moreover, the model is an order of magnitude more efficient than comparable CNN/RNN-based models."),(0,m.kt)("p",null,(0,m.kt)("img",{loading:"lazy",alt:"A simplified diagram showing the training process of SASRec. At each time step, the model considers all previous items, and uses attention to \u2018focus on\u2019 items relevant to the next action.",src:t(60017).Z,width:"663",height:"496"})),(0,m.kt)("p",null,"A simplified diagram showing the training process of SASRec. At each time step, the model considers all previous items, and uses attention to \u2018focus on\u2019 items relevant to the next action."),(0,m.kt)("p",null,"We adopt the binary cross entropy loss as the objective function:"),(0,m.kt)("div",{className:"math math-display"},(0,m.kt)("span",{parentName:"div",className:"katex-display"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mo",{parentName:"mrow"},"\u2212"),(0,m.kt)("munder",{parentName:"mrow"},(0,m.kt)("mo",{parentName:"munder"},"\u2211"),(0,m.kt)("mrow",{parentName:"munder"},(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msup"},"S"),(0,m.kt)("mi",{parentName:"msup"},"u")),(0,m.kt)("mo",{parentName:"mrow"},"\u2208"),(0,m.kt)("mi",{parentName:"mrow"},"S"))),(0,m.kt)("munder",{parentName:"mrow"},(0,m.kt)("mo",{parentName:"munder"},"\u2211"),(0,m.kt)("mrow",{parentName:"munder"},(0,m.kt)("mi",{parentName:"mrow"},"t"),(0,m.kt)("mo",{parentName:"mrow"},"\u2208"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"["),(0,m.kt)("mn",{parentName:"mrow"},"1"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mn",{parentName:"mrow"},"2"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mo",{parentName:"mrow"},"\u2026"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mi",{parentName:"mrow"},"n"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"]"))),(0,m.kt)("mrow",{parentName:"mrow"},(0,m.kt)("mo",{parentName:"mrow",fence:"true"},"["),(0,m.kt)("mi",{parentName:"mrow"},"l"),(0,m.kt)("mi",{parentName:"mrow"},"o"),(0,m.kt)("mi",{parentName:"mrow"},"g"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("mi",{parentName:"mrow"},"\u03c3"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"r"),(0,m.kt)("mrow",{parentName:"msub"},(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"o"),(0,m.kt)("mi",{parentName:"msub"},"t")),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mi",{parentName:"mrow"},"t"))),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,m.kt)("mo",{parentName:"mrow"},"+"),(0,m.kt)("munder",{parentName:"mrow"},(0,m.kt)("mo",{parentName:"munder"},"\u2211"),(0,m.kt)("mrow",{parentName:"munder"},(0,m.kt)("mi",{parentName:"mrow"},"j"),(0,m.kt)("mo",{parentName:"mrow",mathvariant:"normal"},"\u2209"),(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msup"},"S"),(0,m.kt)("mi",{parentName:"msup"},"u")))),(0,m.kt)("mi",{parentName:"mrow"},"l"),(0,m.kt)("mi",{parentName:"mrow"},"o"),(0,m.kt)("mi",{parentName:"mrow"},"g"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("mn",{parentName:"mrow"},"1"),(0,m.kt)("mo",{parentName:"mrow"},"\u2212"),(0,m.kt)("mi",{parentName:"mrow"},"\u03c3"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"r"),(0,m.kt)("mrow",{parentName:"msub"},(0,m.kt)("mi",{parentName:"mrow"},"j"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mi",{parentName:"mrow"},"t"))),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,m.kt)("mo",{parentName:"mrow",fence:"true"},"]"))),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"-\\sum_{S^u\\in S} \\sum_{t \\in [1,2,\\dots,n]}\\left[ log(\\sigma(r_{o_t,t})) + \\sum_{j \\notin S^u} log(1-\\sigma(r_{j,t})) \\right]")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"3.6em",verticalAlign:"-1.55em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"\u2212"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mop op-limits"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.05em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-1.8557em",marginLeft:"0em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.05em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.5935em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.786em",marginRight:"0.0714em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.5em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size3 size1 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u")))))))),(0,m.kt)("span",{parentName:"span",className:"mrel mtight"},"\u2208"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05764em"}},"S")))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.05em"}}),(0,m.kt)("span",{parentName:"span"},(0,m.kt)("span",{parentName:"span",className:"mop op-symbol large-op"},"\u2211")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.3217em"}},(0,m.kt)("span",{parentName:"span"}))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mop op-limits"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.05em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-1.809em",marginLeft:"0em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.05em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"t"),(0,m.kt)("span",{parentName:"span",className:"mrel mtight"},"\u2208"),(0,m.kt)("span",{parentName:"span",className:"mopen mtight"},"["),(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"1"),(0,m.kt)("span",{parentName:"span",className:"mpunct mtight"},","),(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"2"),(0,m.kt)("span",{parentName:"span",className:"mpunct mtight"},","),(0,m.kt)("span",{parentName:"span",className:"minner mtight"},"\u2026"),(0,m.kt)("span",{parentName:"span",className:"mpunct mtight"},","),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"n"),(0,m.kt)("span",{parentName:"span",className:"mclose mtight"},"]")))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.05em"}}),(0,m.kt)("span",{parentName:"span"},(0,m.kt)("span",{parentName:"span",className:"mop op-symbol large-op"},"\u2211")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.516em"}},(0,m.kt)("span",{parentName:"span"}))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"minner"},(0,m.kt)("span",{parentName:"span",className:"mopen"},(0,m.kt)("span",{parentName:"span",className:"delimsizing mult"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"2.05em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.25em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.155em"}}),(0,m.kt)("span",{parentName:"span",className:"delimsizinginner delim-size4"},(0,m.kt)("span",{parentName:"span"},"\u23a3"))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.397em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.155em"}}),(0,m.kt)("span",{parentName:"span",style:{height:"0.016em",width:"0.6667em"}},(0,m.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"0.6667em",height:"0.016em",style:{width:"0.6667em"},viewBox:"0 0 666.67 16",preserveAspectRatio:"xMinYMin"},(0,m.kt)("path",{parentName:"svg",d:"M319 0 H403 V16 H319z M319 0 H403 V16 H319z"})))),(0,m.kt)("span",{parentName:"span",style:{top:"-4.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.155em"}}),(0,m.kt)("span",{parentName:"span",className:"delimsizinginner delim-size4"},(0,m.kt)("span",{parentName:"span"},"\u23a1")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.55em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.01968em"}},"l"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"o"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"g"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"\u03c3"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02778em"}},"r"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2806em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0278em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"o"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2963em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.357em",marginLeft:"0em",marginRight:"0.0714em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.5em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size3 size1 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"t")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.143em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mpunct mtight"},","),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"t"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2861em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mclose"},"))"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222em"}}),(0,m.kt)("span",{parentName:"span",className:"mbin"},"+"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222em"}}),(0,m.kt)("span",{parentName:"span",className:"mop op-limits"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.05em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-1.809em",marginLeft:"0em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.05em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05724em"}},"j"),(0,m.kt)("span",{parentName:"span",className:"mrel mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mrel mtight"},"\u2208")),(0,m.kt)("span",{parentName:"span",className:"mord vbox mtight"},(0,m.kt)("span",{parentName:"span",className:"thinbox mtight"},(0,m.kt)("span",{parentName:"span",className:"llap mtight"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,m.kt)("span",{parentName:"span",className:"inner"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"/"),(0,m.kt)("span",{parentName:"span",className:"mspace mtight",style:{marginRight:"0.0651em"}}))),(0,m.kt)("span",{parentName:"span",className:"fix"}))))),(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05764em"}},"S"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.5935em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.786em",marginRight:"0.0714em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.5em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size3 size1 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"u"))))))))))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.05em"}}),(0,m.kt)("span",{parentName:"span"},(0,m.kt)("span",{parentName:"span",className:"mop op-symbol large-op"},"\u2211")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.516em"}},(0,m.kt)("span",{parentName:"span"}))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.01968em"}},"l"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"o"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"g"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord"},"1"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222em"}}),(0,m.kt)("span",{parentName:"span",className:"mbin"},"\u2212"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"\u03c3"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02778em"}},"r"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.3117em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0278em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05724em"}},"j"),(0,m.kt)("span",{parentName:"span",className:"mpunct mtight"},","),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"t"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2861em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mclose"},"))"),(0,m.kt)("span",{parentName:"span",className:"mclose"},(0,m.kt)("span",{parentName:"span",className:"delimsizing mult"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"2.05em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.25em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.155em"}}),(0,m.kt)("span",{parentName:"span",className:"delimsizinginner delim-size4"},(0,m.kt)("span",{parentName:"span"},"\u23a6"))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.397em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.155em"}}),(0,m.kt)("span",{parentName:"span",style:{height:"0.016em",width:"0.6667em"}},(0,m.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"0.6667em",height:"0.016em",style:{width:"0.6667em"},viewBox:"0 0 666.67 16",preserveAspectRatio:"xMinYMin"},(0,m.kt)("path",{parentName:"svg",d:"M263 0 H347 V16 H263z M263 0 H347 V16 H263z"})))),(0,m.kt)("span",{parentName:"span",style:{top:"-4.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3.155em"}}),(0,m.kt)("span",{parentName:"span",className:"delimsizinginner delim-size4"},(0,m.kt)("span",{parentName:"span"},"\u23a4")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.55em"}},(0,m.kt)("span",{parentName:"span"})))))))))))))}k.isMDXComponent=!0},72295:function(a,e,t){e.Z=t.p+"assets/images/content-models-raw-mp1-sasrec---.drawio-2d285a329e558430aca4f8caacf3d362.png"},60017:function(a,e,t){e.Z=t.p+"assets/images/content-models-raw-mp1-sasrec-untitled-0614ecc2cee131a22e42889b74892e35.png"}}]);