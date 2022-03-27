"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[8456],{3905:function(a,e,t){t.d(e,{Zo:function(){return l},kt:function(){return h}});var n=t(67294);function s(a,e,t){return e in a?Object.defineProperty(a,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):a[e]=t,a}function m(a,e){var t=Object.keys(a);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(a);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(a,e).enumerable}))),t.push.apply(t,n)}return t}function r(a){for(var e=1;e<arguments.length;e++){var t=null!=arguments[e]?arguments[e]:{};e%2?m(Object(t),!0).forEach((function(e){s(a,e,t[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(a,Object.getOwnPropertyDescriptors(t)):m(Object(t)).forEach((function(e){Object.defineProperty(a,e,Object.getOwnPropertyDescriptor(t,e))}))}return a}function p(a,e){if(null==a)return{};var t,n,s=function(a,e){if(null==a)return{};var t,n,s={},m=Object.keys(a);for(n=0;n<m.length;n++)t=m[n],e.indexOf(t)>=0||(s[t]=a[t]);return s}(a,e);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(a);for(n=0;n<m.length;n++)t=m[n],e.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(a,t)&&(s[t]=a[t])}return s}var i=n.createContext({}),o=function(a){var e=n.useContext(i),t=e;return a&&(t="function"==typeof a?a(e):r(r({},e),a)),t},l=function(a){var e=o(a.components);return n.createElement(i.Provider,{value:e},a.children)},c={inlineCode:"code",wrapper:function(a){var e=a.children;return n.createElement(n.Fragment,{},e)}},N=n.forwardRef((function(a,e){var t=a.components,s=a.mdxType,m=a.originalType,i=a.parentName,l=p(a,["components","mdxType","originalType","parentName"]),N=o(t),h=s,k=N["".concat(i,".").concat(h)]||N[h]||c[h]||m;return t?n.createElement(k,r(r({ref:e},l),{},{components:t})):n.createElement(k,r({ref:e},l))}));function h(a,e){var t=arguments,s=e&&e.mdxType;if("string"==typeof a||s){var m=t.length,r=new Array(m);r[0]=N;var p={};for(var i in e)hasOwnProperty.call(e,i)&&(p[i]=e[i]);p.originalType=a,p.mdxType="string"==typeof a?a:s,r[1]=p;for(var o=2;o<m;o++)r[o]=t[o];return n.createElement.apply(null,r)}return n.createElement.apply(null,t)}N.displayName="MDXCreateElement"},37440:function(a,e,t){t.r(e),t.d(e,{assets:function(){return l},contentTitle:function(){return i},default:function(){return h},frontMatter:function(){return p},metadata:function(){return o},toc:function(){return c}});var n=t(87462),s=t(63366),m=(t(67294),t(3905)),r=["components"],p={},i="MMoE",o={unversionedId:"models/mmoe",id:"models/mmoe",title:"MMoE",description:"MMoE stands for Multi-gate Mixture-of-Experts.",source:"@site/docs/models/mmoe.mdx",sourceDirName:"models",slug:"/models/mmoe",permalink:"/recohut/docs/models/mmoe",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/mmoe.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"MIAN",permalink:"/recohut/docs/models/mian"},next:{title:"MPNN",permalink:"/recohut/docs/models/mpnn"}},l={},c=[{value:"Type of gates",id:"type-of-gates",level:2},{value:"Softmax gate",id:"softmax-gate",level:3},{value:"Sparse gate",id:"sparse-gate",level:3},{value:"Data partitioning into subsets for each expert",id:"data-partitioning-into-subsets-for-each-expert",level:2},{value:"References",id:"references",level:2}],N={toc:c};function h(a){var e=a.components,t=(0,s.Z)(a,r);return(0,m.kt)("wrapper",(0,n.Z)({},N,t,{components:e,mdxType:"MDXLayout"}),(0,m.kt)("h1",{id:"mmoe"},"MMoE"),(0,m.kt)("p",null,"MMoE stands for Multi-gate Mixture-of-Experts."),(0,m.kt)("figure",null,(0,m.kt)("p",null,(0,m.kt)("center",null,(0,m.kt)("img",{src:"https://github.com/recohut/multiobjective-optimizations/raw/098954ea18e25506b6320039c85d09385c70f37b/docs/_images/L485744_1.png"}),(0,m.kt)("figcaption",null,"A multi-gate MoE for learning two tasks simultaneously.")))),(0,m.kt)("p",null,(0,m.kt)("center",null,(0,m.kt)("img",{src:"https://github.com/recohut/multiobjective-optimizations/raw/098954ea18e25506b6320039c85d09385c70f37b/docs/_images/L485744_2.png"}))),(0,m.kt)("h2",{id:"type-of-gates"},"Type of gates"),(0,m.kt)("h3",{id:"softmax-gate"},"Softmax gate"),(0,m.kt)("p",null,"A classical model for ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"g"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("mi",{parentName:"mrow"},"x"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"g(x)")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"g"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,m.kt)("span",{parentName:"span",className:"mclose"},")")))))," is the softmax gate: ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"\u03c3"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("mi",{parentName:"mrow"},"A"),(0,m.kt)("mi",{parentName:"mrow"},"x"),(0,m.kt)("mo",{parentName:"mrow"},"+"),(0,m.kt)("mi",{parentName:"mrow"},"b"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\u03c3(Ax+b)")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"\u03c3"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"A"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"x"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222em"}}),(0,m.kt)("span",{parentName:"span",className:"mbin"},"+"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"b"),(0,m.kt)("span",{parentName:"span",className:"mclose"},")"))))),", where ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"\u03c3"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"."),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\u03c3(.)")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"\u03c3"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord"},"."),(0,m.kt)("span",{parentName:"span",className:"mclose"},")")))))," is the softmax function, ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"A"),(0,m.kt)("mo",{parentName:"mrow"},"\u2208"),(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mrow",{parentName:"msup"},(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"I"),(0,m.kt)("mtext",{parentName:"mrow"},"\u2009\u2063"),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"R")),(0,m.kt)("mrow",{parentName:"msup"},(0,m.kt)("mi",{parentName:"mrow"},"n"),(0,m.kt)("mo",{parentName:"mrow"},"\xd7"),(0,m.kt)("mi",{parentName:"mrow"},"p")))),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"A \u2208 {\\rm I\\!R}^{n\xd7p}")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.7224em",verticalAlign:"-0.0391em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"A"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"\u2208"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.8446em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathrm"},"I"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"-0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathrm"},"R"))),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.8446em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-3.1362em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"n"),(0,m.kt)("span",{parentName:"span",className:"mbin mtight"},"\xd7"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"p"))))))))))))),"\nis a trainable weight matrix, and ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"b"),(0,m.kt)("mo",{parentName:"mrow"},"\u2208"),(0,m.kt)("msup",{parentName:"mrow"},(0,m.kt)("mrow",{parentName:"msup"},(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"I"),(0,m.kt)("mtext",{parentName:"mrow"},"\u2009\u2063"),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"R")),(0,m.kt)("mi",{parentName:"msup"},"n"))),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"b \u2208 {\\rm I\\!R}^n")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.7335em",verticalAlign:"-0.0391em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"b"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"\u2208"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.7376em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathrm"},"I"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"-0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathrm"},"R"))),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.7376em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-3.1362em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"n"))))))))))))," is a bias vector. This gate is dense, in the sense that all experts are assigned nonzero probabilities. Note that static gating (i.e., gating which does not depend on the input example) can be obtained by setting ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"A"),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"0")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"A = 0")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6833em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"A"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"0"))))),"."),(0,m.kt)("h3",{id:"sparse-gate"},"Sparse gate"),(0,m.kt)("p",null,"Assign nonzero weights to only a small subset of the experts. Sparsity in gating can have various advantages, including better computational efficiency, interpretability, and improved statistical performance in certain settings."),(0,m.kt)("h2",{id:"data-partitioning-into-subsets-for-each-expert"},"Data partitioning into subsets for each expert"),(0,m.kt)("p",null,"Partitioning based on input alone versus partitioning based on input-output relationship"),(0,m.kt)("p",null,"The MMoE architecture is similar to the MoE architecture, except that it has an individual gating network for each task, rather than a single one for the entire model."),(0,m.kt)("p",null,"This allows the model to learn a per-task and per-sample weighting of each of the expert networks, instead of just a per-sample weighting. This allows the MMoE to learn to model the relationships between different tasks. Tasks which have little in common with each other will result in the gating networks of each task learning to use different expert networks."),(0,m.kt)("p",null,"The authors of the MMoE validate this conclusion by comparing the shared-bottom, MoE, and MMoE architectures on synthetic data-sets with varying levels of task correlation."),(0,m.kt)("p",null,(0,m.kt)("center",null,(0,m.kt)("img",{src:"https://github.com/recohut/multiobjective-optimizations/raw/098954ea18e25506b6320039c85d09385c70f37b/docs/_images/L485744_3.png"}))),(0,m.kt)("p",null,"First, we see that the shared-bottom model underperforms in all cases relative to the MoE and MMoE models."),(0,m.kt)("p",null,"Next, we can see that the performance gap between the MoE and MMoE models increases as correlation between the tasks decreases."),(0,m.kt)("p",null,"This shows that the MMoE is better able to handle situations in which tasks are unrelated to each other. The larger the task diversity, the more benefit the MMoE is likely to have over the shared-bottom or MoE architectures."),(0,m.kt)("h2",{id:"references"},"References"),(0,m.kt)("ol",null,(0,m.kt)("li",{parentName:"ol"},(0,m.kt)("a",{parentName:"li",href:"https://youtu.be/Dweg47Tswxw"},"https://youtu.be/Dweg47Tswxw")),(0,m.kt)("li",{parentName:"ol"},(0,m.kt)("a",{parentName:"li",href:"https://smt.readthedocs.io/en/latest/_src_docs/applications/moe.html"},"https://smt.readthedocs.io/en/latest/_src_docs/applications/moe.html")),(0,m.kt)("li",{parentName:"ol"},(0,m.kt)("a",{parentName:"li",href:"https://towardsdatascience.com/multi-task-learning-with-multi-gate-mixture-of-experts-b46efac3268"},"https://towardsdatascience.com/multi-task-learning-with-multi-gate-mixture-of-experts-b46efac3268"))))}h.isMDXComponent=!0}}]);