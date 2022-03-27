"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2871],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return m}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),u=l(n),m=a,f=u["".concat(c,".").concat(m)]||u[m]||d[m]||o;return n?r.createElement(f,s(s({ref:t},p),{},{components:n})):r.createElement(f,s({ref:t},p))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,s=new Array(o);s[0]=u;var i={};for(var c in t)hasOwnProperty.call(t,c)&&(i[c]=t[c]);i.originalType=e,i.mdxType="string"==typeof e?e:a,s[1]=i;for(var l=2;l<o;l++)s[l]=n[l];return r.createElement.apply(null,s)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},65354:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return c},default:function(){return m},frontMatter:function(){return i},metadata:function(){return l},toc:function(){return d}});var r=n(87462),a=n(63366),o=(n(67294),n(3905)),s=["components"],i={},c="GC-SAN",l={unversionedId:"models/gc-san",id:"models/gc-san",title:"GC-SAN",description:"GC-SAN stands for Graph contextualized self-attention.",source:"@site/docs/models/gc-san.md",sourceDirName:"models",slug:"/models/gc-san",permalink:"/recohut/docs/models/gc-san",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/gc-san.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"GAT",permalink:"/recohut/docs/models/gat"},next:{title:"GCE-GNN",permalink:"/recohut/docs/models/gce-gnn"}},p={},d=[],u={toc:d};function m(e){var t=e.components,i=(0,a.Z)(e,s);return(0,o.kt)("wrapper",(0,r.Z)({},u,i,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"gc-san"},"GC-SAN"),(0,o.kt)("p",null,"GC-SAN stands for Graph contextualized self-attention."),(0,o.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"research paper")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},(0,o.kt)("a",{parentName:"p",href:"https://www.ijcai.org/proceedings/2019/0547.pdf"},"Xu et. al., \u201c",(0,o.kt)("em",{parentName:"a"},"Graph Contextualized Self-Attention Network for Session-based Recommendation"),"\u201d. IJCAI, 2019.")),(0,o.kt)("blockquote",{parentName:"div"},(0,o.kt)("p",{parentName:"blockquote"},"Session-based recommendation, which aims to predict the user\u2019s immediate next action based on anonymous sessions, is a key task in many online services (e.g., e-commerce, media streaming). Recently, Self-Attention Network (SAN) has achieved significant success in various sequence modeling tasks without using either recurrent or convolutional network. However, SAN lacks local dependencies that exist over adjacent items and limits its capacity for learning contextualized representations of items in sequences. In this paper, we propose a graph contextualized self-attention model (GC-SAN), which utilizes both graph neural network and self-attention mechanism, for session-based recommendation. In GC-SAN, we dynamically construct a graph structure for session sequences and capture rich local dependencies via graph neural network (GNN). Then each session learns long-range dependencies by applying the self-attention mechanism. Finally, each session is represented as a linear combination of the global preference and the current interest of that session. Extensive experiments on two real-world datasets show that GC-SAN outperforms state-of-the-art methods consistently.")))),(0,o.kt)("p",null,"Graph contextualized self-attention model (GC-SAN) utilizes both graph neural network and self-attention mechanism, for session-based recommendation. In GC-SAN, we dynamically construct a graph structure for session sequences and capture rich local dependencies via graph neural network (GNN). Then each session learns long-range dependencies by applying the self-attention mechanism. Finally, each session is represented as a linear combination of the global preference and the current interest of that session."),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:n(6035).Z,width:"1202",height:"408"})),(0,o.kt)("p",null,"We first construct a directed graph of all session sequences. Based on the graph, we apply graph neural network to obtain all node vectors involved in the session graph. After that, we use a multi-layer self-attention network to capture long-range dependencies between items in the session. In prediction layer, we represent each session as a linear of the global preference and the current interest of that session. Finally, we compute the ranking scores of each candidate item for recommendation."),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:n(46180).Z,width:"1235",height:"290"})))}m.isMDXComponent=!0},46180:function(e,t,n){t.Z=n.p+"assets/images/content-models-raw-mp2-gc-san-untitled-1-7560e35d7c8e5e1f760710f5e8c207f7.png"},6035:function(e,t,n){t.Z=n.p+"assets/images/content-models-raw-mp2-gc-san-untitled-7ea3a090d360192c82754c70d7b18e32.png"}}]);