"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7916],{3905:function(e,t,r){r.d(t,{Zo:function(){return u},kt:function(){return m}});var n=r(67294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function c(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var l=n.createContext({}),s=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},u=function(e){var t=s(e.components);return n.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},f=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,u=c(e,["components","mdxType","originalType","parentName"]),f=s(r),m=o,d=f["".concat(l,".").concat(m)]||f[m]||p[m]||a;return r?n.createElement(d,i(i({ref:t},u),{},{components:r})):n.createElement(d,i({ref:t},u))}));function m(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=f;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:o,i[1]=c;for(var s=2;s<a;s++)i[s]=r[s];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}f.displayName="MDXCreateElement"},15848:function(e,t,r){r.r(t),r.d(t,{assets:function(){return u},contentTitle:function(){return l},default:function(){return m},frontMatter:function(){return c},metadata:function(){return s},toc:function(){return p}});var n=r(87462),o=r(63366),a=(r(67294),r(3905)),i=["components"],c={},l="A3C",s={unversionedId:"models/a3c",id:"models/a3c",title:"A3C",description:"A3C stands for Asynchronous Advantage Actor-Critic. The A3C algorithm builds upon the Actor-Critic class of algorithms by using a neural network to approximate the actor (and critic). The actor learns the policy function using a deep neural network, while the critic estimates the value function. The asynchronous nature of the algorithm allows the agent to learn from different parts of the state space, allowing parallel learning and faster convergence. Unlike DQN agents, which use an experience replay memory, the A3C agent uses multiple workers to gather more samples for learning.",source:"@site/docs/models/a3c.md",sourceDirName:"models",slug:"/models/a3c",permalink:"/ai/docs/models/a3c",editUrl:"https://github.com/sparsh-ai/ai/docs/models/a3c.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Models",permalink:"/ai/docs/models/"},next:{title:"AFM",permalink:"/ai/docs/models/afm"}},u={},p=[],f={toc:p};function m(e){var t=e.components,c=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,n.Z)({},f,c,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"a3c"},"A3C"),(0,a.kt)("p",null,"A3C stands for Asynchronous Advantage Actor-Critic. The A3C algorithm builds upon the Actor-Critic class of algorithms by using a neural network to approximate the actor (and critic). The actor learns the policy function using a deep neural network, while the critic estimates the value function. The asynchronous nature of the algorithm allows the agent to learn from different parts of the state space, allowing parallel learning and faster convergence. Unlike DQN agents, which use an experience replay memory, the A3C agent uses multiple workers to gather more samples for learning."),(0,a.kt)("p",null,"In simple terms, the\xa0crux of the A3C\xa0algorithm can be summarized in the following sequence of steps for each iteration:"),(0,a.kt)("p",null,(0,a.kt)("img",{loading:"lazy",alt:"Untitled",src:r(73420).Z,width:"780",height:"621"})),(0,a.kt)("p",null,"The steps repeat again from top to bottom for the next iteration and so on until convergence."))}m.isMDXComponent=!0},73420:function(e,t,r){t.Z=r.p+"assets/images/content-models-raw-mp1-a3c-untitled-bfffd80b22d1fd82daa5ea47780573c2.png"}}]);