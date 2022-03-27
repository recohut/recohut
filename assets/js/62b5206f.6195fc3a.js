"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[921],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return m}});var r=n(67294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=r.createContext({}),l=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,s=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),u=l(n),m=i,h=u["".concat(s,".").concat(m)]||u[m]||d[m]||o;return n?r.createElement(h,a(a({ref:t},p),{},{components:n})):r.createElement(h,a({ref:t},p))}));function m(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,a=new Array(o);a[0]=u;var c={};for(var s in t)hasOwnProperty.call(t,s)&&(c[s]=t[s]);c.originalType=e,c.mdxType="string"==typeof e?e:i,a[1]=c;for(var l=2;l<o;l++)a[l]=n[l];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},31797:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return s},default:function(){return m},frontMatter:function(){return c},metadata:function(){return l},toc:function(){return d}});var r=n(87462),i=n(63366),o=(n(67294),n(3905)),a=["components"],c={},s="DDPG",l={unversionedId:"models/ddpg",id:"models/ddpg",title:"DDPG",description:"Deterministic Policy Gradient (DPG)\xa0is a type\xa0of Actor-Critic RL algorithm that uses two neural networks: one for estimating the action value function, and the other for estimating the optimal target policy. The\xa0Deep Deterministic Policy Gradient\xa0(DDPG) agent\xa0builds upon the idea of DPG and is quite efficient compared to vanilla Actor-Critic agents due\xa0to the use\xa0of deterministic action policies.",source:"@site/docs/models/ddpg.md",sourceDirName:"models",slug:"/models/ddpg",permalink:"/recohut/docs/models/ddpg",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/ddpg.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"DCN",permalink:"/recohut/docs/models/dcn"},next:{title:"DeepCross",permalink:"/recohut/docs/models/deepcross"}},p={},d=[{value:"Algorithm",id:"algorithm",level:2},{value:"Links",id:"links",level:2}],u={toc:d};function m(e){var t=e.components,c=(0,i.Z)(e,a);return(0,o.kt)("wrapper",(0,r.Z)({},u,c,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"ddpg"},"DDPG"),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Deterministic Policy Gradient (DPG)"),"\xa0is a type\xa0of Actor-Critic RL algorithm that uses two neural networks: one for estimating the action value function, and the other for estimating the optimal target policy. The\xa0",(0,o.kt)("strong",{parentName:"p"},"Deep Deterministic Policy Gradient"),"\xa0(",(0,o.kt)("strong",{parentName:"p"},"DDPG"),") agent\xa0builds upon the idea of DPG and is quite efficient compared to vanilla Actor-Critic agents due\xa0to the use\xa0of deterministic action policies."),(0,o.kt)("p",null,"DDPG, or Deep Deterministic Policy Gradient, is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. It combines the actor-critic approach with insights from DQNs: in particular, the insights that 1) the network is trained off-policy with samples from a replay buffer to minimize correlations between samples, and 2) the network is trained with a target Q network to give consistent targets during temporal difference backups. DDPG makes use of the same ideas along with batch normalization."),(0,o.kt)("p",null,"It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces."),(0,o.kt)("h2",{id:"algorithm"},"Algorithm"),(0,o.kt)("p",null,(0,o.kt)("img",{loading:"lazy",alt:"Untitled",src:n(5921).Z,width:"624",height:"643"})),(0,o.kt)("p",null,"As far as the recommended scenario is concerned, discrete actions are a more natural idea, and each action corresponds to each item.\xa0However, in reality, the number of items may be at least one million, which means that the action space is large and the calculation complexity with softmax is very high.\xa0For continuous actions, DDPG is a more general choice. For more details, articles by Jingdong\xa0",(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1801.00209"},"[1]"),"\xa0",(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/1805.02343.pdf"},"[2]"),", Ali\xa0",(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/1803.00710.pdf"},"[1]"),"\xa0, Huawei\xa0",(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/1810.12027.pdf"},"[1]"),"\xa0can be referenced."),(0,o.kt)("p",null,"Then the core of the algorithm is to optimize these two objective functions through gradient ascent (descent) to obtain the final parameters, and then to obtain the optimal strategy.\xa0Some other implementation details of DDPG such as target network, soft update, etc. will not be repeated here. Since we are using a fixed data set, we only need to convert the data into a format that the DDPG algorithm can input, and then batch training like supervised learning."),(0,o.kt)("h2",{id:"links"},"Links"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://spinningup.openai.com/en/latest/algorithms/ddpg.html"},"https://spinningup.openai.com/en/latest/algorithms/ddpg.html"))))}m.isMDXComponent=!0},5921:function(e,t,n){t.Z=n.p+"assets/images/content-models-raw-mp2-ddpg-untitled-0023c77cd9b0ec3788e411fe21776b98.png"}}]);