"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3582],{3905:function(t,e,n){n.d(e,{Zo:function(){return p},kt:function(){return f}});var r=n(67294);function o(t,e,n){return e in t?Object.defineProperty(t,e,{value:n,enumerable:!0,configurable:!0,writable:!0}):t[e]=n,t}function a(t,e){var n=Object.keys(t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(t);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),n.push.apply(n,r)}return n}function i(t){for(var e=1;e<arguments.length;e++){var n=null!=arguments[e]?arguments[e]:{};e%2?a(Object(n),!0).forEach((function(e){o(t,e,n[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(n,e))}))}return t}function s(t,e){if(null==t)return{};var n,r,o=function(t,e){if(null==t)return{};var n,r,o={},a=Object.keys(t);for(r=0;r<a.length;r++)n=a[r],e.indexOf(n)>=0||(o[n]=t[n]);return o}(t,e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(t);for(r=0;r<a.length;r++)n=a[r],e.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(t,n)&&(o[n]=t[n])}return o}var c=r.createContext({}),l=function(t){var e=r.useContext(c),n=e;return t&&(n="function"==typeof t?t(e):i(i({},e),t)),n},p=function(t){var e=l(t.components);return r.createElement(c.Provider,{value:e},t.children)},m={inlineCode:"code",wrapper:function(t){var e=t.children;return r.createElement(r.Fragment,{},e)}},u=r.forwardRef((function(t,e){var n=t.components,o=t.mdxType,a=t.originalType,c=t.parentName,p=s(t,["components","mdxType","originalType","parentName"]),u=l(n),f=o,h=u["".concat(c,".").concat(f)]||u[f]||m[f]||a;return n?r.createElement(h,i(i({ref:e},p),{},{components:n})):r.createElement(h,i({ref:e},p))}));function f(t,e){var n=arguments,o=e&&e.mdxType;if("string"==typeof t||o){var a=n.length,i=new Array(a);i[0]=u;var s={};for(var c in e)hasOwnProperty.call(e,c)&&(s[c]=e[c]);s.originalType=t,s.mdxType="string"==typeof t?t:o,i[1]=s;for(var l=2;l<a;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},5080:function(t,e,n){n.r(e),n.d(e,{assets:function(){return p},contentTitle:function(){return c},default:function(){return f},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return m}});var r=n(87462),o=n(63366),a=(n(67294),n(3905)),i=["components"],s={},c="Markov Chains",l={unversionedId:"models/markov-chains",id:"models/markov-chains",title:"Markov Chains",description:'Markov chains, named after Andrey Markov, are mathematical systems that hop from one "state" (a situation or set of values) to another. For example, if you made a Markov chain model of a baby\'s behavior, you might include "playing," "eating", "sleeping," and "crying" as states, which together with other behaviors could form a \'state space\': a list of all possible states. In addition, on top of the state space, a Markov chain tells you the probability of hopping, or "transitioning," from one state to any other state---e.g., the chance that a baby currently playing will fall asleep in the next five minutes without crying first.',source:"@site/docs/models/markov-chains.md",sourceDirName:"models",slug:"/models/markov-chains",permalink:"/recohut/docs/models/markov-chains",editUrl:"https://github.com/sparsh-ai/recohut/docs/models/markov-chains.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"LIRD",permalink:"/recohut/docs/models/lird"},next:{title:"MATN",permalink:"/recohut/docs/models/matn"}},p={},m=[{value:"Implementation",id:"implementation",level:2},{value:"PyTorch",id:"pytorch",level:3},{value:"Links",id:"links",level:2}],u={toc:m};function f(t){var e=t.components,n=(0,o.Z)(t,i);return(0,a.kt)("wrapper",(0,r.Z)({},u,n,{components:e,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"markov-chains"},"Markov Chains"),(0,a.kt)("p",null,'Markov chains, named after Andrey Markov, are mathematical systems that hop from one "state" (a situation or set of values) to another. For example, if you made a Markov chain model of a baby\'s behavior, you might include "playing," "eating", "sleeping," and "crying" as states, which together with other behaviors could form a \'state space\': a list of all possible states. In addition, on top of the state space, a Markov chain tells you the probability of hopping, or "transitioning," from one state to any other state---e.g., the chance that a baby currently playing will fall asleep in the next five minutes without crying first.'),(0,a.kt)("h2",{id:"implementation"},"Implementation"),(0,a.kt)("h3",{id:"pytorch"},"PyTorch"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import torch\n\nT = torch.tensor([[0.4, 0.6],\n                  [0.8, 0.2]])\n\nT_2 = torch.matrix_power(T, 2)\n\nT_5 = torch.matrix_power(T, 5)\n\nT_10 = torch.matrix_power(T, 10)\n\nT_15 = torch.matrix_power(T, 15)\n\nT_20 = torch.matrix_power(T, 20)\n\nprint("Transition probability after 2 steps:\\n{}".format(T_2))\nprint("Transition probability after 5 steps:\\n{}".format(T_5))\nprint("Transition probability after 10 steps:\\n{}".format(T_10))\nprint("Transition probability after 15 steps:\\n{}".format(T_15))\nprint("Transition probability after 20 steps:\\n{}".format(T_20))\n\nv = torch.tensor([[0.7, 0.3]])\n\nv_1 = torch.mm(v, T)\nv_2 = torch.mm(v, T_2)\nv_5 = torch.mm(v, T_5)\nv_10 = torch.mm(v, T_10)\nv_15 = torch.mm(v, T_15)\nv_20 = torch.mm(v, T_20)\n\nprint("Distribution of states after 1 step:\\n{}".format(v_1))\nprint("Distribution of states after 2 steps:\\n{}".format(v_2))\nprint("Distribution of states after 5 steps:\\n{}".format(v_5))\nprint("Distribution of states after 10 steps:\\n{}".format(v_10))\nprint("Distribution of states after 15 steps:\\n{}".format(v_15))\nprint("Distribution of states after 20 steps:\\n{}".format(v_20))\n')),(0,a.kt)("h2",{id:"links"},"Links"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("a",{parentName:"li",href:"https://setosa.io/ev/markov-chains/"},"https://setosa.io/ev/markov-chains"))))}f.isMDXComponent=!0}}]);