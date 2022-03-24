"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2072],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return d}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},h=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,c=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),h=l(n),d=o,m=h["".concat(c,".").concat(d)]||h[d]||u[d]||i;return n?r.createElement(m,a(a({ref:t},p),{},{components:n})):r.createElement(m,a({ref:t},p))}));function d(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=h;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var l=2;l<i;l++)a[l]=n[l];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}h.displayName="MDXCreateElement"},11232:function(e,t,n){n.r(t),n.d(t,{assets:function(){return p},contentTitle:function(){return c},default:function(){return d},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return u}});var r=n(87462),o=n(63366),i=(n(67294),n(3905)),a=["components"],s={},c="PPO",l={unversionedId:"models/ppo",id:"models/ppo",title:"PPO",description:"The PPO (Proximal Policy Optimization) algorithm was\xa0introduced by the OpenAI team in 2017\xa0and quickly became one of the most popular Reinforcement Learning method that pushed all other RL methods at that moment aside. PPO involves collecting a small batch of experiences interacting with the environment and using that batch to update its decision-making policy. Once the policy is updated with that batch, the experiences are thrown away and a newer batch is collected with the newly updated policy. This is the reason why it is an \u201con-policy learning\u201d approach where the experience samples collected are only useful for updating the current policy.",source:"@site/docs/models/ppo.md",sourceDirName:"models",slug:"/models/ppo",permalink:"/ai/docs/models/ppo",editUrl:"https://github.com/sparsh-ai/ai/docs/models/ppo.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"PNN",permalink:"/ai/docs/models/pnn"},next:{title:"Q-learning",permalink:"/ai/docs/models/q-learning"}},p={},u=[],h={toc:u};function d(e){var t=e.components,n=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,r.Z)({},h,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"ppo"},"PPO"),(0,i.kt)("p",null,"The PPO (Proximal Policy Optimization) algorithm was\xa0",(0,i.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/1707.06347.pdf"},"introduced by the OpenAI team in 2017"),"\xa0and quickly became one of the most popular Reinforcement Learning method that pushed all other RL methods at that moment aside. PPO involves collecting a small batch of experiences interacting with the environment and using that batch to update its decision-making policy. Once the policy is updated with that batch, the experiences are thrown away and a newer batch is collected with the newly updated policy. This is the reason why it is an \u201con-policy learning\u201d approach where the experience samples collected are only useful for updating the current policy."),(0,i.kt)("p",null,"The PPO Agent uses convolutional neural network layers to process the high-dimensional visual inputs in the Actor and Critic classes. The PPO algorithm updates the Agent's policy parameters using a surrogate loss function that prevents the policy parameters from being drastically updated. It then keeps the policy updates within the trust region, which makes it robust to hyperparameter choices and a few other factors that may lead to instability during the Agent's training regime."),(0,i.kt)("p",null,"The main idea is that after an update, the new policy should be not too far from the old policy. For that, PPO uses clipping to avoid too large updates. This leads to less variance in training at the cost of some bias, but ensures smoother training and also makes sure the agent does not go down to an unrecoverable path of taking senseless actions."),(0,i.kt)("p",null,"Why is the goal of the agent to maximize the expected cumulative reward? Well, Reinforcement Learning is based on the idea of the reward hypothesis. All goals can be described by the maximization of the expected cumulative reward. ",(0,i.kt)("strong",{parentName:"p"},"That\u2019s why in Reinforcement Learning, to have the best behavior, we need to maximize the expected cumulative reward.")),(0,i.kt)("p",null,"The\xa0",(0,i.kt)("strong",{parentName:"p"},"Proximal Policy Optimization"),"\xa0(",(0,i.kt)("strong",{parentName:"p"},"PPO"),") algorithm builds\xa0upon the work\xa0of\xa0",(0,i.kt)("strong",{parentName:"p"},"Trust Region Policy Optimization"),"\xa0(",(0,i.kt)("strong",{parentName:"p"},"TRPO"),") to constrain the new policy to be within a trust region from the old policy. PPO simplifies the implementation of this core idea by using a clipped surrogate objective function that is easier to implement, yet quite powerful and efficient. It is one of the most widely used RL algorithms, especially for continuous control problems."))}d.isMDXComponent=!0}}]);