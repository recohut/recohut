"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6037],{3905:function(e,t,r){r.d(t,{Zo:function(){return u},kt:function(){return p}});var o=r(67294);function n(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,o)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){n(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,o,n=function(e,t){if(null==e)return{};var r,o,n={},a=Object.keys(e);for(o=0;o<a.length;o++)r=a[o],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(o=0;o<a.length;o++)r=a[o],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var c=o.createContext({}),l=function(e){var t=o.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},u=function(e){var t=l(e.components);return o.createElement(c.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},d=o.forwardRef((function(e,t){var r=e.components,n=e.mdxType,a=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),d=l(r),p=n,f=d["".concat(c,".").concat(p)]||d[p]||m[p]||a;return r?o.createElement(f,i(i({ref:t},u),{},{components:r})):o.createElement(f,i({ref:t},u))}));function p(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var a=r.length,i=new Array(a);i[0]=d;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:n,i[1]=s;for(var l=2;l<a;l++)i[l]=r[l];return o.createElement.apply(null,i)}return o.createElement.apply(null,r)}d.displayName="MDXCreateElement"},30648:function(e,t,r){r.r(t),r.d(t,{assets:function(){return u},contentTitle:function(){return c},default:function(){return p},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return m}});var o=r(87462),n=r(63366),a=(r(67294),r(3905)),i=["components"],s={},c="Shared Bottom",l={unversionedId:"models/shared-bottom",id:"models/shared-bottom",title:"Shared Bottom",description:"The shared-bottom model is the simplest and most common multi-task learning architecture. The model has a single base (the shared bottom) from which all of the task-specific subnetworks begin from. This means that this single representation is used for all tasks, and there is no way for individual tasks to adjust what information they get out of the shared bottom compared to other tasks.",source:"@site/docs/models/shared-bottom.mdx",sourceDirName:"models",slug:"/models/shared-bottom",permalink:"/ai/docs/models/shared-bottom",editUrl:"https://github.com/sparsh-ai/ai/docs/models/shared-bottom.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"SGL",permalink:"/ai/docs/models/sgl"},next:{title:"SiReN",permalink:"/ai/docs/models/siren"}},u={},m=[],d={toc:m};function p(e){var t=e.components,r=(0,n.Z)(e,i);return(0,a.kt)("wrapper",(0,o.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"shared-bottom"},"Shared Bottom"),(0,a.kt)("p",null,"The shared-bottom model is the simplest and most common multi-task learning architecture. The model has a single base (the shared bottom) from which all of the task-specific subnetworks begin from. This means that this single representation is used for all tasks, and there is no way for individual tasks to adjust what information they get out of the shared bottom compared to other tasks."),(0,a.kt)("p",null,(0,a.kt)("center",null,(0,a.kt)("img",{src:"https://github.com/recohut/multiobjective-optimizations/raw/098954ea18e25506b6320039c85d09385c70f37b/docs/_images/L762719_1.png"}))))}p.isMDXComponent=!0}}]);