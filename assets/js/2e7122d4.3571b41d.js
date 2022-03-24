"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5129],{3905:function(e,n,t){t.d(n,{Zo:function(){return l},kt:function(){return f}});var r=t(67294);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,i=function(e,n){if(null==e)return{};var t,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(i[t]=e[t]);return i}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var c=r.createContext({}),p=function(e){var n=r.useContext(c),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},l=function(e){var n=p(e.components);return r.createElement(c.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},d=r.forwardRef((function(e,n){var t=e.components,i=e.mdxType,a=e.originalType,c=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),d=p(t),f=i,h=d["".concat(c,".").concat(f)]||d[f]||u[f]||a;return t?r.createElement(h,o(o({ref:n},l),{},{components:t})):r.createElement(h,o({ref:n},l))}));function f(e,n){var t=arguments,i=n&&n.mdxType;if("string"==typeof e||i){var a=t.length,o=new Array(a);o[0]=d;var s={};for(var c in n)hasOwnProperty.call(n,c)&&(s[c]=n[c]);s.originalType=e,s.mdxType="string"==typeof e?e:i,o[1]=s;for(var p=2;p<a;p++)o[p]=t[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,t)}d.displayName="MDXCreateElement"},32410:function(e,n,t){t.r(n),t.d(n,{assets:function(){return l},contentTitle:function(){return c},default:function(){return f},frontMatter:function(){return s},metadata:function(){return p},toc:function(){return u}});var r=t(87462),i=t(63366),a=(t(67294),t(3905)),o=["components"],s={},c="Jensen\u2013Shannon divergence",p={unversionedId:"concept-extras/jensen-shannon-divergence",id:"concept-extras/jensen-shannon-divergence",title:"Jensen\u2013Shannon divergence",description:"In\xa0probability theory\xa0and\xa0statistics, the\xa0**Jensen)\u2013Shannon\xa0divergence\xa0is a method of measuring the similarity between two\xa0probability distributions. It is also known as\xa0information radius\xa0(IRad)[1]\xa0or\xa0total divergence to the average.[2]\xa0It is based on the\xa0Kullback\u2013Leibler divergence, with some notable (and useful) differences, including that it is symmetric and it always has a finite value. The square root of the Jensen\u2013Shannon divergence is a\xa0metric)\xa0often referred to as Jensen-Shannon distance.",source:"@site/docs/concept-extras/jensen-shannon-divergence.mdx",sourceDirName:"concept-extras",slug:"/concept-extras/jensen-shannon-divergence",permalink:"/ai/docs/concept-extras/jensen-shannon-divergence",editUrl:"https://github.com/sparsh-ai/ai/docs/concept-extras/jensen-shannon-divergence.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Incremental Learning",permalink:"/ai/docs/concept-extras/incremental-learning"},next:{title:"Meta Learning",permalink:"/ai/docs/concept-extras/meta-learning"}},l={},u=[],d={toc:u};function f(e){var n=e.components,t=(0,i.Z)(e,o);return(0,a.kt)("wrapper",(0,r.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"jensenshannon-divergence"},"Jensen\u2013Shannon divergence"),(0,a.kt)("p",null,"In\xa0",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Probability_theory"},"probability theory"),"\xa0and\xa0",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Statistics"},"statistics"),", the\xa0",(0,a.kt)("strong",{parentName:"p"},(0,a.kt)("a",{parentName:"strong",href:"https://en.wikipedia.org/wiki/Johan_Jensen_(mathematician)"},"Jensen"),"\u2013",(0,a.kt)("a",{parentName:"strong",href:"https://en.wikipedia.org/wiki/Claude_Shannon"},"Shannon"),"\xa0divergence"),"\xa0is a method of measuring the similarity between two\xa0",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Probability_distribution"},"probability distributions"),". It is also known as\xa0",(0,a.kt)("strong",{parentName:"p"},"information radius"),"\xa0(",(0,a.kt)("strong",{parentName:"p"},"IRad"),")",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence#cite_note-1"},"[1]"),"\xa0or\xa0",(0,a.kt)("strong",{parentName:"p"},"total divergence to the average"),".",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence#cite_note-2"},"[2]"),"\xa0It is based on the\xa0",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence"},"Kullback\u2013Leibler divergence"),", with some notable (and useful) differences, including that it is symmetric and it always has a finite value. The square root of the Jensen\u2013Shannon divergence is a\xa0",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Metric_(mathematics)"},"metric"),"\xa0often referred to as Jensen-Shannon distance."))}f.isMDXComponent=!0}}]);