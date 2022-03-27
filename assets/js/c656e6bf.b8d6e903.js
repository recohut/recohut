"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4449],{3905:function(e,t,n){n.d(t,{Zo:function(){return l},kt:function(){return d}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function a(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var p=r.createContext({}),c=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},l=function(e){var t=c(e.components);return r.createElement(p.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,p=e.parentName,l=a(e,["components","mdxType","originalType","parentName"]),m=c(n),d=o,b=m["".concat(p,".").concat(d)]||m[d]||u[d]||i;return n?r.createElement(b,s(s({ref:t},l),{},{components:n})):r.createElement(b,s({ref:t},l))}));function d(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,s=new Array(i);s[0]=m;var a={};for(var p in t)hasOwnProperty.call(t,p)&&(a[p]=t[p]);a.originalType=e,a.mdxType="string"==typeof e?e:o,s[1]=a;for(var c=2;c<i;c++)s[c]=n[c];return r.createElement.apply(null,s)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},73738:function(e,t,n){n.r(t),n.d(t,{assets:function(){return l},contentTitle:function(){return p},default:function(){return d},frontMatter:function(){return a},metadata:function(){return c},toc:function(){return u}});var r=n(87462),o=n(63366),i=(n(67294),n(3905)),s=["components"],a={},p="Python code",c={unversionedId:"snippets/python-snippets",id:"snippets/python-snippets",title:"Python code",description:"Clean filenames in a folder",source:"@site/docs/snippets/python-snippets.mdx",sourceDirName:"snippets",slug:"/snippets/python-snippets",permalink:"/recohut/docs/snippets/python-snippets",editUrl:"https://github.com/sparsh-ai/recohut/docs/snippets/python-snippets.mdx",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Word2vec",permalink:"/recohut/docs/tutorials/word2vec"},next:{title:"Unix shell",permalink:"/recohut/docs/snippets/unix-shell-snippets"}},l={},u=[{value:"Clean filenames in a folder",id:"clean-filenames-in-a-folder",level:2},{value:"Converting Jupyter notebooks into markdown",id:"converting-jupyter-notebooks-into-markdown",level:2},{value:"Scraping",id:"scraping",level:2},{value:"BS4",id:"bs4",level:3}],m={toc:u};function d(e){var t=e.components,n=(0,o.Z)(e,s);return(0,i.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"python-code"},"Python code"),(0,i.kt)("h2",{id:"clean-filenames-in-a-folder"},"Clean filenames in a folder"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"import re\nimport os\nimport pathlib\nimport glob\n\nnbs = glob.glob('/content/docs/docs/tutorials/*.md')\n\ndef process(path):\n    x = str(pathlib.Path(path).stem)\n    x = x.lower()\n    x = re.sub(r'[^a-z]','-',x)\n    x = re.sub(r'-+','-',x)\n    x = x.strip('-')\n    x = os.path.join(str(pathlib.Path(path).parent), x+'.mdx')\n    x = re.sub('/[a-z]\\-','/',x)\n    os.rename(path, x)\n\n_ = [process(x) for x in nbs]\n")),(0,i.kt)("h2",{id:"converting-jupyter-notebooks-into-markdown"},"Converting Jupyter notebooks into markdown"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"!cd /content && git clone https://github.com/recohut/nbs.git\n!pip install -q jupytext\n!cd /content/nbs && jupytext *.ipynb --to markdown\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"import glob\nimport os\n\nnbs = glob.glob('/content/nbs/*.ipynb')\n\nfor x in nbs:\n    mds = x[:-6]+'.md'\n    if not os.path.exists(mds):\n        try:\n          !jupyter nbconvert --to markdown \"{x}\"\n        except:\n            print('error in {}'.format(x))\n")),(0,i.kt)("h2",{id:"scraping"},"Scraping"),(0,i.kt)("h3",{id:"bs4"},"BS4"),(0,i.kt)("a",{href:"https://nbviewer.org/github/recohut/nbs/blob/main/2020-06-20-simple-scraping-nlp-bs4-distilbert.ipynb",alt:""}," ",(0,i.kt)("img",{src:"https://colab.research.google.com/assets/colab-badge.svg"})),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'import bs4\nimport requests\nimport lxml.etree as xml\n\nURLs = ["https://www.flexjobs.com/blog/post/job-search-strategies-for-success-v2/",\n        "https://www.monster.com/career-advice/article/five-ps-of-job-search-progress"]\n\ni = 0\nweb_page = bs4.BeautifulSoup(requests.get(URLs[i], {}).text, "lxml")\ndf.loc[i,\'title\'] = web_page.head.title.text\nsub_web_page = web_page.find_all(name="article", attrs={"class": "single-post-page"})[0]\narticle = \'. \'.join([wp.text for wp in sub_web_page.find_all({"h2","p"})])\ndf.loc[i,\'text\'] = article\n        \n')))}d.isMDXComponent=!0}}]);