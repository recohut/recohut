"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2739],{3905:function(e,t,n){n.d(t,{Zo:function(){return d},kt:function(){return m}});var a=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,a,o=function(e,t){if(null==e)return{};var n,a,o={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=a.createContext({}),l=function(e){var t=a.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},d=function(e){var t=l(e.components);return a.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,c=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),u=l(n),m=o,h=u["".concat(c,".").concat(m)]||u[m]||p[m]||i;return n?a.createElement(h,r(r({ref:t},d),{},{components:n})):a.createElement(h,r({ref:t},d))}));function m(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,r=new Array(i);r[0]=u;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,r[1]=s;for(var l=2;l<i;l++)r[l]=n[l];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},63805:function(e,t,n){n.r(t),n.d(t,{assets:function(){return d},contentTitle:function(){return c},default:function(){return m},frontMatter:function(){return s},metadata:function(){return l},toc:function(){return p}});var a=n(87462),o=n(63366),i=(n(67294),n(3905)),r=["components"],s={title:"Object Detection Hands-on Exercises",authors:"sparsh",tags:["object detection","vision"]},c=void 0,l={permalink:"/recohut/blog/2021/10/01/object-detection-hands-on-exercises",editUrl:"https://github.com/sparsh-ai/recohut/blog/blog/2021-10-01-object-detection-hands-on-exercises.mdx",source:"@site/blog/2021-10-01-object-detection-hands-on-exercises.mdx",title:"Object Detection Hands-on Exercises",description:"We are going to discuss the following 4 use cases:",date:"2021-10-01T00:00:00.000Z",formattedDate:"October 1, 2021",tags:[{label:"object detection",permalink:"/recohut/blog/tags/object-detection"},{label:"vision",permalink:"/recohut/blog/tags/vision"}],readingTime:3.165,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],frontMatter:{title:"Object Detection Hands-on Exercises",authors:"sparsh",tags:["object detection","vision"]},prevItem:{title:"Name & Address Parsing",permalink:"/recohut/blog/2021/10/01/name-&-address-parsing"},nextItem:{title:"Object detection with OpenCV",permalink:"/recohut/blog/2021/10/01/object-detection-with-opencv"}},d={authorsImageUrls:[void 0]},p=[{value:"Use Case 1 -  <strong>Object detection with OpenCV</strong>",id:"use-case-1----object-detection-with-opencv",level:3},{value:"Use Case 2 - MobileNet SSD Caffe Pre-trained model",id:"use-case-2---mobilenet-ssd-caffe-pre-trained-model",level:3},{value:"Use Case 3 - YOLO Object Detection App",id:"use-case-3---yolo-object-detection-app",level:3},{value:"Use Case 4 - TFHub MobileNet SSD on Videos",id:"use-case-4---tfhub-mobilenet-ssd-on-videos",level:3}],u={toc:p};function m(e){var t=e.components,s=(0,o.Z)(e,r);return(0,i.kt)("wrapper",(0,a.Z)({},u,s,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,"We are going to discuss the following 4 use cases:"),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},"Detect faces, eyes, pedestrians, cars, and number plates using OpenCV haar cascade classifiers"),(0,i.kt)("li",{parentName:"ol"},"Streamlit app for MobileNet SSD Caffe Pre-trained model"),(0,i.kt)("li",{parentName:"ol"},"Streamlit app for various object detection models and use cases"),(0,i.kt)("li",{parentName:"ol"},"Detect COCO-80 class objects in videos using TFHub MobileNet SSD model")),(0,i.kt)("h3",{id:"use-case-1----object-detection-with-opencv"},"Use Case 1 -  ",(0,i.kt)("strong",{parentName:"h3"},"Object detection with OpenCV")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Face detection")," - We will use the frontal face Haar cascade classifier model to detect faces in the given image. The following function first passes the given image into the classifier model to detect a list of face bounding boxes and then runs a loop to draw a red rectangle box around each detected face in the image:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"def detect_faces(fix_img):\n    face_rects = face_classifier.detectMultiScale(fix_img)\n    for (x, y, w, h) in face_rects:\n        cv2.rectangle(fix_img,\n                     (x,y),\n                     (x+w, y+h),\n                     (255,0,0),\n                     10)\n    return fix_img\n")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Eyes detection")," - The process is almost similar to the face detection process. Instead of frontal face Haar cascade, we will use the eye detection Haar cascade model."),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"Input image",src:n(95769).Z,width:"381",height:"223"})),(0,i.kt)("p",null,"Input image"),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"detected faces and eyes in the image",src:n(50634).Z,width:"381",height:"223"})),(0,i.kt)("p",null,"detected faces and eyes in the image"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Pedestrian detection")," - We will use the full-body Haar cascade classifier model for pedestrian detection. We will apply this model to a video this time. The following function will run the model on each frame of the video to detect the pedestrians:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"# While Loop\nwhile cap.isOpened():\n    # Read the capture\n        ret, frame = cap.read()\n    # Pass the Frame to the Classifier\n        bodies = body_classifier.detectMultiScale(frame, 1.2, 3)\n    # if Statement\n        if ret == True:\n        # Bound Boxes to Identified Bodies\n                for (x,y,w,h) in bodies:\n            cv2.rectangle(frame,\n                         (x,y),\n                         (x+w, y+h),\n                         (25,125,225),\n                         5)\n            cv2.imshow('Pedestrians', frame) \n        # Exit with Esc button\n                if cv2.waitKey(1) == 27:\n            break  \n    # else Statement\n        else:\n        break\n    \n# Release the Capture & Destroy All Windows\ncap.release()\ncv2.destroyAllWindows()\n")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Car detection")," - The process is almost similar to the pedestrian detection process. Again, we will use this model on a video. Instead of people Haar cascade, we will use the car cascade model."),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Car number plate detection")," - The process is almost similar to the face and eye detection process. We will use the car number plate cascade model."),(0,i.kt)("p",null,(0,i.kt)("em",{parentName:"p"},"You can find the code ",(0,i.kt)("a",{parentName:"em",href:"https://github.com/sparsh-ai/0D7ACA15"},"here")," on Github.")),(0,i.kt)("h3",{id:"use-case-2---mobilenet-ssd-caffe-pre-trained-model"},"Use Case 2 - MobileNet SSD Caffe Pre-trained model"),(0,i.kt)("p",null,(0,i.kt)("em",{parentName:"p"},"You can play with the live app ",(0,i.kt)("a",{parentName:"em",href:"https://share.streamlit.io/sparsh-ai/streamlit-5a407279/app.py"},"here"),". Souce code is available")," ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/sparsh-ai/streamlit-489fbbb7"},"here")," ",(0,i.kt)("em",{parentName:"p"},"on Github.")),(0,i.kt)("h3",{id:"use-case-3---yolo-object-detection-app"},"Use Case 3 - YOLO Object Detection App"),(0,i.kt)("p",null,(0,i.kt)("em",{parentName:"p"},"You can play with the live app")," ",(0,i.kt)("a",{parentName:"p",href:"https://share.streamlit.io/sparsh-ai/streamlit-489fbbb7/app.py"},"*here"),". Source code is available ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/sparsh-ai/streamlit-5a407279/tree/master"},"here")," on Github.*"),(0,i.kt)("p",null,"This app can detect COCO 80-classes using three different models - Caffe MobileNet SSD, Yolo3-tiny, and Yolo3. It can also detect faces using two different models - SSD Res10 and OpenCV face detector.  Yolo3-tiny can also detect fires."),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-object-detection-with-yolo3-untitled.png",src:n(29775).Z,width:"1366",height:"645"})),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-object-detection-with-yolo3-untitled-1.png",src:n(99510).Z,width:"1366",height:"640"})),(0,i.kt)("h3",{id:"use-case-4---tfhub-mobilenet-ssd-on-videos"},"Use Case 4 - TFHub MobileNet SSD on Videos"),(0,i.kt)("p",null,"In this section, we will use the MobileNet SSD object detection model from TFHub. We will apply it to videos. We can load the model using the following command:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"module_handle = \"https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1\"\ndetector = hub.load(module_handle).signatures['default']\n")),(0,i.kt)("p",null,"After loading the model, we will capture frames using OpenCV video capture method, and pass each frame through the detection model:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"cap = cv2.VideoCapture('/content/Spectre_opening_highest_for_a_James_Bond_film_in_India.mp4')\nfor i in range(1,total_frames,200):\n    cap.set(cv2.CAP_PROP_POS_FRAMES,i)\n    ret,frame = cap.read()\n    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n    run_detector(detector,frame)\n")),(0,i.kt)("p",null,"Here are some detected objects in frames: "),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-object-detection-hands-on-exercises-untitled.png",src:n(66532).Z,width:"1156",height:"662"})),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-object-detection-hands-on-exercises-untitled-1.png",src:n(45389).Z,width:"1156",height:"662"})),(0,i.kt)("p",null,(0,i.kt)("img",{loading:"lazy",alt:"/img/content-blog-raw-blog-object-detection-hands-on-exercises-untitled-2.png",src:n(6360).Z,width:"1156",height:"662"})),(0,i.kt)("p",null,(0,i.kt)("em",{parentName:"p"},"You can find the code ",(0,i.kt)("a",{parentName:"em",href:"https://gist.github.com/sparsh-ai/32ff6fe8c073f6be5d893029e4dc2960"},"here")," on Github.")),(0,i.kt)("hr",null),(0,i.kt)("p",null,(0,i.kt)("em",{parentName:"p"},"Congrats! In the next post of this series, we will cover 5 exciting use cases - 1) detectron 2 object detection fine-tuning on custom class, 2) Tensorflow Object detection API inference, fine-tuning, and few-shot learning, 3) Inference with 6 pre-trained models, 4) Mask R-CNN object detection app, and 5) Logo detection app deployment as a Rest API using AWS elastic Beanstalk.")))}m.isMDXComponent=!0},45389:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-hands-on-exercises-untitled-1-0a9f162ff084467ebd27ea587d055584.png"},6360:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-hands-on-exercises-untitled-2-24de87f6b567ef8d72bccc20bfc2328f.png"},66532:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-hands-on-exercises-untitled-7bd46bee77946cff46d6a834699b3685.png"},50634:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-with-opencv-untitled-1-d023975b118cde980a874199f93b9035.png"},95769:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-with-opencv-untitled-44fc11078a43a07dc3e4ec7f84197d25.png"},99510:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-with-yolo3-untitled-1-c0c4b05a1cf3256f1a9d0ebdb5f4bfb5.png"},29775:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-with-yolo3-untitled-7c24204b83a8bd10c57c53b4b2423899.png"}}]);