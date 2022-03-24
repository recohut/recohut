---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="t63evfSI9cW9" -->
# Image Classification with scikit-learn (HOG + Logistic Regression)
<!-- #endregion -->

```python id="wz_FrPlr9cW-" outputId="b95bfce5-8d61-473c-f4b5-98cf2016cc10"
# http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# https://www.kaggle.com/manikg/training-svm-classifier-with-hog-features
%matplotlib inline
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from glob import glob
from matplotlib import pyplot as plt

images, hog_images = [], []
X, y = [], []
ppc = 16
sz = 200
for dir in glob('images/Caltech101_images/*'):
    image_files = glob(dir + '/*.jpg')
    label = dir.split('\\')[-1]
    print(label, len(image_files))
    for image_file in image_files:
        image = resize(imread(image_file), (sz,sz))
        if len(image.shape) == 2: # if a gray-scale image
            image = gray2rgb(image)
        fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True, multichannel=True)
        images.append(image)
        hog_images.append(hog_image)
        X.append(fd)
        y.append(label)
```

```python id="ZI-fPDvn9cXg" outputId="1173ad09-7457-4e52-dd3c-2228d6c1959a"
print(len(images), hog_images[0].shape, X[0].shape, X[1].shape, len(y))
```

```python id="qJEcLOXr9cXo" outputId="1f18a126-ad59-4f63-cf1a-2a26ea638680"
n = 6
indices = np.random.choice(len(images), n*n)
plt.figure(figsize=(20,20))
plt.gray()
i = 1
for index in indices:
    plt.subplot(n,n,i), plt.imshow(images[index]), plt.axis('off'), plt.title(y[index], size=20)
    i += 1
plt.show()
plt.figure(figsize=(20,20))
i = 1
for index in indices:
    plt.subplot(n,n,i), plt.imshow(hog_images[index]), plt.axis('off'), plt.title(y[index], size=20)
    i += 1
plt.show()
```

```python id="QNn-L0MM9cXu"
X = np.array(X)
y = np.array(y)
indices = np.arange(len(X))
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, indices, test_size=0.1, random_state=1)
```

```python id="5fIG_gdM9cX1" outputId="15e99003-104c-4774-a34a-6c03e2218096"
#clf = svm.LinearSVC(C=10)
clf = LogisticRegression(C=1000, random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)
```

```python id="NqKziz-L9cX9" outputId="08f164f2-bc52-42be-f633-a998aab901b9"
print(X.shape, y.shape)
```

```python id="9P4ruEsL9cYD" outputId="0259d4b3-b045-4bfa-d60f-74402993496d"
y_pred = clf.predict(X_train)
print("Accuracy: " + str(accuracy_score(y_train, y_pred)))
print('\n')
print(classification_report(y_train, y_pred))
```

```python id="mWXQvXv49cYL" outputId="6ef386e6-4b3b-4c1b-aa70-ba7da7120f61"
y_pred = clf.predict(X_test)
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))
```

```python id="RtbKMA5Q9cYb" outputId="a1e5efec-adf7-4c02-ee07-0d0bbbf06f6d"
plt.figure(figsize=(20,20))
j = 0
for i in id_test:
    plt.subplot(10,10,j+1), plt.imshow(images[i]), plt.axis('off'), plt.title('{}/{}'.format(y_test[j], y_pred[j]))
    j += 1
plt.suptitle('Actual vs. Predicted Class Labels', size=20)
plt.show()
```

<!-- #region id="8iiOKzMQ9cYg" -->
### Image Classification with VGG-19 / Inception V3 / MobileNet / ResNet101 (with deep learning, pytorch)
<!-- #endregion -->

```python id="AHzUccSN9cYh" outputId="90a35ec9-5a9f-4faa-f5c4-3256f8bf7bc9"
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pylab as plt

def classify(img, model_index, model_name, model_pred, labels):
    #print(model_name, model_pred.shape)
    _, index = torch.max(model_pred, 1)
    model_pred, indices = torch.sort(model_pred, dim=1, descending=True)
    percentage = torch.nn.functional.softmax(model_pred, dim=1)[0] * 100
    print(labels[index[0]], percentage[0].item())
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'arial.ttf', 50)
    draw.text((5, 5+model_index*50),'{}, pred: {},{}%'.format(model_name, labels[index[0]], round(percentage[0].item(),2)),(255,0,0),font=font)
    return indices, percentage

    
print(dir(models))

with open('models/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


transform = transforms.Compose([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])

for imgfile in ["images/cheetah.png", "images/swan.png"]:
    
    img = Image.open(imgfile).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    vgg19 = models.vgg19(pretrained=True)
    vgg19.eval()
    pred = vgg19(batch_t)
    classify(img, 0, 'vgg19', pred, labels)

    mobilenetv2 = models.mobilenet_v2(pretrained=True)
    mobilenetv2.eval()
    pred = mobilenetv2(batch_t)
    classify(img, 1, 'mobilenetv2', pred, labels)

    inceptionv3 = models.inception_v3(pretrained=True)
    inceptionv3.eval()
    pred = inceptionv3(batch_t)
    classify(img, 2, 'inceptionv3', pred, labels)

    resnet101 = models.resnet101(pretrained=True)
    resnet101.eval()
    pred = resnet101(batch_t)
    indices, percentages = classify(img, 3, 'resnet101', pred, labels)
    
    plt.figure(figsize=(20,10))
    plt.subplot(121), plt.imshow(img), plt.axis('off'), plt.title('image classified with pytorch', size=20)
    plt.subplot(122), plt.bar(range(5), percentages.detach().numpy()[:5], align='center', alpha=0.5)
    #print(indices[0].detach().numpy()[:5])
    plt.xticks(range(5),  np.array(labels)[indices.detach().numpy().astype(int)[0][:5]])
    plt.xlabel('predicted labels', size=20), plt.ylabel('predicted percentage', size=20)
    plt.title('Resnet top 5 classes predicted', size=20)
    plt.show()
```

<!-- #region id="Q3c6D5of9cYo" -->
### Traffic Signal Classification with deep learning
<!-- #endregion -->

```python id="hB_dNzxe9cYp"
#!mkdir traffic_signs
import os, glob
from shutil import copy
import pandas as pd

image_dir = 'GTSRB/Final_Training/Images/'
dest_dir = 'traffic_signs'
df = pd.DataFrame()
for d in sorted(os.listdir(image_dir)):
    #print(d)
    images = sorted(glob.glob(os.path.join(image_dir, d, '*.ppm')))
    for img in images:
        copy(img, dest_dir)
    for csv in sorted(glob.glob(os.path.join(image_dir, d, '*.csv'))):
        df1 = pd.read_csv(csv, sep=';')
        df = df.append(df1)
        #print(df.head())
        print(d, len(images), df1.shape)
df.to_csv(os.path.join(dest_dir, 'labels.csv'))
```

```python id="gm9h2tBC9cYv" outputId="f1300632-f80f-418c-8d3c-e912e32559c5"
df.head()
```

```python id="IUFtOEbb9cY2" outputId="0a1e3fad-61ae-494f-ac40-a08206e6cbf1"
df.shape
```

```python id="ECUoVX6R9cY_" outputId="2af6fe70-936a-4387-ab9b-7863d2cbce24"
len(glob.glob(os.path.join(dest_dir, '*.ppm')))
```

```python id="YGrAUapk9cZE" outputId="75434f57-e811-4261-95bd-ed3e1939c1a0"
import pandas as  pd
signal_names =  pd.read_csv('images/signal_names.csv')
signal_names.head()
```

```python id="Q_OcfkzO9cZK" outputId="46eacee3-a307-4310-89ce-6bb0617b3426"
%matplotlib inline
import pickle
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

training_file = "traffic_signs/train.p"
validation_file = "traffic_signs/valid.p"
testing_file = "traffic_signs/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
n_signs = len(np.unique(y_train))

print(X_train.shape, X_valid.shape, X_test.shape, n_signs)

plt.figure(figsize=(12,8))
# plot barh chart with index as x values
ax = sns.barplot(list(range(n_signs)), np.bincount(y_train))
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 20))
for c in range(n_signs):
    i = np.random.choice(np.where(y_train == c)[0])
    plt.subplot(8, 6, c+1)
    plt.axis('off')
    plt.title(signal_names.loc[signal_names['ClassId'] == c].SignName.to_string(index=False))
    plt.imshow(X_train[i])
```

```python id="xbpRE1Tu9cZQ"
import cv2 
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
from livelossplot import PlotLosses
import torch.nn.functional as F
import os

class TraffficNet(nn.Module):
    def __init__(self, gray=False):
        super(TraffficNet, self).__init__()
        input_chan = 1 if gray else 3
        self.conv1 = nn.Conv2d(input_chan, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ClaheTranform:
    def __init__(self, clipLimit=2.5, tileGridSize=(4, 4)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
        return img_output

class PickledTrafficSignsDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']
            self.labels = data['labels']
            self.count = len(self.labels)
            self.transform = transform
        
    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return (feature, self.labels[index])

    def __len__(self):
        return self.count

def train(model, device):
    data_transforms = transforms.Compose([
        ClaheTranform(),
        transforms.ToTensor()
    ])
    torch.manual_seed(1)
    train_dataset = PickledTrafficSignsDataset(training_file, transform=data_transforms)
    valid_dataset = PickledTrafficSignsDataset(validation_file, transform=data_transforms)
    test_dataset = PickledTrafficSignsDataset(testing_file, transform=data_transforms)
    class_sample_count = np.bincount(train_dataset.labels)
    weights = 1 / np.array([class_sample_count[y] for y in train_dataset.labels])
    samp = sampler.WeightedRandomSampler(weights, 43 * 2000)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=samp)
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.7)
    train_epochs(model, device, train_loader, valid_loader, optimizer)

def train_epochs(model, device, train_data_loader, valid_data_loader, optimizer):
    
    liveloss = PlotLosses()
    loss_function = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    data_loaders = {'train': train_data_loader, 'validation':valid_data_loader}
    
    for epoch in range(20):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for batch_idx, (data, target) in enumerate(data_loaders[phase]):
                
                if phase == 'train':
                    output = model(data.to(device))
                    target = target.long().to(device)
                    loss = loss_function(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        output = model(data.to(device))
                        target = target.long().to(device)
                        loss = loss_function(output, target)

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t{} Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loaders[phase].dataset),
                        100. * batch_idx / len(data_loaders[phase]), phase, loss.item()))
                
                pred = torch.argmax(output, dim=1)
                running_loss += loss.detach()
                running_corrects += torch.sum(pred == target).sum().item() 
                total += target.size(0)


            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / total
            
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc#.item()
        
        liveloss.update(logs)
        liveloss.draw()
```

```python id="BxSgYXMj9cZV"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TraffficNet(True).to(device)
model.share_memory() # gradients are allocated lazily, so they are not shared here
train(model, device)
```

<!-- #region id="rCYpn-999cZc" -->
![](https://github.com/sparsh-ai/Python-Image-Processing-Cookbook/blob/master/Chapter%2007/images/traffic_learning.png?raw=1)
<!-- #endregion -->

```python id="9jS6x7ah9cZd"
data_transforms = transforms.Compose([
        ClaheTranform(),
        transforms.ToTensor()
])
test_dataset = PickledTrafficSignsDataset(testing_file, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
for (batch_idx, data) in enumerate(test_loader):
    with torch.no_grad():
        output = model(data[0].to(device))
        pred = torch.argmax(output, dim=1)
        break
        
plt.figure(figsize=(20, 20))
for i in range(len(pred)):
    plt.subplot(11, 6, i+1)
    plt.axis('off')
    plt.title(signal_names.loc[signal_names['ClassId'] == pred[i].cpu().numpy()].SignName.to_string(index=False))
    plt.imshow(np.reshape(data[0][i,...].cpu().numpy(), (-1,32)), cmap='gray')
plt.show()
```

<!-- #region id="W8gZvVX19cZi" -->
![](https://github.com/sparsh-ai/Python-Image-Processing-Cookbook/blob/master/Chapter%2007/images/traffic_sign_test_pred.png?raw=1)
<!-- #endregion -->

<!-- #region id="M7ABq2Ya9cZi" -->
### Human pose estimation using Deep Learning
<!-- #endregion -->

```python id="afllkXXI9cZj" outputId="1ad6c348-7c6f-440f-e4d6-f50420eb579a"
%matplotlib inline
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data

#print(cv2.__version__)

proto_file = "models/pose_deploy_linevec_faster_4_stages.prototxt"
weights_file = "models/pose_iter_160000.caffemodel"
n_points = 15
body_parts = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, 
              "LShoulder": 5,  "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
             "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14, "Background": 15}

#pose_parts = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
pose_parts = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
             ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
             ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
             ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]


image = cv2.imread("images/leander.png")
height, width = image.shape[:2]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (368,368), (0, 0, 0), swapRB=False, crop=False)
net.setInput(blob)
output = net.forward()
h, w = output.shape[2:4]
print(output.shape)
```

```python id="fDbF1TzQ9cZp" outputId="d8bcaa7f-6f1e-4c8b-d2c2-68fbad255ba5"
plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
prob_map = np.zeros((width, height))
for i in range(1,5):
    pmap = output[0, i, :, :]
    prob_map += cv2.resize(pmap, (height, width))
plt.imshow(prob_map, alpha=0.6)
plt.colorbar()
plt.axis("off")
plt.show()
```

```python id="Rdx1tXQW9cZu" outputId="247e9d7f-8f43-4b13-b9a5-74d53f8bb128"
image1 = image.copy()

# Empty list to store the detected keypoints
points = []

for i in range(n_points):
    # confidence map of corresponding body's part.
    prob_map = output[0, i, :, :]

    # Find local maxima of the prob_map.
    min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
    
    # Scale the point to fit on the original image
    x = (width * point[0]) / w
    y = (height * point[1]) / h

    if prob > threshold : 
        cv2.circle(image1, (int(x), int(y)), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(image1, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(image, (int(x), int(y)), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)

# Draw Skeleton
for pair in pose_parts:
    part_from = body_parts[pair[0]]
    part_to = body_parts[pair[1]]

    if points[part_from] and points[part_to]:
        cv2.line(image, points[part_from], points[part_to], (0, 255, 0), 3)

plt.figure(figsize=[20,12])
plt.subplot(121), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Keypoints', size=20)
plt.subplot(122), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Pose', size=20)
plt.show()
```

<!-- #region id="jgABqrJL9cZz" -->
### Gabor Filter banks for Texture Classification
<!-- #endregion -->

```python id="k5zHJMG19cZ0" outputId="24a62c5e-d97a-4552-8955-73e0624b076c"
#http://slazebni.cs.illinois.edu/research/uiuc_texture_dataset.zip
from glob import glob
for class_name in glob('images/UIUC_textures/*'):
    print(class_name)
```

```python id="Ddvch5Qj9cZ6" outputId="0fbce1ba-6592-4bb3-8468-adca1a0e4fc2"
#https://gogul.dev/software/texture-recognition
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel

# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

image_names = ['images/UIUC_textures/woods/T04_01.jpg',
               'images/UIUC_textures/stones/T12_01.jpg',
               'images/UIUC_textures/bricks/T15_01.jpg',
               'images/UIUC_textures/checks/T25_01.jpg',
              ]
labels = ['woods', 'stones', 'bricks', 'checks']

images = []
for image_name in image_names:
    images.append(rgb2gray(imread(image_name)))

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
plt.gray()
plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
fig.suptitle('Image responses for Gabor filter kernels', fontsize=25)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(labels, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=15)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel))
    ax.set_ylabel(label, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
```

```python id="Q_D7CP-79caA" outputId="ff6ce847-fd6d-4808-9e2f-a12370bc5511"
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

# prepare reference features
ref_feats = np.zeros((4, len(kernels), 2), dtype=np.double)
for i in range(4):
    ref_feats[i, :, :] = compute_feats(images[i], kernels)

print('Images matched against references using Gabor filter banks:')

new_image_names = ['images/UIUC_textures/woods/T04_02.jpg',
               'images/UIUC_textures/stones/T12_02.jpg',
               'images/UIUC_textures/bricks/T15_02.jpg',
               'images/UIUC_textures/checks/T25_02.jpg',
              ]

plt.figure(figsize=(10,18))
plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
for i in range(4):
    image = rgb2gray(imread(new_image_names[i]))
    feats = compute_feats(image, kernels)
    mindex = match(feats, ref_feats)
    print('original: {}, match result: {} '.format(labels[i], labels[mindex]))
    plt.subplot(4,2,2*i+1), plt.imshow(image), plt.axis('off'), plt.title('Original', size=20)
    plt.subplot(4,2,2*i+2), plt.imshow(images[mindex]), plt.axis('off'), plt.title('Recognized as ({})'.format(labels[mindex]), size=20)
plt.show()
```

<!-- #region id="5l-JZTFu9caI" -->
### Image Classification with Fine Tuning + Transfer learning
<!-- #endregion -->

```python id="-pW0S9z09caJ" outputId="19dba3ca-7fcd-482d-e78e-9358f1c58f36"
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img
import matplotlib.pylab as plt
import numpy as np

train_dir = 'images/flower_photos/train'
test_dir = 'images/flower_photos/test'
image_size = 224

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the conv layers except the last two
for layer in vgg_conv.layers[:-2]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

from keras import models
from keras import layers
from keras import optimizers

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()


train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split
test_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 100

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
        classes = ['roses', 'sunflowers', 'tulips'],
        subset='validation') # set as validation data

# Data Generator for Validation data
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical',
        classes = ['roses', 'sunflowers', 'tulips'],
        shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# Train the Model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Save the Model
model.save('all_freezed.h5')

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
```

```python id="iA4e859P9caQ" outputId="72d5934a-82fa-4727-d336-6304a9c2025a"
plt.figure(figsize=(20,10))
plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0)
plt.subplot(121)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy', size=20)
plt.legend(prop={'size': 10})
plt.grid()
plt.subplot(122)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.legend(prop={'size': 10})
plt.grid()
plt.show()
```

```python id="D6KNVi0H9caV" outputId="b6dc2e24-7b07-4c7b-ba95-6430ab4f306a"
# https://github.com/keras-team/keras/issues/3477

test_generator.reset()

# Get the filenames from the generator
fnames = test_generator.filenames

# Get the ground truth from generator
ground_truth = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
index2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(test_generator, steps=len(fnames))
predicted_classes = np.argmax(predictions,axis=-1)
predicted_classes = np.array([index2label[k] for k in predicted_classes])
ground_truth = np.array([index2label[k] for k in ground_truth])
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))


# Show the errors
plt.figure(figsize=[20,20])
plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0)
for i in range(16):
    pred_label = predicted_classes[errors[i]]
    title = 'Original label:{}\n Prediction: {} confidence: {:.3f}'.format(
        ground_truth[errors[i]],
        pred_label,
        predictions[errors[i]][label2index[pred_label]], size=20)
    
    original = load_img('{}/{}'.format(test_dir,fnames[errors[i]]))
    plt.subplot(4,4,i+1)
    plt.axis('off')
    plt.title(title, size=15)
    plt.imshow(original)
plt.show()
```

```python id="QLXXACZK9cad"

```
