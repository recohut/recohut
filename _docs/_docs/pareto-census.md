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

<!-- #region id="yxrhlMIzU66O" -->
# Efficient Continuous Pareto Exploration in MTL on UCI Census
<!-- #endregion -->

<!-- #region id="0khWmYd4U66P" -->
## Setup
<!-- #endregion -->

```python id="oASPpO1AU66Q"
import sys
import os
from pathlib import Path
import codecs
import gzip
import urllib
import pickle
import random
from itertools import product, combinations
from functools import partial
from collections import OrderedDict
from contextlib import contextmanager
from tqdm.notebook import tqdm, trange

import numpy as np
import scipy
from scipy import ndimage
import scipy.optimize
from scipy.sparse.linalg import LinearOperator, minres
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torchvision.transforms as transforms

%matplotlib inline
```

<!-- #region id="S6iVriJvWrQt" -->
## `min_norm_solver`
<!-- #endregion -->

```python id="0SvDb-GgWrxD"
def _min_norm_element_from2(v1v1, v1v2, v2v2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2)
    # v2v2 - gamm * gamma * (v1 - v2)^2
    # cost = v2v2 - gamma * gamma * (v1v1 + v2v2 - 2 * v1v2)
    #      = v2v2 - gamma * (v2v2 - v1v2)
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def _min_norm_2d(vecs):
    """
    Find the minimum norm solution as combination of two points
    This is correct only in 2D
    ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
    """
    dmin = None
    dps = vecs.matmul(vecs.t()).cpu().numpy()
    for i, j in combinations(range(len(vecs)), 2):
        c, d = _min_norm_element_from2(dps[i, i], dps[i, j], dps[j, j])
        if dmin is None:
            dmin = d
        if d <= dmin:
            dmin = d
            sol = [(i, j), c, d]
    return sol, dps


def _projection2simplex(y):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    m = len(y)
    sorted_y = np.flip(np.sort(y), axis=0)
    tmpsum = 0.0
    tmax_f = (np.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return np.maximum(y - tmax_f, np.zeros(y.shape))


def _next_point(cur_val, grad, n):
    proj_grad = grad - (np.sum(grad) / n)
    tm1 = -cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
    tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

    t = 1
    if len(tm1[tm1 > 1e-7]) > 0:
        t = np.min(tm1[tm1 > 1e-7])
    if len(tm2[tm2 > 1e-7]) > 0:
        t = min(t, np.min(tm2[tm2 > 1e-7]))

    next_point = proj_grad * t + cur_val
    next_point = _projection2simplex(next_point)
    return next_point


def find_min_norm_element(vecs, max_iter=250, stop_crit=1e-5):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
    as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
    the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
    """
    # Solution lying at the combination of two points
    init_sol, dps = _min_norm_2d(vecs.detach())

    n = len(vecs)
    sol_vec = np.zeros(n)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        # This is optimal for n=2, so return the solution
        return sol_vec, init_sol[2]

    iter_count = 0

    while iter_count < max_iter:
        grad_dir = -1.0 * np.dot(dps, sol_vec)
        new_point = _next_point(sol_vec, grad_dir, n)
        # Re-compute the inner products for line search
        v1v1 = 0.0
        v1v2 = 0.0
        v2v2 = 0.0
        for i in range(n):
            for j in range(n):
                v1v1 += sol_vec[i] * sol_vec[j] * dps[i, j]
                v1v2 += sol_vec[i] * new_point[j] * dps[i, j]
                v2v2 += new_point[i] * new_point[j] * dps[i, j]
        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc * sol_vec + (1 - nc) * new_point
        change = new_sol_vec - sol_vec
        if np.sum(np.abs(change)) < stop_crit:
            break
        sol_vec = new_sol_vec
    return sol_vec, nd
```

<!-- #region id="vDsRD3v_U66S" -->
## Random seed fixation
<!-- #endregion -->

```python id="JlkZhWLPU66T"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

<!-- #region id="FWpopuvjU66V" -->
## Dataset definition
<!-- #endregion -->

```python id="cSJEuWSvU66V"
uci_info = '''
age: label.
class of worker: Not in universe, Federal government, Local government, Never worked, Private, Self-employed-incorporated, Self-employed-not incorporated, State government, Without pay.
detailed industry recode: 0, 40, 44, 2, 43, 47, 48, 1, 11, 19, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 4, 42, 45, 5, 15, 16, 22, 29, 31, 50, 14, 17, 18, 28, 3, 30, 41, 46, 51, 12, 13, 21, 23, 26, 6, 7, 9, 49, 27, 8, 10, 20.
detailed occupation recode: 0, 12, 31, 44, 19, 32, 10, 23, 26, 28, 29, 42, 40, 34, 14, 36, 38, 2, 20, 25, 37, 41, 27, 24, 30, 43, 33, 16, 45, 17, 35, 22, 18, 39, 3, 15, 13, 46, 8, 21, 9, 4, 6, 5, 1, 11, 7.
education: label.
wage per hour: continuous.
enroll in edu inst last wk: Not in universe, High school, College or university.
marital stat: label.
major industry code: Not in universe or children, Entertainment, Social services, Agriculture, Education, Public administration, Manufacturing-durable goods, Manufacturing-nondurable goods, Wholesale trade, Retail trade, Finance insurance and real estate, Private household services, Business and repair services, Personal services except private HH, Construction, Medical except hospital, Other professional services, Transportation, Utilities and sanitary services, Mining, Communications, Hospital services, Forestry and fisheries, Armed Forces.
major occupation code: Not in universe, Professional specialty, Other service, Farming forestry and fishing, Sales, Adm support including clerical, Protective services, Handlers equip cleaners etc , Precision production craft & repair, Technicians and related support, Machine operators assmblrs & inspctrs, Transportation and material moving, Executive admin and managerial, Private household services, Armed Forces.
race: White, Black, Other, Amer Indian Aleut or Eskimo, Asian or Pacific Islander.
hispanic origin: Mexican (Mexicano), Mexican-American, Puerto Rican, Central or South American, All other, Other Spanish, Chicano, Cuban, Do not know, NA.
sex: Female, Male.
member of a labor union: Not in universe, No, Yes.
reason for unemployment: Not in universe, Re-entrant, Job loser - on layoff, New entrant, Job leaver, Other job loser.
full or part time employment stat: Children or Armed Forces, Full-time schedules, Unemployed part- time, Not in labor force, Unemployed full-time, PT for non-econ reasons usually FT, PT for econ reasons usually PT, PT for econ reasons usually FT.
capital gains: continuous.
capital losses: continuous.
dividends from stocks: continuous.
tax filer stat: Nonfiler, Joint one under 65 & one 65+, Joint both under 65, Single, Head of household, Joint both 65+.
region of previous residence: Not in universe, South, Northeast, West, Midwest, Abroad.
state of previous residence: Not in universe, Utah, Michigan, North Carolina, North Dakota, Virginia, Vermont, Wyoming, West Virginia, Pennsylvania, Abroad, Oregon, California, Iowa, Florida, Arkansas, Texas, South Carolina, Arizona, Indiana, Tennessee, Maine, Alaska, Ohio, Montana, Nebraska, Mississippi, District of Columbia, Minnesota, Illinois, Kentucky, Delaware, Colorado, Maryland, Wisconsin, New Hampshire, Nevada, New York, Georgia, Oklahoma, New Mexico, South Dakota, Missouri, Kansas, Connecticut, Louisiana, Alabama, Massachusetts, Idaho, New Jersey.
detailed household and family stat: Child <18 never marr not in subfamily, Other Rel <18 never marr child of subfamily RP, Other Rel <18 never marr not in subfamily, Grandchild <18 never marr child of subfamily RP, Grandchild <18 never marr not in subfamily, Secondary individual, In group quarters, Child under 18 of RP of unrel subfamily, RP of unrelated subfamily, Spouse of householder, Householder, Other Rel <18 never married RP of subfamily, Grandchild <18 never marr RP of subfamily, Child <18 never marr RP of subfamily, Child <18 ever marr not in subfamily, Other Rel <18 ever marr RP of subfamily, Child <18 ever marr RP of subfamily, Nonfamily householder, Child <18 spouse of subfamily RP, Other Rel <18 spouse of subfamily RP, Other Rel <18 ever marr not in subfamily, Grandchild <18 ever marr not in subfamily, Child 18+ never marr Not in a subfamily, Grandchild 18+ never marr not in subfamily, Child 18+ ever marr RP of subfamily, Other Rel 18+ never marr not in subfamily, Child 18+ never marr RP of subfamily, Other Rel 18+ ever marr RP of subfamily, Other Rel 18+ never marr RP of subfamily, Other Rel 18+ spouse of subfamily RP, Other Rel 18+ ever marr not in subfamily, Child 18+ ever marr Not in a subfamily, Grandchild 18+ ever marr not in subfamily, Child 18+ spouse of subfamily RP, Spouse of RP of unrelated subfamily, Grandchild 18+ ever marr RP of subfamily, Grandchild 18+ never marr RP of subfamily, Grandchild 18+ spouse of subfamily RP.
detailed household summary in household: Child under 18 never married, Other relative of householder, Nonrelative of householder, Spouse of householder, Householder, Child under 18 ever married, Group Quarters- Secondary individual, Child 18 or older.
instance weight: ignore.
migration code-change in msa: Not in universe, Nonmover, MSA to MSA, NonMSA to nonMSA, MSA to nonMSA, NonMSA to MSA, Abroad to MSA, Not identifiable, Abroad to nonMSA.
migration code-change in reg: Not in universe, Nonmover, Same county, Different county same state, Different state same division, Abroad, Different region, Different division same region.
migration code-move within reg: Not in universe, Nonmover, Same county, Different county same state, Different state in West, Abroad, Different state in Midwest, Different state in South, Different state in Northeast.
live in this house 1 year ago: Not in universe under 1 year old, Yes, No.
migration prev res in sunbelt: Not in universe, Yes, No.
num persons worked for employer: continuous.
family members under 18: Both parents present, Neither parent present, Mother only present, Father only present, Not in universe.
country of birth father: Mexico, United-States, Puerto-Rico, Dominican-Republic, Jamaica, Cuba, Portugal, Nicaragua, Peru, Ecuador, Guatemala, Philippines, Canada, Columbia, El-Salvador, Japan, England, Trinadad&Tobago, Honduras, Germany, Taiwan, Outlying-U S (Guam USVI etc), India, Vietnam, China, Hong Kong, Cambodia, France, Laos, Haiti, South Korea, Iran, Greece, Italy, Poland, Thailand, Yugoslavia, Holand-Netherlands, Ireland, Scotland, Hungary, Panama.
country of birth mother: India, Mexico, United-States, Puerto-Rico, Dominican-Republic, England, Honduras, Peru, Guatemala, Columbia, El-Salvador, Philippines, France, Ecuador, Nicaragua, Cuba, Outlying-U S (Guam USVI etc), Jamaica, South Korea, China, Germany, Yugoslavia, Canada, Vietnam, Japan, Cambodia, Ireland, Laos, Haiti, Portugal, Taiwan, Holand-Netherlands, Greece, Italy, Poland, Thailand, Trinadad&Tobago, Hungary, Panama, Hong Kong, Scotland, Iran.
country of birth self: United-States, Mexico, Puerto-Rico, Peru, Canada, South Korea, India, Japan, Haiti, El-Salvador, Dominican-Republic, Portugal, Columbia, England, Thailand, Cuba, Laos, Panama, China, Germany, Vietnam, Italy, Honduras, Outlying-U S (Guam USVI etc), Hungary, Philippines, Poland, Ecuador, Iran, Guatemala, Holand-Netherlands, Taiwan, Nicaragua, France, Jamaica, Scotland, Yugoslavia, Hong Kong, Trinadad&Tobago, Greece, Cambodia, Ireland.
citizenship: Native- Born in the United States, Foreign born- Not a citizen of U S , Native- Born in Puerto Rico or U S Outlying, Native- Born abroad of American Parent(s), Foreign born- U S citizen by naturalization.
own business or self employed: 0, 2, 1.
fill inc questionnaire for veteran's admin: Not in universe, Yes, No.
veterans benefits: 0, 2, 1.
weeks worked in year: continuous.
year: 94, 95.
income: - 50000, 50000+.
'''
```

```python id="-57E7dxWU66X"
class UCI(torch.utils.data.Dataset):
    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pth'
    test_file = 'test.pth'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.data, self.labels = torch.load(
                self.root / self.processed_folder /self.training_file)
        else:
            self.data, self.labels = torch.load(
                self.root / self.processed_folder / self.test_file)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (self.root / self.processed_folder / self.training_file).is_file() and \
            (self.root / self.processed_folder / self.test_file).is_file()

    def download(self):
        if self._check_exists():
            return

        # download files
        (self.root / self.raw_folder).mkdir(parents=True, exist_ok=True)
        (self.root / self.processed_folder).mkdir(parents=True, exist_ok=True)

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = self.root / self.raw_folder / filename
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(self.root / self.raw_folder / '.'.join(filename.split('.')[:-1]), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        name_dict = OrderedDict()
        property_list = []
        for line in uci_info.split('\n'):
            if not line:
                continue
            name, values = line.strip()[:-1].split(': ')
            name_dict[name] = []
            if values in ('ignore', 'label', 'continuous'):
                pp = values
            else:
                pp = 'normal'
            property_list.append(pp)

        self.uci_preprocess(self.root / self.raw_folder / 'census-income.data', name_dict, property_list)
        self.uci_preprocess(self.root / self.raw_folder / 'census-income.test', name_dict, property_list)

        for i, (name, values) in enumerate(name_dict.items()):
            value_set = list(sorted(list(set(values))))
            value_dict = dict()
            for j, value in enumerate(value_set):
                value_dict[value] = j
            name_dict[name] = value_dict

        training_set = self.uci_process(self.root / self.raw_folder / 'census-income.data', name_dict, property_list)
        test_set = self.uci_process(self.root / self.raw_folder / 'census-income.test', name_dict, property_list)

        with open(self.root / self.processed_folder / self.training_file, 'wb') as f:
            torch.save(training_set, f)
        with open(self.root / self.processed_folder / self.test_file, 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def uci_preprocess(path, name_dict, property_list):
        with open(path, 'r') as f:
            raw_data = f.readlines()
        for line in raw_data:
            if len(line.strip()) == 0:
                continue
            words = line.strip()[:-1].split(', ')
            if len(words) != 42:
                continue

            # make list
            for word, pp, (name, l) in zip(words, property_list, name_dict.items()):
                word = word.strip()
                if pp == 'continuous':
                    word = float(word)
                l.append(word)

    @staticmethod
    def uci_process(path, name_dict, property_list):
        with open(path, 'r') as f:
            raw_data = f.readlines()

        images = []
        labels = []
        for line in raw_data:
            if len(line.strip()) == 0:
                continue
            words = line.strip()[:-1].split(', ')
            if len(words) != 42:
                continue

            # make list
            image = []
            label = [None, None, None]
            for word, pp, (name, values) in zip(words, property_list, name_dict.items()):
                word = word.strip()
                if pp == 'continuous':
                    word = float(word)
                    image.append(word)
                elif pp == 'ignore':
                    continue
                elif pp == 'label':
                    if name == 'education':
                        label[1] = int(word.startswith(('Bachelors', 'Some', 'Maters', 'Asso', 'Doctorate', 'Prof')))
                    elif name == 'marital stat':
                        label[2] = int(word == 'Never married')
                    else: # age
                        label[0] = int(float(word) >= 40)
                else:
                    # normal
                    one_hot = np.zeros(len(values))
                    one_hot[values[word]] = 1
                    image.append(one_hot)

            images.append(torch.Tensor(np.hstack(image)))
            labels.append(torch.LongTensor(label))
        return images, labels
```

<!-- #region id="PsS-_V7YU66Z" -->
## Dataset Preparation
<!-- #endregion -->

```python id="yz_EMRKoU66a" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723932255, "user_tz": -330, "elapsed": 102901, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad195158-27e7-4bb8-9814-4c90ea9bd955"
dataset = UCI(root='/content/UCI', train=True, download=True)
```

<!-- #region id="mYvIDYFgU66c" -->
## PyTorch initialization

- working directory
- device
- dataloader
- utilities
<!-- #endregion -->

<!-- #region id="ltxpr3mqU66d" -->
### Checkpoint paths
<!-- #endregion -->

```python id="lKmrd6z6U66d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723932260, "user_tz": -330, "elapsed": 73, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d207b4ee-3c2d-417b-fb7a-75eed219f030"
ckpt_root = Path('/content/checkpoints')
ckpt_root.mkdir(parents=True, exist_ok=True)

sgd_path = ckpt_root / 'sgd'
sgd_path.mkdir(parents=True, exist_ok=True)

mr_path = ckpt_root / 'minres'
mr_path.mkdir(parents=True, exist_ok=True)

print('Checkpoint root:', ckpt_root)
print('SGD path:       ', sgd_path)
print('MINRES path:    ', mr_path)
```

<!-- #region id="eZARD4SyU66f" -->
### Device initialization
We remove all random state.
<!-- #endregion -->

```python id="1t6Zjuu9U66g" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723932267, "user_tz": -330, "elapsed": 60, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4dc7a9fb-2205-4a3e-daa6-7476fb574bad"
if torch.cuda.is_available():
    device = torch.device('cuda')  # use default cuda device
    import torch.backends.cudnn as cudnn  # make cuda deterministic
    cudnn.benchmark = False
    cudnn.deterministic = True
else:
    device = torch.device('cpu') # otherwise use cpu

print('Current device:', device)
```

<!-- #region id="ksM-yEu3U66i" -->
### Training and test dataloader
We use batch size of 256 for both training and test side.
<!-- #endregion -->

```python id="5XXh2j-BU66i" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723960820, "user_tz": -330, "elapsed": 28602, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="457e57ae-5035-432a-fff2-0d84c2a3c8d1"
trainset = UCI('/content/UCI', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, drop_last=True, num_workers=0)

testset = UCI('/content/UCI', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

print('Training Dataset:')
print(trainset)
print()

print('Test Dataset:')
print(testset)
```

<!-- #region id="hxMIY8rHU66k" -->
### Utility functions
- evenly distributed weights
- top-k accuracies
- evaluation
<!-- #endregion -->

```python id="TpoH0PMLU66k" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723960823, "user_tz": -330, "elapsed": 50, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c6411aa7-d5ae-44d4-885c-6fb9ff409c83"
def evenly_dist(num_weights, dim=3):
    return [ret for ret in product(np.linspace(0.0, 1.0, num_weights + 2), repeat=dim) \
            if round(sum(ret), 6) == 1.0 and all(r not in (0.0, 1.0) for r in ret)]

def topk_accuracies(logits, targets, ks=(1,)):
    assert logits.dim() == 2
    assert targets.dim() == 1
    assert logits.size(0) == targets.size(0)

    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    targets = targets.unsqueeze(1).expand_as(pred)
    correct = pred.eq(targets).float()

    accu_list = []
    for k in ks:
        accu = correct[:, :k].sum(1).mean()
        accu_list.append(accu.item())
    return accu_list

def evaluate(network, dataloader, closures, topk_closures):
    num_samples = 0
    total_losses = np.zeros(len(closures))
    total_top1s = np.zeros(len(closures))
    with torch.no_grad():
        network.train(False)
        for images, targets in dataloader:
            batch_size = len(images)
            num_samples += batch_size
            images = images.to(device)
            targets = targets.to(device)
            logits = network(images)
            losses = [c(network, logits, targets).item() for c in closures]
            total_losses += batch_size * np.array(losses)
            topks = [c(network, logits, targets) for c in topk_closures]
            total_top1s += batch_size * np.array(topks)
    total_losses /= num_samples
    total_top1s /= num_samples
    return total_losses, total_top1s

print('Example of evenly_dist(num_weights=5, dim=3):')
for i, combination in enumerate(evenly_dist(5, dim=3)):
    print('{:d}: ('.format(i + 1) + ', '.join(['{:.3f}'.format(digit) for digit in combination]) + ')')
```

<!-- #region id="tgOymhumU66m" -->
## Empirical Pareto front generation
<!-- #endregion -->

<!-- #region id="xBlALifrXKpw" -->
## SGD preparation

- hyper-parameters
- network
- loss function
- optimizer
- learning rate scheduler
- inital state snapshot
<!-- #endregion -->

<!-- #region id="fnSv8JbQU66n" -->
### Hyper-Parameters declaration
- num of epochs
- num of different weight combinations
<!-- #endregion -->

```python id="kXh43GL4U66o"
num_epochs = 30
num_weights = 5
```

<!-- #region id="__lXmgZ_U66p" -->
### Network definition

We use a double-layer MLP with a fully-connected layer for each task.
<!-- #endregion -->

```python id="yObDDhlJU66q" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723975385, "user_tz": -330, "elapsed": 14589, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3ad4a14c-5365-445c-a4f7-fb8034610782"
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(487, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_age = nn.Linear(128, 2)
        self.fc_education = nn.Linear(128, 2)
        self.fc_marriage = nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return [self.fc_age(x), self.fc_education(x), self.fc_marriage(x)]

network = MLP()
network.to(device)

print('Network:')
print(network)
```

<!-- #region id="WNngb6lsU66s" -->
### Loss function definition
We use cross entropy loss for two tasks.
<!-- #endregion -->

```python id="AjSgN5PhU66s"
criterion = nn.CrossEntropyLoss().to(device)

closures = [
    lambda n, l, t: criterion(l[0], t[:, 0]),
    lambda n, l, t: criterion(l[1], t[:, 1]),
    lambda n, l, t: criterion(l[2], t[:, 2])
]

top1_closures = [
    lambda n, l, t: topk_accuracies(l[0], t[:, 0], ks=(1,))[0],
    lambda n, l, t: topk_accuracies(l[1], t[:, 1], ks=(1,))[0],
    lambda n, l, t: topk_accuracies(l[2], t[:, 2], ks=(1,))[0]
]
```

<!-- #region id="gwIQkOi7U66u" -->
### Optimizer definition
We use SGD with learning rate of 0.001 and momentum of 0.9.
<!-- #endregion -->

```python id="zJqnhdW9U66u" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634723975391, "user_tz": -330, "elapsed": 75, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9fbe578b-5137-4ede-a707-4cc2cc313c4d"
optimizer = SGD(network.parameters(), lr=0.001, momentum=0.9)

print(optimizer)
```

<!-- #region id="GWWUauhGU66w" -->
### Learning rate scheduler definition
We use cosine annealing learning rate scheduler for training.
<!-- #endregion -->

```python id="XaQ8dz02U66x"
lr_scheduler = CosineAnnealingLR(optimizer, num_epochs * len(trainloader))
```

<!-- #region id="u-KqZ0yEU66y" -->
### Snapshot for inital states

The initial weights / optimizer / lr_scheduler are saved for further training (we removed **ALL** randomness).
<!-- #endregion -->

```python id="jSxoJIHJU66z"
init_ckpt = {
    'state_dict': network.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_scheduler': lr_scheduler.state_dict()
}
torch.save(init_ckpt, sgd_path / 'init.pth')
```

<!-- #region id="g_uCN8M0U661" -->
## Let's train it!
<!-- #endregion -->

```python id="B8_6RP8eU661" colab={"base_uri": "https://localhost:8080/", "height": 337, "referenced_widgets": ["f34fd8a7749f44289a5f35c23e322a73", "19e9ed7a2ca24f818bd3aca6320659b3", "86dbb43ee79f461e822bbfcbe07be36b", "fad52343bfaf4b55ae63c8e067bd60fe", "fa227a6dcd924852a152192e537a519c", "4d8506b4010942d3bdbcd6a836f04fc9", "301cb5084e2c42a480f7e256a602c24d", "90f4731e0fac4ce4a280a4c67eba60d0", "afa9fd08ba9345afbf65b63436df0c98", "156abb7382d34cb09fc32f1bcafb2f1a", "c26f04370cb443378dcaf431f8ad9ad9", "8b57c1654a63470a9514b19217e4dc9b", "6cedcac24e904fa08373fa4eb9764b1c", "187e773bf04944408a88e44daf408233", "d82dbafaab0e46b8a15c84bb66448b1c", "fb23a0539dec4d45aaf029a86cdff5c7", "5e596d5f417b4965bc33dfb7e941303b", "47a8ec4f42514ef397accce721841237", "b223b5ebfca746388844f91cd07a18bb", "290374bcb6a948229c142035f8250816", "c3387c1a42fa465b954b915e8188a1be", "7e35081eef13490d986d6cbf2828eb6e", "68f22bc927e54472839b1136fff44d5c", "e5b2b7ae27a94c1bad7f4c1e4973c254", "bf2635b169a740f6a6a7dc144c5f2c57", "820f08cdcf1148cab2b9cb846f12516e", "e375ad53bac14605b61980ef50712a38", "96a9e7f7f8924c14b28f5b1a7a5301bc", "f0c49e9dda54441b92d519c46e1fe093", "d4b8905d2a4649bea678841cc95ec5bd", "835757212b4f4fff97a1b7f6e7c12102", "c82580e05a5a4f1895f245c03c0d4a08", "f01a3783e3dd48a9a349346c348184d9", "f1d78cf7acf541c293ccda40dad95ffc", "a976403d99d14a72b521f8d22776d781", "0f8dfe956dfa48df88793641d57d015e", "66bfa68bc1bc4b0982ff1491ba731e2a", "bf488fcac1f04ceab2b4aad333d1d6da", "c503ccef6d384b2b9ebb6bf6265082a8", "fa7fdc43e932404ebabdfe93062c678a", "09994bf51e0544f68f86becd9276c15a", "692b0176e16b413aae941baa8f3edbcd", "f1bc578d685940408317a11948a5ecee", "40a6a7f83e6c4fa88ac29ef60c5efbe4", "b3ae91a2207d46b597baf6287bfa9af8", "5135dc031ff2433880ef85633fcb2090", "e053e970b42249d7bce5a9aba26f4803", "cb7aa91c465c47b5b1a11fdd109039e6", "d1cbd57d4b714ccababbfcca576c6a35", "a4b0776c77584e12b7fcec326ccc47f4", "713a1723f9d04cf5a6868304f9848335", "26eac31b59ae4318bcb951aee1196baa", "fd360fc60b9c4b8c9589a9a8f82834bb", "8a0ef887768644e484635c294fa0359f", "c2ddcad0574c4937a9159772945b8d03", "40aaf5da585c475c88fc791bb180f373", "e4b9a476c0ee466ca92a752e3bad2692", "da4198a74df44c6488b36f10668fb36c", "96b1d2dd842342d0a980f52a13507607", "40c15e2e5e434fb1af125e503e2dcafa", "7926650342b44dc3bd1c153698af9c5d", "03c2fcf017144dc69867e5466aae89b8", "2043a8ed178a456c84392b8953dc4e43", "0559146a1c9d4327a2b449da7644462d", "6a47fa4d4b8142e4a69bb9bdcb0de24e", "3f813e969f6b45cb88d8c6ac51977a59", "5eab04ea63024901a2bdcb374da4d061", "e1054cf9b0564d9bbe71452a7f6cc0ab", "b1b363a293664bbbb5369a82eea8ee2f", "b1c5386a4f904882988f4cc8fa0ac800", "487ac25cf10c4560b75ce1efbf9d03f0", "ccd237d1b09144dd97b3e30110cd3ee8", "938f73546fd24ec085728f1406ea45fb", "5df613452f1a4fa38a4b9ec0eb952c00", "45675111e41946a4b3b17ac5bae6d65c", "bec16023c5f045b2997c681edcf3660f", "7a6fa9db74bc4a0c86ede1416dc5356d", "c1f1fa706b0645109bb318a5201215ec", "7c9f640feff649509cebf09a2358b247", "5f6a7721f08444a19499d3eb07a9f633", "8b6db62d4a274116a1d8d075bf227d9c", "dfb8adc31fb241858cb78648a962d157", "f1843729a3ec4871a2401aeb08438f5e", "80a9d3854fb9465d864477057f97cc62", "34a13ab184c54ba7a2f18db5de40e8a4", "66a10fd528f14730b1a1c6f555a7f937", "da6f8572fa9e402e85dc76fcd8a4e8bc", "76a5766fcae4479fb8c893e1d9a25067", "b502bb24035442bf8cc745984b0781d7", "980584d9d29744a2a79414878acc8ee1", "6ab3b5444e0d4d5eb62548e29f03e800", "8013713876e84808b125b5179c5f6f7a", "39a20cd6e5304ba0b9b773ceac00a1f0", "452b835ff6594cadaad407a85183fda1", "e3ba8e44d54a4abe9592de6d4e6715d3", "21939bb9dffb48e195886ff1387072d6", "1c5da44858a64932a0ee2d37bfe91c55", "d9b8c6b959ff455eb168c032550e88bc", "2bc8a5fac4f045ed8415e297dec74cce", "835f65429248466580b7c83b8f351b8c", "38bc18b2c3e1461db5d27b9f06ade486", "215d590971e04d798ce438c96d2ec083", "20dde3a5511349a989354d20eb6e0cc4", "c4af61da8fa541889747488010866118", "11f74018829a43a9aff67b424aa831ea", "80448eeab76e4075bedff06e95d1c7f0", "d507ffe074d542c18df4eafade9ed26c", "f7007ac5a525450096d13e9a1a220c5a", "12103f68ede3412cb61a91f29b772a19", "46b47725086445718f641880a509cc91", "6a8f660c32d14912a986489f546b05b3", "5353ea5cf8f945a69fef4170ef2c915a", "848479e5631444b08364aa17c6e173fb", "39f4489d798f442386d2e7009711a06f", "1034dab686be4e389a652ca706a7c449", "5e8fd4bed7f24c8c9780f8cd613519cc", "4237e82d3e334a2480bd20b3c76d8159", "cd911b1cf099456ba2b0bddd44fa9edf", "256dbc419d5143c091fa79c7fcfb67ea", "20e9a5662342456d8d46b3a33cd78527", "d6bf5344cf2245d29ccfe528805c9ae9", "461f7aaab8bb4fc9ae025e84d85425e0", "d8c204ece1d04e4bb683fa33d540f599", "e19128cc4f2743cab9604856d9e38a44", "2baca76fc9014f1ca60b491c503a07da", "bdd8d3ad1b9e4cb2ada754f9fd5edeb2", "9b74925682bf42b980927e0c9ff0d2c3", "3439187f8f81446ebb888c0aae2609de", "c79e2a40d5954b0f8f6a964f80634636", "ed85f1bf600f45f48e05dddae9dd1b2f", "7eb51c58916d48d9ba29ea9a0c1561d6", "f7b70937f801412da92510d95a8bad68", "8468a90079ec40818ab6f915c3340273", "4cbfb727e20a49a9ae14668e7552950a", "1903c324cf9546448b483d8b21fa2c90", "834fda5afd274fa7b96ddcb60468cd7a", "38c7a20e010443cf9235e858af05f9bd", "da05733dcc634f5d904e47383a35b775", "082f2f8530a94b7dab82e49e09fe2e48", "eebd614343a943c0afef061f373ad8f8", "e4adb94b60aa4e178d7c0c4b5c598600", "66cdfb2e817d4f7282e2acf28b6ec5d3", "7c830baa3ea246da89a6daaa771c183e", "888d0376f1da4af3a93aa004f822e1e7", "a7979829379348d4ba7f881d4ab67641", "6c2968690eb245309bb1183c8815e511", "93003309f9b342bf848a495fd10eca9b", "da01e9374dd14b89846342e4d4e93af6", "905a11a483df46328637513def601a6b", "79fb41d7d9664e89a5987db64322019d", "007793e1c68e46c3896e77f8facd9c5e", "855624328c704a6c89fc31766bf6de9c", "512975dbdf3247bfac810b687b2048ea", "6b3aac5ed2f643f79cb1b15a5853cf49", "448df487ee644030b5f53fc5cedfbbd6", "a497e73cd13842c488ca428485600238", "243f686cdbfb4bd0b264d41eb18faa1c", "6c49cfa5a57343bd860667f372d1b6da", "69b9e88ae24c4afea009acd042584523", "bb9bb194dc5b4b17925fdefe8e1103f5", "7aedc7f8fde74f4393fc60854d23200d", "d92c6e79299042e39499c9c6565e2178", "1fb744171a9a46738cc1fb477101a632", "f87a9e9913214aca8ac2b015afe09472", "8e8e26a2529e48be8d2565ad89a695cd", "2b27d019bb094016ab4b0cd53ab9e00e", "b9ee3df664174ed4a2a699e1285402d5", "90918bb8a52d4428bb0898ef3c0392f7", "fdaa989f935043a8a2cdf8396606ab4c", "63d3d3da95374167a5898bf2a533564c", "4d55f150a2234fc0aeaaef943e005ef9", "9d4bf1b7af044a9d96ccfaf36ac95cb1", "f4ab589ab950400792c25c326463ff32", "b8cd10d9a3174914b7fb9b1fd1b2c5e2", "2e2d847a374a486da9eea87499d55118", "8b599d6b9a3a4eb0a74ddf56b3789c41", "a2e77e3fc19d408d9c974c4694540aa5", "bb1cc9fae0de430da0a9191b4180a035", "6f849f4402454ef98a79c61cd29c2c9c", "bb5fba0efe234f40b95981aee9d85a02", "33c83a1445c64999a9d8e0c9a7093473", "25339dc6ff1545338fa06e95b9a435de", "17dc8ed2cbe84853a6523d7be822e02f", "6962be048ad54ed1ad1c00521a3f1d04", "557dbc5ce3f44345a016779b1d39e28b", "92d53258abf841d78054b8f21b2f1eed", "bc5632d39fec47a79eca05b8c40c118f", "08c73fb6595246918ad785453145d6cf", "57d5616b47b044ebac6f0a6c48d646eb", "e6db5a3ceb8540368c66e54525e1a005", "05b8ddf2e94244018e8676af7aac6664", "bf1a2a27a7314ab99d1ad40aceeabea3", "b124cf59f3ae4d9294fee79899b3f068", "d7cc1ed9771547cf840a0f3801132da3", "f1ad1146599c4854ad3d88eabc4c466c", "5f4714086db64e96994bbc7d256ad1f6", "4157e8e64d10458d8053307c830f1b3f", "ed029f81f7eb4fc39c7b2890edee59de", "7cba9e028dab451badec09b59ab64b00", "776652fe681148a2998aa8a764d3a2c6", "d38383f024234a9a8790f07f5aa20d5a", "e0ae433188dc49bbbd7e8fc6dc28c0c0", "2a47cf2eb31c441a9b7aca0553155873", "d1cb1460f58a4437b160994fa4bb77f3", "f7ec4dc62b7546d2b2b887529ba8df14", "a59aa2543c034164bc1c14be0f80a457", "3b47d8758815420e8d6b8eb727167c40", "374fbc92db314cf6b3d240147df74c04", "3e73101a5ae643d2bd38dafd554a6025", "4865b7d8bb1447379aeb571c283497b3", "cce62c0ff92f455fa32fd9b7e02179b2", "a9e5f83564ca48d0a368403c452bea88", "9b261a2dbe8945239cf41f739e113ca4", "0939ca12f8c64e208c6c4c4c387d676e", "8fbcc03e45e342e4acdea6538d081cc2", "eb707dcec7dc4ff4b2ca0809c15dd72a", "ae296fa4350c4f4984915c1d296f994c", "ccf8238969b141fe82c566fbf6f6ee00", "e74e30ba38074f73813ad2f90c3e9b61", "eab2f2ac695d4086a9ed692bc4ad3792", "29913009445d41d8b085dc33147b9031", "7ccde24e25124307bee6f00a9896b47c", "1f3d5b6944b44d07921163ab96ee0368", "1088daf45a6a4daabd9c3df6968d8c4f", "8616093cddbc48f29e428cef35a5e839", "0e0b433944e44250b7c46734eec59390", "78a102ea74254b0aad171a02cee2a886", "02edd3699d954cba80b60e8468f39e8a", "be813ad94ed948559a11d28cd51edbd7", "89636f906acf4998b649e5e24474a0b5", "1946534446964a459ebaa1687b853e33", "2e769b691fbd4db596a2ba69c389bcbc", "cf65236e23d5441d81d497d519b0b6b1", "bd63c76b094643c69caf51cfa1777d2b", "5a8ad04a6c4b4a37ad4fdd24d54e69f4", "9745bb6c25af4c028f94e71031000ea6", "63850df50684452ca7e53292d760e07b", "1062445b0b304586991a18cb71684764", "fb7ea7e8d2f6408787736dd1411de693", "577323f385e64623bb6a09073f4d3903", "94b8acfff9f84268a3279af65b273547", "b186579639654fa1918c1156d0cfeb7f", "2306c4a6d396470ba9ab56d55c4181df", "ab8f92c28810417b92415e0795615a5d", "b8e2e08e7e734ad99d202f19b38524fd", "b2babef5bbfc4721992c2a0b28d8f4eb", "f5630c5660f546ccbc7a9e427834dd47", "89158b660da5441c9e37982c3be89f63", "e543b78abc294c6089ebdec31623db8b", "d411d1e5a45b481c8710c10a37d7a63c", "34b8fb3643ad468b9288e98f0dee8579", "e2d6f40a1970455e970efc1bd8417906", "affaee9135bc44b3b6c99cde7d78ab7a", "c78a7a726bfb457c9ce57a31160c5bd4", "bda020d5a2344088ab437ca4575e075c", "48060df06ae341be9f41aaa413649d1d", "b46ef8ea18944f319e925cf09e194f48", "b981f4f549a74780a552162fa71f819e", "2ca4dd2915324d29b7482d6e7c1b7a35", "b66f55724b0b448396cfd76418a0942e", "fb1bcde366c94fcfb90f199429108b0e", "841bf73f6d824d0a9e6489343236cd86", "dcea443bf9a94e5d96d3104bde163329", "2284b60c94f94eae8098f8547c231071", "e2ffb666191844a3b27a3c33df335398", "bdb7a4d7f0204c6892175898f2b73160", "941d83e2095644de85c0ce667b3d28c5", "68a95e42cb594c1bb853a43f56b473a6", "1b62896667d044e5a7f3f63fed6c0c66", "b17bed16818547028094581b7dfdb350", "0e8c19f145d341a08ee83d22f278910d", "f4c2f6846519407a89a2b476164af220", "d5f3800b2f1843ffa0bfbd287994fdc1", "33f16653333941598d6807a03a5894a1", "a65833732cb645f681641c91b7eee103", "00983befb1774dce94b91081156fc9d8", "af803b3b55a04364a74236c1e7360ddf", "e8eaf17a726c4a4aa6c0372a652bc50b", "da248c02140e41edae0db37a6f0e3904", "f499d843b7a54dd98fa7cb6fb4ab0a77", "2c807896aa7c46fe87901676446bc007", "4f8f075d1919446495f5e6d0ede57725", "bf8d1f4b5d1342c08eec71be0e766181", "cdf76d196b6a4cee909e39898779e6f5", "2189f9a70ce4425f89b9f04b7aa29f23", "7fb5f745a2c346e589f93ccb950660df", "d5ac3f0542fd4605b2e7e3a39e920698", "b7a9fbc4edac45c496913b6622220516", "1a10ef393be44c9e839c646cbadef80a", "bc2dbd1ae29740a489f83d6816736dbb", "f8a59508acea41eb98a77d35f98a74fa", "76c451356b1844109840011c86ffc470", "0fc91f72a32643748b3b4520f2cc30ed", "dacd4f5f5c844ee8a668706fce344516", "731016bad3634f70997fd9c7042c8942", "0cf90e7c564346a187f825132024ad5c", "eb390df79a7a43d392d0cf34b2ef5257", "affbbf60fc37435cbb0c2f0817477609", "d6eda16e86ba49b8b9d7e66d5620f912", "3df332a7d670408c837285be8d6472a6", "88da3558cbb54c5d8262b1ae09cf33e2", "b4c3d1691f1648d8a3ef534495ecb3a2", "cca3fae597e94914bd7cc2859659d257", "893b035b85694b938221957b2a061f1d", "d319f6f40b03498fa72bee5ff8d94d4a", "2d6f88dc0e9747a1aed92114c95b9963", "fc8cb378058a4566b34834f400652d31", "5e20d31a713e4db5b03ace7ca5bdaa52", "2dee30c42e3a4cb3990221fbe1c3d6c9", "e8e9e19a62cd4d179d22688d587897ca", "92798edd69ed43268dd90c6116639252", "753f915c820e4fcf8909c6fd4dd9d5c0", "8684cbfa6fb1449e817780efbc1e31fd", "33f83bc95d3d4d659fcb200bc2d5a3f3", "0649fbed0a0644cc86daa449599825ac", "9e93291f8d9747d790d49a972f454bcc", "6a546e9b8f5f4b60b723c2af3afe136b", "e5a9fe68c4f94bbeac1a15c5d1d65078", "cbdaa931d4b548798851d647fbe15481", "0eb98d80d8024a74bc1f59419ae00e53", "f8e4eb52a0bd44d4b3934c0596c6f9d9", "4c1d7aa6dc3741679d352f723171ffbb", "547185d203c6483a9e7025b8afc19bae", "ad22176e7d9944e5bfdf90b16f92c208", "2d1998ee7a1d4114a45b6b9f7c13c612", "13d71cdece534b48a42c5cfad1b636e6", "4ba692f56d0d46ecb1c1612e85699e07", "c0db53faaa7543ac9c1a408065bdde30", "bf5d30972bb44cc59055a53351e70b90", "cdb11a144a8d489eb26c649f8790a8ac", "35ab1575a2a9458d9be5d9fbe75db7a3", "fec64993437d4ed2b17fea36d7f6a592", "d3a2469c58be490f87844f7f712ede68", "3076d180bf4442dd9e5033de8bfc90d0", "31bb1e9696524dc8a0babddc4c22a992", "a5afbe24ac014eec9caafa9e3fa96ff9", "d74009f1b5ae41a1b9bfbade4757cc0d", "93c06fcc616e4f0c9aea796371892b99", "a588e1e78964499db73cea5872ce33ef", "a9e49ae28174478da84e4068cbd40950", "fbebb6bb78c8417b8bb37e6cfe6edbe2", "161fd88ee384468a807de12501541a00", "291478e34efe4535951c1b380aae6324", "a550ed22f999433fb980999ec2cd8174", "381e72b37e094ba2bf761c07d0b97130", "7748dd530ad24c609ac8b352b7e86212", "f4a0121645b74f23b594829c2fcdf8d9", "72ecb54fddb0428991ba1b083ac5e0fa", "817b55e3dc8f4cba9c3ebc106debf6c8", "f5d4747fc86545449603992109d743ff", "41725ce6fe4f435181c2c4ceee75bd7f", "6cedd03377114b00b06613037b9dfcce", "fcf4834958db406fa0e559dbd5f038ac", "cf78ad9651df44f4a3c426cacd63b53a", "abcbce68f8f646debba594de9f4d2fb1", "6141e5d42ed04d26881e4b3079b1e06a", "7db62e3d55e64f5b89b19811ddb5d023", "8bc50ed1ff6741a1ab6fa1a093520e09", "6f3be0c6c1d743b9bedf4a9c23800be6", "289f61d7ad35497c93d35dbf0cbdccb7", "2abe77e281274e0bbed0ff6fa9d3e07b", "641db926cbc046b5b2df568865d959ea", "93a7b963c9684e9ea0206440bb743087", "4098a6d051f8459490b07f99890e4564", "3e534714671643dbac86207aaac41743", "c53ef43be36c4cc6b3db7e6739a48957", "2765647647cf4439b7bd0fe60c5f9fbf", "5f8dbbf082a3447c91c65ef6bd7153c7", "6b1a81b1017b48dd8c68e4897367c0b4", "0f489211d5344889ba208800c6133e6d", "124992f5ac4c46ef9f0dec7f3e876134", "84f7676eefb6409895dbc57437ab5c62", "150ad3a8c6e444b0b2f73de2f09d1369", "f25cf00d1d5e4ad3be7fd5a7b5f0c0b7", "018ab3af97144dbab3616233aca14582", "2ea58c856cad4dd5bf722d7d5e756a74", "6182291275654985a9aef6200ee29a5b", "a38423e7618543d69a939a4c2c8be81c", "ce3273298bdd45278bfedfa18bfdd4d8", "b60b7d640e8c4e648946b7f2739b83ee", "0987813b147448ee8dc513417896d3e8", "6b3a65ff81424184998f253e0b4ab42f", "4fc5145bb7514b1ca4ceaa7802c55276", "12f2b09b47fa42cc84452a40c33e6a7d", "a3e8b2cf79ad4982b9d63bdd7570810c", "ef80494238e8486d92c096a55e2795fb", "f76f18ddad9146d389b3b7d422235179", "cb66944044d9435d804ed7c3837bf4f0", "53c2ef506d31422a8d3f395c25b837b0", "7aa03b5ce8c74caea49f3119a1e3acfd", "08ee580d0f194ed5b82d2ddaba7aab49", "85ad8e51b14b491abece96e73c3937bc", "1db4537e645245d9bf46a198b8d54d7e", "5d6eb606ff3e4e5baf5d3685a1c3dc5d", "206756ff2ff14f648e3e31f5b961d67b", "882d43cf916a4bb489023248b78e0728", "6aeb4a9543ea41f483eee23310443124", "09fb8ebbe4d94c64970e72893b974927", "25edf94c49554eaa8ae3ee74bfa9e6a9", "331f4b833d484939bf5bdc9666ff0233", "9b46fb6b5ee849cfbb6a756247ebd432", "b00cbb62b1e54a0d917c70e11bb867cf", "2c23bc5d093d4bcf94590755096c2f66", "e4e5ec4560ac465082c3ffea2dc072ef", "1a6f6557e8ff417b8b931ced07712742", "e65426fce5ff47a993fbbdbcf6556f67", "c356e5226ad641b0ac03f0896347b301", "9e833bbe916a41c793ca25e1183d6fa1", "a18166bf3764452fa21ed58f230383e6", "f4ca4b1668c5429f947eca7a6f0336c2", "4f15118a28224addbcc9ec16fd6b7766", "c7dc3be36a29470988eb504acf254719", "5edb891a6a1849bcbd3b7048d32acce5", "98511d2d9c6d4a94ae539e285a92cb5d", "ac148855896d4cd8b32d7cd28cb83553", "685be9693ac34ae3a13c3c18b620c4db", "531988bea3a143e4acf18b262217c591", "23a038e171d24bff9c50b7d960297a46", "b82a0ec61687445198175e88c9c177c7", "57bb636f62924b848d00e5c65d22f876", "3e3183207b2646298b6e88e99c40923e", "0bd4c8c1b9574be5984521bc526029f0", "775ee2571204422bb171f076bcb802be", "29c43b38017e496193cc482a115a091f", "3a224c1a0ec04075b6c6a014459c9662", "56424531ca3c4a49b6ed27496c63fcb7", "7402857272304a11a56c6b87aba4cf58", "62e9e3f16368426493fb670684ac28e4", "62255e9c5f50425893ed6da796e9d4f8", "a5b12d7ab3ef47cf8b4b82cd13cd7c79", "e997319ca7d241d885f26459b5357665", "81288d826432465fa98377367c0d772d", "528800f657ca4db293345598d89c9d9f", "2d8a778fbef4401eacf573c54fdd52ea", "265b8bad2856431e9f7b39018bc0936c", "d7aecbc7d31949a7b3c67e5b4a96c4b6", "d504e060e26b467a806a41e4d0ffc1c5", "a9905d27d8dd4a048d633cdca78ebca3", "621b8bfe03914c08a5a6da6a2dad6f2e", "feeb7c1386c44b3985793377590a0dde", "78b554e78a15407d8f32d3a86e810fba", "0cb97985f63f4cf4af3e626a7f971fca", "1cb44d03283547ccb690e2a27a25111a", "0b2eb977ca3d422f97f678019c47a294", "8ee616c453ed46659c19fadef0c0dff3", "11ffc423cf494a20a1d6987e0d084e74", "8491f0401f5748c1b1edba1dbeaa4be0", "7281a7d6d633442b9c9bdac0e6079796", "53cd32123c6a4c45b0c7ec05b0af0507", "92d79b71528f41ceb302ffb88490dcfd", "073070bd49b44eeea3e720c493736be9", "3fb51009961d46d8971f2ce809710907", "12804b3902c6449c9897d741d48ebcd0", "eb153c482ecb4b4e82b03909fbf79dbb", "4eaf0ef36e1c4954bf3a7a8a3735871d", "39cc08e02395429eb95d5006dc1627a1", "46fbd1af199e468da9b61ee6262f4db7", "f18121fe0eb7467c8af93c2d3411253d", "0c8f78fef3ab4a7c9f9e6a0859db78e1", "935406091a5c4f9781bf4e23e59fa24d", "7c368dae5bed4ac083f5d4c729cd3604", "db76ca97f33349bf887e176fcf607907", "f45a66b0f4a54dc089e8abe837d01bb5", "12a86ac1a951494894c62e895ef8aa5f", "015274c80745457f80ade7597c77c37d", "192311d647554f2394fb843190b57e06", "6afe7f2e65c14c8f926a6e2559dd82f9", "62eeaa061ea647f68b276376ebfa5db1", "107eb6cee4164710bdf8b4f3d0a3af4f", "18ca3cd5b08d470b823f0d70d2f0e1c1", "1561500c8d1c476fb03b6a055a715779", "c2554190ef8b4825be41ed1499321b97", "38664a5f5cb545e2b0b8e70709117b32", "dc8be5bab7d844dcae6e2604dc81d80c", "5c8cd7154bb24f218d1b21484795d852", "69908625fb80446a87cf1858c42871b7", "0509ffacfc37414cab2fcd02b2353e8c", "7fefc8a8a6bc4ea396a804ccece597ea", "1a6955dfb8bf42dcbbb3761b1d6697b4", "e402e489c3424c22bbaaa65ad169378e", "9b4ba6ff51b6431ca055b5e467f40731", "923d4ed3ce52452db2fd61c5ffc7e67b", "829aeb458f524e3ab934fd2130ac1f5a", "fd36e3e97b134e2ea5cf0bca9705cbe7", "b6148feb383841ae869c8ce24caae339", "05ade82b0d4143baa44671648dba93a8", "017d6c400b3f46c2af0c72a13e1fd684", "68e3a704dfe7447e8843544e7c74c823", "14903b6e3382465f81dcca45d41e9cd8", "916e4d928d9e44c2b2ed2c3b46d7a1ec", "c83a03f45d294b5da004bb9000373010", "329718e52b0946efb8764590ed181093", "8c78e3beb4ff4ddd83368c3ad9d83714", "6d71be1925344a878ee87fd027e55274", "3ee40c9bb6cd4174be9da94e832770c9", "a391409e60d441fe850cdb9ff08a204c", "25ac6f86f32a4522a0a1eea860ea0b22", "9b792b26701d477f82cb48dbb7e2fa97", "f6b357d2ea9746a18b36da9e6aeb5807", "310ce8a468cd43d68f6ee9bda5e0aacc", "ffd491e1b34a40c6acdefa8070352a43", "0d1d72e171584e00aa3936653db30b12", "70d0d6463dde4284a5aef3ed9417131d", "969546c1171240cc819cb811802080ce", "36baaf4c84884506817afdf89433884b", "1713ba6563004dca8e924b9466855e4e", "bf58b99158ce4a01a2014199c0dfa305", "78125cf527734ee2bcba0a4858237976", "d46f00d5db184bed975691337681f8a4", "52f1c85fff9a4869b087b6f955040fa4", "89e26cba067c4daf9d993f3c612e4c02", "dda30c18929d40158d60eb4b58c49e67", "5eb37aeac1d94badbb120501a822dc4a", "56974e8e8f774bf8bf441dc4bcddc03a", "142dbacdb46942ef826bb5c64a69c9af", "3f2121aa6bec48d69cc08831e4235e11", "d4331c1f07e44bdd8f77a1e3e4ceb66a", "8f1fb9c2ae47430cbd9490c55625bf20", "d669a420d99c4a22ad750d0b89aebf2a", "df7c430480d54c398104cd40b6dff153", "28b0bdc04ed04a5b963ad229b46afcdd", "99c3352aa1414909be3c80d73e5dd77c", "38fea81936b146d6b4660e2c4a062f41", "5c030534944044d48585df2b5c47de31", "cf0e416a3d024a0e8644b87a2c959aff", "abf78abaa9b144ab94316fd5eb9c1e68", "c503429cb263451ca0ade7c7fd77498d", "c002b1e9cb354b7b880e993ceca60b70", "73a1c0eab3294c2cab7bfcaca959b760", "dd6981cc018a40b68204efbf4c046a70", "d9fca341dffe4cf695464b4989c909d1", "ed47143562174152a1fe72c6f656f416", "e4c5e451ea2f462489ed7152cf4a2e52", "5d26f85c20124b61b7b22b2327bb1539", "aa1ef03ca31b460cb2a6520018cee863", "6acdf8bd971a406897d1f80e0d923709", "5e8eb8acea224b1c8ce3c213da32917d", "88cf8c76a81344be8c1c04625f9f627b", "fd04c990f2be4d7586f2ae4a68a24402", "d9b6836005914eb2bd682cbf919b644e", "dd0981c702e54d9582365dd6177795b6", "fc9a3cc76994489b845705fb3812e264", "9e1df18f83944b2d9ef18775d9803810", "105c992105254b50a9b3481fc675e9f4", "4c1d886ec8054baaa8f63df324a20eb2", "b864170abeda48319078bb669752eca8", "caccdac7e43b4971b809efd48c8354d7", "d147d515fc9549b4b6a71e42d7c709b0", "e47c6994d78f4757b1de76e5269317d1", "9524fa49bd254dc7b5ece9afe33797a7", "c5357621b33541a28da5574dd1971aba", "9f92c996113a46cc80894bc09dad9eab", "fe9ce8f415204b8da9c4dc60305c911a", "ea04eefb0e7f4ab5a0f7cce12e848e69", "490cba476c9a4df3b20252f5ae897821", "10fca555dd4d4302aa6a7138deacc77e", "7fd14b1fa68a416babef74b21f38da3b", "4c77aaeaf603455db4e0b43e980a5712", "c8d197fb9f974a77926aade12e5db51f", "8d430de26bbc4b338081d11b888f3a41", "8524acdd1aa046029c8851921116d855", "fa7d4711012c4a939d3af2a393387d24", "f5f563de9a7f4b25bbee01ce0338f5b0", "ed3a03239ee94c0aa746d18d9c60ef02", "d509b00a12ff4bc2b35cdd347ade3787", "b8ad96c6501b46379e3ba5218e4f25d5", "fa20ccae38304c3f90e41610a9d88776", "41429d5cea90416fb70002079562b00d", "4edb513505a8404bbb556c9c7af94de6", "395d891d7f5b4f0fb632d6541feebf9f", "5ebcea8a04584cdd9b1ae0cf321f8407", "db832bc7a8c94799bfe144a43c73c135", "469c83d405764ed88c0c70a79c0152d7", "83fd4c77396449f08ce98dc656e663e1", "72e38954b4904b809f4e245aaf77e46e", "436435a79295459fb9cfce0162d1a382", "de314aa4f4d9414d8586c14669dbd5f9", "be649954a59d4fe0878ff5460a2f4a76", "0b19c8ee46494fca855235c06cd955e4", "e91e24d3a5014be69f768fcd00cc3b55", "083642c408cb4b58ad6e80f2a8c4ef06", "a6beaa50f2404676bde1c0547990106c", "058a39cb38d242aebf46cc997c9c8cc5", "c2a44e3406fc45729b1b34fc11d96aa1", "63e4db6713e946ed974cd44d563e0a36", "5f895ae8edb041079cf7c1bc15d2ce00", "4d8c8b72d26d4a5286e5674f2ce8849d", "4c8259e27a394b84bccb4c465057662c", "6116813210e047d79791d4a6a623cd26", "ed1f6e0f35be48af80b852146f6d597e", "64e526a0b7d04225951ccc1d678e76ce", "30b44cc040104e84b34779201b18bfbd", "4904315a0bdb4bb99636559461844e3d", "3a0dc69083c248c689260a55a370dc73", "81c3038b79454ee7a833a7e3a40a17b8", "be0ad98e80ac4f4b89f890dd47f828f9", "3e76077ad312450b93798c0c17d37623", "0435076208b242adaa78dcbc96f7d163", "df8daac2864b400db692f08a9ad2d4d4", "a11601f2a45347aab3de517e7af323f1", "b0efb0d0ee2d46ee9cf8fb1ccb46dbf5", "53a8d13b6dff4bc185c8ae7c8abade70", "52bcc275acf440eba1e6451f95c45031", "015e6be0ab4742ee87287cc81ccbb92a", "ed533b0fa2184f469258beb28017b31f", "763d843da37c40769c8d50ba60a2b0cd", "2eb5741efc844dc3aba17d1b15833bc6", "7c29b222dd4846f2af39ab03b0f76d3b", "a073c07ae7d44d288f1b7e5dfa5039bf", "b2773d6df90a4479be31b3470e7eb9bd", "c65eaf77869a49bdb6cf61c1cc8bf792", "a29b5671e54f4bf5a5aa2ac804a8d6cb", "2f0468208ea146419d97120885561882", "9a9b84470ece4fac96dacc3b7d11eb9e", "823ce1ad361c48c2ad6645f080051447", "7f8e033b8c6643d8b18fe0a76346609c", "44aafeb4f0254369b9e02566d16e270e", "8c3ef235208d4bfc8734650c90e4e0c1", "4a60de8ec14c488d9b6513bc84992995", "35bcde38755b4f37a5d695f148537ea4", "7a5827db557842878deebf7b284b6328", "b804289c07db459595d3fb9112db8940", "83fc3c60a4df41d389c2a2b3f568774d", "b8abe62755054d20b2912df08f844457", "892bfa233e334f059fd018cba6cb8321", "9d44269ca9f843e9b096ffe6e56d69d5", "cca80c535f6a4bb8ba5c1b6d82d2071d", "667e37cec6f746e792ec9ed8ea01e9db", "5d436c7cd303470cb30007e726920d68", "3f2b15ef3c6648509b019d0d3aec5f92", "e58267c411a04993b9eddb0bece92ab5", "3f31e45a8be445aeb7366929b36d915d", "262bac06e482412d83a938207eab7ab5", "747ce8f5dac54513a75a834b229f09b9", "43e695b4e21f478fad0710aa74d28489", "06e13fdf5d864991afccfb6f7bde5f99", "abe60a46028949a5b58a730eda379c9f", "8951e1a5c5ca418fb35ceb32a8b21b83", "79dc31cf8470483f9a358f9128770e60", "d5a2496e254f48d99505a83e19dcb094", "c2bbbf7e41324118b2322e57c9594709", "2e9ab4b6c4514f29a2525318f5db477d", "76d595566c434170a8cfae2a31177f99", "8037f88750a24c7390904ab9cb4d966e", "914b56c3e4e04da291b15d821fd1a823", "a99ecbb226e24f16ac4931eb2701b9da", "8cfdde3e30d04e26a84a07ed38c908dc", "e18aee2d6ca44846ab8f7a8f9622408e", "95a39e3f30dd45f8b1e4494a718174a8", "1e3e3ea43bf64df090e3102a0a38d7dc", "9b1308dc012f4c3199ef15a03a85071f", "8a9f0d2afe854d25b6f6c8b3cadfa0e5", "213f650cfdbf4ed68330a85b8593a6e6", "6552da4d919243d581e796751d0e85f6", "2d6d0a631e234f88a171755fbcd22047", "db1a5bd57ff944aaa6c354b4afa041d3", "89440495d24f480e91eacedfc5564fae", "28e3cd155f3642ce86c192cb0871d07e", "a1f572c6e2cf4aa792891ddc31e53de4", "4f22a0091cfd4a8faec13ffc61b70ad6", "1e44ee6443e74086bf5f20eb6f8be7e8", "81b47a912c8a4e9eb058c9a0075bf41f", "8951dbe41a1f461ab541dc2e223e0df6", "f36e5fcb19464e24a0b05de2075267a7", "aba3a5540cfe4b45a308b917885f6a9a", "4e6704614eff4d779e76033748de770b", "07dd9aaada9c456f979e77ff3a3425c9", "3d7ea5ed0cdc4e8baf9565099b06fa76", "d8d7aeefe49642c0aa1564090620edeb", "2587164f6c774e8094869206237c8208", "9241a662f605423db20cecc3314bd38a", "bf60a4012afc4714b253ea14e9449f81", "a42fdcf3ebb543aeace711aa042b72cd", "19711bd425a8429da55321d0c53120bc", "6998b80161f8490f88ef6f5e6d1c185f", "bda176dfe16446e7b52b222980abf1de", "5e62c55cfb06456ab183fe7cc6774e58", "68bcd5531ae44971997cb98a115d122b", "2bb1b108d7c945aa86434b51a37ca210", "2b0a5c29443c40dca7803b55b7327a4c", "b09c0692d9564c269a92fc9f1d68511f", "c69c7c31dd4840da9e500cbe0d754172", "9f7b4344e922415ba3113435ad70961d", "789f5229f0514604ac54515d7381f397", "e4a200b7b36a44059494bcd14abe6ce6", "9a095f5b94be4824b58882bea2687bcf", "97fc1ca3184e4ceb8bf1ffe99facd52d", "1d4468500f464b83bd62c2462fd7a1b4", "2b502a69e1eb408a96d607c124300ee4", "876819f2d2934eca834bf65504874556", "c06e954ef21246ceb1343de238d5b042", "817b9d859dd64fdeb660ead967f88f8a", "86bf83bd699c4230b8c573238f8dbc2f", "48194c48b5f74ca19a8caee169223a30", "270f9d87187b4e17bb1c785cc5a20e50", "71b0f768e56b40b2b042e55a6887453b", "179238b0feb44c098c4656dc78adf13c", "be8d25ad3235491d88634d644fde9849", "3f3ace3839564f62aaa590f9705648f1", "af2a796a2a064b15b0d240e9e2ca3c30", "8f38d1ea8447478a9f37a9f5a2a587fc", "7f19684072b241d3873e19790bb2a760", "139a18d2f1a24206bbbaf9addde90501", "5491bf01b4cc46d19df116600834b28f", "571db36b7ce445dfb7f784a072985779", "0d77cd292dca4c5fb1540e4c1a8d696d", "248f16982725483cbdb52a6459a30f56", "4981199029e54bd58a00ea2d059ab49c", "36527554a7fb4605be3583874828e659", "7b056fc8f0694b28ad06f1131e922bf5", "d71ad6f4d20740d79820e701acce1b32", "1fbed6f8318b4603a5fd30e9806ac978", "75e37f75e432408faac2a2c1248236cb", "3099fa9298fe42f8a3f61fb7bc3e9499", "d2ca159b04c84712881516a51f614b19", "f47036a831324cd0b3dcc50b828ea35e", "c54b08e936124d8d9d2ad1181423fed5", "7fdb3d04165e4ab4a47a7cdd1bb56c67", "1b73a2bb47d34f3c98f7c045f69e3805", "bfd42a1f3a73413ea6e0ea3a06d8d806", "d29ce4707848432bb362ae2f4e3c1ea9", "692af9ec5c29433889b789740084a197", "aa250d39cad347798afd95098e6f1ff2", "b35896a25f42424ebabe7c03679e3a53", "bfd2fc7494814b32a7e27f341c4025b6", "7a1a01107c054b2ea46735831fe2d3c9", "6a16546fd59543e8891ad660ace77fc4", "c251ecebdf8544128e15c5e21c82442a", "e51bebcc62c64da1abc7b507f652f152", "eaf9212794304d7a986f6fb6cb704582", "1bc588e582d647bd802f65ee95681d93", "187bf80fd7dc4c178039fad2fb44f7ff", "ae23ba02edf84681b9e7e8f8e94279f2", "792b15996b234d43a77e7aedfe14253d", "a9a9173f0c7648aabde7a16331958553", "e1f278a855044f6c8976bf8a5c2b3a48", "f2df54b6829b44c4a01100280a60f78b", "ed197b47420c4309a732453e84e8158c", "75894d2cfda84ac884ec6146e07b8b5d", "9d88e334fac24c9280a4fc97ace52365", "7cb23033a5d1461bbb66c8c6236e06bb", "f44e916ebf494128a0fcf84d926e563f", "cb1a64f672654750913ce8f23bce39c6", "869aeae8c6dc4cfbbd47ae30de34a7cc", "f9b175e59b65409f920b4a4dbcbd365a", "aff7110ae74e40fc98a3b7568f3b2e72", "71fc6eeb4e5b4188b911e445e3e49d09", "ea6427713a0f47f9a8d1006fd5e527a5", "6de56a14b6e14a3eba8eda81969a22e0", "08effb2a2e8b44fe8e09378e0fd620a5", "e4266366a282496ba6c7dab9971a7e43", "15d561f09f004ea5a5df9579a1cb3862", "36baec0b1dfc4a8993eab2152d7e0e43", "2d486f5d2dd245c0bd6115f2a567b940", "ccda5ec02f084eb8bcbe3155285e2fb6", "892ea0ef5d77434e84615519d8849d82", "2d906c2f722041039247b196104b0619", "4588a10e3ba945a2b37ebf0a16b568da", "0bf721a1c9f847a9ae82200d7135f570", "f7563caa5225437ca1251955dd8150f3", "0c2205d448e0446face45b9714db6b52", "589fe0fb7a654a248c8d2e014b7808f6", "a5d0a5bbf0f0472296ccc8ff4a852f12", "3a323c70aef64627a4f56c2ea2ce431f", "bec09a1e3db541fb8cd2a380f88ce5c1", "a30e0ca9540447dbbdee257180c5dcc4", "e8e7ab98611a47deb40407c3a99cbb19", "63bacdcfa4d34455b54b9d0a58fa13da", "4564e2a2a38d4176b869ade9bef0e679", "288dcec6a93c41ce9b8d44020dee197b", "56fbeada8d18402bbb126ab1dd267b25", "af1f292db8e347349e4e325c854a3049", "dbc4b782797e4c1e9270497c1c75c876", "20ba0301d91a476db71e255cb1b3db4d", "7a2bf36ab1f94a0197adcb86844cf8b0", "e0eab64c26f24d8c9c504bd5cd12e3e3", "535cb92aeb054e738bd432729c9000c4", "cfa459965dd64795b5588b50ed4984b0", "e34b0cc5fd6b4fe5894746ecffe7f51c", "72ab33fa288048b8a2cd38724c37285e", "07a0cd5f2c4e4243b1d01e4bbc068b14", "8d8a3dff872740f7830c56db4b8d27ab", "b4874aabd36e4f6ca6edf2c08cbbd115", "0d6bb31b86444803b6705fbcb651f1f3", "93427f5640c240c3ad43ca5b1a436db3", "f7d3b22668a944498ea4e6151985da51", "2fac0ad0b9a5453b9072c39ae04e527e", "06a2ae46b17f494cad63874f470cc778", "92b81ddd206046af97823df78177ef27", "4be01c23e35241728e275780c7a40e4d", "afe36acb8b1846769170c89b99c108c5", "759479f325eb40449c8caa1dd4c7656f", "1ef929f0b5f9471c8cb023b049aa85f5", "85cec86d1ee6490ba0898849018f3e2a", "0b0bea3b647542828b7ac612b271eaba", "eee77879320f41deb1b64857f9916cef", "d91bce6b0b7c45bfa9f8a7fad905bd83", "a44e4e08ab754728ae7383fcb53b9d96", "89649096835849f99131a96cd0f97189", "2abd020d0b6e497a86c959ba12e97aab", "16479ee7fb3e4ae98ebb23d0b9d3422c", "0ddc34c0ae5140bdac4919f98df13b60", "00111086772c457e936a10481c1f054f", "318a6d22d749495ba8d00c53e9859eee", "4588830048a74676a6c51efedcfadbdc", "4c23b5e3c5be40ef94ce8e163b6ed335", "9326c205df3e43399a3be6e996c9ba2e", "7ea11af32b404c9296ecd418543c49cc", "663d3525b68449fd95bc438cacf433a4", "d3bf3205f6624fc1824bb1a1c1dbde35", "b1a8cd931b4c416b9601be0d6e2cd1b6", "b4c05b42108442f7a19da5ffab2de3b9", "5b464ba5cc184c9a852e8e3d8d9505d1", "2127a7d9e3694bd095932aadc9c5b7be", "785643a536cc436ebc4f36cf5e27af00", "25e0ff401a9f40348a49747e38fd421e", "f51aef87b27849c4a66772e72fdbf7aa", "af6b0c6fff96400e8756b0c9bb88d3d6", "41ab784fe5d54146a7c20d8a8ba1a413", "25505506aabc4c1d9f545a1d805be5e9", "745c72eccb764a9286e8703dd47aaca9", "2da095a78df94837b72ea703e834a7b9", "d19585e623004fc3accb8b51eaac3084", "aa135bee06814615b18253ac5ad76891", "1383d27e17664ffbb211a5ad5899a359", "2d45eb26331c407699c6d52cfc4d3ddd", "8e8032e083b74b92b6d09ee7a6ddf90f", "0fc93ee2c626444da8db97ae447d3df2", "42cd200558b8404999f40a5cfb9463cc", "07024528997545cba041df5e0b55e2b8", "faa8a22add3e4c88b976da0de2defb9a", "c0edc394808e46c5b6287ecca226f158", "1f2fbe5813164ecbbf8379ea47018847", "a4a84b330c754a779835f59f884f1c32", "942922dbb1954634831175b924a1d551", "91183676e9dd4d43872e23450441d3a7", "edc4b5a5c1a34a7a9e0a46966c3676e6", "055d26a062104d399c62f2c0b427e0b7", "a58e7148878e421e8135044500804910", "004c92d47deb4782a34d523c5c56a896", "06cf2d10bbec43c889c4402d9348a47e", "f759766531d44173a19b549994462875", "b91c296041d04b56a002c7bfcc1ce3bc", "7cdd8dac293d4a428d4d7a379305799e", "54119d28c26a49a984de28b2aa6b7aab", "79ffaea8cd47497aa92e22970eccb840", "0c20a9ae3c254a99bf91407c5cb5a9be", "105660faa2cb4ead97bd2a887764fec0", "42ed5d7ef5304274aafea075f479da09", "69784cc5c94a4bf9b6460dfa1b52d36f", "a85369a2d5ed42b68be94063dfacf343", "64fc43fab8c340e3b11422783f5a15c9", "09469e0f921046a5836b6f035cefe7c3", "253b6481501648ce95a4144b60b86dfa", "5a8e3347151841fd9ef97a3abba15404", "df0aa196f99145a3a6923e7a7185c4c1", "543e963cc90f41eca34d227424733b80", "5eaabeca74904c67b33d8fe77c2bb0e8", "6760caf3d5c8471e83433be89958487d", "895b62a3759147cfa209ef1f01875c4c", "06e504fdfabd45de92f47336e04686ad", "4e7486be7a944472a98f450f24130972", "116614edd6fa43aa9586afdac5d11fda", "6e062b3cb194464696e68f5ca450c4ff", "843f013e122b4621949ddbc12e77784d", "035f566c189045ba945c0511579d8397", "9a04bcd6fe55478c92b2431142b22dca", "72cb7280a7ed4fc0928268d750fcbc9b", "d85656ff45b94f7ab0ce6ba2047cf976", "0efb7a0d9b89438683e32b2ed6dda66a", "72817dcbd4694ba7bfe5e0bb37632b22", "cc50a3a4817448e0884fd729056acf36", "e057ed2a409a402183803396f4293b9b", "0c274d3155d9450b88e05534ff1a0bd9", "6ac89e3cd25246bbbdba2a2b9fa40cdc", "4e4b705ad108463ba0af55e33d781712", "3b86f9a8a59947959c891e0a1e853203", "828c82ba53dd493fb6f28ed4599269d8", "82e08470d98342239d87d1683c168f96", "47be235ab3324b3fa66e749bf53a0328", "5412f63e93f345b4a9bae1a88128c8e3", "b85f6e99acf44bba98828bfddc65e31d", "6347a6a28ba34715aa2d7324e66d3760", "c87240bfcf854329bf4461c5136890e1", "fdcbaa7366d44a408163d8992513f439", "4c9848e5a9fd489f939c2f5517b83644", "e9ec23339a6d4baaae24681d25200fbe", "8b83f5b7911c4c58b0c9f44d3340f60f", "a8b36ed550e14023944a52656aedb5cf", "a3389733759a43579c7156475f6ade79", "217d3b1fbb0742799c0cd851709a5604", "1acea8d75ee84b2898bc6e94bc238566", "c1b9fb7ae46c45f881f868c44a9f9dac", "fd6e5010f3614d94ba8804589d8427b8", "6f20242803094dc9ac6e46f9fca6049d", "fad383707b8c4c7691a94c15a8871383", "9b50fcec7e374b9da226c2e5aa9c253e", "57954fa8c2c047d1b00bfa9d107c1f5e", "9cb9256a161b426a92853fcbde326452", "c83ebbb1a00f4b3fb54717592ba20ece", "ebc58fdd5eb0462a8be85c5977a31c64", "07ec7803517646fbab5fa28b99fce291", "14451c43f1e2462f932e33158b4d6391", "cb927b7fdc4d4fb2aaef4fba3da43a83", "2c2f985471c64643880bfac881fd185f", "ab9afd419b1e4b7dba23982c566d98c4", "b36cdcab4ea74355a2b2957485ac3a82", "44e452cc1c77489197d95cb474ebfb51", "461af7aa54ff4f00a3fe04407e5115a7", "aa7135876bed40b3a4c9c312f7cb50fa", "9de1e9e5a3264f8f927f1e537d3161d9", "da86afa18e30447989b4c10663c6d237", "bc0473bb8d4b425f9df49116555793f3", "8a2d018613ab4b49b74682cfffe0ce22", "51cda4b30da746bc94649fb1c7262039", "7941c3e362ed48049ca0e131310b33d4", "e9a17ed7cd5b48f9899046ac64ccbfa9", "b8a274a566cd48fd8cd37a6c907601b5", "1b0139cc9ad44fb49a191f3a511b3759", "f498daa2a21a47fbb1d8c2e662e6e85d", "1031f024b3af4ae9bcb0df6a92fb3685", "b94e3076361e40dd8851e59a8e7e9f99", "fcc91a39208c4847b6bd84a2aeabf2ca", "bd3124f7daed4c67992d9991575f8dc1", "5bd94dde7a904834ade5ce9f3c01f809", "ecd5a5139cdd4348bf5bc28ae546f9c5", "9e642b022e0c42379ea4fcb1f6234d0e", "0419b1d5b5b144519cadc5a6381086ec", "767fc4067ea343aa851239817c293643", "bd0f1f4e2df843caa6d3005725d19033", "535b6ae53a07406da605ae06cf20250d", "81489cc1caa34ce69d113aab348e72d1", "3fee3ba5c3244fe4b6b90cd6ced43294", "1cfd1ffb6fbb4cfbb7edf0267215292c", "513e5d3487a448bd966d1bba30f8bac6", "688c30f534ea4b3c9e29ccfc8609b6c5", "c10763ebf5674e16a466fbc19f508a0b", "da3dbc5cceb3420e88e4e1fa511c9703", "8a6b82cbf48b4000b5e8d1072d0c7bfe", "8fbd121a272c4986bf38e21b1cc4d1f0", "256d55681282440c80a2500ffaaacc05", "e77d1086a6df4cc496e5d3197d315922", "fc202e6cb92e4b51b6e430fe7cd3706b", "2f137da8e05e48109959efcb0392ef39", "7874fd24d8ef49d5954dc3fdacfb91a6", "73a0297245bb403a9016aca3a2951c30", "9ed62363a524453bb81465e33d9825b3", "6d5510da875e4945a42c3ab6472774cb", "6ea43fbea7b54484a30f8c2e4e7ee0e0", "41418a20a95740fdaf4d94e8f8961621", "6d6553d0d83e4bcf9b7c94b849f9035e", "3f7e176c8042492ca94c3130cd8ab3fc", "69d1bef6877a4bc29e35001033d5b4f6", "6815c5f05c1248edaba7d4196391ae2a", "c0329239a6e849458f4e8a4575e14405", "ad926fecdac246cab3d68fcfb3cbec55", "e093d32659d54c008f8fc9d53982ff36", "41ffedd97da04248b2d7fc91c809786e", "1eca7c040a9f4c84bca320326206aee0", "7b919334d0a142dcad24589c94e17253", "96f46e65b3574fd29c685e2f33fd173c", "0ef1d68429de451399cb996c1d845d8a", "e22e150f77a7494b91727bbd0a3b0f6a", "a833a9c4e90d4bdb9e1f791b7e3f3dcc", "dfcb3f578b3f4cc7a064525658dbdbf0", "9df9b000d8c44228826df3550077ebbe", "cb220467c22547a2bd052459d351b7f8", "7b4d8281f6924af1aeb6b8c2057cd61e", "0d35656083fb4d74927e7f2f5a844c7e", "98b1cdbc78f74fe8aa0a6439d04957a4", "7be5fe091d0e43379b90590b42a7c675", "5d53ba13b12743f3a95ca373933b82ea", "9894b02cd8ec4590b35a1f6b9155f286", "6ae855a936574328a6470a2afd6e88f1", "fe3d484d557b442a82406dbb157e3b7f", "ea22fdfcfee54970bbbeed3c81e67e43", "9c86942f5fa54cffb74c4b2d17136648", "86f6977caf19455c9a5cfa4a648b7ced", "b40a28f8b30b4d578db34c78b6e6ee33", "2646f79eb72d47329e88414aacaf4554", "e7ee5f5b9acb4a87a9f51660bacd0eaa", "5afde6672125409ea602b08bc9d6bace", "f7da18a8bd744434b53cfa89325665f7", "8ed8c91292e24757ace723da396c412e", "561ee3dd962e43b089b455c2054b3ea0", "bd832b54ee064df2b109cae76b7362c3", "e7f46775aeb048e8827526d4b0c702cb", "19216bf1bf6a4d6e87d299cd328edc52", "73ef88c0102c4372951388c08fd241de", "aa070fddc62e4010b5ba714999c47698", "3c4bec3303bb4806b9701ad49a51b378", "002487129a464e689fd3155d3a452d3b", "bdd94dd79df0433bb92910f410ad0df1", "eaac11c16e604f478c04de90fc88fd00", "7f7bda189e0e45bea9913b6c841dda33", "15251dda4ce74b4185e8240da0af8694", "ba6f08ce1ad1474b81d976a22d672927", "fba156f8c1f3474fa2fbcd6482b219fa", "7a3550ef2ad24c4c83d283179128502f", "9dc877c23bd846958dae7b9a81595de4", "3829e9fa1ee945499ae68ace12dbec70", "da60c51861ed4b59bb0d42ec8100aef3", "c3d4fce30ada43faba7fedae52db1805", "4fc4cc503c374f8b887ad099fb830285", "a75c3ef61277473cbe64f4c0f5e51fa2", "fae4a11faf364fa68b2246de0f289ee9", "1b205ff7c8de4f69a29b4a2847f756b4", "5c160b8be0634ec39bb68f275510c8a4", "fd8331c0c3974129a4265acae03c44dc", "78018397ccb946c2939cb79ead9e33d2", "e40e8354fc514a13ab27231f1599c808", "9aa04a81a3544aa7a5cbf9bb09e876f7", "e999f684c2ee4c08aaeaf964fd52d9c2", "3de83f88ae574d9e862a0bc0536dc041", "6fc785b14dbe4af69a236826d5744797", "cbe6447ab61b4bb097beb819edd7c30f", "44b639de0080429eb0c63c5802752156", "58fd5f3d8ebe4a07a519f450b48f5639", "af6a5e3f9fb54060a5a7f36ca1d7b11e", "01373789f7394643a4e8a9b14a524547", "f2e40a96a7e340e6a52a12d4e67f394f", "973b216290eb458c8f35e02877c50c3d", "cd2f79311c9b431ea924555da2879043", "91531b63c00e4c17a791ba53ef030558", "d438e3eb207543c2804a03655dd28309", "018b36c2104a478ead9602b6419ea1a5", "8f2edabdb7a048c4b32ab8f505f360fd", "896fb10a10a94238a18ee4b0f995d6eb", "6c2b55396ec641bfac4559a0ee3bb1e4", "e83975613739410faa54f30d81296cec", "391e0bd7b0804257b09b8ece9ee1cf65", "79e212e5d37f48f9908413edb31b6096", "05242e8e6b444c16be8d9fee1bf06b04", "aa8b9129de754ff0b82167210cc0940a", "9c5fe5d015124f3d87a674856ea180aa", "99bdc0c62f2248419bfcf0b2a7c79978", "26c5558b557442bebb3da5ba575f0e4e", "9e76f50dcf5248c6bdc7e7ecc761181f", "1bd12c2cfa02461eb1e1f9a8fbcf5022", "84d8afd6fb714089a6ca7c6a27d6f6e5", "ece47356941b4c4b9c1fd19fc9bf072c", "f04316f4b1e0475ea00b8535d5d19422", "8bd72a069c8d475cbbd832f16178cf7e", "84c1289efbe54965b2dcef592eaa44b5", "3d2eeb8a47aa432db31158ed1c0d1694", "e71f7b0be81640f5abde1d517b9ec7e8", "7c10862b3ee8435f9416977c2a59c17b", "2459df3aefc2426e86bca56530f7836e", "dc18c29cda9c4750a55db3e512f7433a", "548cccdbe6ad46fe936742b7ffeab0e9", "558b0c777cda4c9497b2f1558511e9de", "699d0aa9b16b4d668409811da4f8151a", "da19fa8cdc734c45b77e6defd848c51b", "2e2a50c1ad564fd189e3ec863605a9e6", "f7985faa4a844378af43ce65d475f825", "0f216f2cd5ba4b54986d6d860531dd7b", "2a92ff9204b74beb87c686192651df07", "0ba94a03165d4cb0a208be406c682f37", "eb8af27c8eec4356b5caf146704c2755", "24a244ad2b8e406590d620e658e9d81c", "8cd2310c1dde46f2861eed90b2934986", "0035781db5f64d98899e796a16f63c20", "2ce2d4a028394acab498d6d5c4fa805b", "5f32a7f0c279491eb9defd7eb151bb67", "2046dc74f2ee46ef92e68dcba54b9e18", "90dc922d991149c6b5bc0b2f68306d38", "2e0e0ee184064c63bbe19138afc841d0", "81562388f15c440f82e36000238826dd", "ec97ed520c114844b21239312f071f61", "6cfb8cb0f5f4463e958374e2c0eb5811", "7c4ea615ce1d48f9a7453b071b7fb487", "a6ecaa10b2874e199ae70eb0a7695fa6", "b3f3db29774c44aea461a5e8bf1afec3", "877a2119d94d4579ae6ebf9ef292ca4e", "4e53ba3566a14956badaab0cbaa43ba9", "8e528350193b405f82f3288ac2a4c46c", "b39b3ded88314af08e44542f63934f4b", "80a462d31ea94f1c926424710b9a758b", "d9818fff1a994bcfa3981d92d0220146", "ac46dfa371b94ddfb79a7716ab9e1962", "7bfaade9be3440d49159748bd0eced4a", "59f45d5534164b1ab86c7f78c5bbf0ff", "b0c54bc2cf8f40349fa438be8627caa6", "d0889adfe648431cbe3b1fd11107daa7", "c16115d8f57840ce95a4678fe60967c7", "97b98e3d2ae34cedb55039322c1f5730", "f78fce7d7df7469fa6edc6a96cab4454", "f2eea6fbfe95487684954e013feb954d", "4345cebb7a7342c7b89ae146ccf5bffd", "c9ea6639aad94ff6b15548869c4de15e", "6e145a7cdc7f484f95b2853ec7d19dfd", "2b3720f5771949b480f1d120c1a96e7b", "c66169d500804b1e8151d8589a3aa973", "c8066c59482b4d3bae97d9da24396016", "84347dcafb7f45f488ef99017e81e75c", "4a8c347bf33a41168b6661e10983e179", "6a96e2473fe7432c8af8f1699ed5a498", "78843c9ee4b642bc8d5b2ec06b53fb1e", "37696293094443d4aecea8ef89d6de4c", "d76e56feb85f4f9d975124b410ff1cb2", "78628137ea9542ac91852652459f37f3", "d9c7330548cc49488b3edb3332d16d40", "cbb525f5301843a0b0dcecef06b07029", "ab02de690f8b4d509d94abc016d651f3", "a044c29d09bc40a9b238456b0fa80cc1", "f17f4e0c18b24021a16badf1a792cffa", "237311313ba84817a67ba9b69220e2a7", "5381b41f0f754b15842ac039aeb71971", "2fb62eef3ab44cf39bb7428a623db646", "2f7d8d746de7491faed3c2ee1574b3d0", "07569cebf8da47fdbaa61aa687b984dd", "919785aa65334eb6872651318da7af37", "fcf43ff4b0f34baf91e1af22c837deeb", "62e12ddaf8094386960d90da57e5a8bd", "064a795bfeae49309c0c6f1209a1b65d", "b3493123269a41498d65d7d756e1e34b", "0ccafa6bb61e46a0b732c418dc0c65b1", "1a620811e12e4ac5a50ef0a2ed7a0b42", "69b4ee5000e74718a2cd59692da4103c", "16ec59dcdec3481c87c622561801d91b", "c0271ab08eb74bed969f91e0ed5710f8", "b7f183d321b24802b14728e7edfdcad1", "37f713ac988644269ef984591846fb7b", "d2aa6aa4ac2d4a9ba366255508ef93fd", "da159a7d60b54091a30993e43d5ba47b", "6dd748ce72094a319521f06bfbf04df4", "542593d99d8147039e51755425a53dd5", "6d0efd7a035d468196cdf1484e0ea77c", "cb0d26e0387d456b90c87c5345e824b6", "2b328cca21ae476c853b348f5e51e9ad", "33d41c46b6774aa2b6a065f4bde7bbc0", "0f44917b5a6f4e0389214958464a18f0", "08d4d2c627aa404cbe3b293ca87ca003", "95b682f592154ebd94a913ec22fa8c54", "ffa399ed5f954d0e95396f3f8a4f036a", "f0a6bd7a2b3c4aab96b32867c6e5402c", "a0d4410e574a430995aa2587c21707a7", "5731dc0704d14cd7bd89b6f3a061035a", "610cdd080dfb4947bbf48344b4a4eba3", "5f2d0c81cddf4d5f8e9d970e4d083379", "d569cacaffdb4356b256b13f52890786", "28b9bc07c2b3424ca88ebfcd481f0814", "235776f5d958468aac58f0357d686c62", "606e482748f74cf0843a0d7e5e7f06c7", "f55e65215dce42d9a4c2a71eabfb6f49", "9f957cd2f4404adb86cc810e7c9b058b", "cf95527126d64c3baf9c442f28bbdcea", "6ea7ca3b95844f7ebdc2bfd590bf6abd", "c45ac0201f4e41378b761114a0fb6efd", "a67365a767a14573852f7b214edfd0b6", "c0a61b34af4f409683ec6964d0fd8927", "facf7bae229f42b4a5e6957991971149", "0b1a1381b2d84f8c8fac9f76270eaece", "7774a9897f17407da02098b7f2973b36", "909cb8932ea44a8cbc71be6baabdae66", "bed44050dca84d4db3156142c6b4455b", "c44d0914805c4f86aeaf773f56fd9f8c", "961d79c9aebc4397861e5ba8132938d9", "cb56ec9688ad49c7b7ac9cc1e91a972b", "1d2404e5847946358c95c0c54c99de87", "85aa54f28855437da40275bd89012a03", "3d0ada75e1ff4dd2a37109413c653a16", "2f5b03f4ced84e788e9080c41a76c6d5", "d3bba4eb17f6408fadf5ca616d52d980", "663a18b13db04a2690ef339bc3cca055", "4798273f90764691aea60f0a2e9e28b9", "555736f336d445cbae11a43f4ece666e", "9b93724dec8d4db197570ad5c5b91a98", "b82233ea8f984310a5dd69c8a658d0ad", "455d088add8b47bcb59cf3680c555e4b", "acc526e52d1c458a83d2ba0d4089e703", "d0793a3edb1748f7b8f9db26ff509ceb", "62c3af1968194563a69c91d452893a4c", "a00f45d7d4944d2b802bc903659a4a09", "891335d20c3e435ca499f71c589a4dc8", "7e435ad646064b65937cca654b3ce8f8", "4110283df89e4485aa11b190244e55ab", "f842ab2f0f80457a8406deee54700f4a", "1f2bc7c7c84f41a5abe14446e4782a63", "d9231992074044b0b308bf8e412a386c", "380deea3859b47bd86729bec140fe7b9", "bb7634ab41f64f61a273d948c4ed19bd", "3647834eded447c0974ac30bc4ad89cb", "dfe7082dd62942269202f29affbf0bdd", "5648c4bb92dd4d91ac8aeca445565cc8", "ca75b9b220914713b65290282519b76e", "2d2324dec1b54b0689747076fc6ef20d", "20004055572b4c8ba587c891ce762f98", "3530e6a620234ce68274472c61587722", "04040d9eb1e541acabd6fab7369ae16a", "cb1e3b5afdac4d229f7cdeebabd770f0", "7316f960f18a411b80ae37a09ba4b95f", "a335f19ccc8c48a9844f3ee37039d1c9", "50aa67d172b14a75a3d6d05dffa4f26d", "d49a322d8088403e8fb682717fc48a3e", "de60574d63bf4681a0f2ceba746a936e", "b2b026026452485a80a4de9f291d8a7d", "22661f70148642a18257896b71dbae73", "b78f62a7254f4db6beee1516adbfd821", "8311b55f8159487eb1a10af239ebbdf5", "bdd23fa9aeb04462bf17aaea4fc6ea05", "a50c44a18f1347f6b5208d66c578584b", "6bbd7373529044f3bc38f5feb3055f58", "929281c90bbc47779a0e3a41f0908342", "59ec234ff1cc4eec8f5b13e99314b03e", "ee4934e3990746708210d1ae1b3e2d71", "5c0650496e7740e6a107c88b90dbda23", "7a5b3c9d779a43d3a5ae559378a99d06", "6f6b272e8475499187b7ac83eff1b5ad", "af0af93817504814b169b556eb8c6812", "1157c9c6a16640099e9e31c55a8eb388", "9103bf86909447fab93e9fce2a192539", "fa89fa8eda354926ab35398967c39cd1", "d28b9d11f98646908ba3c734e27041ac", "f730787a68cd4436858d5cd21fc56d61", "a2e15a8c88fe4929a12fa5e1d9a16815", "37031638cb59433abf44700d0de47b8b", "887ae750326f4615b0a4121385c00f36", "54b464bb5cb244d3b3d4a1de7158e4a2", "5d117ed80fdd47949f0337a23fbeeb36", "4ff143f086d84d43853c8e2ba80673bc", "719542ce49c54685a200f1d571a61b2a", "fce728cfbc2242a9a8a6aa06590e6001", "428eacc890264eb8876aef9f4fbeafe8", "85a50aada8db46358ee209490e1cc131", "08f4fd4be10f48e7b1818a733ba9551e", "14737451310c4837b983093f0e374a65", "96419e2239494639ad0f8682c071b620", "0ab34ac810e74f91baf723cb5a8a5153", "4a6775f76761474297810ef83a10cccc", "ce1be35c74344e9cbcaec784375fd3da", "d9aeeb8d47ed4504ad8e4be5b1224ebd", "9bee117faea94bd4a693651224abe30c", "916ae392719544fba7259e56260ed0a3", "775405dafb834dcba718d9ff05a04577", "deb78a638dd64c9bb10e593e7b5cd5d8", "bcdd011de4274269a3ee6862d7e59e41", "0956ffa924674d138ee7454326e5a4d9", "d2be3546500041488a30f95f5427ef8b", "41eb014aefc74fb5b7d6dd00f753c83f", "aa1c093ce55c41a88c1ebc409d464ce8", "9de5e968b00145af883e06960e206cf9", "99a97e7f04bf4e0194a0542c3995ab57", "dcab01e712db4d3284857d63f6677bcb", "09b3dec5ca92491fb06eb44851a366a3", "dcd4569f4557456c805522463f1909c1", "b1a19b1089fb4514b4e5863d54b3e23e", "7728c6b96da441718e8dd467cc61ef9c", "47927a0854e34d578cdf1c3b16c4076a", "f65468d2cdf64e80b3ef05f00c751512", "a339074169b944c0b4971ec5754f9739", "c0624e6e622d4ba9848c9306add91932", "1f360ab524344ffb8abb3717cc0a3c68", "2dfdf45008134ae4b6f54945933d66e4", "800895a797d84dd18fb35253152ca3d0", "5c1701a6dfb7420ab346c3e5bb1680b2", "f5410b5509a7441987e63d91c3f8e946", "cf6a59e673a14437abdbc85df66667b2", "a07d9f558af143bea5176e4f15668eca", "05072a0eea3a47c7be243a170ab2bd9b", "1d7af03366cc44e8ac2e7edc96a44535", "d9d21362ff7b471493ec7ccdc5dd8fcd", "208312a2a47d4a02a3ab102c30a36593", "76f4e5e403e94862a1b85fd9cce6011c", "b67666a97942461d850e2f7c6080fe37", "e119536c51964cc0b95abca1c897a452", "4cf8926a48a547398003fa0657aaeb2a", "7f6de813d2e74433acce5b3fd857c2e2", "b8d95aea45d0469282ebc9569b91e818", "28ac6c8b14e248cba0917d678ce7eefc", "f799b61dd5ae4434952127af14566e86", "51f33fbba9aa456fb22d768a6a833331", "f3dff79eb5c548d492b3f656f8fc54b2", "25b773c0d1c74abfba0bda3bf08d415e", "eb4267337f1d48568c5423dc960ed50f", "b7086c97bea6498bb4d144bb31e8b1f5", "0d2560d2fa8e4713a24f5fc7fa9d1ffc", "408eb4924fb846c2b7e3e78fff791a68", "cdcef5e0bda64d53b049d1d392f3611e", "297aa47e20b947ae9a65e8c261dde650", "429bed443e48426982dc2cb4fc43a755", "72dc464ff8d34276865407b8f6a62e22", "ad02547678254ad1b783cbf60c696310", "1b196e18feab4b879487a45433383093", "886f818fa7464f4db79e54ea7c0f3151", "f5fc70541ba747a5b125f5e5881fc1a3", "1b448db079854606b47163742ad3d3f0", "843aeeb5dcc24479a31846f68e89d048", "7264a46669854159a9e4218bb464631c", "fa0929a19f784f0fbc95bb12a226116b", "23ed03302c9742daa8cd5bda3cd1f3ba", "b45d8c701615441f9331235b66b74fe5", "66d7a3a49edc426db4d5a4ae18e70d9f", "7cd7aa3aeb554cf792d0ed8d594684a3", "45f759f9ea3c4028b4208c78e53cc5a2", "8af916a61aa740b58c99a9644c50f772", "988e32bde5774e4591e0d67eaaf873d2", "47c4b49365f145228514ac1e1d8a5c4e", "b6e590f4114d49c99f37ab4bb97bf8b5", "455d0ff7ff564e03bff3f1c688556aae", "f5a673d6688d46c39a065bcd1ddcbffa", "7d07623d66e14905b05a59e3129b96ac", "e89173b2d00746ac87b7116f7936cb3f", "40f3f5297eb94334a84bbc7898d1d1d7", "f2e18fb1533f4f29a95918362e66d1cd", "dd4a52088cea46f2987c51958a0e872e", "327e9db16dfd4a459d18f982d42d38e5", "d97ce11dcaf1463c96ec3b488f6ef593", "c2a2f508734f4464870242e1e106878a", "48d41a216d60466cb439ab591fb51faf", "5b70ea6dafdb4b11b332d6d98846e500", "4d78c625e965438dabaa1fc42e8801bb", "6510c60cf8464d039273d0e843d20b17", "9db3e34af53241009a8ab8e793dc12b3", "d196171dab424eac958b1126530b0bbb", "5cd23854cfc6424685ad3bc6ab580de7", "5f726beb4a4a48d2be4b899ec5fc860e", "53b126b8f57b40d09ce5c295f236b818", "455d4515c7324657aedea11f41efdf23", "ffd483bddc204d8cbb365a221f51c5cf", "2e8ec390e1e04bca92f1abc26d24b7ea", "1ffa086f9d094b4e8adf30557b38582f", "af28eca48cee4748a38c7b81b9829440", "02d627f677e84b0d80cef4d1ed08525e", "2edcb5efe9214174a31b1f21a364345b", "96c3245ae26f42688b83446f35ca84ab", "062b9820ff9f48d583830132a6e280a0", "62b961874f1247caa76d7f02b2300e3c", "5e88427f6ae1430d884637eeecc470ba", "d1448c84e2714f02aa684870c309df0d", "7814a7251ff348fcaef01ca85ffb25f9", "80f03f1ee8a44bff92df6b39e720185f", "acfcda74d4ba4451a9ab3c38fe28bba8", "7111302054ef4a3cb9f6ff227b06604f", "f859d6489c3f4a34ad9dee275f542ee9", "5315264b298c476db286cffef780ad9b", "ba7fc6e299c547d5a57bc6b40b85a710", "1d5e91574faa47a496580d9e00db6565", "208c76bbe708470787d90aec5806cfe6", "3a1f0db228e541b99ca0e287e2a8288e", "2e24503b284943d0b7d1c798bf07324c", "be511acb8d3e456dbfcd41550aaa88b6", "d27e78fc9f064b7e969947a711aa9418", "c8a2119f13d64ba69277acf324479f8e", "9c0ddcd905f44b07aa21d6d8f0731e93", "c6296c493cfa408895226e79d1c147fe", "9d07ad55e9d648a38824076af65af98b", "59386b541bad455c9e3cc3a0b721df40", "c632865f8f5f4a06930f48159a41a6da", "d285efafe405423f8c801650380fcba4", "18b24ef65cd84c9589eff5893a143052", "d3540397e836438fbc6965542d13103a", "634fde492df4435d94d81413fcb14908", "07bf64cabf5944a58365682933fc9e1e", "79ab8695176742f9b08a16e18ec36f5e", "dcbeaabdc6ee4ca6acafb434aa86177a", "b31a49bbe28441678bdfc747c4f6e8ea", "cce33ed8e8424b5384fccc645ebd9f2d", "043643e6008a4f61a98b6d06ff9c2ea9", "4e94bbd2bed845bfba4da558978819b1", "8168b118e3e24582b2ffaa80e99ea584", "671e83d61deb4d1c8093507d984fdbd5", "9061a8aef60c4f758eb331141f29cb3e", "b0e046a983f24b0384e8d9f9ec020c0a", "bb9752b2671e4855aed9169434f44d97", "9a516a53051a43d7ba42663c4d1a0da3", "b759bb193a95456f9caccadd83a25e77", "c88b3bab8cdf4a079959b67747e8739f", "3eb4f5275c5a41018d346963f3f50f88", "5b4f17d5e2604ab3a7a457280e36bc00", "9308939d06d74e05b13a483820daaa1e", "0d2cca04653e4c839a079c1a253571d3", "317064c53b294ad195ea5c50f8d29a8b", "98736fecf9d84ee48db804ff2ddb71c8", "6e5c6b364c824458a192145520ddb8df", "a4c0ed8c796e4e3193cd08cc9f191e85", "2f248aa54aef43fea620d3c0638d62d0", "0ea0f39e23494011b93dda4c14e6c43e", "2b64e88435a8487eb613a5745a1d5a81", "f047995f6ef246519f0c5a256f0916ac", "89500e3889f94e03a2c259c91e057941", "41b323180970438996e4121fddc1b9a4", "69db9f01f82e4a07a48f1929ae1dae5f", "2ec74c90a31d4aa7a73253a24626a609", "ce33783406be4b17ac9bb06ca8962124", "4e6b3005c7a941958186b525dd1c6cff", "795f9f7eb77d4be1b088e1a0e3f6ac42", "a5545def858e4e438f2f9bb84901d1f8", "37cb23cb64564088b444b4829b341c11", "0ad47d652be24cb99a3db244b96476ea", "f4f9c74fcd6840e29fa3b8d3f2c290ab", "d555b5feb3934722963a09412b16b99f", "eaf24de5b8a147c7b4c4057566247b5e", "041334a9c12d4960931d44c317963a93", "e6f51e85dcf5470485f8a2d86f71e681", "037c765342564e6280e485b30e5c0392", "14629303d22b41f2a03da6a4bbedb7f0", "3ef923b65e4947c18163d861a58c0887", "77948fcf69c44ba7b58c328175e1a0af", "ead9bff163844357922e1dcade24c433", "0d928c0c8d3d44a1ab3209b11d5f5982", "2b52abf377364c0382fc4358d223224a", "5011eb276aba433891d483fb5e30bdc1", "cfb16faafb9442c68a25f219de0c6181", "712a79e4fe654042974329a4754d28c6", "a8b09b437c084f89a8a58c1fb0a9ec0c", "dbfd2a72f42b40a0956200883877b104", "d704f50baed5484cabd5ae4b06ce9f61", "96f4b07dc99c4652ae38a4b3f5083905", "8219a8eaf8bd4461b7777d7509fbd5e3", "5e3cb9dae7324d96919d1ad8564c7b5b", "45a7cc02ded8448f955f684560c86c8d", "aa3da435969e491e8a31d5ff7930dd31", "d47283b82b41463cb997687bb1607bd0", "2bf7d25871ab4170af69d5ad220b5437", "4aa0f3eb7f644a2aa6e3870ce3c5b71e", "6f51fd38f6ac4c23b0c62b63732528ae", "8008d3b0597e4cdab1e94df954dc0821", "9584d58f5e754ecfb6fb109545082eda", "58f4e8958c37492692c8ff23a7fba0c8", "19f088a58fe74b879cd05a5db7db0a93", "7ba3356065b04768a05451af3d88352f", "c52ed4d733ce46c7b321f3933b6ae565", "021c78ec8f3a4abc9f6e95474c55bf2c", "37570978825d463c83b0b89d7943bde7", "3d6e994ad6a340daa6260803b24abdf3", "b0cfeba0d1bc4cdbb3cda46a93bbe563", "4fe9218c2bfc46d4a90a769d83018c77", "7720a0ef64264a66b183a92891aee934", "a5a55f2152df42f9aacdc727c6e272a5", "966628cbbad44ce98bf9c8911988c564", "02b07e055f5842fea00f27469e2e4d44", "14e53bc690f64e8c8e65504b8ab4d315", "0b1507675f6642c386d8604dbfac4c69", "8b7d8f7c8d47483b93859ac6bcc07678", "2e79166e6004468e963327d77877ac71", "36ea9be60be14f62be46c682995248c4", "409e669f356b44b38e0b33cedd1e4f58", "606fa33ab1fb433facefcaa5f1703407", "afd2f8e0488f4e14abaffbaba4cd3985", "040f025ada82477cada6daa8e05d9bf6", "7370cda0a9c54d7db4196ab3f5d43d76", "f5fe4de98f2645d59d04249c299c4045", "450002a4dae84e3e98821458740f8808", "8d66baf7e7d148c8b2fe9472d091b28c", "d7586fb4b33747f1938e2b54f9b7d750", "de442d8bb02f4682a22ff090fc505f53", "e5b4ec4c63c14997bb3cc4fee4201543", "9a03531aa159460bb8ee18f077a67c0a", "9c0567990a7e4811877ac5b8bff5a835", "0fce5a9127c64742aa8b42136b39c745", "450ede5431f948cf98ba9c80449a8938", "a3b3fbb4324a44f59e68eed1df37d60d", "9d78ee9f99c143ec9f51aeef74ce066e", "fb39b3ac44504441823d0ba60197e5ac", "c0ec481d499b452c89d36a58545a719b", "375a60a4954b4c7e830a6494f2dbc5a5", "94e76ea3879645ca99c7ecb4d63a29cd", "f882a8a30ac74a9aac60b678fd1bea21", "35cc9ec48fe14f7697adc1d21c7901a9", "f36223d599b7430ba6a2019faec74ce0", "a9c2c82b09c6495d8e2f7006ebc5076a", "ba41a49276454fd487fad809466c4647", "81baa3fc90034bbb9fe9c6660717362f", "e599513d26584d1595bc0497a9c53e14", "54153f9079c74645b25d851fe73856b0", "026d1eb17e97450fa4e3042f62030fe3", "fdaa63c744284db79d8d559ecb251d3e", "3e7f67dcb1614117bc8c8790d311ae3d", "83347b14610444d7a49c843f4d3cfa84", "edc50d6d10cc4284b50cf9811aac9891", "0c82aea118494ec4ab63f1d20f5972db", "7a8df5fbf7ea4d078d653113cd855c46", "1ef00d005023417197cf689dbe5515e7", "e99dae6bb5f940aea3f3fca0583c2371", "7d9537aec6bd42588568902b1f823ca5", "878e2e8dad824c2c855e7b8b7217c1df", "222d68bb0da34cfe8a9e2d0fe17777e8", "326f1aaf2aab4af0b231f1acd3358779", "d68014f90f37490e9fb9736d6aaa81c5", "d3fbb6dc90264968aedfa153e4d38263", "cd4ea966b1184d99bcfc225265c46d9c", "4aa98e1a2be848519b76ae89fbe2036c", "f3edcd439f334951b4ee51adfc699472", "9ca6833c9b14466fb16156e35076986a", "b2cbf1f1ef4244078faa9572d602f12b", "a752a0f9fb134b43ac9b492882ad0203", "94fd6174b00e497f90ff33e96eafa678", "4bcd46ae5db140258183797cf5a5b2bc", "cfb26dad4b074a598696c38f37d80237", "bb69e38b03104b199b522e59ecb4089c", "b0cc0afb524e41ed924427f2b7e26962", "f8026fb4946048368d9bec43885e6e9f", "6e220b9b88884e89949da5ceb7915209", "900ec560c9924d62a56e4868be6cb23d", "4e2dea6659d84fa58fc1e1f8aa8702c1", "24433387f6824b12a2478cc2a593a7e3", "688f087ba0c0443289c5d07cfe3f596c", "8a5f5157162a405fb06721e6fa492771", "c6d595753169443aa1dc75994c930e2e", "cf888a72d63b4ea4b4776685a075d5c7", "75af7940d65b47169acd64aeefa81c4e", "d77983098c8e48b0ac42efcbef67cd29", "f7222a6e77d640df97e8fa620c7ef936", "66a140981b3a4bd89711220e03b912a3", "fa35511a2c834ae398da70182d07fcbd", "9e349ea7a50f4fa98a2e5fca32abd190", "e409abc7a9c44d8abf392c5bd54749f5", "12e88d08a64248f9beb5fe53a8cabf5f", "f1e33149074c4740b2674b37a4a75994", "22c02449b8f1453682a382288a1df8f3", "395d4f2a84d742e997f3a6bd03dbca2d", "6aca7b18205e4eaf87ebcb4dea74297c", "9505b29628364967bff52330c9ada5b6", "da9e6d32373141569e33cacc006c4651", "324726f2ea364a8da608e85c170ead8e", "6b437d7372b14016a0e3a0afb3f6f4c2", "37f9ec055a7241de946624f6a7dabead", "30b180f5514743d98f89638ac31540f3", "7c5aa3e794164da4947fee418274b41a", "6283a9f25ddb4052a7e7e91c5833f6c2", "bb0505796f5b4a87959bf9fcb539418e", "31475955e7b544f582fa986dbd26b6eb", "e54e52909fe840739fc6f2f5862ae3ed", "a1a4c0cb3a644114a8e711af73fafd01", "448315382e0641d9b29c78ac7d6793e7", "6b482f4855ed4ae39a24ec863c5e9ad8", "5a62498d99af4a879d0d62d8ea74ef2e", "229718cac55849e78e12decf2adcb8cf", "a9673d605adc4b04bf73b46a735f8cc9", "721fcb71a2ee408498e8db1eaefb357c", "74dea826a5474a7d9d0a6423e8e6beba", "a48029cbe36a4a43a5926fe192888655", "e591f00867704da8b795382ab8c31d18", "843893b8c8514e0b8fbd568eba8980f2", "987251e0320c42b5a3b74238755f0fae", "aa652dffa7ac4d04b9dfc80439c0584e", "c59aa9970d6e4faba3894cac7c4cb47b", "753bf4d5d5d142309acb3efc2cffccc6", "05d6cfc6d973437a9a11597be1778184", "54abf38af52a4091abbef8984c454098", "4c892cfea96e4b20ba89221be6484b76", "61d76cc0ed3a40ca8ab41604e8be7d7e", "1196c5ba164d41b2b6bb8d6d9d95049c", "7e85e402f71848f5b54337fb92d02581", "62ad1a5f61354591882b280a200fcdeb", "b0e6342d7be043d6b97a6fc538f598be", "e612941a91664c9ea5d6931dfb64c597", "2439226f9f234c3d947e4cb62b731ab3", "faa1b390ed4c434f9c5730f018c43165", "f891502dc7e44e16ae6904abe71ed4ad", "f9247ce2275c45868f9401058e135cf5", "29c432fc0c8d4b429990f6f4d995a1bb", "b79cf78c2da1445c9682388a82142001", "d84a68701bbd473c827729c30d23515f", "c020edb9dc984a8e8084efcf7286b6a6", "b68633a04bce46a9a627761eca98b1f5", "9cc5be74b79a45aa8b8c7ebdd7c055ae", "6410f29b79bb4bc4a161fa4b54157e5b", "7e7efedc1f9d464c905ed257f993f8f7", "07cca34ef71841de96fdd739b1f4daf9", "f7d7ec4bd3f349a68de76993becfd77a", "af746474045e4b4782247d183a632eb8", "e41281a8aa844356b78452ebc4807632", "d43aa60863ca4fb5af4e68ae38af2777", "cf1b143a51614213942ccb4e86fd5a68", "4101509ae4944aaa8b662558bfaeac4f", "9111c410dcd249e290526c22e3b2aebf", "550499657b6047d88d236177d0d127c8", "89ab6c874d444fad99588de092e229de", "7c48c4581e4b494993334b3d4a9768f0", "d8f17532f8d44dceafc6a7e7c2eff192", "7b8ccf50783646d0a7fc24429bd308f6", "058ce5ab4ad94c9e9833b68f2ad54006", "729df9e911b0473cad4d25211f92a6dd", "061a0f79d61b4d13bd1860241a649922", "124ae21fba46423094475e7bf7720a47", "351407f538964921befcdacaf7d5ffc6", "80f9b7f17f6046ffb50fa4cbea511b3b", "c2f14c6c8324485a81aba3afc783a55f", "8daf681a70f449bdb16fb1b057b95db8", "5ef6975718bb474587efd7bb41a6307d", "39b4014fdc9b4a25b1a697d644d51300", "11177df6c6d24cfeaae1fed271c7a08d", "6692f32a5b5448ed8dc23de9c13e13db", "de3058f358944f848798d3a25675ea46", "7cc04cb9de784a9593ac95ecb7d085d0", "51d063ed42f545dcb59a92119bdc3d9c", "2b6a3a4be48344fd8d1f311146c7e5f3", "79bee1838b90470eb14a7c38a6eb2e22", "0dfb1fce382e47df964b625a4777fd3c", "099fc8a3bde140f1a5c088b607333956", "90dd60b80312428ea264e7aac04b56ba", "3cded05c537c4bc69f2cecc1ae783df2", "0a6633a736a349caa82972fea1168faa", "7fc38d207d6944419943b4854206431d", "6fc3ce0d837843e1af070d269625120b", "875361d0c41b430e81c6b97c2987a5fa", "dbac733d7a524105a7f85e2a50dbf66e", "022668ea20f34f95a01effc42bffc361", "ffb2ef71b0d1461d80b0ce7aae74164a", "ae11cabe5806430886e9b14631d2822b", "dbc34244daa4424585749ca41da04e74", "7737da396ad54d26b4f4adfa0094aa5a", "052cca93769f444eb66b6a26dfd9ace3", "03767ffd8c7a4166b6314b0a5b8c3c27", "4399451c09094f15a827f17b97b340ef", "f1a9a5100b7c45318236fa861dd944c0", "aac14e929ba144bf8e474b126479951a", "12b7fa3a69c64a4984703ad241da3d9a", "ed51127808aa4222917f0a2742753f7c", "c72d1f8ff38d44f3a4582e042253eeb3", "88ac9211bb1d4c33870b3038b974191a", "a83592909062456996ac8ace8df4d139", "44fe25c516284c51a4dcca00f3313ce4", "17b66d6324b44195a812894760fd7b22", "5df66e78671b457ba76b7b6853e5e3c1", "35cedb3eb62a4e429b76e4f597a0534e", "ca59f365b79445dcb081d7f8eaa65207", "de04d6a3eb464736b411ac920bcdf3ad", "b9cd8fbc473b4dbaad417823d263d40f", "d9c920692b264681a27538e9f6592b34", "6df5eea04c03425eb0d30b49b2d843ef", "890ac672f6db4bf5bee33e898d98d7d8", "de24370fbc1141a692fbf935a6879c16", "18fad02391484738bac700cf4c25f209", "25b0249fe7944e14846fa5b506bf2be6", "274c5562cee6410695b7d9fc8f72f48f", "8acaff00e05d4145854b730d49f6312e", "8d3d6194511348c383917aae3e084f45", "6c16df1e993a41159405587e43e4f6c5", "a925caeda9764870b23c1faba2628235", "808fca98e3b244b79bd7c2ef06031a1f", "006d64b5fae54d999b3d9e31c1262713", "7db73ac2b245442b818323fbfa11aa6d", "d121e1c50a4e42d19b009508fdd1b5de", "1166d58a994643c3a51bcc4ef5f1634d", "215c31b0f3c6409fb77692d7125c31c8", "36de09badc3b46378d24dd4f092a786e", "f96a7b410516457c99dc8e597ac149eb", "e730b2cfddd84c7c87fdfa74751a5727", "2783c93f52c64ea9b409de4213bb3e9a", "befaffbc6dc64969bf9639c1a8d32a57", "76f341842f524a88bcb42699f2a0d4de", "54b658011a0f4175936f1e9ecead3ca5", "69b1205f0a554340b078035fcc809df2", "5309c16f12f2421d9682553dbd4a2c9d", "46c06ec80a564d61b00d4f2ef6c95586", "a7018be7a3df40dd97a56406e4f4face", "a74036bed4294d1dba67f5319b33b9de", "efba0223386b4331b58458858711ab5c", "1b9388138f2a4d498c5e01757dbc76d4", "9b8fb46d4e7044d19ea58d8eebc69826", "0327198e5c8d41e7ba9ea6836887dae7", "c24ea3595f1b4b6aa8972c7bb4cf02c8", "42bc2953e1ec48c1a4a0de74f82a02a5", "c895bb8406c54d599ee4ae0e11e9e0e6", "fab4685df0964858b2f2ffc5c8133e2d", "703d11495d0f40dcabbc2cd7a8185903", "5eab34f9f01045b592dd4a0b74f53bba", "68bb6b51c4aa4d0d86cc24f036311dbd", "81c1badedd314ff0a44f09e01afcb37b", "3ea60bd70410469c963b6b6137a950cf", "c4267955517841f6b05eda8532d3c018", "4da73f4763104e2bb9f80d45bfa2be4f", "4c8953852dc24b78836578a132352aee", "f2cac1fd4f9d4b8db61b5ccc922f0509", "55d173b79eb34b7daf45d2e3227a8f93", "82b651e4300840a3b40f888b0ecb7cdc", "1b95f18408b04609bc10112a09c93d15", "0ce05587a3114465b3217c5b43dc814d", "82b8081e215541509c41b50c9c8f382a", "0ea11aeb3f3845dd85e7688b11cf83ba", "c352421a489647b1801c935524043a04", "ab3cac3babff45a8b898bed268e2b01a", "d8a7a8dd341244b9bcaa4dfd2730ae38", "3c18111cbdc145a293f0b945e392919a", "dd870d07ed244045b2bd4cd42a4fcb29", "68d315efa90a451390ede84f4d3bc116", "2af9e36ab41945fe9e9a6f3c59bef202", "beedca8f962446e9b4ceaf89631f3995", "07cdfa7cb57942ff8f1bbcf2885d33dd", "219f3ff017cf45749b72ae535c81d168", "eae78ee678814620a3a996e00987668f", "6358bed529dc492f95efc59b29cd0a05", "48d99e9e18f74445a911dba29b27f0b0", "0e4c4f4b535449529a7e8c58eb30f0c1", "e36092f6119b4cc9a5024a6231ff983c", "59f57ee8d5b942b5b79c2c9cee842bda", "525b885d819c4c0a8b40783572434fe5", "3e9e6e3e3b7640f68be1065b70eed09b", "2032ae446f704d2e9880cacc440fbf03", "bc6096dd932d40ceb5ed670dd9d6a23e", "d01b494f92184737944f3096b1572127", "a0e1f4decacc4f50acab1b3b76a230fe", "07069dcdcc8b450c84782afb9371ba07", "0bce15f85f7044c5ac64f43df6d7476d", "8bc39084037f4fdcb1d96b4feeeb8c6a", "2f96587f5080491d940275fad0377153", "d2f0c52c82b64d53aa2e866da86958f4", "13161ca53e6649cbb0eb840c350f4e20", "b53246ca165f4daa877db18c39f85992", "556f579e6b444d0198c196b60b0b4ffd", "041db86e7ffe49b8b821ad0cc8be37ea", "57c5f255f73844a3a19ef84cae750235", "a53d163a61ef4a6c83be9ce4ed7f2f27", "5310ca80625142a38c744759c09f9f00", "d81eeffb3260420b861132afc87a94aa", "86c76f9702b6465182dd902c19b52489", "d07a58fb3f474dcd9fd7e71e446fa84b", "27e3be915706414483ba7a19ebb8f7be", "a4ee31355884471a8dbda7687a8cf69f", "fc557b8fb9224eb08fd60665f3d93828", "ce326e0813cc401687cd422a3973d0a8", "9252f7048f054a71af7f35e2647e4310", "025bb49e09c54a8084dfbe66f68a2a97", "984804abd3314d7ba99cec7d2e6d8f43", "b720f2ce78994d9fae132a30c8aad618", "222ca993ffcf46988225b39f5b244fe7", "92f61dcc5d6447a49ac05ac86938a1b2", "c298ef478b4d46cbac69a367f9ea6916", "228e452965bf405c807169a953a4ab90", "93ddd7d6d6ea4d57aa0f28b9ae6bcab8", "69120e5b009943f3be2b45ed088b9991", "3dc7d244773245e69a6d1825a7cdb8a2", "ad3ce16f859649439dca081eefa59026", "7eed27c686c640968dcef64435d894ed", "05fc7a20c5924cc3a08620d71dcab4c2", "ccc32033cbf14cfc928567d78adcdb50", "5ea03f07e4834c8686262ebf7146db1d", "6f5b0156dea641188fbffb9156390b7b", "7fbbea8919664744b763a6ef155bbaab", "4c18c60f3db94a3b96702313a3deee7c", "967ca4360d5b4d2ba5a2d633a4c7750a", "b9340ee21c7246e88adb1cd955695f79", "751a35d61c88427aa7e72e5e2c97af80", "6077e46c8fde49788626ac27d976e326", "14f40100b16a4309aac94cbb05527c91", "a36c7973f447408da912ef2154c94b26", "1c9eae6a1b4740f38f813c196039a46c", "8e11cd8dfac54121bbe5ed5fe3065af4", "65805897286b42f2b3e9bde69b4466e8", "a21b1c531e7c48258ac4ce56e9f15338", "05d79d05cc5d479baed100ad2df0ddd9", "af7c371cbef44c83a8b34dc906378b4b", "6b39f91ab1f342e7baa1b72e2ab27889", "c3db68a5c6b841fc8ff7ec0c867e9330", "33d63cb319d64de1bae711bf239f270b", "32f93ece93b9427593a5a732e481066e", "cf3d33c044ad453e8131b0f90217cfc3", "af7945413b844ffe97d915da55b17fcb", "cfc6824226124b779008f3b0a5823517", "cf94d5d90c9b43668763444f41c47f11", "05228d0f37274f248d2c7a3f792f3f80", "c089fdef2b4a4eebb9f776f9c8b8fad1", "0cda6e0a147749eb8d130a77745531e2", "6621af718c6b48629ac8638fb9a0e441", "6f443cb30cc94000be3f27e02659b7d3", "ee6122a3114148eeb7b27e3b75e2498d", "0417598ec69b49ebb537759dd78be9a8", "49987aac21c94cacb626621c4abdf67d", "63e2d420c1f64faa91013c85c636ff58", "ebe6a99fc96b403bb6be8b8bf3d16ee9", "bd34a52659df44f192a974a79824260d", "7a54e584ba3d4da88f3e4ae3f8e2dc28", "9d57dc14740344ba8c8786d07c5b48c3", "09a2e82e967d48c08f75d8ec7df214a9", "ec1a6e6292a24c2a90a7f7721d85cba2", "45150bc50c7f4ca78bc8acfb092c0062", "327d0149260a4f8db649413722b1e6e5", "ecddf903118144f2968336431b68ea1c", "c3f62e9bab1540149b8b12f9accb5055", "6daa12c0e05d456ea26a40a541d5a2ae", "8abf281576ce4efbb0226ee3f1f85dd0", "07e1098c42c44f3d9f39f4f4ecf93a29", "5297b4794b764b51b371fc7863e6135d", "2ee6996d93194c5a809672b954bc5376", "a2a24ffccf7b439094b89489badbaa85", "bab8ff8f37a44a85875b9f6922ba7229", "630a8f2c01c246fea1406e6350a70037", "6402b3ad24bf44b8bc479d3c4676c411", "24cbb58ea8c04c00aae69d247870b999", "e20507566ec544bbba99de2700bf6eab", "3be064bfcb984228bd888e28596e6fd1", "81183bdbc4d9457d885c7efb5008bf50", "d71d296512f74a3ba1c318541ed07a76", "e78b2cb52b2a4a519c6ff44c715bf496", "7c8c11f759b3485e83f0e7840c2aa31c", "246c1ca3436748cb882ea5f8f4b2e21d", "c4ea5ab211fd43bb937fee859df0db8d", "17270f38709643dea3457349ce606924", "ab11c15cee7e427d8f1002cd92bf0ee9", "0a05512968744e358161cc466efaf044", "b77c3424dcf24bd096c2c953a9782fa2", "c938c6960318408d8764c9e2c3be8dfb", "e82519f536404865b9f3aa5e51c2c118", "d79362bda0c8460eb8b702c405ec2eee", "3024fe129c0c4dd2b6852f9fff1de92c", "2e8e93fe68bc4ed4b5f7f7bb75576ee9", "7916934c60434155b9376cd009138b9e", "40fde597721348f6892fc3330e61e254", "68f648a50c864acfbf0e84a9c933d414", "d8e2a9a67ed74c91a47721505e2ed3ba", "e46da87a6f614b84ae0574b00bf6acd0", "3910453421964d94898f0f4069d90924", "323f60f1d6084dc3a8869f1bb5f85deb", "9960beaacd9a4814b9a247d13b227eaa", "15d23371b5e64aa696e6fade1b96773c", "3e5d71c5deaf427bb5652b3af7c33e35", "b1be2ad96fa54fd8aa5360211e2e4fdb", "53064ba546794992889718a2f684fe71", "6e92f145825049e4848b6b850d22c059", "cf529e21283143c08fac13b1095d4779", "1d128027903246439eb7c371eb15b623", "3285570fdbc1419a8ca897c1c795f30c", "d286b1ae68664fccb404b61603212401", "67a91c9fd9f74981940c8f159ef8f493", "68848ef0001f4cdda27d70a8db1087b2", "d3ced6c07f2f43a8b72820ab29a26e4a", "c8df56ec08954878a7cdfff950d2dd9c", "5c7e89440e494770bfcf413db331b636", "6146d848072c4d369c81809964b3a152", "e2f106decd9c44ee812e4d8019e6bc7d", "66e70362aa984acba76ed2d5919b5fa2", "770beb16fd984761ac825baf54f8c359", "868fa3d1c8354541bbba46264a748f15", "32ccac30eb9948378b6b9b793ba2148a", "6917d1d2d7844a569ed5e517b1efcf0e", "2528822c5cb846d485c82a5e40314592", "5fd71611be824325888662068503d998", "bf642b5212ca496895b66c02edb0bf3e", "deac4eb9940247a1ac5301534690a8b8", "338a21f6b5aa476f8deb1810c694072e", "18195b3497e043d8ba0b25abadd757b6", "322d0ec3bee44b2eb97d9d459230089f", "d5de8a1134ac47dab0109e4e841d9573", "61b473f53008446fa7d3c21b414d9778", "250c04fbfebf4ebb99413ad29fab99a4", "91774dd08131452e8201d3747c040c29", "e3c3ff6ae45947208d074001d8a31087", "28c8e59835794b36b0681b72c932d11c", "9ace4daea9474a799b5eb0b2d8845262", "126dc3a1cbf347a3a30449e54a380e51", "42832789e66c4fc699ad720cdd15878b", "c413c395001f4df5ad286bedc1124bc8", "4c5262e089d740b2a573628f2eab54e8", "402b0cc0ba81438aa25d0d38faba972f", "4c714597d46e401c806091a4998902e5", "b7ffc9e6ebc347debe54b44a4886211f", "c819409ba1494b2c90f8ba4b197cdeae", "ceb6406bc1d44fdbad90ead9bb182d36", "37aaa2eb124e449aaeb7a44d03867664", "7d19e5fa99ec4fdeb3a5470e9c7f21db", "1400c25f45594a8bbdee42e4521439aa", "77ceeccbfa42437eaafd8f920444d0a6", "d3660f313f8949de8dbaa8720d983566", "8f87344051374984b29169b9e0366603", "7e06dd3947314e0aae9540ba008d6125", "6ae58d5e321c468fb9f51df0d3fb438d", "017c63eb82034e478e3d8d0fd3d46ccf", "0caa184145d74ba88686c798ace7c52e", "e10ccc2443034c9780d768bd240609ee", "7fda011ec6d043059c97aa508fe3f473", "5e24b97591e34c4493a356c22aed48ee", "92a16a03dd3e46d49590565e52f6d867", "ae6511e7c2a64fb6a3614866a6f832dd", "fec037a2ebbd4e9ab512b224dffb82af", "c35866cb8af74800be9edb7980240888", "f17fdfc3c8ba40daa1a92f8de1980458", "6c940302abad4d0f9af1cb2861cbc014", "fa9a69cc271d40cd8accb406d60e2199", "4e36a59dfc074e009f955ce7329dfc77", "eb646a49f26042cbb2046e2c882bc4fc", "3bddc55e1d304fefa71a4eef709bbee4", "9a6de7259f2943a69050994b80bf5bdb", "e44d4b55c68647adae22107ddbd35842", "e1ed0fb9765f4ec784a32bc26fa63ec5", "84670880f3624af5ac342f9dbe133007", "3be3acdf38b34e61bb2272570368011f", "62712d6b00264c60b2a1b68f2ecd96c5", "3d7b1a9874ee4c6f8a980729815bbdb8", "7804c1ec0afd4a1c9093df1f495480eb", "2444a76df4ce4679afa96f1768b032ec", "df67291f10bf4744aee7e3f03ef6536d", "7990349caf47421c90b4a972bda9a934", "4215368ecb554048b9ff0886658337c9", "729c9e12d2b14234bc4d46baa6c0fda2", "ef507b731c084530bf6275e0def56d5b", "0e19c5f1070f477daf380ba3f28a1d13", "e88c215dcf0d42379612ebe742ce9e21", "8c6a66edff73432689f81104355042c0", "4d99610cd79747868c561b918c92e1ec", "5b7d9486c22842438c96bba7549015fa", "79081cfc51714eb89f4fb4d1b3738018", "c23ebd48e83944aa9da4cdadbf7927a5", "3c68a344b1f64e908f45afcba7960d38", "3524a723c8944ebfaeae1d6af5a62743", "7f818b7d8fe141f696e9ebaad95bc5c6", "a366d0a173af4a549c6b71102116c22e", "e3d0e3948c73435fb841a40f843d8524", "bfaf814fb1a349aa8262a26e515126c6", "c282978b170f42d3816b9fb878da4c40", "ecbdba0fe63246c8a6db6e615370d4ed", "9aa50e8ae42147e9bbc292c54cc26444", "b2941d7101424d49aae450b58264b262", "b879accc202a4650aca852ce3a896aed", "4738dc7e59e54e6584b80710621dacc5", "3701979c0a4e405d949a4938e259fd73", "bb955e25fb1c4a8a9fa7b5823efbaff8", "e347143885bf45ffbab06db8df71e794", "8fdaa77665804798b4951bbe2463d017", "6ed148f802e147aba7ea21ade77016ff", "2abd74e374b34b268f73e8a187eded2e", "b122ff733ad446cb932b1200edebb2ce", "7c4c0ca46f3f4e868e0fb28634921534", "9e64c2709fd7489cbd3e54c178385645", "88ed077e8c934a708c29257f5d2720c1", "9a06367d4e254980b633181ac21c9bc2", "5ec765f006af434a825420626fbd5f46", "3c747795f73b4970ae64a4eb550c86e7", "45eef1172d944af6b8f5bc6300e06e12", "b58859f94d4445c9948291488467b49a", "7e5d254bc38d4162832a8c3bf8cceb95", "237ad09e37dc46bab83f5b0c64839433", "e903922efe144ce393f653985fcc6b88", "bae68b74ee7f4567932d96086687c9f4", "e77caef54eae4538abdab7cd52d71aea", "bae7a7cd8be04a4f9d8001adcb0f2d22", "2efa805fa711486f9671a96cc5901728", "7557918e2eed4efaa1c105c28694f9ff", "32ff3380f6df477ea1e55088d24f1ec1", "06da871e7d0a4d3b92a5c231aa3fae40", "489ab355cd1f4012bedbbf61eede2e12", "edcda9664ef54cf2b09543c73183046e", "5103781c30d542d4a2811f6f2a9d94aa", "3ac15d5a7b7c46d09d88cf26e01a9bc7", "041894b7d4ba45a6a63d74dc8441e602", "44ad248a5cc14891bd62af380b343a3c", "5e87d5550d284ab48e6614d86b97e0ed", "ca7ae43a76854d759290e9208c9d995a", "1b28a2b971294b08ad0651e30746190d", "14f72475c7c14d4499f1b00ba4e40d85", "4bf370792fd04d69adc77e350768f921", "f78a9776f58042f18bc206613d770293", "0e723a7deaae450b8cbb404962e6819c", "1310840161824bcb8e8626971af04172", "5078ea8c54654f86a0d5dcee76a21164", "098227456eeb46748a4c22628e03e1ed", "f984d067521748bc83ad19fc9f93a713", "22921478311b477695cf863071876123", "4ef366628d9c4f6f81b2df535c4a2c8c", "8548201797f24f499ae4d1d3fe7f34b0", "06c5cbd81dc8428396968c1aac43085e", "010d70696a904effaa33fab2ccd7f5ef", "0befc59b8c9f4950b4a0755851faede2", "3015d9b72b524741bc0d254378b45055", "3585add9a8f4430e9985c819f258ab8a", "48531e35d7884687bfd03a89a0dbd101", "36a5bfd8654a43cb8aa45a4196d6a74b", "850829a4eadf4439a333abd71b148fb1", "42688a01d4d54a2d87f7071d23bc6766", "90fbdf504b4b413c9f94ded1f7acaee4", "7fe27f4465af43c287f055f8497bfaf6", "741c595459894b04aeed7f5086b9b462", "e2f88b8a82fc4ff38c921495e5899e21", "bf322b7e478343f79b3e3cce96606549", "b434dd6e198c4feeaec307f47aa52bd1", "84a1db6db7e8468b8e5a018485886f4e", "ed842a03894e4c54a3c5f58a05f71340", "9bf37af7909c4ff59f99986ae22e2143", "a580557c836f48e38c50c0d705694f3f", "8359190b17cf49d9810b7e9511fcb814", "01279c15ad0a4c27a477ea9970438b0a", "8b39dbba8f8c414d9e3e2a23e1618ca3", "0f3cba6d66cf4ea2a27df5399aff42b2", "76b5a16f06254e59a4b935dacf7edfa4", "d73f1a4a6a1c4d7091c577c08863801c", "ccfae6e9ae5743388c92b711f4b0f9a7", "30253da95f464332b4a1190e6565c840", "6eaa79c360124b1f8b7cfa8c601471ca", "77974a436b3f4444906c19f1f1b8fa26", "7d9433fd2fab485c9390f10cf063eb14", "df48fec576754bf18a51ce1b6d465999", "6d9b938e86c14eb1a1f76392792b642b", "e5cfef71a26f4150b2884b8c64d8576b", "cc33b847b2df411caf7cc332a4650312", "fb952386cf454c7eb0f45093497ff0bd", "7a45da92840047848f120bb90dcee9c5", "bbec08ef69b94419ab4a3e6db749828d", "34d200e235ed4b7782ff39e29d7c94ac", "03ad5e79ffea4028a7d2b49f9bff48ab", "4cd7aac0e3874e8cada9015da788036e", "610afa065159479c86d516296d120655", "1ad7f56fa2a04d8b8501018eb270fff7", "da2aadf40b5840da8dae0ad71e7421a5", "02b9f9149af7462a8b9985855bfdd035", "d5ea26d0c57641a49936c714c283b596", "4e5f855f131e4df8b11646148ee071be", "71e2d3ca6f3f48b8b556460430e688cd", "cc7c3248cc4e4871a6461ac2d975f0a1", "5726c17c42cf414a906ff34b3c88cf6f", "d381fa10978a446182483c1fc8e23428", "02ab2cf5edda4c918a3e706c0e3d177d", "589ef320b3af4cabb5f1794fb9f4028a", "870465e107e94a08af6f28cae53243d9", "bd114918b24e4102b3c2b2c44d2a4481", "a13bc56caf1f482c82658b6ca53f041b", "5b02be8ccb2240c3871942166858c1fc", "f37c9ca95dc84f71b24a26c3fb7620c3", "a01074addc4242409cef696564e7a40b", "75a990f7fb0d4bd2a07adb5ba5fb361a", "b37267662529422db542df5de29f4682", "d689c9cd2c9f45778a70ada015f163e7", "18cc3c1d7ba44c25a79e8c8303b14c5b", "a0bd54ce39e14d688a43932d7ddb8e18", "4c8e4c2518a54fc98b7096329f87f5a4", "fa17e239d0f345e8ae621ab103d45f9e", "44754c5b72da4cfeba06670623a66db4", "a7a2889326794fed99d8a810f25df725", "f610d5c1df2449c595b3bc1442112b1c", "371b965ad83249e99bc88a6e2d944b69", "db7a7783a51c46f1ba4615cde81df622", "8d6f5fa3771d4f5ba85bbe830db3b124", "0e1e1cf3e5044641b23abd316bb3f71f", "11339a3bc62f4509a4106d85b05d2292", "ae9bd0ae3d364af985a974252abfd67f", "a89e9a577b3f4d35a2edd66dc807c097", "586fbc04ca9d4619b52a7c671a80d2f7", "31a1f2c8338a423e8070dbf6bc0c2cfe", "3960eb2e65ab40869dfeea061e53b4d6", "1aa52506d9d54b8e8382289de7986431", "15358b74b2974b0e892d4edcf817236c", "67073a0d15e2452cac23b1e0acdf0f76", "75fdfa618c1543d6ac15c4ec077bfccb", "f71d9a20b033428f866bce0a20edbca6", "0e9789725724469aa18632a3cdd42071", "a37632583fe04e2691b57d51f29da5d3", "46a06998ccc14ea981ac16eb950c3140", "bb8fb2681ef6406795850e22e12a8362", "4c45c9ebed2b4d189f6463a90705a446", "b3996feb306a40ac8dfa237d596ad865", "1d6d7350820347599df3b17beb10ca5f", "e54b12694a894831b175f5736a4b98bf", "08d3ee0e89f54f8b82dd202cabf5b717", "c5ee214b8c414febaa218fdd4adaf053", "d4b0b13a4b3f48a4b13509ad7b755f26", "563d642a8a124f8abf9e2cdf2257baeb", "54d5085d129f46239c442946f1c3e9ee", "6bf49acba3834aa380ed3432bb234f9f", "48963983f2ee48f79a78a4ea03b79839", "d145ad2524f64e1c98e14a9a162f97c7", "bfc0065ddb2c4f9aa37425013ec4552f", "a2f3c2dfdb6d43738c00f1f9a6c9759d", "aebb05583858435c904a51c6bea2f5e5", "4ecee26870fc4c2390fe9548561c53cf", "25c80e2ab4204364a5b8c995e1cad3b7", "936c6f34c001427d85e6f18a3305e871", "6048201ea28641cd808e0e2fde4417cd", "0c0284c9be474e32af0b6fa36a8451ef", "c7dcafff9c5e486dbf3833a98bd9451e", "27b4aea34e4b4e5ca94b479eb45d9c94", "c63d4711c4d44a4eb247793de078e38b", "a7a607128184450298aee49dba62a083", "842207c6cdd64013a4e351fd9c8ce246", "3fdcd89b69ed489493d7acf5a85d20b3", "1ffba4341b034dad878fe9951d917f7f", "b4ed0b2d816e4efb9a6579b67d608b22", "8d96d185026943a69a1ec9b42fc40ebe", "dd8b1d01ad3742368ce39f2e0f466b16", "f1dc41ad998240cd9ae01623b27bd695", "13403a6d2ccd436bbd6628f6d822d3bb", "e50468a8242b4cc98f91226fd92b76c4", "08599e13bcc54b91a73ac086d9a146df", "016b18b085064f4d812c615c7a5949d3", "564d12c74b4c489f8292ae83882c2eca", "175543a823354caeb9858ca976177101", "9159d0519a8f4c9480d736d7c709b13d", "cf7be3243592416db020e59442280f16", "d379dd04c45e49d4b3c0be859cc9061c", "4ca0062896094e69b5c0e95fe3c3fa16", "9b9cabfdf7c44f88b07921ae6765155b", "4e3f2896864a4a75b264a3670ad4c324", "022737d3039e4cf6a8f48278a3fb7ca9", "02dc02e14f7b47fcbb341695df7ae2c4", "a1cd341a79864d838fce59b481d84bb7", "28a403a98c2847c9bcdd778c23c2109f", "52b0a851b1224a2aa6b0a4512eafc6d0", "fba0987bfc904169bff658498264af0e", "10a1eb50b5634b5c927aa6aa00eb3814", "98baa252b53f49b5997532e20ae498e8", "7f9dac9c293a4ce187da0206a402af81", "487d7a07d5fd4fb59a826339873c761e", "02957f6e5ef64aa9b2811123fe3a5052", "197abfd49d864b51a3eeb22f290d6bf6", "d8dcf8af7718447d8a78cf4f0deeca02", "c80049d14e0d42fabff957937318c925", "14287d6fb4084feab2d3e3c5ebf8eca9", "f7e2954b33974e11bc3148d97b43a3b6", "e7919b48591d4aff8414e050d69b5e02", "86e6dd616ff341259b455b111d975a19", "caa9d1f13665465f925c6346f71cd784", "082193d21781485984f71ad73b593db1", "1f7c1390c9f645d088fa115110b71459", "9bdc270897074c72af989b62c76e2b4e", "810309da027745eeac56cf0349946edc", "75d6d62605554e36bc0bf0de0bc968fb", "fb62e7f717ca44c0bb8749d6bee3c927", "bc9e45e173c14c9199a5e3d5dd84267f", "f99c58c792bc4c44a1387f0523ca6f57", "65df461c4e204f15896dfcdc6024998a", "a9f3a217f9344aafbd7abcca1cc1aa6c", "6147c9504d4b4e0a9cbf4c05aaaac6d0", "8aaddbee2e68499f9caf8df3c5f16e49", "6c907ea56894497a906fd9eb473a4f7b", "026edb04581f4030ac3e5c6e334754d5", "66271529ff92401cb0e8d8d3f6fa6955", "b8091d27d69246019a4b8f25244eeb90", "105747c277f44480988842a78d1261c9", "b6ed5b0b113849ca9bd87e09bc003bd2", "9e39039079ec44a1a996b1aaa80588f7", "743ef36ed3474dadb813dd9c36581e8b", "53c4cb97747347d6ade1207fd96f8054", "3820a52cd28142c6bdc6b535bd5b030e", "9cb30dfb894a4fd2b28fd4edccb7c45e", "cc945387406f426fb247637ab199b7e7", "2562ae7c93d34edea4278f37866b0687", "b420301078854207b33f2898e124d66f", "06233079b2264dae9dbed72e73ea8233", "0153bcc3721540e49f389161af443cdc", "c64c8bbb2fcd49e2a0a05e1fedf68a76", "ece45f50469744d18db0b3ea09e1800f", "f1059e86d3df4ebc9032c9e50a9904c3", "542fe7806d484a38b2c942800559fbd6", "564bf24d6800497a878496384020ef9b", "1eaf85b1a2044950b39597a89d6838a7", "1c164d7768cd4dc08b36cb90422e22cc", "98d2e017f757463a9537eb44c154bffb", "4079ce5bfb2e43c398a48eaf9d3ab5a5", "8e3405c8ad574cf2b1042bcfee75161c", "4d89863455a94e048e93e8e730c0ec27", "fa1f8b7235d64cf6a43c4095a8e83e8d", "b046243804c546f0a452dcf6a8e53771", "a32fe052038549158aeab58186efb726", "9194388465e9477d8150be4bdc7c0eaa", "3facba1b4758485a91f571e23ac0d031", "5930107966cb4a508291ba4e767b9675", "2383a4640f6149e9b69f60fbdbf5a7a3", "273c22c6ab954cca98dc1ae7f0f7e0de", "65dca97b38934bf8b4664ec0ceb3912c", "61ddbb6ba643489db3c3a81c06bb2b16", "d517e704a73f4a41adf8f2327a41ef80", "12d0098d6de8424fbf8e1a43684e0710", "d129720420a94bcb9f84b9cf1872f47c", "104441e77c0f44a783ef3e63ed61c7fe", "1635a8c7676e4ac891fb8a6112b3759c", "c0253d178d1d470bb862d8a1494b00c6", "742134948d374785a224e83dc901567e", "6410a65dedfe4f768f758d1b8e991129", "0b30d950a0b54bb1aac25978efd018e1", "ebccbd48a5114b88af0655afa1c3b22e", "dd196e46054745b88501b0ccfd1b9433", "b74a88eb6b5e4e66b6040482279d3cca", "8cf1a8e5f0c3429ba7de2cd5995034c6", "f0c1f47f39c14e70b9e4e3eae5941a32", "54098ee3f8184a60aa2c2572edb2fe42", "e3bf8e0f78d54831afd4e6d7d1b51966", "725e6a0e680d413f803e42421713d013", "dbcfb9a1e4e14d4baa66952681374224", "5f0dfd4dcdd64067b79c3b21ecbd30db", "ccc6cbfe90fb4e508811114d577ad79c", "d8dd7a1f9306485fb13a27d4e48510f2", "98492d5c8e8d4396beed579163aadb0a", "334bd08acf5c40458ae96d63e451b55d", "7eaf87f1cda24a56a93c55d6cfdcb5fc", "305fdcb1dc0244fbb57bdf64b3f61d35", "4fb8dae8ff284954a8ec7e3d6ab8d605", "30304671906b437cafeadd631090a85a", "d0ea61b7fdac47bdb18e6b8cc628273b", "4b4920ccc5ea493a921aefea9299a799", "d8dcfe677ffa43b7b2b34dffa661b231", "27d93a1a51f44f9cad5d955903dda4d9", "f62cf8c43ccb40a79526a5db0a58c59e", "d470e17180bc4a469bcc7c0f71adf32b", "2f477734f078464da2b39504c164ea74", "e8159d04cff5417a83eca21299b85412", "58e596bb291140e298509517c4d0922e", "639d28fc568647dbbe60c87756dc2f2a", "1e66e426a039480783ed80428ff29703", "76c7195409b24715a729c7e21e552ff6", "86495fee2b304c97963d35dd0f3780bb", "aa87089e5dc44dfe82640585049c1783", "0e25532fba1b439098c7fb4c9d56e438", "4a87b46d5cdd49d08ad1d72b3b4c4d21", "9b4905276e9f42588e5a4bb58121cbcf", "d6e7bdae49d2468c93903fa68516f1ad", "1da0ed1355834cba848b8c28e17b3cef", "11e908db7cf74c46882a2aae415ea1cc", "cf87d5c9752a4624881605ad3a05bb6d", "0c11878f40c04c329a4d0ae348809fdb", "32874a57240b45cd9cb8b6f57c2140ee", "82fbe0a9433c4ed6aba2ffbd511e9894", "867164b1f62d4553b5b17b81be824b54", "9dc856d2b5d244cb99882938d1157d95", "e9e184fb8b0b4f7ea8a6ba2fc509c203", "68d59b6def5849b5b1237dc355321d18", "dac5d3d7890845b9bd53a0077a2495f3", "87f3b8c239894a6080db38597f54e9df", "640b655338484ddeafc1fac2e9e0b3d3", "f9e9b8a456404d67af8ba5992d713aeb", "c6f029c9afd24aa79df52862d38959f2", "6d2c49b897d341d18c9c734cd01d8c95", "8b9bd301bcc148e2a438913a370a5642", "ebe73692b01e48769f783aaa4fb4f330", "ba4f87ebec3447aaa024683e20534680", "9b3f52e44ca54cfa887f4e3391f35d17", "c651c066eb6d4683942c92adcd5f9c6c", "421a5b3b415347df8e02f63de95471b3", "e6d401952b8b43858c6e87445db8df2c", "a028a25b31ca4e9e84c7aa9e218b8260", "6e42a71713d74ff5ac74eaff2fd1b27a", "0cc2e5ee24654c04bd60b769d2eec599", "a19cde244e61439eab192f12479734ed", "73088d3b2aaf452bb3a9d4ffd3f16cf1", "e3de8407b2424d1a8f490ee6d707b0bd", "5fc520e75feb45f0b656c39dfc2f8cd6", "23c62c286b5045628e3537e69e1518e9", "7db4e29d8e024a7a806189038152497d", "02482cc5d8b84d63b2ba86577726d114", "dbed49ef25a549968ed672add7bec97f", "a137c119b04b4ab093cb88fb9ecfa411", "37486e22c4834265b50b22de6434a9d0", "6e8a92ccc4be4d65984248207688253d", "cdb658645fc4493fabb1f8bc3979d60b", "7850fb5265d04da7b69c58d361be9883", "463be85c16a34d78b78ec19e963eec3d", "48424a67ef1a49aa850ff4b10a731e37", "6f630595821a452d9ac627fe86c6d4b5", "e50fe69b986b4db380f8288e04f2cc04", "37a291a093f041518c8c25b985906560", "8bf65379ba0e48dd9af30679ab697933", "83d1ed19f6e44db3a6c3da51e79d82a8", "db8625d2b83349ef94fc766c4b3df581", "534b121e519c4dc991d1995f1cc7c41e", "04fdff080c0f4d29ad7a521517bb127e", "a5853a82f393448ea9584745ad423e3b", "15ad7c072f2344f5b71e5906f8a83b16", "e4128ec564df42aea7fd7f8f63a29459", "777cdfd5eb0640bb9d5d84b2fe6ec074", "0227880a55794b978beec4f1e174a8f5", "0ba9aa3499a14bb2a670c3883b7ccdba", "e731622da0324bf7ab1e700d8dd64ff7", "5afa25557358415d85abec4cb36caa4e", "72f3a4c253ff4e7ab8ec5740c06f5a95", "28137ee5f99e4c95874b1adbb543de5e", "bbada9789c724306bc830a4d388e802b", "af81644a21e5483e877f63dd1729fd59", "bde54700b82e40f1a0f0d47edbe7ab80", "2348c75f796c4b7dba4cb94834cdad4f", "17990fa7927d41a9b5d1842361170468", "108302a4d8c34b959c5e50a645259ffb", "c20beee9d425419d850f577b4dac4791", "fbc2177b13504a14b4d01c5a246033aa", "c044b4d340b04898a5c42a97e83433e0", "64144062c2294b91b0f9cc8cd7f5ae34", "86c31216fe6e4e4e9d8b5e6d97200b45", "8e5631585bab470ebb66dfb4d97ad253", "84e97cb26e1a4e7ea2683c3adffaecec", "11db06927df349b9971992adc4821a83", "a007a21360494677a3d9677cf90b4dd3", "d641fd6a1e8247d5b8ab0f3508ef2fa2", "7a2f8509fb5a424b9c43d1b9a38944ff", "60f32347c66c46c3978bb2810bd6d7c9", "6cfe878d08bb4b8db2d4b9b0c0d14407", "e96030667eb743b98257f3464a974a0e", "716168d4154e4a3b96c61fd913a3c25c", "84f78e61108845e2ab62c322a37f2560", "3206b6e4465940e99efe39e45a3bebec", "4e6eafbae65f4f118ac92b7650f06037", "e463b6d7ae8a428faeabde71403094d1", "f2d78e5d69a94a5fb213654a81dd171a", "e1daf352d8bd49818ca9badbe07a25b7", "dab27d6fa2984137bb4a33d03616bcbc", "36df37094eef4e80a3ad84df12218e91", "3aa2f2a5e12d4ef39214de78f27d3681", "4491adb6f3554ad3aac3681bd47da116", "93087ef58fae43109e2a9cdaa58cc54e", "9f2f4353f236446db2dacea03412b85a", "d14fc792a77a4162a4b1d894934e84cc", "37bcf68824a84b829ebe68fa934e16ef", "b3e3081a2fb84638bbbf4773f97171cf", "884bb390fa724e96bc27065c90ac110a", "07e117a691f745509f85ca21894b4f84", "0123ac738b6a429f9642ca64771afd05", "dabdf862b93b40379848f710303db4e4", "c622980459f44478ac368279ac997ac4", "681fd1a734924f219c54f543dbd5c84c", "04843932b58d48f08a217b4e802c8391", "15489a34efa44ed2b75545c81003180c", "e93dfb30f411468caf28baaccb324250", "8dd680965f4f4c2186ec43d0389ff88f", "430cc63c6e3f464a8c7c42be584ebeb7", "879811d420bd4fc09921119f2af4756e", "0b577149872a42d2af2f77c599f19a7c", "64f34395e0624d9681a3e333b347b513", "8c1826103fd5410d92bcc09313bc1f9d", "d5662b7c2a1842c1bde2abae0aff98bb", "8f2f37474fb94e8a8dac2f7e636a653a", "746e4b0775254571befd379bcac09818", "13b1b114e23a45e1863b013e1b3bac70", "f67574a57aec46a683b3e554881908bb", "71f034ccebce4ec7b900ee92dcb4fd35", "24c152ad618a4dc4a60545ff7415a996", "dc07b2461d004133b71ce673641e27ad", "e2d907534f7448c493a9e4f60ccd3b3c", "a009a9a3191f438aa7802c5077aa03d5", "06fa2275ad13433ba8a261f861968c67", "db6e1d04863c4f1d9ce4f5b75034c692", "fe0520cc4b434da79ba9e45e161c18d2", "2a79b9b82c694f9a98673585a4035005", "1866605d8b3745e981627c338b0aac7d", "83faa3b312e844868722a4610004df97", "2a527944315b4b5094dac015c6631305", "39a54d46dd62460cbff1c52a86915c0c", "a44afe831a094918bface99d7fc65efd", "8e8a2fe4e8274a06a253d461545c0d8f", "644e88516a6d404aa57c9ccdd1199c24", "87075a0458ca4045af72e3965ac7e782", "ae98f839dd67465fa10002b72fff3209", "c91eab6ba38b4ffbb2a0255d84d79e6c", "c7e4e044f8984375a8220c7b832ef28e", "ec60d82c787140cf92db8e291b74d234", "819c06c700754413b70aafdfd7590e8c", "274acf60390f4bd0b619d0f4a8ec91bd", "febacbaecb954169a39e3e67da989dde", "dae3eefc7d6141a5a80d1f8c686fb5ff", "9ccfcb82da66480b99716750093224f5", "4d6160fbc8dc41b6b069312472a438e0", "01801c605b12405bb79d5ea577563f5c", "a88c9c050cce433086f22da64fe01578", "bfe4c4dc44e0420b8be97498bb118042", "a7ec7bfeb56d4103b82eb342172cde1e", "90fe484f00334edea6b62931286ec0c0", "52876b93fad34ebba1ad7cdddb0fa676", "400fc5d4aec7414ea100232ff90e1dad", "50b0d1280f464106aa88ecb3551632e1", "330018ec623f46948500f165bb636245", "1333e8917b8943efbe2ed5947c25b880", "d8c421713f8f46ae87ca32fc35ed519a", "b05f2b11b481427983cac2ebb98be2e6", "8b7e3c03ef68468bb558611651e7fd61", "c84bf3476c1e43ab8584dba14ccdb9e3", "d1b2d68334f44ae794be88eac02e091b", "38cbd97e3c554832af268380727fc96c", "e5be4bc9e04040ac9e20a50fd0965f06", "cfe43b93b58d4ab1afd70b97966f48ee", "b3e064b5b11449428b52248c56387b47", "56383d6c27ff4fdaa525798e9913cf84", "899c185dcbed41b1a394258769559a1e", "e598f1a16820408d94bd430cc0d15d7b", "11c3ba7f63bb4a408dc0de7264d30181", "c95fb79d55164114836d62f472a0270d", "c4a4b1c2f28d4bcf83bbd6cebf73ab51", "7d7694b99bee4ac6a1b5ba7f66b52b65", "900234be538641df841eae5dbd091671", "dfc4a8dfe6de477e9b174987e597d56d", "3ae1794ef1da46f39a4499661b3c1c80", "cdc9042d2b9c45f088fe89779d611cb0", "1532cd9399ef4486aedcffb8be70ebf2", "b19374923c4e40369913ddecc647f172", "f3f5ef6cff8840ed8e7c9483811243d8", "c2105b85f1334a5eb92f931aae9337ec", "41f5179423dc41549259f2a19e69d65a", "362cb3d4362b466c978b62194f1b08c8", "7ad1f74bd3884665b99b0feb1d3086c0", "442261d9cc8c4a07988e1b180a92f12a", "30242323fce9451eaa310e876e162f47", "6316d39814164517abbefd07b86a8060", "4e63142b96f049f9a611e1d15e246c62", "66ff9716527b49c39e4caac0821746c2", "751e19d2003d441fafb9484353acb879", "0e2aa3c988044b4d99676f4b47d4fdb3", "618509038fcf4d4a957d584146195d8c", "2606c51fa004434899ba487fda409a4b", "3b61b2a90b964fe382c199ef19545111", "8ba7b17dd66e44c88b21167bf9f6e56b", "2c7f8ad2603044928710c247b6b51437", "0591759be5c644278cc4cd1a4556d158", "420705b16057403b989c8b218e123040", "1ce47324c2564f948c7ef57a92b5180e", "88c9f14c63db429d9bc90f4cf31323b4", "6b666e8c85e242cb952bdba7b224ae5d", "f1eb315248a74bbdaa7573196a3c1c64", "a77ed4a29bee4b03b247d5cd4433426b", "9b9810cb36634437aa1ee652f1d3bffe", "e24c23d0a0f64a22bed56e6d0a08ee2a", "0fcfc9c1aadb4403bb604f9a33a6d7da", "8eda52f9579c4ba3bf3ec610df3c6c09", "321eb95970b841748e4c9e99e33290ba", "34631023fea840e08879a9ed23cd27b8", "4cf3162a1d464792b23ce8c9396b6ed7", "814d95ff3f9b473fa02b565b1e948259", "6dca76fb908f4aeb9289b3ed046f28f3", "2f3039a9e7c54373b3effc316272b6f3", "d6e87f6a723844d49380d225cca4ed2a", "5d06ecb8d1764382b7fa765d631f0f9b", "296ede4dbd354fbbbe9c03bf104fd5bf", "660b2786455c4770864fd8f01430f4ac", "93c0815b1b7e44d4a3be35a89a5913cf", "c012817defff4bdb906423a71af40057", "68b2ebaa53014b4494ef51404f55a5cc", "bd6acefe3dce44b3837021e7124d99ca", "efbe9b9790544179b36bcdc7e7411e8a", "9c4566cbb5d3416abf052bb969a5705a", "35cd42b864884978b33ae1bcb1305bcd", "272991eb92e24ffb9e46f715dfcbdbef", "d71649f8ec1343aeb974e1d5feacc360", "1f36ad6fc6434782acbdba5461f58f87", "c528f52b93144e56945b58cd7d137a22", "ad1b83e9448c4c3198da5a43526440fc", "b7cddad50b144c57b805dd14ae3e4c9b", "9a4331d16693410dab52dd5bdad12795", "6524cdb6a7a549708f23e5a3806752f2", "c6aa3151d1bb4e519e5df9bd1346ffbc", "fc80f6769df84afc83521a1546e2cfa9", "b52f60efe2df465a8502ad483699264a", "ea4b4cadeac94cfca26d1021b7b400e6", "444fc47ae09c44178e22f033981a2da4", "ef3a0ba0928d467db9e1678d467ec4a5", "d15d14e4c62746da8b54ba5ba706b0d4", "ce87f8dc8cfc4feaaab7c1ccf357ef8c", "7b88f30b568a4067b890c91b4291ef0d", "f68cbfd328cb48b9836e96c6f778acd5", "2a7e01e1a39c4406902b33c047f9edbc", "2b2a3b5e051e4e1bb89c95fb5deb7cb2", "c8f52fc557354fa1af5f62101284c043", "e0dd3da43f284dd9a6f84e3e980328b5", "5f4cb9c437e14736ac1f31bd2fa8d8cc", "7c26148bd8b548a6a62c6be445ea8976", "47bf08704ba14fa59961b17d69e6986a", "a37b2cf9ba8d4f0c8545e64768356cc2", "5390922b243748c3abbf6e0c03491606", "8ffaaf731cfe445a855dfbd0bbc61f94", "606f893c08634003aae4d04dc64923a8", "b1e58bf1d1cd452cbbfdaaa77b17dddf", "f8528476717c4723b3c8585d59d7d200", "48f6cfc2933f48aca84553261873de42", "b5b8ba6915e24e87a28b51593003907d", "cf9d9698e030425391a0595567eb1710", "fa394221e7624b9d82c115fa1d1903fc", "cd46323f56174b3d9968854985d17435", "5b0b1f8866fd4d7a9157da1895a542ea", "c39d1e0bc79e4fb997e546be8802f973", "4b1a0fdd82984856b0ad02b1f707f478", "5d2c9a4b87a14a03adcdc7731d2ce8f3", "1376fc3b2da048cd9c0e3b1b08f76f73", "b6a3f2640d5e4694bb5d37d8b705ad22", "9755a73e54d0456bb4bd7bece83325c8", "ae2d6e5b7b6b4dc7b7a2d45b95cbc389", "4ac8661ec46247878d54e8796b9d3ffa", "2d858d6b606b4b81877f7a2554f234c0", "b14035fc341147dc84d485185a6e8116", "24a09d0ad0ae44728882c8065097eb1e", "27a0add0ef4c410699e348b40ec5f0ac", "2ba7b15836554ed090d767c317ce607d", "5af554cf770a4ca2a203959d77bbca3e", "eb9961bb96f445f099ef0df607255b62", "43e4eb1f81d7476a8fa3059559b85372", "07a37ab37c534430a319887db5681ebb", "68656082f3cc4d899e86f26a26713381", "06251253bb7b431ab69f59381f5a9bf6", "cb2730a2eed34e99a1b8c0b920b58308", "aafcbafa53ae4d73941d4014feb39c49", "8d2955228d8142b1a5e40a372e6eaa82", "0d43150ee20e4c899c6410ed82e68b60", "2c2f560f0abb4074a7c1f1fda07eac8b", "25dbda762ca24ce1b8f5af05d3cbcba7", "b8e65dfe9661413fb8e9e435e8a3b34a", "8eb5c58e84334f20a503943f3d577fcb", "4d8280deb2324e0f97e596152fa6331b", "3202779f1b1b4584b981fa41f5e24dec", "7be6da7ff9f4478b8e597102e11fd202", "f9662fe5f60946e09f6ab2a822263454", "f15cedd82b514e368cd4eb78a4399981", "ee5998dabb2646eb940f0be7a941c05b", "826bacac70d34fd89c94ab77572321c8", "a022cd1e3c134efa969b8e03e0f4b967", "2c97ebce9f89474a83eae234631f8e43", "c871c5fa97ec4febb550560a11090ee9", "21e0ff58a04b4856be1387afb4796f94", "ba4353a869c54a23ae6f2fff0713d594", "3f42d32ad067401ab05f22c55d58aaa0", "46ef9c08f3f54bdab558ed977801d36f", "b2242d28c39e43ecabfee1b4d3a06678", "6c22fb2f8cae40949f2d643bd06b5e0d", "0a118f29b7714d34a469836254d84bbf", "cc7f191665fb4566aa471de610eb6150", "d04cbff28c5c4dbe88fdb40b334d3bdc", "966eb7340f464156b4dd5e508c0a65f5", "2d30e439bfb041e3af37f90b6d69aea9", "5612df13e7a645b88813c87c877e8985", "c4c13977dfc641608670428b52783a21", "9b3cac8b1c6646f681f229a2d96b8651", "2559e79331354031a0169da6620f5541", "f0062c7ae65f4b509ecd482f9c767781", "ea0d8c2405f44dc783e2563eb435d099", "717816c292314898bc6c08aae5069a3f", "be6b5e6b8be8475caf2b09cf0b5e177d", "955823f405204417ae4247bcccb42cae", "b0b1694957d94a44bd35a25c916243c5", "2e5618dd5fc54b38b70745e7e7ee233f", "7de51a2ee4e646ec9f60aac0652210b5", "4a9425fac68b4dd2874410be9f043665", "1b8eafeaa5e449048f362a204a1ec68d", "612ba2f4559b4168b9343e499086b9d6", "3c519b7bac1b4d50bf32c81785b4ef82", "1c3b4c99f38143578b504b75f7ffe5d3", "dba14b846cae4c21922f0990dfbf1744", "3d8914c661cc4ee6b23ed59c401d3044", "328640fd062d47dbb807eb30b5fb8b62", "501074151f6144fbbc35b4f50fc85fcc", "edfe9b0d4e1d4bfb8fdb0cb1b5a375b8", "b0dacfeb9d1c4667ad7ffba63958ec43", "f2bd198882ca44fa999dfddce23fc499", "c0981d0bc1044a0bb2dfe21671af035d", "0d7b6ead69c3452ea55eafcecf67694c", "47b3b8d42063484dbc511d0ca9a2f190", "7ad25a6eeab248a4984c3a9b58d3a1ac", "50f2056d3f194db5be28d2aeb58ab208", "196bdd10702f410aa0dcca78c79f5f1c", "ae024ca33d8a4003aa6de356c531ef8b", "9f6eac1941fd4854bb132f97cb07a7e5", "c24aa41ca9d742ed990dcc5501d07e73", "f60909f9eadd48518cecab185d78ba86", "c3aa17da0b3c43d5894275d5fb5a7ad2", "97481bdad66f45dca0ca56d6afc14936", "abc99e736e9e43d7b05eb8b80b8e351d", "aa5955109fc047cdad32f0cc071b75b9", "8a7ea78e8dc24514884a765855afb6a5", "3efc70032dc342f6bae7640b89f0942d", "8286baa8bcc84b69b54b34cfee9e6785", "152e39becc5241e49e867a711e90c0c8", "4a99e16e59ed4df69e77ba886292b4c0", "56531aafd4bd4278885a8c1418b3607a", "8ceb40b3c33b4e76b290671ec6562210", "0b5c121c693b496e8727cdff94a5744c", "d710b9737d024fd29996e6e47a6b5ffd", "3f8cb01077154b789533428e3dbd181d", "66214dec937747d68fa75028e976ab03", "3400a06a01e243dab2c980ba8526a352", "5260c762b7854e8a87abc40884d554cb", "afe82bfeecc04b6eba4cdf67fe3ee560", "87bc2cf737204d71892bb72a764ca854", "24ca927d0a504c11a0fb6aac4afbdbfc", "92f20e893f28446e94fc8f235f5edc74", "4140b4798fb74bc78ada8a9e160e7293", "6e0841e68cca42ac88af0ddb0a8e8a2e", "9e7958ec30c746e881459a4df67811ee", "329ffc3396f24750ba770b572f9eb179", "3b646906d5c54e638775133dfad33059", "9a7d5abb0b484f608400a4b36f805f71", "9beb0742059c4b6caf7067a8e37fbf93", "8db144f6b32d466d99055488411ab974", "53c8e0f41b764cf89e32718b03241de5", "3cf9b7c4552f4b5c96ede0a61d101329", "4b45ce3340744536aeaa1d72da274dcc", "f986c7b1cb5d451fb6d69fde3bdf23d7", "6065f06af9cc4c478c46ddcca4ba6e0b", "355d203c20d348d2a257a0ead6527fde", "048901aad08e4029b02a950489b14561", "efb61ab7da2c4ba9b38e745b055158a1", "aecdadbd71b9464fafad32f8476c92ba", "b8007fb44a3c43eba2c7a208cae049b9", "f45fba14505247599639c9565e1770e0", "c8ae96310a6a4d1181d4053ccb9670f3", "49a94dc946aa4026a243aa398a00a2b7", "d5ea2b52d80b498292b54f5d0afeec6f", "372e99b4fe3b471e891a8bd0e31e8e7c", "6fb8c34b16d54767bdcaf6f8dce18f59", "3cac06de94f04b4c947c03f8aad31f34", "6a1ddf55ce0440e1b7efbd06791a1524", "dac8ad53cc2f406792080fc01cc23da8", "0d1273ebe84046e5996a01660cf2cf46", "5a153966340f4d068dc6cd2b5561c32f", "a4862eba166c4182b91483fc46c99792", "6b6c2466c9474e56bacc875f999761ab", "df42ec6dcffd4e5bb10a4bbc8a63b37e", "8bb2412fe7314ebd873e918e85b7cdb5", "69f9512d5d60471f8c299c3895d533c0", "f1447ea2bf2f407498a681c979f19f87", "c777354dc2a74c989aea1a7e124f8a0a", "33364a8ef70a42a180da52b588e24c2d", "d98c0e5e5036425fad31fcefc0328a35", "63d239ba47464fefb501cb55a0809247", "bf645066c3ae4988b089f011ca4652f9", "c30551b397a74ca7bf1781652879abb7", "b9c4abd3e3514eaba653c5e8e6cba3ab", "67f779d1dbeb4c19b93234c661d9fc81", "bcdbdeb2d9884dcc9bb2513c766489a9", "e5d7cf18a8c144eb8b34cf6aec371ab7", "aab800e3f09843ebb7a90dda116f9284", "934fdaab2f7a4cc4800c143d2fef2d0a", "3591b4b8cabc4bbca174454612196f8e", "69fcd6dc337b4d609b0e08587709d149", "ed2539e98fa54296870b5c4077fa58c5", "026334b1def343fcad67012912ace919", "6b083921fe284af3bfab78ba312af1cb", "93a9ce08dd884ffdaaeed57a37fe11bc", "3da464e22ab04d7b94e25cb2082a61b1", "7de19600755345bbbfc475d0378f80e9", "6e55195ffa484f928d28b4871a5940b4", "ba36fb3c2e6745a8aff0de5b2c356d5a", "deaa43b6c569486194db11071692836c", "b63fd09f9c4f42c39736fa72955e084d", "055c1f62f129464fb3f25d1c6e973c8c", "45e7f4805be541e2911e76901f554444", "471837856f5e473b91bf208667e72272", "a5a85b95460a4dbb84bee632d148ed79", "aeef93728e2c4c4ea50ca96a2a688858", "0223038c63944b419512825e5e82c9ef", "43ed100b88374b5693ea7175fb315127", "09d8739cd6eb4e2d8820118b04d58cb6", "2673c4fbf6d64f888e10c3671b06a647", "c23781f26be3431cab9472f6b45431d7", "eb5f407617e24574be6ed92b665d6826", "b94d0590737a4652a855892aad64b9d8", "5f2987b175564dd3975f1d14127f9449", "7b33b0a83d734236bf958df899d5f2b0", "1ea5f52a204649709da0a53a3e861377", "d34603ea9baa4aacb56662d6d18f4625", "370b4d4331a24faebdb2a67b0f02cf7a", "46611103085d429aae0b0faadfba0669", "446438be19d04df1ba909fe7e5b02158", "be9778f32ad44df39e77d60c7da384ca", "98e034779ce149e1821c15de01fd4626", "b17a2ea4bf3b490cb5c4875fa4c9765e", "6d1593c3684d4d31b6f5697b7869990e", "45aa6719c28741d3adce3cac7772ea5e", "e411fb88e3f14e7bbb27216b9192eae7", "f646146d80964311b5f038e2da1fc890", "f19a3275e1a54edbb65ef288c1c32306", "593b45981b0f4ad1bb1b99a529ca7165", "141312c886e04640ba26cf90aeae96f2", "9d1c7f77312d419aa558db179fca65cd", "ac91412c52984567a8783b883ddc25c2", "1465754c4cb44b23a3d9e7aaf2c1c9b7", "c2c89dda842648eaace966dae7ff8dba", "ae8e07c69e8140448a4e5c4746fc5de3", "aa533a42bd5c4550a66091e18b60e42c", "80c4a0e3300e427dbc73e58275d2f84a", "2d2790be7ee24eb8b3bf9390a4a03fc6", "ee262de1565a4e1397f917de56d4e26d", "8b6633e049484d60862a583c16921fef", "c00a2b46673642a28c253d6a95fb48aa", "3332d6366e3141a781d986259da74f48", "61e4f42ce57747258c340a79ed12ac18", "ecae70fde35e4ff3b7b0d06886d0fad5", "d02e1f95c443462c9ee5149df9ec377b", "3d45a62d9b8a469b8bc28bf08e4acd57", "3befc36e19dc4beaa0fc6e69e95f9f87", "3b794f642aac4396854e593298409f45", "6425bd194c0c44658d843bba5942324e", "df35c7ec3c25494689186ceccdaefc3c", "998fc7d4abd549c7b75137122a35286c", "95f1c9b4a33d4ebfaa336a3a0f8c666b", "4f05032cacad430ea7cb1015d43e37df", "05b5e186eba54618878333e97f4f5bd9", "9611fb6ebc0149a68bec06d472b0d38a", "00225718e63c450ba94ce83f16cfa1d1", "6989a603c3094193a2044aeb32f2244b", "c9c0eaf6fcb147ac8d864feea4fe953a", "9dba38128da848cfa60b41687f6323e1", "5d7248d7e5a1408aabe2d7e4c884f8ae", "596f5ad2d1644143b639450d4f3083d3", "723eefc8ec1b411cba50c9749f99a1fd", "8e509354820241ae8feb7e08c85eb6a1", "af7fdf195cb9453eac5a56897976b8c3", "64f259fc1f1c421c8c4a5f98752c075d", "a3fe2e3f27d24ae4b8486d64c8578917", "16c8942edf3a401ab5960c7c31fbc86c", "4b9ab7e4c6fd4bbf9ce0e59010135e11", "3c547cdc4c09471ebb2bd34aa9a6b5a1", "a3c755bb5acc42fba5c13ed568ef0e2e", "84adc0a4bdcd4b8a94c5c6c9c90bbe69", "c43b4d304d774cbdb2041f758caf951e", "69ed50ae3a4342d6a63a363dcac1c244", "e4526d8a4fcd4d0f8172898cb64bdc7a", "891b5166e8244340a963984389563062", "5ebef0d8067f466bbf7055e2c99cfd2f", "bcae53d0d8ed48cebb265c7bb3b621e4", "6a0639171a444877aa3420105a5dec4f", "438b9427808a4cb1b6c29d54f3656d53", "270489aeb557431eacbd1cf161933c02", "12cb8fc21ad84a2f96adbba4fc107c97", "9d642a6bfcfc4cef89067392e3b31003", "87c651bec9fe4fc7b00827e1f1898e1f", "7b21dbcd8f2343db9ce9f042900168f3", "faf1cb3b6d05458593e040256633335c", "03eb6b85b5c44fb992e92dc3114ae93a", "f450a298bec34cb980f0b1437a8097e3", "f174bb28ebf64ee48354e2e64043120a", "58036167d24e47e6814338f721e8f96d", "6025168b2d9047939c8429243f6b32f8", "f7ffe5e1525841d185968b01132f0b62", "8d9a089a2a0a434d9ab51df587bff4d5", "69f161ded57c4d25b07dd44909b7d29a", "76e32e2a2f324c7e81010b1f1d11091d", "e170e3239d47463cb9bf1f2489a04e23", "ef55daec61e5492797ee34d743aad2c0", "aa560dee4a204954a2a9c164cc952383", "d846746544ce4861a983e992f61b066e", "7f9851b3f55042fc8260019765406b36", "bb52d51d641b426c91e980fe9b6f7262", "4f3015440f58471cb92a9446e8fdb10d", "48682de084874048a2da532203f62b7e", "c8ef7c17e45d42199c22122e3bde0e7e", "c90aca6811ed40359ff092169b594f8d", "7d7e3e224b354a77a47f55069ede8583", "897029c308294e119a34098792fc59d9", "7745426bba9042d98acd63cd038c4235", "e464a0f9f1904355b94bc4fa6fdafa49", "856b7943748147b2870c565aa31e7a17", "9a2c12e433904b438541d668282b80eb", "878115852a814d55a41f93b4bce342fc", "7d0e667329c24dfca573d84ab9ad1faa", "50d178b5ee974420b6fc15cee922fb57", "cf926be0e5ab49df83210069b895154c", "424a52e6a4d849c088c2105c0a895dcc", "8a67e81983014be6be61071d499ac471", "3fe14c046e7847408cb7d5efd62e5a67", "0f9a7412ec994443a07bee967d90b7a3", "826d9fb751df461a8154f4b71bff78bc", "bacee6ae58ea4b44aec03cff3fc1c975", "3d9df96ba8884232b88fababa8d14a99", "0f9a1ab760ec4338a689601f88dd5d36", "c0d78b2f10f6414a8a2e9d4a6acbad17", "ff94292056044e5c84caeffb4616eacc", "b80323e530924c2d978663c06c629504", "e7c6707a2b19460e92fa875acbacfbaa", "bdd5cff107d54ce1b243409d49fe16e4", "ebef4649aaa54d10a01518e368b81203", "2f1a77a04c9144a3a7f2a5be3ec6ba7e", "b62d07c2178d420080214535a27c542f", "859c6fd196ab4cfaba2290618b263f95", "3cef2efb55c6432fb7427ff9b7958733", "bdabe155221646558b7e93b2274386c3", "5a64a0e2b870478ca6e0b742028c33b6", "b6659266b2b444cebd0829321021be25", "cc121cb685334df8b44ae527c0ab0e40", "955284111ea84a83810539368110a57a", "eeb87a459cf94197805a0705a6615bc2", "5723daeeac94484a93c4f5e0613abd66", "08d2e339d4e147b6a21c68c7a94d9360", "b5fcbe2e61af4539ab15f6fd0175c99b", "e5337137399c45fa9564bef2db548535", "2d815afe983645c1b4e35816f9edfeab", "090b1277116b46baa33cb2bd45c57d8f", "54984cd912d04c88b2393ff7139f48e9", "5a70bf308a7b4354a849c12b41a6e964", "87529d11d50845ffa5fecea51c843ccd", "b463f1b4e8e24051a71235d0283001f1", "bc6b3bf973914599a2b5ad77d851a2b0", "ace3925cb52547cdad2f3ffebfb7d274", "c8f410d70d714f4b900ef8f1613a4ff3", "919e382b5e9046ae8041734f513d16d1", "e576cfad55c341ac9ea1cfbedaeb9a8a", "c9b7fe1568774261a19042c40f32eec9", "dacd09f2f15941a085cf05b00ecc7864", "02f16b2977bd444386edbe36dc17fc26", "0f6b2663643c410687eca6db82eeb30d", "a6c93158ef59498c999a96dd37a99b82", "957ba1f1679241f1baea494891a191d8", "7df6ddc5667e4da7a52c9feaeb29ace6", "4796c1688acd46c78b390a6e0291190b", "774379f397f84cf6bafaaac26f936ff2", "a183929af9574e6ea560594657068c3c", "77f58e006afc453bb0e9a2dee80ffdbd", "2373deffd5ee499e8392146343f5d33f", "b668275712aa4bd38a0da873a1f3ddbb", "c7a5c39f042e42439ff7cd98007ba05d", "37b1d57f5e1d49f79394ea3ad5b39848", "3b9e97cb75f84dc78e18003d5953be39", "cfaecfd32b3e435b82e3937744684c69", "52348acc658f4f3c9ed9e545427cbc80", "a26d35a2f7044b268a37c8621afcccec", "53772726d7d044af8771706da415c1aa", "b71e3d1bf5974efda97f34e5bce65272", "da5a2de2df8f4a6586436f9fcac96fde", "e2fba62326c549a6952d459892cecbb0", "a6f028f970264d85a41762af4878c6c0", "588281dfc0094448a18e0e1730c978c7", "868e859fa4434ae2964d7a6189a37a9f", "da036a4cca994b8385057f3082fcce29", "30c1d8cbe7694898877983309f3537d4", "4fd669243ab24b48b2ee1b216cfd94bf", "08ed9f4007204f9dbd59e51ad0574b63", "6a00f8f807684d85ab0f8e0f80fcfe75", "07bbfb48866d43369d03eb473a322e76", "aa6e8e40c21b401cacca1a386ec2768d", "e228e8cdc961475fa023ce39822ca9ea", "57e7cf186b6a4e249f61f16ade026e71", "ad913711aecc414f9270c8afb024b839", "77c12b090e894d4fb3fdbe27d6c97a6e", "83048e8be93745ddbecfd9e83634076c", "677916667e50405e96f03271bc27a976", "1dd4e09d32894fbda09eb013186b4fd2", "118cb347c5bc47ddbabc91c95ff48d0e", "83e9ba3135a243ccb13532d425d8127d", "d3e8ff4cbf3441ec832a15751702a412", "a268549f63cf462dbeea73c8c14587cd", "82e1eab92a634509b3d37e032d9e37d3", "524f61a5380d43719bec661694762f22", "e5fabb91d212429babf838784ce5e509", "06f2ddbf84c34e428fc335b7fd406528", "fe20a5188a9c40e69a8d1127ded70ab4", "815f57e74c0246e481af9abad78ca229", "ee97c7cabec24cbdba691c084fa3153f", "36903124da564e13b23a6912acc723fa", "5e4c79bc2ba546c3915359321489b039", "742a5a509a004e8ab21fc7b00b07a5e9", "e87068f3ffb542249345ee12d60428f8", "89d6773a02d64eab8d32a88e17f45566", "7f2f32c45e3a4e2ea519c78402afd757", "d2925af8ca47422087c6cc14a41c3da2", "1394dfa7f90544808a169ffa32ea74c3", "933e3044ec8a48a7826f6d40b3b7c05b", "aaa7e0c06b914f5a9962f70cf02f1010", "f3930fc135eb45c18c7aa2d040fa0d6e", "5861e2216a55471f972fa3824c7b8197", "5ab033e597f445788fde36eb20a2b016", "d5cfad4c958e4c698fcece27376b4a10", "d3bbf6b0f4174677af39039736359139", "89f5a687b63e4059b682ebe5a83ca2e4", "e2ec7effc76740efa61371c27707de5b", "d1c8096aea9d469e84dc6274336eea39", "94399df20907465b86a1693e531e83fb", "2ffa0ba22fe74242a123e0f6f37eded0", "fc10392b51af478caaa79bf7b0837138", "e2a3da7ef39c47609e9e21935d34025a", "5254854b0da9453f87d1342ce49e3316", "dad4c4b847504f46b7570d76fb4d0112", "41ebb69575ca461e87e95e6c4ed8037d", "9bdfbe93e7f34effaf21f8a181444219", "f5e1c43ef75541e2a5cba1f9d6feba7b", "e33f2d1a9478410095747c0c8913081c", "7d8d8bf1bab640a7aaf5aa004ae7230a", "734ee7cdae544eb9b51cc322e0a92906", "bde4a5feaeaf4d298f008af1ed76bcbd", "5731cbace2e4443e86910765b0dfe19b", "8a2977ee469f40a787560e0121516007", "13eeac211b2d49a1ad75b40158f10396", "483fe24d121d428ebb757164073c49e0", "69dcd80aca52455597aeff9cc559b187", "9e3d6c79828746c1854792fa86d1ba80", "baf381ec85234ba394e1029bf672e347", "b40a334f6a0641ce9c64546a40a19729", "1c9e53a22dfa4037b3e46928816e4558", "895db22c82474312bcdfe41152126e03", "a34c3afedccb4ec2b7b57ed5279b26ce", "3f416348689d4227ad87fedb6e53ca3d", "11765f4eb2e94649818f29926fbc0445", "d330c7479ab44658b5bfc363633b1357", "dfc5ff44928246daac76304c9b2e1775", "92f50f7db12840378ecdaa9ec3f07d35", "499ea137c7f34cc59aa1959113a6ac8f", "a46952b26f56411b90a7fa8e8b8f4a37", "1d701174cd3d4504a5d1f29ae8494039", "fcec5a55c7564d21a11613ee7b04532b", "dda3c9fc910b44d0811f4aa6f2a22ed1", "4f34a181c691424aa81a2c5087100617", "5852b2d51f7743ac90e5347395cfdfbc", "92bc239197964f009537f85186aea9cb", "fbe94ba7527c488194cb0d50a79739f3", "8482e896786a416587308813864c6e1b", "289d4583705541d895e902cedaa65955", "09c572471a764de98b92e69a91925cce", "8312e4af5f2849deadae2ad5b04a2fd9", "710e281b33eb436f936132f5fb8001cd", "d4963a0d858f42ec95d5c247a77e2c14", "acc5b4652da241ba9b3c58e5eac67f16", "a9c66b4c1957499db84b5ce595014bc8", "90ecb282da874065958050b4fb8de815", "56b8b51915b14f8d902d5f1079f8bb60", "7de7996138aa4227abed4729a23bf79f", "41d5e69aa6f44a488f93a9e248751402", "3ffd97dc19b748ce98851eb03bbc3f7a", "e9b7c18d01a64aadbdd09eaedba20d7f", "1cd04afca8cb44d995701224cb38268e", "6d0b878bb31a462ea06c26a52b4fafd7", "a63a35bbbc6d447e97f3ad28102a35f2", "c5c410705df74b7d975054ad6fe53072", "0522985c18e74e05ba7015e24cd01bcc", "2f9b0a66347143f8ac8c4877dac56793", "1884971724d2401ea39a261f5e692f5a", "8118923078294687a207606056bc3977", "b50c47da46754d98bde01fbdb9fb2be8", "47775e834b6d42bca4b82908868baafb", "984c1aacb2914c2693fe34db5c655713", "cdd7e0aded044c839bf0de578afebccf", "d1658fd242b04c5f92e5377e9ff7eb3f", "41e6dae6ac1243b6a5dbe97bd0db0782", "c3013199b0c84274b2022d079a29631e", "3d63f87ae6e44c389a9216dd75d49ff2", "bbfa03c25d7044e0ba7b40dd24ffff5a", "d493e6ea11484282809193f2a1c3f0d8", "536ef61504b447d28d7d1bd7a153362b", "f47e0ac42aff44c7b175c283b2d1d5ee", "dcdfd50be05a454298bd752fffc0f208", "d4468c0e25b54aabafb838572c9a1c59", "23069005c64a4779b0bed854e9202287", "f289ad3087c24b6ea6628271f4bc79fd", "d84d4473603f466ea8d6068111dc36a3", "d98dfb1fbfa543ef8da8d725b2360547", "28fc4212560742ccbe66b3a222685b3b", "1061b4ba525744bea041dfe6e7c0645e", "81b163f2de1a4390a1c83a7837e2fbb2", "5029005634db436aaf8b9a50b9af9423", "20d1ff7fef734436b3ddacbbd00c2476", "fd31958f622d48d48e534eebc3d71fd0", "505a83b6e1664b709e43892518e7189a", "803699cf855747d2bd6f0497db353649", "b4904a893b45499cb5030ecfd5b686f3", "20b16b9db17c48d084f7cdaf4eead9c4", "9a0d352e0f3d459ca5d31689ad7dd82c", "590c870a1582433da07b936cb86b8eea", "e445ef349f6743bb89e384aafcdf10a7", "60be188e06464852b77c11a86e9920d2", "c81f4238924749f6badcf64ce44708ce", "f1425c4e64a24b15b716487861d3b3e9", "f5eb06a469744b389d0a7754bdc429da", "747d3b4c2f0a41d8abd5c02da5d58493", "43c3ccf2d4604866ac2f4ba6c827ae9b", "e03506e411984dffaa337386f2149086", "42de9f418f47450db8a38632a6f4e3d3", "f2ee96fc12a64ff1aaa9a3947598332d", "7a49d2e1e1f5401da27591206d8a80a4", "5da7e2a4e8c745b5a8d63b33978f104f", "f74bdc5ef0e848fba1a23fa1aebfc991", "b09231283bac4c8eb7b9c3429f4e6a52", "5b2e5fbcef5145208048650033f5e22b", "58beb05b344b4111ac6b926f08f68d45", "6dbb6946ab90437aab59ab4a58d0d6d4", "da572d0b1ee24bc394f1e5cf803532c9", "6a6ebb6d5a8e47d1bb19f3bdb72ac0b4", "094b548a03204277ad935c6d99148c2a", "4b593a69ea2444a3953b77c03760c48c", "6d24738250a84c61a2cad6ea3842300c", "c97e63913683439da17f9f35046f870b", "ffbb127876d641c1910bf9741dcef5a2", "98d43f6402b94079a2b5ea0e30e18f59", "0b4e4666ee1342b0a0ad19913b3b39eb", "122e5ecfaa384edcb0a28884b001fb6d", "0d459b31ce334dcfb78541771dc36311", "4fca8816a30f431eae13bd8942659865", "d03596c448654ac083f8c42d426c4f69", "6bf8060d00f34d7c87f05b61224ccc8e", "e80aeab6c0be44fbbfbdc61753f36059", "383d2713c4594fedb85bc6166ce23763", "152edf7f19994a88807c9e13bf16bbc9", "22145e99da0f4dc6b28d5190167f4941", "2d8cb1cf19fd4649b0307490f0b71d3d", "51a8db6fb30045bdb2153f13da8ad58b", "5967b85cc2be41a788a3100e87a5aafa", "0f0c966e4a414fdb81a863211b46ea10", "c1123c0125e24532b8fb6c623bc6f360", "43f0deba12284e32b32941c01b82bc50", "5097072b055640a7abc27825f2f6f02f", "e4a7dec0a01d4058b005c470d55fa17f", "d55f18ac82a64b669a6eaa5d80edf1a8", "e47f0056725f46a99633c21e078fddce", "efb7d457a519411bafb793c20a127f93", "328a1d699ad6491ead743e53b01b5458", "97d265e110f445d194f6bf2a90131c89", "925249634f5c4d2ebc31d1060b0c97c1", "bff245c7d6ce4e88951de8c18605d6be", "f06817f12ee34de4aebc09226d5b18ef", "8b944d4641d9455c886c2dc78e03cbf7", "d945c69ea45b46f99bd8ad21f34b83cf", "cb4bb3a90bd94daeb0b38f773de6cd02", "bcd3f99f692d47eba46bc381a74dae14", "454418ad77eb48f587e6239b691963f7", "233add51bd8d4aa7af34027c6b4f151c", "6e946cc521f1413dba7d3bcc5dda852d", "92518e00cbcb4bd8849c14404af720ab", "5c570d0c956b49888388f3739fff663d", "3fb7eab5913b48dfaa1bb1615cb01860", "5624fefb95d94171aef501c20664142e", "0f471a8ecef04020865c2fb6cf571d2b", "bee3c2dc169d49838086c6325815eb4f", "5c9c9ba26c0543a3885b91d735c561e3", "816ecc8bce81433b80be761271bb6485", "cdf7f69e87cd497aae14368bc1f0d931", "d9cd768418894019a5eb112c62afde8b", "2a9d05a8c0a84b20a4fddd515825e473", "f5440af091af401d9837f09ebc9f6cf2", "916689e8d9e641b0b13decbadb5fa50e", "c5045480fbff469b8ce940700a121401", "b924366e49144d0fa0e777788021264f", "57b6a0395b484df3be0e6a199c98c034", "37facab53ad24f7ca6993da045c4ef0c", "69adea971aa34bfd95865ab320bce997", "669f2730199a4d488bbecaa9884519ee", "692a19eec0154c9aafd25d21a6f15da5", "4849420eb80d410eb23583ac097c53a6", "ea807e42150241b2924452a7da0a2fd0", "9393fd8dc8744f36a82583d86832fae5", "4f065ae01f0444e8b307f398d31d8f91", "926cfe08c4f740168c13efc412419156", "70b9557673984c74895973f7e088e612", "570c8ae5e6a54dd789beb14ca8ee2034", "02537061327e4d2393ef9c5038ed451b", "3cfd054f1ee24c3b9fc49bccd6a0de9d", "927089dec652446c9fc74cbd82de7945", "cc768c55c6564c35ba49c735ac8f680e", "8fc645bc7e124fb7904ee39fc454d355", "8a744dd6925f4b6aa6adf7934cb61dd1", "5e110b94076441ad8a1faf0f646a43c6", "8c1d57f87f9c4cf8ac99b9d10d3ccf5b", "f65475eecb14437ca1cabeb0217e1359", "d012472bea6347858ff477dd21adfaee", "5a59a833ea84447e93ee805266291853", "1d0bdb7d89704f4aa8f487b9da6ca20b", "8bb3b1b500564a4c876a7db4481f293a", "6dedf5c80c244f729ff6d9d156229087", "6067dcb19f354b6184b968d1374bdffa", "7cea24de3e194b6799fb65106962b21e", "b17d92721e864d55a50d7c25a17fd214", "6b65656d21254cd4ad28469d0ec92883", "cf4ebfb62bd34f31b67d321d3b7d5d63", "85b2b011ea5144869102b3984cd969bb", "d0d423c0275f41d0b038ee258bb61277", "adb3165e091048a4a0cbed90e96a8887", "f1312878aab04fbb8d9ae3945fae631d", "17fb8c0eb0eb4a01b2d628f01d38cbd5", "956572cabb254724a2e3e039188b9e0c", "5b984014d33b4ab496bcfb151fabc3ae", "ff14ecee9285482ab0d107bc10803e66", "ed2e4d1792a945c7aa6a242b71f9d7ad", "70fbd2faa1494e0182690a64be41d24a", "890ca402be59437c919ea9ca89343e65", "40773bcc542d46fb9c2347680a117c58", "34a0937164b540fd85a72a04d955aca4", "6c8b6aff2813496fad4af28b014acc33", "5d542f0fe6b4463e89029990b3264cac", "68bd9ac797564e719d31750b05f96ed4", "b15a21be83c94fe484a5ddbe480c9acc", "54426131ee364a53b2a025fe3bf7e0b1", "39071f4d084b45a29cfe0a0306f3de6c", "395444bb8e254a048ac2397e2124ceab", "32325ba3d15a448587f0488f6db4c466", "7510ee51359e46bead2d5314f4251e9e", "02298b42f0a542a58f2c766649b4457a", "5a3d443f4252402fb273f4a46ce64374", "6ae4934e8b504e288799884c9d879ac5", "d1d72e74ab8b4f5ebf26597465335d1e", "fdc1c7e5d2904e9490095fea5f35d022", "b7e6e38d38984dbb8f81996dfaf8112d", "f5370a0b67d54cf88bfc5842a8286be5", "c86cd4b3134f41df9f86ad2c56b82f73", "0621a48fa6ab41748e0821094384a85d", "db7a2c1973b64cd196179e46bf672be9", "d4f7d0a79821480c921495138fc4d99d", "3609f252419e4d8aad94ec0e72ee6ce3", "a16b2ceebb0243a1bff240987c66e4c3", "f868d6061e364710b1327a492df980a2", "5ac4070649fc4a5f81387e984ff6053a", "b0fe6932e6bc49c88d11547cedcb88c0", "aefa0e17cd8d40bdbd8e33ea0f4da75c", "1ecc9e9035a047b4a5bfb679cbb67b7b", "9be518ca864946be8361770dd2efae18", "64ee547f3f4547efa75d3c92cee981ad", "aaee8fcdb4be4c308a788b1c283c6085", "a0116ef22f6e4fa2b190dc1cca47a512", "304386d456ed43aa9a12f004a0b2ecd2", "31009a84346a4fef903fb826ae4c33ac", "95f79762b5b64514a24cf78b193683e1", "7d2f87366a154a3087b3a5c5570b384c", "490e23ea44bc432eb510fde55093d6a0", "5a5b084fd8eb4b5481b12e73ec463c26", "8327666852a84cb99c623f5c910848c6", "ce290c4dd27d4ad2ba3bfc3cdf922817", "1834fb380c894ab88130b3f90de10a88", "3e042c1ca94d42ecbbb42be08272ec26", "fd938f11b67c4ae0a166dd4168adcee2", "3323a1f1ac594767b2683f210e391e03", "480a8d8d06a741498b32c7b69b1b14bd", "06c18c797bfe4c12a9391c453c4ee1ea", "57c5e895c0a54ee995c7c56f98cc520a", "3eae539c7f2242ff89ee48165e8ed8bf", "1e58d67b38574081bb11be37bb062fab", "25404b7af6f0444095f321f3c14b2c3d", "770fdc93f68d4b88a2a9f8eaaf58f762", "db716ed68b3a4d668968bca1dab2efd4", "16f94b2cdd7141e58305f21687881587", "5bec320cafb9437fbb8deb815bce1600", "4372f6ec85a54d08a7313e9ecec0ad00", "0c82a243545d48558fd9a0122ed941c1", "a9a5b46051a04e93afba51b33231b642", "9a22940fb74f40c0bc97a32f89e84b86", "658c4964f2eb44448c915a9dd73c6ac9", "e94cab97447d45e0bd6bfdbc67d04afd", "f6d833a60879448eaf9b093c5ebcf93f", "f3d28a34c1b34c8d9b4a0793189636d8", "137e453d9e274780ae5c07ceda32ffa4", "47d7a76093b0484ca3726a92a1da2e28", "5dccb58f31174a8ab295c532de4f39d6", "982182a8e3e340d3bbdb260d0001f5f1", "68d1dab48c75421baabe401a87b8b0de", "7f4c4f86b472463698ea328cb0c620a4", "62ca07d1e21142eaa522b04d70d83046", "e700680034444f64807a01b68058fc64", "d36953f2467a4f478bb25c024d71b77c", "966acbad5762404580165e841bdb08f2", "187ad54b35d04d6ab6948d2fd6ee612c", "d24976319bfb4b879311599b439b661f", "644ac23a7b844d9f8b1bc3406f7aaa50", "759bb65496124859bf4483c6f87b4ebc", "eec869eda6cf4534abf87cd8aa842809", "29b6caeba86545eda2f60cd2955e2814", "cda5852ba36d44938e6b5dff7254d0b6", "662fb30109b14707a15d81080e0b0e90", "3d467471e81b4960af1a4a02c41578f1", "3ee1c7982fec45dba06a0141aa6e1945", "d13598cc28594718896472ce80477266", "fa3c683c18e94548b4be85c49cf0fbe2", "80b6aa2facb3444b8acca1141df503d1", "f65c964ff79b4bda915e19f69e07057f", "2ad93720e12d4c7faa1166f483cf954a", "0008a68a808c4db587f0e12e013c725c", "26d29322bafb41fdaabc5bf8a625af85", "01620b606947487485dc15516acd86ad", "23e071b2357547c1bf4fa12fb601cda0", "d85fee3750974499868ae09a4730399d", "867a1aab1fdd4c5ab0846e4bbcdb1dfe", "e098bf80cba94750b0e836bcc3b028dc", "2b75938a6b86443eabfa20cb555a887c", "2604cd8618e8487b8621710e5a57d718", "ace1718d5bb8439f9e2641a9d04c058e", "f9af04a0afdc41189a235bd9dd850b2c", "ffa206e5164440a4be9790dfb98589ef", "5180326af3df4c8f8b91a2e29a3fceae", "2990fd5f4f304d0ba9f17e70fdcf4f72", "ea5eb75c4fa24eed9952591427960cc2", "92c4d4ed735c4688949a9302b55f3c20", "a67c2d19c52b48b18fdd05e7374ba6f8", "b19597122d774e8ba1a47a06580ad6af", "6e29a9500b5d4450900008bcca24c1a2", "40a1fb1ffded4cca87e2b9152d324684", "5b67dcbf224d4d5dbcd4130d374df5ff", "42d47d9c395f4d5fbf19489c5c637d6e", "e9ac26e297474ee197e52da0eb26c317", "20b07adcfddc44beb2fc23022c622c06", "dcd469687d4e49ef8cb906b6559a2f13", "301058b4bc244586a40f540cd4ea94c4", "ca6fd4212ecf4d81b749450f3e91a773", "245990aafe2a4c4b98e400b7616b2c0b", "4aee3b280cf24d95a321f37a2dae8ce7", "e266557d278046f8a28b95a8052f4067", "8dc220d27ce34cbd92150494c3266b67", "0361387c1f7d463f9cd5478605278447", "5aff1eceaa53469ea0fb46fb5683e0fa", "c30d907705e14ca287ee261f72c2bed6", "d349ccfedf0e45a79f4cdf722db36e26", "c1b9ca578d1a408584667450a6a593b9", "b3936045aa364123b48b423f6f7291ca", "976760bd0f744192beaf2ec1785db3de", "4b648832d1764e00b8b9cf9487538210", "dac54815376d4111bf7a84dee4bc7ea4", "18937af54e8f47f9bcb49d2bb123d5d1", "d48cc64e912c4e488a60063be3c24985", "b7a586302fbd46588147af10c9da671a", "69e6b23818e647318abcee8d854b7ca6", "fcf210f4ea76483aaac7993a80b31597", "71eed2de51bb43b197668f369fb03b3a", "ce586d00d1d04353b38a5d69740a6947", "e164986d30ff4b60958ccea8658cc1c4", "54a6a19275cd49b588d23a325af76cb5", "544fb08fd8c54171a4b61b8b4abf9a7b", "a495bde8f94041308fdd90756161f4b4", "2535c23f7ac943448164419cee4a4828", "f6accfd56bd54cb8bef8d1d752513ee2", "1e468339d96d4327a285c41612c994b8", "984c750e6f7644d6ba912657652ef74b", "e129931fe8eb40e0ae6a15241283e87e", "82315fdd195b44b1bc1f64c77a6740b3", "e058b3b999c7494d940fc37240a44f06", "e00f31f5a25d45e09f1b9d0684913477", "81d9e43ebbea4794bce2f9329a9ba099", "63a1b43fb3474293bac2a6cedc1c1061", "5fa54003e6de412e830324d8495b80c2", "b1f01fca5d294226a30afd6b0722a5d6", "2fd346f2fa5f43a2b6d67ffa7ad53273", "286680e769854027b1f9c723f1242066", "a6d4fbceed5346638f9d7f3b5a8e163f", "085b90d949644b6e81a837401a642d86", "b3cdb59454954551b691b8b62641eecb", "13ab0a61a355457cb9f927bfd9cb21ca", "8613cdf6848c4bfc9f2ab3d347c7b334", "720341b863be47efb072563cab99f2a6", "fbf2ba452a344444b47e98eeb8d06dab", "10ab5f0d4d434236a7f135dfac9d2c1d", "43e44bed78e74e94a6b223c872f99788", "57a7be9aee5c42b6b9aebee89d99187c", "793a25266fcb4af28ceb4760ff08f14b", "11d28436bfcd4d46a8728acf17f305c0", "54b65bac24dc496b83f4ef905afdad37", "cd39ae1a293647af93a8d662dcaa895a", "aa5ffe54d9df41898af584062c2ddc78", "0e29c483f1a9448f9280c928e09c6be1", "c4082c8d8d10427abf33d3a7cf30073f", "76d88e5917f6455b8ec49036c82eba8f", "6567859dd9fb42c08300408c7129edc6", "e51c5880c9e649828dd62741741af92b", "6be0a009caaf4299b96c9032fa32153d", "be007d2da6ea4ad2b3da1c9eaa7a998f", "bb3d63841e5842a091b18e54cfb0d41c", "bf1bf738485b42189c98a5e6ab3301ad", "f2dd7bb72170404d8e528b25a7bb47ca", "c291210009a94aef8f4a5a8cd65b5467", "549c4cfd3e3b4df4af95d18b23409235", "87ca8c50ea2445c9a56eb69df5089ea6", "b9d78c388298448b8fd0db4a23c9767b", "6402d6d9048649d98057b64c56699f10", "dbeb6c2086024844814ae823df9e0e8b", "0b275f8e7dfb436f956f5fd686d9cde0", "2dcd7770724846c49ede8d8faf2add61", "99acc13ef3e14a0b9865d3d757f352bc", "c45d0d40915e47a68c111c71cdc48ba2", "0890a8517ae14bfab6071698d681ffb3", "0b88a5ed769c4a74bd21d175f0331f52", "7260be2369544d01bc515216b0b2a350", "ff3a643a55e046bab11d312c70ea3a18", "19c625e0c77f415d8eab877115fe1009", "0de7462ed1de406bac30c925bd5f9ce6", "8f2d1e8b143846eca4d13721bac02dbd", "35819563d66a46fa963175434402d7fb", "abe789c214bd4bceb0446b8179c70d82", "1abe5ad4f4d64823ba8baf28c1f59b0c", "ae0d37a2936c4904a3446fe669f6596a", "7d5e0f75675740609a8404685d4c5884", "7a99cf95025b43ebbfb89ba606344a24", "c12c36f2ab634eda8317972c22695bde", "7c91c2e5d14d40b69444ddb8a6197ac4", "387d69e6d8984ac38f91ea701061e844", "9b854b4180fd4863b13e1bd46e72dc05", "558baec73a314593ba01ec59248b47a7", "5e97b0f9672b494e8ee3c545ee877dad", "3b3454a92a0e4017b566b1937769786d", "e9709304e70348f28040e681f015ec9a", "94a0d363684d437084b7e68c72746766", "0f6d44a640c8435d803a3cfc9a9fa57e", "a9fbe893f5be45388829d1cc5d91a68a", "755d0b9410f14579908cd223f2f95180", "96defa4323234d5fbd49c5d610d2a65d", "bedc646690a04b1d8225810585d00ea0", "ee83881817994f8b96e2d2029b79eadd", "fb80b8353ccd486fab7bc90c27a348fd", "0e3c4f21647f43849e2b3ea9e0a5e89b", "5949353c79cb4afead8d5804628b276d", "f97a3b2cc51943fa884a97b481d48ed0", "4785c76afb364e09a41d0fa717e328a7", "aee1519cbeee4d62ad57286df484bcec", "72a9d999b4f8432b84f3af295cf28667", "311f034ed1ff4af993c554e3b1e1ff72", "64c0d1d64d29476c9ea427257c032c4e", "4dbaccc003744fd89579d97eef93bac8", "56e7852fd5a34deba4bcc89750553898", "1a78996412d34217b698d6fa7cb12a7d", "fc2d81df460242f981c6a0039e194ae3", "3ab23cad95ac43dfbf2a24a3401b0bd2", "b1cb95aaa5fb469f8d3771c97d12de95", "5e834146626d4a63a350c9ef26045c78", "b5075019748c4bde84c8d303b0db1665", "4996bba4f9f44b26a47b1b3940b7987b", "abb127c7571f4cfa8affdc8b322828ec", "2cc108338d274941a1136a9379db54b6", "40cb1ad2715e4190b4a05a91dc9fd174", "83da4adf0fe547a28c106f40a0078149", "20b7d5f0520b41449c575d419c129c50", "23e0e94fd0fe4135a8b7ae69bfde6ec2", "6678ed2bf6e6448d8e73f0236d920a49", "6b0d32a408cc48508666a5589eed8f1e", "8a04c994d24b4ef38ac705a84bea478e", "e7e2136642284d85bce33d8807f4d04b", "d13cf8bb9acc45829f84ea269d1f4097", "f74236865c634c24b4cb03be4b1ed06c", "abd4ddc9daf3490880a8b7438e673227", "a94e5c434be94dc1be8e7b181506868e", "5fffbf0fd090412c86efb6b019e5808e", "559e6854c48946c19e71ba26108520d6", "affe49b757094278ad1ed5127adbd3f3", "76935c3ec0dc4bc9b111985e104eb9b3", "5482a50c8fc84d9cb6ccec248975df4b", "c5003aebac9840ec8d41bc4629390a76", "7399d558e7b84c68a7bac5e0aa040762", "b5ed94eb86cf4c07b1413fac260513b6", "f0ef5416eb7c4798bcd63e73d4a5faf6", "c55d391ac1bd4a89b49e26642845e172", "2accb259743b4506ae2fd69dff933e03", "9ce80e1f0ba54e13b3fa760127e9dba2", "7352a7e53b9840ebb70b9d42cc854169", "98ff70a14b114d38b16fc6a031ced131", "8dd588dc655b426faef24a35e7d00b38", "947fda31086648e2a38788c609309286", "1a4b394e9a89405f9ade743954bfeeb4", "85926c0b98164f479abee4b9f4a19d79", "03bd6e03babb4a8ba88f71a9cc7fc647", "998cb40c939f4d9da2984d80e819f8a8", "73be366ed3db4f3fbab793bda75678ed", "dba72c48a663470eac4ccd9ae6827ac4", "907036983d804cbbac0211ccb88c8390", "a7afd654ce474193bdcbb6239dadd7ed", "9599f8ee8ee148d7b23698188fefa25e", "cb4f03e85f014ca995801762c92af9e9", "cb9600b738104da9901775b9bfac97fc", "4ca0dcd9dcbe4c848b1429037311a2d3", "ded7376121db455f940c8a39112ba01a", "6a93eaeda138472d852790ac5cb8eaa3", "d2a78025125544a1883280433c019b9c", "056b1b74e56b4c0db98fcb3187db8024", "6ca72e7e2f3d473191d29a2d60996474", "e582bee3d8ab4216880c1d2741210a23", "9dfd634adabd41328b275b5d9ccceb6a", "654587e1e4d5444f8f028186dd82ebb3", "66454f12c9114d2ba01d9d35149923e8", "7bee0c3538234019a715bd5c279f6ca0", "b569f1e62bd94a9c9eedc0151e55dda0", "ae61ab3b034e4cc2984bff73b79a4379", "029f079e0c3a4a7fb24790a47d7a8925", "d263cee996f34127bcac6cbe307172d4", "693a892b4729450c9477815b761366b8", "7e71313f1d364f5592385946c2f37e55", "17677e64a4ae4fe292c5480130a749fa", "b8f6fb1174f04e7184967b5edba195b4", "4e35eaa8291c4aa4864341500a06d847", "0abd4cb1399947a89b7866f6b6672237", "11b70e2bbe854eb2b77e9c9e402c7440", "b2c7413d906c41e79c05f34237b97392", "eccc2372aece42239c53d8289f5b426f", "e5dbef79c9ca44f3b4a2df6a52ac6fa1", "8f80177a15734c9887e06100b90d4445", "c9149d2b9f91433da017eac3bc6ae215", "5208e9b02f6c4b488964fb369ec8370f", "4f3e1e921ee945beae465bbccb3fe9d2", "de7c34cb11d848c4ad726a0ec1a60593", "e5a09a2de16d4fd38e1354382443c32b", "0a6114e0f6774f79ad33d31daf3ef48e", "003d30440a4d4276aeb57b24e3ddf882", "1b98541ba69b4c1dba4dea252304352b", "4285bb6df0f34fc0bd57d517ce194cb4", "552196fcacfa4db29112caf070cf739e", "82ae1e6088f244fda4b511622b7281be", "0f49c309047b4ee5be5161c1c3fce2f5", "432dbf74979141728b54beaf0327060b", "e320a32111474c62857347d857d2ad53", "0a78fba2089c4c0ca81a288866c48912", "a929623453594944991cb42b4061487d", "7e95948ed0b44920aabdbe9984e6f4a5", "1e255455398a4d8d8f33a8baf7f6549d", "459aa3a6fadc47078dbdd1fa860f5fbd", "695df7dcd4214cc696ff084d229d5de9", "3785cdba15914e02a989e11a2467ddc7", "01f61b19e8ba4f809de1cf6717c4b85d", "c49cb859b4fa4f488a935f4ff0d29ff9", "771bbb185bdd43659aafb7cab732ea40", "a0911121bec04c71a6d4f2b07aba422a", "817728eaa6e84730954833d450cf9f61", "acc647fc06ee4e6aa7f6dde045aa5d77", "7f30a3d853394954bb9fc2eb1eb4b113", "9b971d25e73940a5a6c2f7fe336f7865", "27f0cb70f6884e118cc05bbe945f36a7", "aa50385be11b4d6fa9d65d8585776505", "d71038b769c4435db4dbf9b0a19acc18", "7b7454e2fb88480297e146645d4aae0c", "6b1f02dd4e534cddb3c6ea47d8e1bf6f", "8120c8a812c4460caa7fd3dd4be9ec42", "3961693e8455480e8d352553c3179a5e", "9207eb661a544ee890991844a3a228ea", "66fcf5936c0e4833be574498faa9a845", "0e097eec761543d68cda68c561d7d589", "26697ac552f9498e977b94dd86347f36", "2b7ac8e9fcb6424cbe039d9d3e350089", "e7d48b88609a4ee78b0b49110451079a", "67752f5f568b4d77b88669ab053a6ab2", "594bbf5544b746549efa66e0473e2c91", "40c72267934243f68015ad0bb298d440", "30cb3fa9c741422faac5cac6ed5a18f9", "6dd6f97fed594a30938044b3dd0e2791", "bfbd60c73f82425badec5dc9c505568f", "0879d31f03844c1fbf66b1095306fb6e", "1d478f47bd314e8d82af7fab148aae67", "9092321ecc994bf6bb523bd7685ab0c0", "f37094e937404dee931b12cbd4ffd488", "1493eaa2b28f4725addf8c37b77be24d", "d8046fbb24014281b7702e3c7a8fd0f9", "1049dec3b155479a8cc1933f5dc6b589", "f8a1521a7cf5423abedffdd2121b7c1b", "c388b43a380842b1b8def6bdb7d6e630", "c4751a5fd4544fb9b0eb3e4da3b81fca", "876df6a072134a068aa594919b71f636", "befb92e669814e759bd29e94abb2b052", "f14288370a044b229f24002dac658df4", "a15a5a943fe345ea86ed520057f1dfa7", "3d2586234a594fdb8181046e8b6ab18d", "837f7546c2344078b1ad5bb11dd4caac", "8fccbe243f3b4bb4875263cf1f84e47c", "20c1de748fd745de8354999b31cb51d4", "dc0bb27443c4401f9b068dd3baa47b47", "1592fa6c4b1940a5a440d573babec2b4", "e1f282eee19c433b902304d352fcaf88", "00f2171c9cf848028cbbaab8ed2c6d65", "57637619de204c27a96039266f3aa311", "b70e8c68e287425ea8f795b7ad14454b", "573caf02e2fb46c58f94112cdb6042f7", "2a89620d658a45a58cb8a6ef2b0b1a55", "00f7652164684c3796ae8b443c5932d9", "426a234a5ab046cfb6f181f6a2e6eb8e", "ea294233d63b4f71813b6ac7ae40e79a", "b26057241b32416a9d3384c5f5beaade", "00d61cd426d546babd4c0963e0556208", "6747ccbb17eb4277b1f99d087571f5f5", "87061d774aad402583ba13efe0b62ea0"]} executionInfo={"status": "ok", "timestamp": 1634725526805, "user_tz": -330, "elapsed": 1551471, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1161735b-71cb-47c8-b943-7b4e85deaa4c"
for i, weight in enumerate(tqdm(evenly_dist(num_weights, 3), desc='Weight', leave=False)):
    init_ckpt = torch.load(sgd_path / 'init.pth', map_location='cpu')  # load init snapshot
    network.load_state_dict(init_ckpt['state_dict'])
    optimizer.load_state_dict(init_ckpt['optimizer'])
    lr_scheduler.load_state_dict(init_ckpt['lr_scheduler'])
    with trange(num_epochs, desc='Epoch') as epoch_iter:
        for epoch in epoch_iter:
            network.train(True)
            for images, targets in tqdm(trainloader, desc='Batch', leave=False):
                images = images.to(device)
                targets = targets.to(device)
                logits = network(images)
                losses = [c(network, logits, targets) for c in closures]
                loss = sum(w * l for w, l in zip(weight, losses))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            eval_losses, eval_top1s = evaluate(network, testloader, closures, top1_closures)
            epoch_iter.set_postfix(**{'acc-{:d}'.format(i + 1): top for i, top in enumerate(eval_top1s)})
    ckpt = {
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'metrics': [eval_losses, eval_top1s]
    }
    torch.save(ckpt, sgd_path / '{:d}.pth'.format(i))
```

<!-- #region id="x2JMTL4ZU663" -->
## SGD results illustration
<!-- #endregion -->

```python id="M94DIzshU664" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1634725528071, "user_tz": -330, "elapsed": 1278, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9fc79c12-8ebc-41a4-ef2d-5cc2fd62f0e1"
fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='3d', proj_type='ortho'))
total_top1s = []
for i, weight in enumerate(evenly_dist(num_weights, 3)):
    ckpt = torch.load(sgd_path / '{:d}.pth'.format(i), map_location='cpu')
    losses, top1s = ckpt['metrics']
    total_top1s.append(top1s)
total_err1s = 100.0 * (1.0 - np.stack(total_top1s, axis=0).T)
ax.scatter(*total_err1s, color='tab:red', marker='*', s=200, label='SGD')
ax.set_xlabel('Task 1 Top-1 Error')
ax.set_ylabel('Task 2 Top-1 Error')
ax.set_zlabel('Task 3 Top-1 Error')
ax.legend()
ax.view_init(30, 45)
fig.tight_layout()
plt.show()
```

<!-- #region id="7_vM423uU665" -->
## MINRES-based Pareto exploration
<!-- #endregion -->

<!-- #region id="83YBZu9RXaDu" -->
## MINRES preparation

- hyper-parameters
- dataloader
- optimizer
- Jacobian solver
- linear operator
- utilities
<!-- #endregion -->

<!-- #region id="LQ9-Je0HU666" -->
### Hyper-Parameters declaration
- num of steps
- damping for linear solver
- maxiter for MINRES
- momentum for Jacobians and alpha
<!-- #endregion -->

```python id="XbL9Ay1jU666"
num_steps = 20
damping = 0.1
maxiter = 100
momentum = 0.9
```

<!-- #region id="0XCW_NzvU668" -->
### Dataloader definition

We explore based on 2048 data samples.
<!-- #endregion -->

```python id="UZ6qJ0PWU668"
mr_dataloader = torch.utils.data.DataLoader(trainset, batch_size=2048, shuffle=True, drop_last=True, num_workers=0)
```

<!-- #region id="Qwou4anHU669" -->
### Optimizer definition
We use SGD with learning rate of 0.01 (**without** momentum for fair)
<!-- #endregion -->

```python id="gFGLUtvnU66-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634725528077, "user_tz": -330, "elapsed": 54, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9c5159b7-fb25-4db9-ee9f-c2e6c1a94625"
mr_optimizer = SGD(network.parameters(), lr=0.01)

print(mr_optimizer)
```

<!-- #region id="AeUOaLU-U66_" -->
### Jacobians solver definition
We iterate over trainset to solve jacobian with respect to each task.
<!-- #endregion -->

```python id="CGhhFoaBU67A"
jacobian_trainiter = iter(trainloader)
def compute_jacobians(ratio=1.0):
    global jacobian_trainiter
    num_batches = int(len(trainloader) * ratio)
    jacobians = None
    for _ in range(num_batches):
        try:
            images, targets = next(jacobian_trainiter)
        except StopIteration:
            jacobian_trainiter = iter(trainloader)
            images, targets = next(jacobian_trainiter)
        images = images.to(device)
        targets = targets.to(device)
        logits = network(images)
        losses = [c(network, logits, targets) for c in closures]
        param_grads = [list(torch.autograd.grad(
            l, network.parameters(), allow_unused=True,
            retain_graph=True, create_graph=False)) for l in losses]
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, network.parameters())):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
        sub_jacobians = torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)
        sub_jacobians.detach_()
        if jacobians is None:
            jacobians = sub_jacobians
        else:
            jacobians.add_(sub_jacobians)
    jacobians.div_(num_batches)
    return jacobians.clone().detach()
```

<!-- #region id="AQKPSwKyU67C" -->
### Alpha solver definition
We solve alpha by its analytical solution.
<!-- #endregion -->

```python id="e5DLdIVwU67C"
def compute_alpha(jacobians):
    sol, min_norm = find_min_norm_element(jacobians)
    return sol
```

<!-- #region id="qIYQroCrU67E" -->
### Linear operator for Hessian-vector product definition
We warp Hessian-vector product into a linear operator to prevent explicit computation of Hessian.
<!-- #endregion -->

```python id="TKuaXs8bU67E"
class HVPLinearOperator(LinearOperator):
    def __init__(self, dataloader):
        network_size = sum(p.numel() for p in network.parameters())
        shape = (network_size, network_size)
        dtype = list(network.parameters())[0].detach().cpu().numpy().dtype

        super(HVPLinearOperator, self).__init__(dtype, shape)

        self.dataloader = dataloader
        self.dataiter = iter(dataloader)

        self.alpha_jacobians = None

    def _get_jacobians(self):
        try:
            images, targets = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            images, targets = next(self.dataiter)
        images = images.to(device)
        targets = targets.to(device)
        logits = network(images)
        losses = [c(network, logits, targets) for c in closures]
        
        # Get jacobian with respect to each loss.
        # `allow_unused=True` to get gradient from the unused tail.
        #     It returns `None`, which will be filtered later.
        # `retain_graph=True` to retain forward information for
        #     second-time backward.
        # `create_graph=True` to create computation graph for
        #     second-order derivation.
        param_grads = [list(torch.autograd.grad(
            l, network.parameters(), allow_unused=True,
            retain_graph=True, create_graph=True)) for l in losses]

        # As metioned above, `allow_unused=True` leads to `None`s in
        #     jacobian tuple. Now we replace it with a zero tensor.
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, network.parameters())):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
                    
        return torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)

    @contextmanager
    def init(self, alpha):
        try:
            alpha = torch.as_tensor(alpha.astype(self.dtype), device=device).view(1, -1)
            jacobians = self._get_jacobians()
            self.alpha_jacobians = alpha.matmul(jacobians).squeeze()
            yield self
        finally:
            self.alpha_jacobians = None

    def _matvec_tensor(self, tensor):

        # hvp = Hv
        #     = dot(∂^2(f) / (∂x)^2, v)
        #     = ∂/∂x(dot(v, ∂f/∂x))

        # dot = dot(v, ∂f/∂x)
        dot = self.alpha_jacobians.dot(tensor)
        
        # hvp = ∂/∂x(dot)
        param_alphas_hvps = torch.autograd.grad(dot, network.parameters(), retain_graph=True)
        alphas_hvps = parameters_to_vector([p.contiguous() for p in param_alphas_hvps])

        if damping > 0.0:
            alphas_hvps.add_(tensor, alpha=damping)
        return alphas_hvps

    def _matvec(self, x):
        """HVP matrix-vector multiplication handler.

        If self is a linear operator of shape (N, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (N,) or (N, 1) ndarray.

        In our case, it computes alpha_hession @ x.
        """
        tensor = torch.as_tensor(x.astype(self.dtype), device=device)
        ret = self._matvec_tensor(tensor)
        return ret.detach().cpu().numpy()
```

<!-- #region id="iaXPczBeU67G" -->
### Utility functions
- assign parameter.grad from vector
<!-- #endregion -->

```python id="ohwL-lI-U67G"
def assign_grad(vector, normalize=True):
    if normalize:
        vector.div_(vector.norm())
    offset = 0
    for p in network.parameters():
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.grad = vector[offset:offset + numel].view_as(p.data).clone()
        offset += numel
```

<!-- #region id="XbT8HrVFU67I" -->
## Let's explore it!

Executing this section takes around 30 to 45 minutes depending on your GPU types. However, for the results reported in the paper and supplemental material, our training process is typically a lot faster than here because of the following reasons:

- For reproducibility, we disable randomness in this script as much as we can. Therefore, parallalism on GPUs is not fully exploited;
- We frequently evaluate and save the model inside the innermost loop, which creates a lot of overhead;
- Since the MINRES implementation is from scipy and is on CPUs, calling `HVPLinearOperator` creates a lot of CPU-GPU communication. Ideally, a GPU implementation of MINRES could completely remove this communication and expedite the training process.
<!-- #endregion -->

```python id="bjRysgSjU67I"
linear_op_template = HVPLinearOperator(mr_dataloader)
```

```python id="t2mRpEL4U67M" colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["cec77f6eea4143cb87e21b175d6ffbda", "e652f4b363414bb8b70e5769283af4c6", "f1052bee35b8484a8814076b911245f9", "4d35df80fa524ad6a46bb47fa1319d2a", "59d46390eb684bb4b57e668acfa0d47c", "25d5e734aaed4a65a6f8f58110fefcb8", "2558dcb63cee4c9fb895535efad2f85a", "1f3615cc14094d16af1526bbeec3d497", "e17fb05cc62e4f7982703f0569ec6050", "ef90d099a3474ca695bb45dcda7da95f", "5509596d27584deca175259155f11bb3", "d10d69e74332469c92e80eae0370f319", "5ebe178abee947ce9942950e682bb6c0", "7e9b92a81ba74cf1895444a2c2a83d45", "f816b4e04437439b9e7acc474f9cd211", "39c751ebb838430ea9d282da307bccc1", "a76cdd5db80645fcb58c03ab5f6e3863", "6931ecf5e91d469891beef6c3a3c65a9", "a2636ec9fce24b7aa3f48bec9276a704", "60a17db6746a4feca9b350926b3bc5a3", "a35c3e33a8364b0f8cad65e7a7704d4a", "b942f89a363b4e288775bfefcbb69bbf", "e1d9bc7582704803ad06504b1d501105", "b789ccbe00ee47eb8aca74e0a64b96e8", "44c0330224504bf2951c9c076ed2487b", "2a72564d6cf54624a8fbf803d04bb1d7", "00fbec0fce824d849a3de62e893fce9a", "4c72e5ef4c3940b5bd0e8c70f41c0dd4", "92a861df0812467a90f6320c87a6287e", "5ac3c1c21ce9422497187e488bade6d2", "61ea877d560548f09cf7d0915972a5d6", "1b358050a2184220832f3f9fabc17775", "302cd86a304b4cee918597536c5dbc70", "14b426fee0f541aabb052ef9efe95986", "ca46ba4b455d4e97be3f186bf7709f77", "2635c573d0c64694a4e40fcffadb2c22", "104aa1e37b61447bb05857914cb8a8bf", "47c3992a6eeb4f178d97160cb5dc023a", "fb8e49e975ff413ea053f741fbe909fc", "57bde0ef39d74b20b5b1aefc899e3073", "21f26c68149947f78b40dfa547eb6dca", "24276ec68946498b9bd4839b9104df70", "ff46a8c98db94e9e80699f10e4569f3e", "0020582e7a6c4c079012ed87f3827cb9", "f07d2e40374c4327ae68f9ffb93a35d8", "61d731e4e5564c12bde9e7f0596cb879", "4ec0d4ae60924e4b90c4df80334ad19a", "6a0819f7ff65458da9ee23c8f668dd8e", "abbbf617c0e942ed90c4901cbc0537b0", "bc7b805a23f84b79af18461a6bcef823", "fea1bc075dcb44819a899d1255f422e1", "87fb7f6ad1e04cae8d66edbf2155bdcf", "e4ab6ad6ca3948a3a0254a9d0ed7c3b1", "2dbaa9b144084addbca0a897705692d8", "ad5caffc94154de09a49892c5d2f5e55", "c7b07fa08e38440d937298ff63f70048", "c1906aa98e8545edb7511fd30432fcac", "8d42090fa9ef4534ae8501139b3e2838", "2b2e2dd87e2c40a184201fd88533ba0c", "daea4eb2974d44ee9c0424ebcccb6f24", "eaff2edb8a7c4db0a29968735662a6c5", "25ed3519dc3e4e36a7d5e651cb74ca5e", "a90a69bf1eae49d481cd86389a1451db", "b139853b4b2f4c21a7c5389896d911a2", "b49e3cc6f9aa4202b1124726a5e9eeff", "382fccb985d7455f898d4c3e6b556c8a", "f9304006cd434237b100588bfde74d74", "74fb0a0b998c480e9f3d6a166e8fd9ea", "d502858829d94db5baf633313c5cc204", "50fae80cc0ce47abbf825a61a26dc9dc", "de0cb1f033ea4155aa2a2b22c0806ca8", "d942be2569dc43a2a87b97f21b79adac", "f239ab60c6984ba8af2fb40fae066bd5", "1bccebbee1dc4309b18a5e9903f1926f", "6c5a03f94b73431e87f99760c3aae045", "09614d177a8c48958a6360676189bb8a", "eebd425aef214299a4893c6d1c985064", "8efed44bc8d34d489158cb4d1bf3dd13", "e7bf8d8bc51745fbb4ebdd590a179ddc", "52737b25901548bf99e9a44addcdab68", "6ddc1f1ffe554df4bc81b3cae743aff1", "dcef980269e645838cc66b6725e68b60", "92b43431e36f4389baf235b584dc5a81", "5ed81cc7582549b9ac60fd7493833a55", "5ba927d29e254661b0f2e59702671c00", "896759a318da4d9eb3b446ba773526c3", "3a7610a993cb4cb787578e34e8cab26a", "e60bd0f932f84b0d8be09ecfef39cb1e", "4e97fc228fe34eed8edf441f28a59bbf", "a49a53d478da41c3b2a0640eb4605fb7", "73618cf5142d4c079181d571925a3ce8", "5af696dcbc0b45ffbc871f4881ffc834", "b822898fe13e4b45995efd742edb38ad", "13d44c4f6af04a8587dd4e54a3c2c62b", "c2374146b386493f9e232467ad960b43", "6105f759f1e54216a8d0e0241172f660", "a7f15f07b9b640718bdda1330cb1ad0b", "2a758696d78946a3894d81aee9d856ba", "a035af8651f344e69be4c4132f8d2db4", "a76525990d0742e892a1f2205bdc9559", "845836ccd6c243daa2ec39a2b26e28c1", "9e5f25fd67a944f6af5f5d7ecb4e5587", "f0b7c518326d45a6930a1fb5117c75e5", "5fdf581fb7eb412c8dcb24fa2714d24a", "7f5a088cf9f54d928fe53f35e4edd89b", "fd94fc91864449198d5b3ffc70f57e96", "224640d3ae1649b29ac02e103236fcd2", "535e13e1a0fc459ea2b6d1be8b3cba72", "aef6169fa0a5445f8d7dd81259df84cf", "d94ca69d022243709625786349b3e605", "9728b061614644db84ebe440ac802a00", "d2e51c8c2924416db1e3ef16c72e62fe", "eda6365c2f5b412683bfb1bbf6a20465", "63065320540f4852ab5b9b7086968a86", "a5c7e960ee8e4ddebfa75841eab47e9e", "66311080d7584ee6938fc9aba848f754", "7b30c8e57f364235bb8d12e51eda7dbf", "48cf159975aa4f1c99505c5af4a9e946", "0ad6a2a8021f4828a870f2a08d332e7d", "18d9f0467a624d3f8cfb751afc4361f5", "2d98df4fa2f542f0a93ddbfee3fce52a", "adc848fa23f448429e48297c23d4ca29", "cf9edc02b3e449c9b0bf94b4971c8dae", "2fe454c95d39441b82d23fbcf2607885", "a0961400b97a46b49e2209040d369771", "8fc8f9c366f34b5b9aa94ad4924eeee7", "ec11643e1f104f1a8a8f0538392961d0", "df72139d36394d6d8b235a773a214a92", "35177b697fff4e20bb1b70565e3754ba", "7d7de3ddfb134b328f941a7db3594797", "95a0e4c3035248f7a6989121cc9f5e6f", "323a888ac11b48889a117b01b0b34d39", "fe9fdaed3c114fb494148b5323892530", "994431ec71c14abfb5833eabd4262474", "241d95bdc2e5408bbf465650b72a36c2", "2fefe125a5984a77840dd02461b14fe4", "962702efc9a94afc9ea7093337e7e5b5", "841b3ee302854aeb91c01affeafca577", "9184a74e87d54e22aaec2bc96581311f", "3e422255be64474da0ca109042539ff0", "8019a411b28743b8803359515bfb45ad", "092e4c12f8704d98994cdde7c5bb5bd4", "eb5a7d5f3ebb432a80534063d3a72eff", "9588731592ff4732939877a609c4bf0b", "e2af288919e947efbd75652f1b8b374f", "f15c938ab41e4cd6a63e5ae04bfaaf70", "0c4e90dbf3084a7e80b288fcb5ac4518", "618cce42626a43c78c7c6bc9ea4959db", "b886f1641af8449d9e58db970fcca4ff", "296bdc6497944443abb106c801eb8dc7", "8ca9c563363943598028e6446f1c99b6", "5003e200a61140228cc85cf4fca43f22", "c7bfef52705a4e04a73f3173f227e290", "3bd153e9b91542b1b2dfaed1c7ee5595", "42216e9a8af244fa9dcee50854bf5aa5", "e013615f8e90457db5d53d556a272bf9", "cc695339978242d696ecdfe7623c7d6a", "03702053a3d64923a15afadec8989952", "81073e839f6b49a5980c287cc1688c16", "27b9e89a6c4a401d888a9abeeb95ab9e", "d8ef666da19047839caf1609aa1d5495", "c7724d680a6a4031a34286d2bf2149fa", "601297156c0b4ec2b54c48cd6f9f8886", "f9a12343f28d4b40ab51c44aed23731c", "6f2d80db140943ce8d4bf5da459646ee", "9efe883c74564eb4865e6b65f132566b", "3a3a8c8bcd764a81b945b30b001cba6e", "78a7a112a6984453a5287672f4040589", "ac5e135eac39424fb288e8a892776e03", "570028c222af43b4a6a082175364d3a1", "60ec10cacc5d4b10a47ab8aa74a0ec9c", "afc9648f9db746a28c94f8d3bd043dba", "4b048ede78a54d9fab817080bc679811", "68f1bc7299f142a28e31e1d39a152c7e", "b0788b2cc5c149d198ee60b990d865cd", "ee21581d28444351af9590eb70ce2f67", "04cf993319dc495fa07c77407153b9f7", "ef0bdbd25ad84a61852fc0cc3e78ee2f", "8062e7333bec47db97b3407c926d8f5e", "0da31f33520945f2bf9aae462def84f8", "4a67878ab377499da3d79b5fca02f7b5", "48f91753c59a45ed86a6fdadfaa149ea", "f05ca0a06c6e4bea81bd03d044f1ca82", "e850ef04163143c390be4948a70703d8", "d6ce148bb73f486481a12c45da237a0e", "5efdecee71434c758d8789347dbf1b1f", "c729006329824740b2eb771140cbb34c", "bccddb55a64b4c348e177ad6bc81d1e8", "2661af95098d463bbaa82820fadee4f0", "66f17f4a3bfd4f4099e185dfba07227f", "a61f9bc6b20e4112aa89b53b93de6723", "8ba064365a0646718a0af4885964c1b1", "c93bf494a0f4467e9eb32df8d5462ed2", "6db0bdc3ef484b74a62383a68102be5f", "5bbd3cd38fa641c7a61a2cb4c41f5cdf", "3448ca6222ea46e6ac5ef91bab7c723c", "5ce6ef1ccf4e489ab5d3617df27180b3", "78870044600241ba9bc9e4b193d8fd78", "2f20bf0f94d944aeb3372fca75823e66", "4185661862ba42819286117bd5000ad4", "53ee99b3c16f46b5808f08d4dc5e027d", "95557010a7d64aa9b81e08af0efaa91c", "7c62a159bf5046fa90bf20c3736ee43c", "e20c05f76fee4778a743244cba8b1776", "b86f290a9772496ca9f12d480f75839b", "2749f2525ec24b67bdda995f337788e0", "4a3d313789fc45c5b611cf8a3de7ccef", "bb4656cbacb046048aa27c312d24a589", "8ffd15d0b7104ae0b31ab858a78dbb6d", "c5834939cce442d6a261281ee5486eb8", "c3067806a9784547bee7627d0be122c9", "9bb6c728705e4b48b85868ba7620cfdf", "7928af25afa04df1bb9f6d2dd4cba385", "8b3eb540514a4bd4af33d061d20ef3b1", "e4afb7cc88a64e9aad23a8eee62d16ed", "b740bae37b6d497ebbb0a9c4a7f0df32", "d8b1aa193a1b471d8aff2bd7ef9fccb8", "cb2e617922bb479481d0812d79a472c2", "aa4fd151adb445f5ba0a2961e2877faa", "68e1c10c0de548318bdedc0ed899d65a", "e17504a6b1a14aa09ad39739a3378a7a", "c3a625def5b544b58fbe2b9e9bc4f1d9", "619a56c3e66244858ee927727fc6a85a", "b88f412c1b344d28b608d2289dffa4f9", "0ded3160d4f9450bbf53008836d3d1a5", "0b98d44e9cb841a9b07854e84f129c23", "0f6030f3917e478b923e6d658808cab3", "9d150509097644298b6e42aa505a9f15", "57f08ed2c2fd40419cfe7bce6f7f668e", "af8e3ad30f8b46529495d95ebe8ecf12", "68a711209f86428e829aac640b3775f2", "e481bfd03a99486681ec1a1e4fc0df69", "1876feaa5f0c475b9d67ddcbfe8f9ce9", "9338b0df1ac2466b846e098c4a6a17de", "e74a342b0ded4c79b234b7b6e015b0fe", "c17b401f3ba14147b3b3ac9806673c92", "c6d4601117b7498189c69650cce0a023", "5fd18198b3f04c3d8a9f4994f39f85f7", "f64dc171343d4e4a9c0afc33dd2fc7a4", "75cf447aa4c44f28bb3c7578e61d7119", "1224e5e9ffc14ba999eb7afcc1a7e036", "76210574d71e4db19c35f415c69811de", "66194aba21a9494e9ea238fd6e80d7b1", "a3444f83085f4e2db1943bd9097b1f29", "03aa079ea1d845bdb97205051b04a773", "d8a4ddb2f07145929e6c580337b502d5", "7605209416ea446e9e7e09a9e90b68d1", "cde2b06ccc374ce39b6bfcd2d40bb3fc", "5aeafa5381d9409d83e42929532f705c", "6ed95003e5934bcab76b07b4ea7ac3c4", "cbb946c8780e49baa861a37780908515", "7727670fa5284c338930c3831644ea26", "27959cf9068a4f79a9f5f88c63f31fcf", "6f3c40cfd8c74d21b716a626ef064a32", "9de99afd934440f2a850d5a2ec340f6e", "d87c08ec6afd47949bf79995023550f6", "4e34a55347334bc1b825c56801ce5df9", "63e3138851ef442cb8f194d881caca3e", "2b83ba338ab344758714e4b1115bfbbe", "996ad96d52db4f768a0d2607b221f7f7", "2e764830cad84f22a571d00c4bec9ef9", "f5732d66f5724acaa7184f0b7b592de9", "8ec67a724c2647f6970c06042e4f76e0", "cdfcb90a696e4a01a4a972f8a97c7075", "c939fe7a91f145daa7919bdc3ae3c1ea", "059032a3ed8b4c239f7a7991049fb864", "fe7df40b7d6c4ff78d9e891eb6f65850", "f65bd6ad377642e6b19968a15cfa0394", "c1b9b2a1e86349c0856cb59dbbadb24d", "b02c5986f7c74a3c934e526db59a9dd8", "49876852c84f4547839ca27b0aa109f5", "a49a96d62f274ae5b21f38839a9ad719", "3d3601308e634ec596567027918d5385", "abd393d193964c8ab28e0c597c693280", "7ac6b4e056934e25839c37caaaac7a72", "2bec89e2e9c04ea6b86acdd52cfbb415", "44a7f9124ca646f2b56906087855703f", "6d42972e99b64011a24925c9873181e2", "303e8f6babb54594a1fdf8e86823eb35", "cc04291b4c844705a94cc2355a29febb", "078db3d1052d4028896c194fcae3a1f6", "6b98691ec82944f5bf1c6a851678cf3d", "9bd053658a9c4101b4ec8081b7410787", "17ecef36a0314140b28a6068a57f4f9c", "e490e9b12c1e4a32840b7cbb645f6e78", "7992d195d890409790be1681eec42d75", "68aa826eee2841d1be7d2f3e142ce05a", "9e318b6f46df4e6f9d770a24f670900f", "f9bc3d655d434d89ac028a0a44426c87", "89e971e4edc742468540a2d1c64e7e4a", "bf85acb461bb40bcbaa087357208864e", "4bc7d0e1b1014f6f931a08a4d40095e4", "25451fa3e17c48d0883704154f6e3580", "2e67710f3e4544199a2b52a3d5800f17", "1cf0a246009e480b8a00690d2d99c3ec", "30cced84b29e4b89ba1ee4cba532c95f", "280e4c8b138d42c39bd1660a537f1882", "838b5837d84147c4a0eb39d560acb65c", "a4e660029c62443b9c0c32445b260e78", "25db4607488b4c78a6e64f034df9952c", "2b43740b8c04481fb9d1d6c27309c9a7", "141bde23202e431d8354a629d82b6a0a", "4f589df125b3406a98b0024539552db8", "528a17cb62e54679ab9d71f3dd359a54", "fa41bc83f0bf490d940c1038baaf0df7", "7a35ca14d16f43cc825f4adae224e2ce", "c78a9a26d35b4a9c9d5f476e7b4e9913", "e3e3821045f745a197f510c986a0bf0f", "a6f799d51750401c99593df88a04f63a", "b6f23fce62f647a9890065a2f41f8888", "08ffea58036a46cf96c344105c5b3e9c", "79efe9dcb28b4b5d807769d9cc6ae88d", "a70b30b9a53e4bf0a219f0a754806467", "35c8093394ed4a97a6f2a76c8b571b4f", "b503d70902334e1d85e15c6445ff40d6", "1cbc461ddf244b4595566e47cf098a11", "82447fa136a84e33bf3f7bfb686a8158", "eefcc61b3dc84d3baa4082abd693a17e", "2dc19c63dd224404991cfa95e624c182", "4d0934e650d64acda673099e06643d9c", "f313d6c001a444e5a7fb3a1b91c54279", "7b2ec390047344b3bc04a46e7f8341f2", "ab39213269534f93b843c056a05a861f", "f69940ecf4cd4479b5b92c4b9bac2d6c", "42580d04677c435fa4fe858d58decd86", "76ba1577a0594f8abf22810eb3ec0dad", "9bb37e712d434e47808c8eee563a43cf", "5c5494a367ba41a998bca742cd89215d", "91ed47a35d8f4e729a2e215200d2ea1b", "38116103469d46268c77076480bc5ca1", "ac773231ec4f47d5a917cd34bcc80961", "7d68cdeef58240b28c27c9e8823d9995", "7aff569769154096bdf27ea372e8d308", "37d578362c2846dda92bae562022a42b", "a93a4a4e0997434ca2452b103e94f27d", "86b9823a5f6f4fd1b9f91706da4bb0c6", "3b6f3960342e4e2c8fd65011ba36b2bb", "90434b78b17f4fc9ae716a89879015bd", "e6da7c11fb624ed9ac2b911ec3aa38e0", "7004cdbe21074d768b37f9c4cb29c647", "058e8d0c376d4af4b03799b01b4848e8", "8182de330d4b4753997d74fa08330c99", "ed503da307e947dab97f795a1809b05d", "f38c57f3fbc04acd8772913dbdaf1a7e", "d5be86375e0c49b6b7761a553989039e", "4f4fa3daf58c43508630e1fee520bbc8", "79db451315834bb6a1174d198c950c1f", "5890384dda56469c97f5c5883f905d6b", "2b2f088c2d364d48a504888d4386da73", "a97fac3d214e467cb02798f1047f4042", "46fe09e4322c4aac932bed2cb9580a03", "614267f95ddc407baed1b67638e88759", "09e0993212584334b177024040b42964", "21d318b2b6b049559639334ea167c2ce", "80449364661b4f8bbe9d61672e867131", "7aa63ff3502f41b384ba4f34af527bfc", "ec1ea01ff05a4bea98c09ef836581460", "898dfb637caf47eb90fd6fd21f664563", "ffba548d7f354862a65fbb3cc2c7b464", "c810850f17304bdbbabe82ef6f4c4d9d", "fc8affaa309b4693a549af1402cacab4", "c3b0fa8c96ab4ef8bfb7c3e9a617bcf5", "3265a7868ed1469398668efbb51eca2c", "67fe516fb3554ba988afe638a10e188c", "f8a764ccb94c4acf9e4ccd1a54a18b17", "f2612ca372564e1cb3aaa3b070821ace", "e8c6506f08d24a489b3f6a3ca16b84db", "36916a5ccb354395aecb6d649d023bff", "a07074dc5cbe4084a4ed59b2763c3ea6", "19f0c5eaa25a4ef4a3e7570ddc9e604c", "a3f80ee3d96d44c1b6e50ab105bb3026", "50a10f123b6d494b928301e0189f9659", "150e5e83cc774e6eb26e57b44532d3d9", "0f54e5fcc487456eb18ce9cd5722b26e", "2df92561fa1f4dfe96526ae48c929af7", "5c01523a1d1f4c19a2e2e1eaca77d3db", "0db03e6a85954ea39ad4be2cdbf08668", "1fc4cce6983c48d8beda9c6146435deb", "b985a372fbb14945a3d5758113a7031f", "540539f0749f4f9d95c69b6b16e1f931", "47c2fee6d34a4f24814c8954cc871e58", "42f3aa6acc7a46c6ad91a995221a67dc", "babb54ab87b64c48ae16eefd93a507ee", "362a4c46716d4859be3716ee2a3c4fb0", "dba4419c233043ee887f797b4b52b60a", "c1076efc41754184b6e4888c479e9092", "3957e48891c143f5b80dd2f441b0a51d", "f671e480b7314e208cf58c8c26d4b965", "4155d32930f244eca2b2e70679288408", "7debb75df561412fba268fd6c7c32668", "e4691164aa634172b1c453dde516b8fa", "a931bfcb30d74660a74cd0d6f9e98ba2", "cc8befefb074497bb6121d3672032387", "e82e593dde31474ea9febd622612d874", "5fb881e7d7bc4d6db97946b1a870dacb", "62d892fe8a7e468e844b7776e1381b40", "a02239aab451417c9d6ea4c35c116d59", "6ae02ac3562e4b79924968f2d811d6a5", "e5aac41b58af4f38b51c5c883fc9fc59", "6e0eb300aae84d2baba517e18c284d7e", "51f6c1d1a77e46f696c773ae5fff1c56", "09d29086708a44f5923b7f0a446e3f5c", "77c250d78cf0423b9a355f133c80f09b", "dab071547083400ab7b190e8a564c2ab", "91f676c999ba4f12809eb0e7c096ab7a", "728cbad4d604401498f166a9fdcd1509", "ccd8a330f4c34277905fbdf6b0ee810a", "23fdbbd58c3b413aaa091965e194e0b9", "dcfa30cb19a04b919c445a767dac4a2a", "3a6d657d021d4aae8e1cb5c8b5573883", "b8493c0bceda4f428be5ba1b223c7cee", "f83f5b4d91dd4326aaff00ea2b91aa2e", "3398642da52947388f6ef76cbc46f553", "e8a1db3aa0904561868c3e314bf16b5d", "f419a6c1cecd47ed97cccbfc35536d87", "ed2f4f2cc93a41f4b2be141990be5554", "2b552d5e3d9d4508894c704d1f89e651", "997442de005f479bb6c4c45ab89992b3", "930107dbf2044409a92e2b1522d738b5", "8a5012f272be482ab6472c388bf56519", "c7bda1af140241f098d4d85e2d57fc9d", "966b085e990549428bfcc36ea7ce113d", "659b9216702641eaacae831f37d87f39", "fa292536425b4f03acbb1190fbc4903f", "e5cda82504804eab94761616313afebd", "c9560f028dc94ca5b32b97636fa29141", "dd7fda997ca04ccca6dd8366a01eb702", "b5da5cef46e645a49c7520c34f4b73e0", "fbed0a53672e4ef89cc4b0643709c657", "c6f748afd2114c5f9f8b177e1ac9032d", "60d8a2cb22404dcdb8f907d8af174528", "6fd50a1f62004f20876a89b912ed9cf7", "d4ff4ef9de8541388d39d973955087bc", "5a996b6f190c4f3590ecbbc43b2f2d1f", "27a9531270464ca5b725b483a1486213", "68c5203fabee411cb57693273c51eb10", "25311fb5828946ae95dd5c840d8aa13a", "2dcf355bb05d48959b5e28fe084a9cbb", "af45efed54ec48458e8ed984fd12198c", "ddf0888e285646a4bd2df803d1815d91", "877e6a5cd0e74d18b73d8572ba851dea", "25d42183efaf4a46b798d9814c415f7f", "7bd6bcd5407c4e4090c8f850d5c18350", "3d62186874924ca0ae314f3f8baab1cf", "45aace89b4c64b48b2aa01589d271efb", "1ee892643dd243edbda463724abc47b3", "b2c13f1d1dc2449097ffd798cc779b27", "3cf2a6ba5a904449b56f0259322f09d7", "c7b034f296404e2d99f7d3863aa4a386", "601b22b8b0c94b8686c55995eaddbb26", "060b3b0a694a4a4fbce702ea4607e013", "4ef70ac799bc4bf894b9e3827852e78d", "6acd6abaebc74d98a056570136fb23d3", "103da5c3bf3840a6a2c9b498d4e28b01", "3404d3475992464cad82680345bc133b", "32e1b43606df4ef7a1ac328c871db6bf", "be6daa30352a4e2fa0b32a4bdee391c6", "e357b00de6044fc98ddef284736afeaa", "e68710d51d7648e8a8acb3b102e04bd4", "d78e4f7a62ef41259803afb9883d67d9", "fccad81088414b108f1b6364299e6103", "52320ce96dde4175a9571ba4ef0c1519", "21b5c8f19b8d447abcdf702a48738e25", "2fb87c015f824bda977cd20954af0d79", "9830ea89e1ad44e3ab5c1977c9d784de", "88662e828ff845b89605cab673dd8ca4", "3285b539a5aa4af79515f8de9bd3423b", "d34f20358c7e4b20a5b72f951deb8323", "8ce701475c474ebba16bfeec5bbef387", "d1314db4612f4afc866efa56699efee0", "277435dc510b4d69b20bba9b669ff664", "8b39ea1b54074faa8609957b9073107d", "412631de34e54d97af1423dbc32c4791", "0b8bcf94ed5c47578bbde3ed4323dab8", "fe9f10218a384f55bfd164d004dcbf38", "9b41d678dbb44388b05819b1b1d29430", "5cd4aa14cb554338a288ba967fb52892", "2c7b240cc74745d0a0265a7ca7712c74", "419ac9280319459c9c3d4c7a0fd6d6cf", "272353d0aabf4e32bfe7536e0ac0f1f4", "2fd74d49075a47c8ae5d123652b45665", "03412070fda14b01b4fdb67049f31d73", "51a5e1552d064664871a59f181fcf5c0", "a63dbfe04a1949d09a46d4077a92e2ca", "f24bbb19d4214cefac6fbff91fc347ef", "fe9e3b3ecf6242b1abf55828a34fa5bf", "5df16bc1fa4d425b9df673b682563e5d", "b2212e465eb54990929ab7165c3ec267", "ae83b3f63fdd46e8980e3363a662ab2e", "c0a79bec751e4ae4b3939ca2665506ae", "809187aee2444af4bb4d9869f433a5f9", "7a813a348bc94e518076c9b990d15be6", "d07224f4d9d14866b238f2274e4b19bd", "6a86beb4c62f4faf9d85fc36c5628a8e", "846a51044a524d3e9b3e51d871f4dc5a", "928f3e9c758945af9c0d4462c072603d", "2498058251494afdb0f2537426ff52a7", "bf89c43315f541dda073090e7360aa5f", "8fcb202c562e4e53b423797013e84718", "c2eccea20f134039b71c5c97570ee9f5", "485f3fb1dde846039fabc28378ad478e", "c2dcf4f27fc3407987f271a9917eebc1", "dde9519b551d4e2b91d3dd8881d78274", "12bcac6a5215403f8257e45a8c10fdd6", "e2e52b46d5e2439ebf06897db9f1e9ae", "ab6ccaffdfd14263993038613f152691", "b4618db1c8fc47dc8ad2ab889f002a06", "5aabf3573a1f46e6aeef3b49096e22c4", "a63aeee62b934188bafe20d66648a52c", "58bfe45e9b8c49ddbfd6d962b81f967a", "5c4be7ae2bf248d88b49bf3ecf6c5d8f", "dda0acf5d9aa471dacf2c422f79326f9", "f81fa4b6d9894a9f96eb4de6ed526a38", "3df42017047b4b7594550ce3ed053991", "7961824a8a8142a7992c6b4b671b547b", "fd1443b519f048d89e89572a77743bc8", "999d7fbb2487498192e15b3f0959e108", "aafb22b7156341958d1695ec9f0718a0", "72e0823de26843f088fe9cc06927b883", "fdc52be90a4148449fffaf7deb106622", "762145d414d14c88af9eefa2c60cbfed", "6f4d3e060aae4ec5a2ed64c0bc89b681", "ec8f2b9ee97347f7a8a66bdf5f8f858e", "b17a9390ce874679a3d24357d6d9d566", "0d9fa986744543bea954ef247e3b74f8", "ad036fd3ace64a97a2849af29eb26657", "02173e2dcccf4c98a9139b6b930865c0", "7c1421dc06fb4eb082e4ca8618c74d0a", "9958436a4f244c13af9a2499ab428f4b", "085ea6607a2646819e208ae4d6fccd80", "a2cebfc81bca4fa9868b83a4d2ca2baf", "0df86e36250742cba09f539e780ffda2", "86b0f36c2d9c452ebe3636d784e6f67b", "44156d850b314c38b63c05c3feae99e4", "db8557577e2b4957baafea1161394430", "3e1c9e5bfa004ee999d56db689259f6a", "0d1476f8d2454c7d92c864985fbc6c05", "903370ca5d2a421bb5a0299fdc6df487", "de946e40d4614c4187680af83a7937a7", "6b7081fd42944f72ae31ded08765633d", "8f912b5dbc944503a162ea2b67cf7182", "5abe228a0ed84b6ea2d768603e7b8c36", "a768538200e14cfcbefbe4b5c57f2d8f", "33aedd58aa1d4c4fa99757381006f13f", "b4254624256642baa31057e3c289135c", "ecf780a3c6cd4b2892c007e330e44af3", "ee65719ae37f4f3ea69e542374ccbf81", "fc859bd86f1f4093b7121c22fbdb9f44", "fd2af33b2bc5407c9ecad1f8358be81d", "b7eaefb71a83405aafa5bf10aba74c96", "3f3df0ea9ca147cba8c7c9bf4d275908", "a0ff1826b1f94b9e875e3e17d44397fa", "6411defd8f184ec4898bb42668e01ff7", "e8e2ae23327f49e99abae1c4088e442a", "48b8f7ef702e497d9217113ee34dde19", "5a3fe068596c402b93ff8e204cb18aa4", "f973797467cc47c2a846b9fc8eb4f8ff", "4cee0e51a47b4678bef46a020e35147f", "9e03467dd096458b81f84d4356e456ef", "b2b0a6c4802d47ff8a66a051d9e560ce", "622809396dcb45b194a46f54230f247d", "8ec85158099d4879a7470b2250827cc8", "12653c3f12e44ffbbb1917db2249456b", "350eae34244c4b6b9cea8b3c4045064e", "1dc9f6d8d90d4f248c2772e929e85f81", "bdecccf7c1fe41ef8c6c20c2d2c80675", "1a9b0f19e8554184b61500e2a2d195c3", "2435b92a1f7a4abb988d328aba89735e", "dd62da1198244a31b684c837840c30fb", "8e3f7cba2d0f4c528c847d5638ce73b6", "a425b48a4fbe4003ab3dc468ebd25d61", "58d95b1f86d74f14b0bf6151bac4e95b", "8aae5039005845d4901ae890cf17454e", "804398d854144b48b30f1a062599e77f", "fd3e2667897a46d5b000f6eaa03e1616", "11fcb037f0a84737969fb6df558dd5f7", "a46c0c6cea65480bac76e43963a60b64", "61e6cfcd85824c8692012bf4c68181e8", "914d26e4f0cf482abc1dd4a265a3d4b5", "a691539e4f2146fd8b83cf32d7a9eb70", "4ad5fbb9373740549bca98ca4ae5e1fb", "a408ca4286a5456987374e6a38b0c60d", "d427a7b0afa24249bc3e77887519515b", "937097c4bdc34fbb8101081b6ce03aa6", "5f8b85f7ab1449a594f3442376b05880", "cab1a8e8caae41fd9d52895b1c7ef7a4", "e4593c7196814d9591556b348b186ec9", "857d7d8c1a1544faaf52c858359f6d72", "51fb516caf14490f8c76496611ed419b", "9255c9c40f2c44d0a2a2f432aeea10fb", "a249476847024879bcddbb87395eaac5", "bec2bd8c9aeb4707baf8451a4f1511f9", "c4ba1722117a4f22977f7bb24b4c4a5d", "2e1a1e7512264406ab989bb3af819ee2", "3d38e38ec56b4208a352c20435446e3d", "62e91235a136499ebad12de1f800cd66", "8b31708ebf2445bcb663feaeac1c22a9", "4d79d7dd1b1c40feb44d291881c23594", "38dbbda146e64f388156e58cdb98dc4f", "77511873a78d426e9073de9f977e62ec", "0918baaa24944379bf3f276e2ecb94f6", "447ea8e1a4094c479e984ab1619e5d7d", "50bb49115d1b4af0b658cf30d947af0e", "4baa4fb65e6a41d49c991ab154d2cdad", "1b41024069204448847478a4e57bcac2", "d3beea952aad42b598ef1c0f08b18281", "ab9e29f6426549a9af71e34de1de97e6", "fb307cbe52594871b8908ce0ce6b5bb1", "f117d1fd16e8478ba957550e48b0999d", "9abca075286d44f5a6f5ca891e37f3d8", "33b7b06abe1846bda475fbf50f366afa", "95338ccff3b94f3cab9bf4d40f88ab21", "741c44fc1acd4aa8ba4ae1aba0d392a9", "d80419522c4843c095c2d2ac29a4ad80", "2135da24fd754edbafe3cc63ad09363a", "481711c45a2a4ed79bb616fa51bea9c4", "c5ae881c597a422fa430d4086dd5c2ce", "4880f5aa6c60444f8bfd8a571bbb037a", "262da405016e44cb914343381bf50338", "03ca8ddd4f224c59970884d536e66fae", "5f6e8f2f2be44f4f9c05ae031b007d26", "ebcc1e8dd0f84d1bb9388a7888035472", "593d875c5674446597995a6e3930fec2", "3281f9a252c24699895e93251920a30c", "46fd41815347496fb126d242ff46ce49", "1b3391a1dc9c4fa591eda9d25c0a0331", "c41e126631574113aa7c4a27b249aecb", "616b6c79101949babc2ae121d5d326b1", "7f6e6f1916b745a68c355184032aface", "3f37e6b930a143a1b84a2fc6dd5f89d9", "64a87e81339d4e3baec3ec321939d38e", "5e67aee730694db1b47e714363093aec", "0bce8240af574b5d9b4037b2b9769753", "f744b6255f49493da82df62f6a7d51ba", "d38338d46f584f7086262b4b9870da22", "47ab8d79921e41fe98afbaa5cfaa6ca3", "658e20b4a2b04bbb8933d34973c5a165", "3c9c97942b8f4c0f9c06f6763f0f2a56", "e8553be9add44a36af106f06fb179291", "0a9938b95c4543f2a99c4559861e3028", "9dd875d8ce8a45c6b2bc4e13a6c9285b", "25bd92afdeb24a2f87479ef6476c8193", "2e00657ab98a49259541d0f4dbddb77d", "d1f73ea817f84d1aa834a2fc6fb6770e", "db007527d98c46a1840d4030e72bd47a", "5502729f78be4ad1ad89d8cdbe960959", "5596a4f2a1c14ff08653a5922f906f8d", "42f0fc125d0742c9bda068712d0401aa", "0e392cc6a5c7476e9d87888441a2c286", "0bf0aaf1aa534beea101c77681e4dd78", "6937098e1d9141e9a87c13a15afb15b1", "7a610d254d904b77b9e2616e31140356", "b3cf58e68a584d3bb5a8220562112e1a", "d0ce7132b11448e88c59bea382b13acd", "e501ad63f91a4afdace899c7c2a0b804", "f62dae7c94e64ccb9b3eeb9396f7d3e7", "0cbbfb4cc2ca429c900ac58eb0bd23f6", "2e8ecf4cf0b947b4b2f23576b17c8b49", "1a9e9055c7c44b3e8d105c9f3d45f9bb", "c5decfdd55454b7593e9c3cd250c378c", "18003752d46449db8efa334c5057b25b", "04138e31fa824c2da18abdf02f7c182d", "12051119824a457bb39aa9daa8a54772", "003b511b777a48d3974f41e75ba16947", "52ea4c5fc47c4aa19d81ae0287241c3c", "a7e54860919c4b478b7da0f1c5d6744e", "5be757e1e2324df48339f0d386cdca6e", "677d3cd80afd456eb443624bff7279ba", "d9d1f3b40936475faf96bd50ddb9495d", "61584d7b68584ef994389afa19149eb9", "9be9101a5bb94955bddb76615fb10086", "420d14d8d51e4f0ba932dfa7328d6d44", "e3bba87b060342dd8c5777ce19c609d4", "a142edd72d11448da72275e49c1be851", "5693bd4b7b3f40b3826b3fbb31f7c241", "55e6a6b34a6a49418dbb25d23aa7de85", "830c6e29b15b4be793707b4000f55d16", "8322c517de664b1294ed46c10f3563f5", "13713789acbd48219d450eda3cfe3f60", "786e1147cf1c43a3a3d868420282233f", "312b176de0314cdbb7bb1d1fdd8e3dc4", "7c7a2c043865423db21018fd7a653c4f", "052ce97803c24167a244d130d3f95b30", "6d1dd31d44174ae4b1b8a9671cc73e46", "8c1e971804184c1a9f57d3e221bcf38f", "29310b11bb644111b6ed14ee3550a209", "027b31bb283642d2a0891fb70e8e53d4", "3ba14ec6ac1f4e228151ba2de350acd8", "c99df9961c164983baf4f4d0418eadc9", "f02f62c5539e456e87e30317ddf5fe49", "9d996c75e629492b94470ab249d2989b", "ceffb87f8a314d5ca62699463f174212", "d96ae1604d42452a8ea706d58c4baa8d", "18fbeec85b324845bd11c0f5a3e5563b", "dc0797bf58a448f3a7afe8272e4ac585", "0d12cabc15e94f8ebc72c11d98ad6f10", "baef70fc6a244446b875cff391450bbb", "a60f11e446c14b97970244fa8259b024", "013a47165cde4546bf63bec1cafae7f5", "1af2cda914bd49489f4c79ee13088a1c", "8c94f28ec13741d1a76babdf261aed9c", "1146c377170c4152886b4119539e0c22", "b1a24c2e59f24c2fb20e524386770952", "47cbe4f32fe94648be2e45caacf7a8ec", "e19a3311a616441abb150a5ffb14992d", "afc0ee6605264f748ec85e0d0491f121", "d0dcc67c9a1948aaa975fa160f264617", "0ac536ce6fe94d7e8205a5e28c43fb0c", "31c9aeee929640dd86acdcb957b41557", "e0266e562d864e0ab6ab2c875801cee5", "2c3e93c8da9846778cd0273ca2efd2d0", "b0945799102c44068cf577a8e6195981", "9d43787e4f4941bd8a141a6f664bb5fe", "dcb73e3905a04428b65e052d8fee53b5", "948ed06a5ee34f27b2f697e86609d3e9", "4a711e22dc7e445faf045ffe5aec0366", "5617a4601cb14765a2127cd5126009d5", "20ef425e86444ce9ac5df3cf4824163c", "1e942d4f02c041f8b4341c2ff8888f4f", "42435e9b2eb04f5d97912035884eb03c", "f1f0fcf19ed140efafc19429c69be496", "5c598e12e3ff4032abfa485014bbd8d9", "d78f634ce5dd47eca98ed32a13981490", "de58005b5b9448d781c2869cc223c573", "d569d03eafcf4d6bb58650cafba9204f", "56628a984752470b9606e9970932c806", "b827371db1ec4c1b865cc3420526a0fc", "f7199793201349628680c088c0994767", "44ba58a07c5a43ddb84227888014cb34", "60ded3e99e3e44c599a80a3c245a0641", "ba456273f7994ae59f533d5f1cb23107", "f52e5689bb3840568492d41cf5130c41", "ad2c2f384c46499680e72549d40c4981", "b81d71aeb14d4da19a3be1da8a4edc3b", "942d930366c8443680b21405fe8a6b62", "798fa952d9394d02a6e3be183d3379ef", "f40110d7043a4cec9fd877320b90237a", "c5e5cea2625a472eb1f23e371c77d369", "e6749cc013614d21983074daf4ca244c", "d7ad1402f71c401c9432c63d9590029a", "f1272da00e2f42a394ba32bd0d80c475", "2f3f47ccc05e4d5fbc5d756118a32f45", "031478ad0b0a42c5a00a017beb87a0b2", "e168d14aa1e5422a924546e9c1d7f2c3", "b855b3e935464867adec1f792002b84d", "f1df4be3c61846a684bbcc1c80bd33ce", "aa07bda824364bdd9c30d97903f34fe4", "cffc3e6fb414485e9d39aad0f0cd390f", "a6412551881247838ee4ee2cefe2eb62", "9a75d6c87cf8488e89f3e8df9d401fbf", "ff4690fb73b34ad8a15e899120ebb4eb", "82dd7b468b77460dba2369dff0d884c4", "1ead7f50ddbb4f4ba9b856e63ce6ff72", "1d7ef7384e384be0a0eb2c9ecb9ec233", "c90924554dcb42a9a78eb80ccacac75c", "51f51983992b4d5780a123f245c574bb", "db73f802d7d84c5f96840423fc64e669", "ad78a946ada74835ab9191a0b22fcf64", "19e881e8a062477883d95e661c4b5246", "3bccc2e180cc4e2093aad79188362c60", "a948d21f3e8541e3942065147114cf43", "904031803b26464db90f72bbbf594fd3", "3d6b783252a344cd9a7786a4a8ade474", "c01d11afb8694529b8dbdc4cd174e192", "9655b04983014d94984ca6f9893a1e73", "f8e784878b374513a9dcd33a9b8d249a", "1bbcd235e0884ef8a6fd2b2a81fa0fdb", "e3b5f8dfedfd45deb948b4aa43e001a3", "724028d68351480481130fa1942c2c7e", "420e9b3616e843d0bd35c07067e4f403", "3945160bf0c74544bf84c00a35c09bd0", "1ceb9cb4896a425791fb356309ef4ef9", "d76517ced7f940a8b67ff6b269db571a", "a8080eebdfeb4afdb6fca8a42ba9a714", "a5a05365b3c542acad47021f5c9caada", "f817c47e89e5476ca83ac00c3a7cd54e", "3085b46d75f44f3ca001020ed180725f", "1371406395ef4b68b0ec9a1d45e20104", "07450a34fabb4eb19ae373fe22d20e2f", "79cd443582714b24a383a360688e54d4", "f9708cfa1e6f454a9bba2e3a91434c27"]} executionInfo={"status": "ok", "timestamp": 1634729182264, "user_tz": -330, "elapsed": 3654228, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="757fd871-dd6d-49cd-973f-326a519fe31e"
for i, weight in enumerate(tqdm(evenly_dist(num_weights, 3), desc='Weight', leave=False)):
    for col in trange(6, desc='Direction', leave=False):
        
        mr_weights = np.array(
            [int(w) for w in '{:0>3b}'.format(col + 1)]).astype(linear_op_template.dtype).reshape(1, -1)
        
        # load SGD starting point
        init_ckpt = torch.load(sgd_path / '{:d}.pth'.format(i), map_location='cpu')
        network.load_state_dict(init_ckpt['state_dict'])

        # initalize momentum buffer
        jacobians_buffer_tensor = compute_jacobians()
        jacobians_buffer = jacobians_buffer_tensor.clone().detach().cpu().numpy()
        alpha_buffer = compute_alpha(jacobians_buffer_tensor)
        with trange(num_steps, desc='Step', leave=False) as step_iter:
            for step in step_iter:
                network.train(False)

                # compute jacobians
                jacobians_tensor = compute_jacobians(1.0 / 4.0)
                jacobians = jacobians_tensor.clone().detach().cpu().numpy()
                jacobians_buffer *= momentum
                jacobians_buffer += (1 - momentum) * jacobians
                jacobians = jacobians_buffer.copy()

                # compute alpha
                alpha = compute_alpha(jacobians_tensor)
                alpha_buffer *= momentum
                alpha_buffer += (1 - momentum) * alpha
                alpha = alpha_buffer.copy()

                # define rhs and x0
                rhs = np.squeeze(mr_weights @ jacobians)
                x0 = jacobians.mean(axis=0)
                
                # fill jacobians alpha rhs x0 to MINRES
                with linear_op_template.init(alpha) as linear_op:
                    results = minres(linear_op, rhs, x0=x0, maxiter=maxiter)
                    d = torch.as_tensor(results[0].astype(linear_op.dtype), device=device)

                # optimize
                mr_optimizer.zero_grad()
                assign_grad(d, normalize=True)
                mr_optimizer.step()

                eval_losses, eval_top1s = evaluate(network, testloader, closures, top1_closures)
                step_iter.set_postfix(**{'acc-{:d}'.format(i + 1): top for i, top in enumerate(eval_top1s)})
                ckpt = {
                    'state_dict': network.state_dict(),
                    'optimizer': mr_optimizer.state_dict(),
                    'metrics': [eval_losses, eval_top1s]
                }
                save_path = mr_path / str(i) / str(col)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, save_path / '{:d}.pth'.format(step))
```

<!-- #region id="0hydrXw2Xk5t" -->
## MINRES results illustration
<!-- #endregion -->

```python id="4tek7rqHU67O" colab={"base_uri": "https://localhost:8080/", "height": 369} executionInfo={"status": "ok", "timestamp": 1634729184750, "user_tz": -330, "elapsed": 2497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a2701ace-755d-4d7c-e931-36eb84305bbf"
fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='3d', proj_type='ortho'))
cmap = plt.get_cmap('autumn', len(evenly_dist(num_weights, 3)))

total_top1s = []
for i, weight in enumerate(evenly_dist(num_weights, 3)):
    ckpt = torch.load(sgd_path / '{:d}.pth'.format(i), map_location='cpu')
    losses, top1s = ckpt['metrics']
    total_top1s.append(top1s)
total_top1s = np.stack(total_top1s, axis=0).T
top1s_min = total_top1s.min(axis=1) * 0.99
total_err1s = 100.0 * (1.0 - total_top1s)
ax.scatter(*total_err1s, color=[cmap(i) for i, w in enumerate(evenly_dist(num_weights, 3))],
           marker='*', s=200, edgecolor='black')

for i, weight in enumerate(evenly_dist(num_weights, 3)):
    total_top1s = []
    for col in range(6):
        for step in range(num_steps):
            ckpt = torch.load(mr_path / str(i) / str(col) /  '{:d}.pth'.format(step), map_location='cpu')
            losses, top1s = ckpt['metrics']
            if any(top1s < top1s_min):
                continue
            total_top1s.append(top1s)
    total_err1s = 100.0 * (1.0 - np.stack(total_top1s, axis=0).T)
    ax.scatter(*total_err1s, color=cmap(i), marker='o', s=30)

ax.set_xlabel('Task 1 Top-1 Error')
ax.set_ylabel('Task 2 Top-1 Error')
ax.set_zlabel('Task 3 Top-1 Error')
ax.grid(True)

handles, labels = [], []
handles.append(
    tuple(ax.scatter([], [], [], color=cmap(i), marker='*', s=100, edgecolor='black') for i in [3, 6, 9]))
labels.append('Start')

handles.append(
    tuple(ax.scatter([], [], [], color=cmap(i), marker='o', s=30) for i in [3, 6, 9]))
labels.append('Ours')

ax.legend(handles, labels, handler_map={tuple: HandlerTuple(None, 0)})
ax.view_init(30, 45)
fig.tight_layout()
plt.show()
```
