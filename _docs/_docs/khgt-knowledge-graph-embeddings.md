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
    name: python3
---

<!-- #region id="PN6o0n4FLmgR" -->
# KHGT knowledge graph embeddings
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YrCCIZJL0RHf" executionInfo={"status": "ok", "timestamp": 1634137431404, "user_tz": -330, "elapsed": 17526, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="858d4b73-626e-41e2-82dd-0b55ae2690bf"
!git clone https://github.com/akaxlh/KHGT.git
!cp -r KHGT/Datasets .
```

```python colab={"base_uri": "https://localhost:8080/"} id="Hfxn2KIl0VHV" executionInfo={"status": "ok", "timestamp": 1634137438694, "user_tz": -330, "elapsed": 7321, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="94150b40-d43e-4b02-bd9e-b839458b14ed"
!apt-get install tree
```

<!-- #region id="MwvRbg_N594z" -->
### Data
<!-- #endregion -->

```python id="gYHG2lwu5_dO"
!cd Datasets/retail && unrar x -e trn_pv.part01.rar
```

```python colab={"base_uri": "https://localhost:8080/"} id="XWWTd3-F0bI6" executionInfo={"status": "ok", "timestamp": 1634137440493, "user_tz": -330, "elapsed": 60, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3836699b-cec3-4df0-d771-6bc33b7bccb0"
!tree --du -h ./Datasets
```

```python colab={"base_uri": "https://localhost:8080/"} id="So1K062h1lce" executionInfo={"status": "ok", "timestamp": 1634137440497, "user_tz": -330, "elapsed": 49, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="917f04b9-7711-4f6c-e16a-092cdcc50e2d"
%tensorflow_version 1.x
```

<!-- #region id="SgLNJEgF14Dd" -->
### CLI Run
<!-- #endregion -->

```python id="fLU7xmNX1XHy"
# # To run KHGT on Yelp data
# ! python labcode_yelp.py

# # For MovieLens data, use the following command to train
# ! python labcode_ml10m.py --data ml10m --graphSampleN 1000 --save_path XXX

# # test with larger sampled sub-graphs
# ! python labcode_ml10m.py --data ml10m --graphSampleN 5000 --epoch 0 --load_model XXX

# # For Online Retail data, use this command to train
# ! python labcode_retail.py --data retail --graphSampleN 15000 --reg 1e-1 --save_path XXX

# # test it with larger sampled sub-graphs
# ! python labcode_retail.py --data retail --graphSampleN 30000 --epoch 0 --load_model XXX
```

<!-- #region id="cZs_LlNf2E-M" -->
### NN Layers
<!-- #endregion -->

```python id="jXrWU2Sg2E7u"
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	# else:
	# 	print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer='zeros', reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	K = defineRandomNameParam([inpDim, inpDim], reg=True)
	V = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets

def lightSelfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=False)
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	tem = rspReps @ Q
	q = tf.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(inpDim/numHeads), axis=2)
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		rets[i] = tem1 + localReps[i]
	return rets
```

<!-- #region id="dN6PkpzA2E5l" -->
### Logger
<!-- #endregion -->

```python id="mEEETzOA2E2t"
import datetime

logmsg = ''
timemark = dict()
saveDefault = False

def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()

def SpentTime(marker):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker]

def SpentTooLong(marker, day=0, hour=0, minute=0, second=0):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker] >= datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second)
```

<!-- #region id="dvjuYIYx2E0Y" -->
### Params
<!-- #endregion -->

```python id="FRqvzMDZ2Ewo"
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=32, type=int, help='batch size')
	parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
	# parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
	parser.add_argument('--epoch', default=12, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=16, type=int, help='embedding size')
	parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
	parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')
	parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')
	parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')
	parser.add_argument('--slot', default=5, type=int, help='length of time slots')
	parser.add_argument('--graphSampleN', default=25000, type=int, help='use 25000 for training and 200000 for testing, empirically')
	parser.add_argument('--divSize', default=50, type=int, help='div size for smallTestEpoch')
	return parser.parse_args("")
 
args = parse_args()
# args.user = 147894
# args.item = 99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
args.user = 19800
args.item = 22734


args.decay_step = args.trnNum//args.batch
```

```python id="criRetuA3-No"
# python labcode_retail.py --data retail --graphSampleN 15000 --reg 1e-1 --save_path XXX
args.data = 'yelp'
# args.graphSampleN = 15000
# args.reg = 1e-1
args.save_path = '/content'
```

<!-- #region id="p6_37HdT2Eut" -->
### Utils
<!-- #endregion -->

```python id="7T3lIkbp2Esn"
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

if args.data == 'yelp':
	predir = 'Datasets/Yelp/'
	behs = ['tip', 'neg', 'neutral', 'pos']
elif args.data == 'ml10m':
	predir = 'Datasets/MultiInt-ML10M/'
	behs = ['neg', 'neutral', 'pos']
elif args.data == 'retail':
	predir = 'Datasets/retail/'
	behs = ['pv', 'fav', 'cart', 'buy']

trnfile = predir + 'trn_'
tstfile = predir + 'tst_'


def helpInit(a, b, c):
	ret = [[None] * b for i in range(a)]
	for i in range(a):
		for j in range(b):
			ret[i][j] = [None] * c
	return ret

def timeProcess(trnMats):
	mi = 1e15
	ma = 0
	for i in range(len(trnMats)):
		minn = np.min(trnMats[i].data)
		maxx = np.max(trnMats[i].data)
		mi = min(mi, minn)
		ma = max(ma, maxx)
	maxTime = 0
	for i in range(len(trnMats)):
		newData = ((trnMats[i].data - mi) / (3600*24*args.slot)).astype(np.int32)
		maxTime = max(np.max(newData), maxTime)
		trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
	print('MAX TIME', maxTime)
	return trnMats, maxTime + 1

# behs = ['buy']

def ObtainIIMats(trnMats, predir):
	# # MAKE
	# iiMats = list()
	# for i in range(len(behs)):
	# 	iiMats.append(makeIiMats(trnMats[i]))
	# 	print('i', i)
	# with open(predir+'trn_catDict', 'rb') as fs:
	# 	catDict = pickle.load(fs)
	# iiMats.append(makeCatIiMats(catDict, trnMats[0].shape[1]))

	# # DUMP
	# with open(predir+'iiMats_cache', 'wb') as fs:
	# 	pickle.dump(iiMats, fs)
	# exit()

	# READ
	with open(predir+'iiMats', 'rb') as fs:
		iiMats = pickle.load(fs)
	# iiMats = iiMats[3:]# + iiMats[2:]
	return iiMats

def LoadData():
	trnMats = list()
	for i in range(len(behs)):
		beh = behs[i]
		path = trnfile + beh
		with open(path, 'rb') as fs:
			mat = pickle.load(fs)
		trnMats.append(mat)
		if args.target == 'click':
			trnLabel = (mat if i==0 else 1 * (trnLabel + mat != 0))
		elif args.target == 'buy' and i == len(behs) - 1:
			trnLabel = 1 * (mat != 0)
	trnMats, maxTime = timeProcess(trnMats)
	# test set
	path = tstfile + 'int'
	with open(path, 'rb') as fs:
		tstInt = np.array(pickle.load(fs))
	tstStat = (tstInt!=None)
	tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])

	iiMats = ObtainIIMats(trnMats, predir)

	return trnMats, iiMats, tstInt, trnLabel, tstUsrs, len(behs), maxTime

# negative sampling using pre-sampled entities (preSamp) for efficiency
def negSamp(temLabel, preSamp, sampSize=1000):
	negset = [None] * sampSize
	cur = 0
	for temval in preSamp:
		if temLabel[temval] == 0:
			negset[cur] = temval
			cur += 1
		if cur == sampSize:
			break
	negset = np.array(negset[:cur])
	return negset

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def transToLsts(mat, mask=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.int32)

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.int32)
	return indices, data, shape

def makeCatIiMats(dic, itmnum):
	retInds = []
	for key in dic:
		temLst = list(dic[key])
		for i in range(len(temLst)):
			if args.data == 'tmall' and args.target == 'click':
				div = 50
			else:
				div = 10
			if args.data == 'ml10m' or args.data == 'tmall' and args.target == 'click':
				scdTemLst = list(np.random.choice(range(len(temLst)), len(temLst) // div, replace=False))
			else:
				scdTemLst = range(len(temLst))
			for j in scdTemLst:#range(len(temLst)):
				# if args.data == 'ml10m' and np.random.uniform(0.0, 1.0) < 0.1:
				# 	continue
				retInds.append([temLst[i], temLst[j]])
	pckLocs = np.random.permutation(len(retInds))[:100000]#:len(retInds)//100]
	retInds = np.array(retInds, dtype=np.int32)[pckLocs]
	retData = np.array([1] * retInds.shape[0], np.int32)
	return retInds, retData, [itmnum, itmnum]

def makeIiMats(mat):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = list(map(list, zip(coomat.row, coomat.col)))
	uDict = [set() for i in range(shape[0])]
	for ind in indices:
		usr = ind[0]
		itm = ind[1]
		uDict[usr].add(itm)
	retInds = []
	for usr in range(shape[0]):
		temLst = list(uDict[usr])
		for i in range(len(temLst)):
			if args.data == 'tmall' and args.target == 'click':
				div = 50
			else:
				div = 10
			if args.data == 'ml10m' or args.data == 'tmall' and args.target == 'click':
				scdTemLst = list(np.random.choice(range(len(temLst)), len(temLst) // div, replace=False))
			else:
				scdTemLst = range(len(temLst))
			for j in scdTemLst:#range(len(temLst)):
				# if args.data == 'ml10m' and np.random.uniform(0.0, 1.0) < 0.1:
				# 	continue
				retInds.append([temLst[i], temLst[j]])
	pckLocs = np.random.permutation(len(retInds))[:100000]#[:len(retInds)//100]
	retInds = np.array(retInds, dtype=np.int32)[pckLocs]
	retData = np.array([1] * retInds.shape[0], np.int32)
	return retInds, retData, [shape[1], shape[1]]

def prepareGlobalData(trnMats, trnLabel, iiMats):
	global adjs
	global adj
	global tpadj
	global iiAdjs
	adjs = trnMats
	iiAdjs = list()
	for i in range(len(iiMats)):
		iiAdjs.append(csr_matrix((iiMats[i][1], (iiMats[i][0][:,0], iiMats[i][0][:,1])), shape=iiMats[i][2]))
	adj = trnLabel.astype(np.float32)
	tpadj = transpose(adj)
	adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
	tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
	for i in range(adj.shape[0]):
		for j in range(adj.indptr[i], adj.indptr[i+1]):
			adj.data[j] /= adjNorm[i]
	for i in range(tpadj.shape[0]):
		for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
			tpadj.data[j] /= tpadjNorm[i]

def sampleLargeGraph(pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN):
	global adjs
	global adj
	global tpadj
	global iiAdjs

	def makeMask(nodes, size):
		mask = np.ones(size)
		if not nodes is None:
			mask[nodes] = 0.0
		return mask

	def updateBdgt(adj, nodes):
		if nodes is None:
			return 0
		tembat = 1000
		ret = 0
		for i in range(int(np.ceil(len(nodes) / tembat))):
			st = tembat * i
			ed = min((i+1) * tembat, len(nodes))
			temNodes = nodes[st: ed]
			ret += np.sum(adj[temNodes], axis=0)
		return ret

	def sample(budget, mask, sampNum):
		score = (mask * np.reshape(np.array(budget), [-1])) ** 2
		norm = np.sum(score)
		if norm == 0:
			return np.random.choice(len(score), 1)
		score = list(score / norm)
		arrScore = np.array(score)
		posNum = np.sum(np.array(score)!=0)
		if posNum < sampNum:
			pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
			pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
			pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
		else:
			pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
		return pckNodes

	usrMask = makeMask(pckUsrs, adj.shape[0])
	itmMask = makeMask(pckItms, adj.shape[1])
	itmBdgt = updateBdgt(adj, pckUsrs)
	if pckItms is None:
		pckItms = sample(itmBdgt, itmMask, len(pckUsrs))
		# pckItms = sample(itmBdgt, itmMask, sampNum)
		itmMask = itmMask * makeMask(pckItms, adj.shape[1])
	usrBdgt = updateBdgt(tpadj, pckItms)
	for i in range(sampDepth):
		newUsrs = sample(usrBdgt, usrMask, sampNum)
		usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
		newItms = sample(itmBdgt, itmMask, sampNum)
		itmMask = itmMask * makeMask(newItms, adj.shape[1])
		if i == sampDepth - 1:
			break
		usrBdgt += updateBdgt(tpadj, newItms)
		itmBdgt += updateBdgt(adj, newUsrs)
	usrs = np.reshape(np.argwhere(usrMask==0), [-1])
	itms = np.reshape(np.argwhere(itmMask==0), [-1])
	pckAdjs = []
	pckTpAdjs = []
	pckIiAdjs = []
	for i in range(len(adjs)):
		pckU = adjs[i][usrs]
		tpPckI = transpose(pckU)[itms]
		pckTpAdjs.append(tpPckI)
		pckAdjs.append(transpose(tpPckI))
	for i in range(len(iiAdjs)):
		pckI = iiAdjs[i][itms]
		tpPckI = transpose(pckI)[itms]
		pckIiAdjs.append(tpPckI)
	return pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms
```

<!-- #region id="q55PIcHK2EqT" -->
### Run
<!-- #endregion -->

```python id="va31ijFN2Elx"
!mkdir -p History Models
```

```python colab={"base_uri": "https://localhost:8080/"} id="tQMAToyz1ZLa" executionInfo={"status": "ok", "timestamp": 1634141291431, "user_tz": -330, "elapsed": 3843860, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="677ddbbe-a690-4c44-f6c5-c51f9f6993e5"
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle


class Recommender:
	def __init__(self, sess, datas):
		self.sess = sess
		self.trnMats, self.iiMats, self.tstInt, self.label, self.tstUsrs, args.intTypes, self.maxTime = datas
		prepareGlobalData(self.trnMats, self.label, self.iiMats)
		args.user, args.item = self.trnMats[0].shape
		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train'+met] = list()
			self.metrics['Test'+met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * 3
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Varaibles Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % 3 == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % 5 == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def GAT(self, srcEmbeds, tgtEmbeds, tgtNodes, maxNum, Qs, Ks, Vs):
		QWeight = tf.nn.softmax(defineRandomNameParam([args.memosize, 1, 1], reg=True), axis=1)
		KWeight = tf.nn.softmax(defineRandomNameParam([args.memosize, 1, 1], reg=True), axis=1)
		VWeight = tf.nn.softmax(defineRandomNameParam([args.memosize, 1, 1], reg=True), axis=1)
		Q = tf.reduce_sum(Qs * QWeight, axis=0)
		K = tf.reduce_sum(Ks * KWeight, axis=0)
		V = tf.reduce_sum(Vs * VWeight, axis=0)

		q = tf.reshape(tgtEmbeds @ Q, [-1, args.att_head, args.latdim//args.att_head])
		k = tf.reshape(srcEmbeds @ K, [-1, args.att_head, args.latdim//args.att_head])
		v = tf.reshape(srcEmbeds @ V, [-1, args.att_head, args.latdim//args.att_head])
		logits = tf.math.exp(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(args.latdim/args.att_head))
		attNorm = tf.nn.embedding_lookup(tf.math.segment_sum(logits, tgtNodes), tgtNodes) + 1e-6
		att = logits / attNorm
		padAttval = tf.pad(att * v, [[0, 1], [0, 0], [0, 0]])
		padTgtNodes = tf.concat([tgtNodes, tf.reshape(maxNum-1, [1])], axis=-1)
		attval = tf.reshape(tf.math.segment_sum(padAttval, padTgtNodes), [-1, args.latdim])
		return attval

	def messagePropagate(self, srclats, tgtlats, mats, maxNum, wTime=True):
		unAct = []
		lats1 = []
		paramId = 'dfltP%d' % getParamId()
		Qs = defineRandomNameParam([args.memosize, args.latdim, args.latdim], reg=True)
		Ks = defineRandomNameParam([args.memosize, args.latdim, args.latdim], reg=True)
		Vs = defineRandomNameParam([args.memosize, args.latdim, args.latdim], reg=True)
		for mat in mats:
			timeEmbed = FC(self.timeEmbed, args.latdim, reg=True)
			srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1]))
			tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1]))
			edgeVals = mat.values
			srcEmbeds = (tf.nn.embedding_lookup(srclats, srcNodes) + (tf.nn.embedding_lookup(timeEmbed, edgeVals) if wTime else 0))
			tgtEmbeds = tf.nn.embedding_lookup(tgtlats, tgtNodes)

			newTgtEmbeds = self.GAT(srcEmbeds, tgtEmbeds, tgtNodes, maxNum, Qs, Ks, Vs)

			unAct.append(newTgtEmbeds)
			lats1.append(Activate(newTgtEmbeds, self.actFunc))

		lats2 = lightSelfAttention(lats1, number=len(mats), inpDim=args.latdim, numHeads=args.att_head)

		# aggregation gate
		globalQuery = Activate(tf.add_n(unAct), self.actFunc)
		weights = []
		paramId = 'dfltP%d' % getParamId()
		for lat in lats2:
			temlat = FC(tf.concat([lat, globalQuery], axis=-1) , args.latdim//2, useBias=False, reg=False, activation=self.actFunc, name=paramId+'_1', reuse=True)
			weight = FC(temlat, 1, useBias=False, reg=False, name=paramId+'_2', reuse=True)
			weights.append(weight)
		stkWeight = tf.concat(weights, axis=1)
		sftWeight = tf.reshape(tf.nn.softmax(stkWeight, axis=1), [-1, len(mats), 1]) * 8
		stkLat = tf.stack(lats2, axis=1)
		lat = tf.reshape(tf.reduce_sum(sftWeight * stkLat, axis=1), [-1, args.latdim])
		return lat

	def makeTimeEmbed(self):
		divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
		pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
		sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2])/4.0
		return timeEmbed

	def ours(self):
		all_uEmbed0 = defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		all_iEmbed0 = defineParam('iEmbed0', [args.item, args.latdim], reg=True)
		uEmbed0 = tf.nn.embedding_lookup(all_uEmbed0, self.all_usrs)
		iEmbed0 = tf.nn.embedding_lookup(all_iEmbed0, self.all_itms)
		self.timeEmbed = tf.Variable(initial_value=self.makeTimeEmbed(), shape=[self.maxTime, args.latdim*2], name='timeEmbed', trainable=True)
		addReg('timeEmbed', self.timeEmbed)
		ulats = [uEmbed0]
		ilats = [iEmbed0]
		for i in range(args.gnn_layer):
			ulat = self.messagePropagate(ilats[-1], ulats[-1], self.adjs, self.usrNum)
			ilat1 = self.messagePropagate(ulats[-1], ilats[-1], self.tpAdjs, self.itmNum)
			ilat2 = self.messagePropagate(ilats[-1], ilats[-1], self.iiAdjs, self.itmNum, wTime=False)
			ilat = args.iiweight * ilat2 + (1.0 - args.iiweight) * ilat1
			ulats.append(ulat + ulats[-1])
			ilats.append(ilat + ilats[-1])

		UEmbedPred = defineParam('UEmbedPred', shape=[args.user, args.latdim], dtype=tf.float32, reg=False)
		IEmbedPred = defineParam('IEmbedPred', shape=[args.item, args.latdim], dtype=tf.float32, reg=False)
		ulats[0] = tf.nn.embedding_lookup(UEmbedPred, self.all_usrs)
		ilats[0] = tf.nn.embedding_lookup(IEmbedPred, self.all_itms)

		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)
		pckULat = tf.nn.embedding_lookup(ulat, self.uids)
		pckILat = tf.nn.embedding_lookup(ilat, self.iids)

		predLat = pckULat * pckILat * args.mult

		for i in range(args.deep_layer):
			predLat = FC(predLat, args.latdim, reg=True, useBias=True, activation=self.actFunc) + predLat
		pred = tf.squeeze(FC(predLat, 1, reg=True, useBias=True))
		return pred

	def prepareModel(self):
		self.keepRate = tf.placeholder(name='keepRate', dtype=tf.float32, shape=[])
		self.actFunc = 'twoWayLeakyRelu6'
		self.adjs = []
		self.tpAdjs = []
		self.iiAdjs = []
		for i in range(args.intTypes):
			self.adjs.append(tf.sparse_placeholder(dtype=tf.int32))
			self.tpAdjs.append(tf.sparse_placeholder(dtype=tf.int32))
		for i in range(len(self.iiMats)):
			self.iiAdjs.append(tf.sparse_placeholder(dtype=tf.int32))

		self.all_usrs = tf.placeholder(name='all_usrs', dtype=tf.int32, shape=[None])
		self.all_itms = tf.placeholder(name='all_itms', dtype=tf.int32, shape=[None])
		self.usrNum = tf.placeholder(name='usrNum', dtype=tf.int64, shape=[])
		self.itmNum = tf.placeholder(name='itmNum', dtype=tf.int64, shape=[])
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])

		self.pred = self.ours()
		sampNum = tf.shape(self.iids)[0] // 2
		posPred = tf.slice(self.pred, [0], [sampNum])
		negPred = tf.slice(self.pred, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batchIds, itmnum, label):
		preSamp = list(np.random.permutation(itmnum))
		temLabel = label[batchIds].toarray()
		batch = len(batchIds)
		temlen = batch * 2 * args.sampNum
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			negset = negSamp(temLabel[i], preSamp)
			poslocs = np.random.choice(posset, args.sampNum)
			neglocs = np.random.choice(negset, args.sampNum)
			for j in range(args.sampNum):
				uIntLoc[cur] = uIntLoc[cur+temlen//2] = batchIds[i]
				iIntLoc[cur] = poslocs[j]
				iIntLoc[cur+temlen//2] = neglocs[j]
				cur += 1
		return uIntLoc, iIntLoc

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(sfIds)
		pckLabel = transpose(transpose(self.label[usrs])[itms])
		usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
		sfIds = list(map(lambda x: usrIdMap[x], sfIds))
		feeddict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
		for i in range(args.intTypes):
			feeddict[self.adjs[i]] = transToLsts(pckAdjs[i])
			feeddict[self.tpAdjs[i]] = transToLsts(pckTpAdjs[i])
		for i in range(len(pckIiAdjs)):
			feeddict[self.iiAdjs[i]] = transToLsts(pckIiAdjs[i])

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			uLocs, iLocs = self.sampleTrainBatch(batIds, pckAdjs[0].shape[1], pckLabel)

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			feeddict[self.uids] = uLocs
			feeddict[self.iids] = iLocs
			res = self.sess.run(target, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batchIds, label, tstInt):
		batch = len(batchIds)
		temTst = tstInt[batchIds]
		temLabel = label[batchIds].toarray()
		temlen = batch * 100
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uIntLoc[cur] = batchIds[i]
				iIntLoc[cur] = locset[j]
				cur += 1
		return uIntLoc, iIntLoc, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.tstUsrs
		num = len(ids)
		tstBat = np.maximum(1, args.batch * args.sampNum // 100)
		steps = int(np.ceil(num / tstBat))

		posItms = self.tstInt[ids]
		pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(ids, list(set(posItms)))
		pckLabel = transpose(transpose(self.label[usrs])[itms])
		usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
		itmIdMap = dict(map(lambda x: (itms[x], x), range(len(itms))))
		ids = list(map(lambda x: usrIdMap[x], ids))
		itmMapping = (lambda x: None if (x is None) else itmIdMap[x])
		pckTstInt = np.array(list(map(lambda x: itmMapping(self.tstInt[usrs[x]]), range(len(usrs)))))
		feeddict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
		for i in range(args.intTypes):
			feeddict[self.adjs[i]] = transToLsts(pckAdjs[i])
			feeddict[self.tpAdjs[i]] = transToLsts(pckTpAdjs[i])
		for i in range(len(pckIiAdjs)):
			feeddict[self.iiAdjs[i]] = transToLsts(pckIiAdjs[i])

		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, pckLabel, pckTstInt)
			feeddict[self.uids] = uLocs
			feeddict[self.iids] = iLocs
			preds = self.sess.run(self.pred, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	datas = LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, datas)
		recom.run()
```
