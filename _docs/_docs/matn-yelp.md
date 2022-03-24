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

<!-- #region id="B6APV93JvrV0" -->
# MATN on Yelp in Tensorflow
> Multiplex Behavioral Relation Learning for Recommendation via Memory Augmented Transformer Network
<!-- #endregion -->

<!-- #region id="2hoHcDfpumh7" -->
## Setup
<!-- #endregion -->

```python id="SwuiBSWoqctn"
!git clone https://github.com/akaxlh/MATN.git
```

```python id="mYENSPmkqdsP"
!apt-get install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="mYqs8GfEqf6r" executionInfo={"status": "ok", "timestamp": 1634112044005, "user_tz": -330, "elapsed": 614, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f6d1527a-0efc-4b08-ce0a-c240ef5a5b70"
!tree --du -h ./MATN
```

```python colab={"base_uri": "https://localhost:8080/"} id="kb2v5YbGq-tO" executionInfo={"status": "ok", "timestamp": 1634113312265, "user_tz": -330, "elapsed": 1031, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fbcb80c3-1df9-4efd-e28c-054d02671cbd"
%tensorflow_version 1.x
```

```python id="6F9nZ0BDvD_e"
!mkdir -p History Models
```

```python id="fRo1wpS-uo1R"
import pickle
import argparse
import pickle
import numpy as np
import datetime
from scipy.sparse import csr_matrix

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.contrib.layers import xavier_initializer
```

<!-- #region id="qjOKVmpErJU0" -->
## NN Layers
<!-- #endregion -->

```python id="WwWuxkJ0q8P1"
paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.01

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

def defineParam(name, shape, dtype=tf.float32, reg=False,
	initializer='xavier', trainable=True):
	global params
	global regParams
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

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False,
	initializer='xavier', trainable=True):
	global params
	global regParams
	if name in params:
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

def FC(inp, outDim, name=None, useBias=False, activation=None,
	reg=False, useBN=False, dropout=None, initializer='xavier', noDrop=False):
	global params
	global regParams
	global leaky
	# useBias = biasDefault
	# if not noDrop:
	# 	inp = tf.nn.dropout(inp, rate=0.001)
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		temBiasName = temName + 'Bias'
		bias = getOrDefineParam(temBiasName, outDim, reg=False, initializer='zeros')
		ret = ret + bias
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer='zeros')
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
	elif method == 'twoWayLeakyRelu':
		temMask = tf.to_float(tf.greater(data, 1.0))
		ret = temMask * (1 + leaky * (data - 1)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
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
```

<!-- #region id="xlLh1HkrrLB7" -->
## Logger
<!-- #endregion -->

```python id="v-wrKKZfrOyd"
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

<!-- #region id="8-W6urrWrPKE" -->
## Data Handler
<!-- #endregion -->

```python id="6eoRj_smrTR9"
# predir = 'Datasets/Tmall/backup/hr_ndcg_click/'
# predir = 'Datasets/MultiInt-ML10M/buy/'
predir = 'MATN/Datasets/yelp/click/'
trnfile = predir + 'trn_'
tstfile = predir + 'tst_'
# behs = ['pv', 'fav', 'cart', 'buy']
# behs = ['neg', 'neutral', 'pos']
behs = ['tip', 'neg', 'neutral', 'pos']

def helpInit(a, b, c):
	ret = [[None] * b for i in range(a)]
	for i in range(a):
		for j in range(b):
			ret[i][j] = [None] * c
	return ret

def LoadData():
	for i in range(len(behs)):
		beh = behs[i]
		path = trnfile + beh
		with open(path, 'rb') as fs:
			mat = (2**i)*(pickle.load(fs)!=0)
		trnMat = (mat if i==0 else trnMat + mat)
		# if i == len(behs)-1:
		# 	buyMat = 1 * (mat != 0)
	buyMat = 1 * (trnMat != 0)
	# test set
	path = tstfile + 'int'
	with open(path, 'rb') as fs:
		tstInt = np.array(pickle.load(fs))
	tstStat = (tstInt!=None)
	tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])

	return trnMat, tstInt, buyMat, tstUsrs

def getmask(low, high, trnMat, tstUsrs, tstInt):
	cnts = np.reshape(np.array(np.sum(trnMat, axis=-1)), [-1])
	lst = list()
	for usr in tstUsrs:
		lst.append((cnts[usr], usr))
	lst.sort(key=lambda x: x[0])
	length = len(lst)
	l = int(low * length)
	r = int(high * length)
	ret = set()
	for i in range(l, r):
		ret.add(lst[i][1])
	return ret

def negSamp(tembuy, curlist):
	temsize = 1000#1000
	negset = [None] * temsize
	cur = 0
	for temcur in curlist:
		if tembuy[temcur] == 0:
			negset[cur] = temcur
			cur += 1
		if cur == temsize:
			break
	negset = np.array(negset[:cur])
	return negset

def TransMat(mat):
	user, item = mat.shape
	data = mat.data
	indices = mat.indices
	indptr = mat.indptr

	newdata = [None] * len(data)
	rowInd = [None] * len(data)
	colInd = [None] * len(data)
	length = 0

	for i in range(user):
		temlocs = indices[indptr[i]: indptr[i+1]]
		temvals = data[indptr[i]: indptr[i+1]]
		for j in range(len(temlocs)):
			rowInd[length] = temlocs[j]
			colInd[length] = i
			newdata[length] = temvals[j]
			length += 1
	if length != len(data):
		print('ERROR IN Trans', length, len(data))
		exit()
	tpMat = csr_matrix((newdata, (rowInd, colInd)), shape=[item, user])
	return tpMat

def binFind(pred, shoot):
	minn = np.min(pred)
	maxx = np.max(pred)
	l = minn
	r = maxx
	while True:
		mid = (l + r) / 2
		tem = (pred - mid) > 0
		num = np.sum(tem)
		if num == shoot or np.abs(l - r)<1e-3:
			arr = tem
			break
		if num > shoot:
			l = mid
		else:
			r = mid
	return np.reshape(np.argwhere(tem), [-1])[:shoot]
```

<!-- #region id="UijaFqykrYpR" -->
## Params
<!-- #endregion -->

```python id="hvhp5wkNrbbG"
def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
	parser.add_argument('--batch', default=32, type=int, help='batch size')
	parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
	# parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
	parser.add_argument('--epoch', default=12, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=8, type=int, help='embedding size')
	parser.add_argument('--memosize', default=4, type=int, help='memory size')
	parser.add_argument('--posbat', default=40, type=int, help='batch size of positive sampling')
	parser.add_argument('--negsamp', default=1, type=int, help='rate of negative sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
	parser.add_argument('--trn_num', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	return parser.parse_args(args={})
 
args = parse_args()
# args.user = 805506#147894
# args.item = 584050#99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
args.user = 19800
args.item = 22734

# swap user and item
# tem = args.user
# args.user = args.item
# args.item = tem

# args.decay_step = args.trn_num
args.decay_step = args.item//args.batch
```

<!-- #region id="HPwctu5nrf5h" -->
## Run
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Qgl9jdg3rjk6" executionInfo={"status": "ok", "timestamp": 1634114080098, "user_tz": -330, "elapsed": 763277, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="67f20ec1-b10f-47a8-bcee-85494b51dabf"
class Recommender:
	def __init__(self, sess, datas, inpDim):
		self.inpDim = inpDim
		self.sess = sess
		self.trnMat, self.tstInt, self.buyMat, self.tstUsrs = datas
		self.metrics = dict()
		mets = ['Loss', 'preLoss' 'HR', 'NDCG']
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
			stloc = len(self.metrics['TrainLoss'])
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
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

	def multiHeadAttention(self, localReps, glbRep, number, numHeads, inpDim):
		query = tf.reshape(tf.tile(tf.reshape(FC(glbRep, inpDim, useBias=True, reg=True), [-1, 1, inpDim]), [1, number, 1]), [-1, numHeads, inpDim//numHeads])
		temLocals = tf.reshape(localReps, [-1, inpDim])
		key = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, numHeads, inpDim//numHeads])
		val = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, number, numHeads, inpDim//numHeads])
		att = tf.nn.softmax(2*tf.reshape(tf.reduce_sum(query * key, axis=-1), [-1, number, numHeads, 1]), axis=1)
		attRep = tf.reshape(tf.reduce_sum(val * att, axis=1), [-1, inpDim])
		return attRep

	def selfAttention(self, localReps, number, inpDim):
		attReps = [None] * number
		stkReps = tf.stack(localReps, axis=1)
		for i in range(number):
			glbRep = localReps[i]
			temAttRep = self.multiHeadAttention(stkReps, glbRep, number=number, numHeads=args.att_head, inpDim=inpDim) + glbRep
			# fc1 = FC(temAttRep, inpDim, reg=True, useBias=True, activation='relu') + temAttRep
			# fc2 = FC(fc1, inpDim, reg=True, useBias=True, activation='relu') + fc1
			attReps[i] = temAttRep#fc2
		return attReps

	def divide(self, interaction):
		ret = [None] * self.intTypes
		for i in range(self.intTypes):
			ret[i] = tf.to_float(tf.bitwise.bitwise_and(interaction, (2**i)) / (2**i))
		return ret

	def mine(self, interaction):
		activation = 'relu'
		V = defineParam('v', [self.inpDim, args.latdim], reg=True)
		divideLst = self.divide(interaction)
		catlat1 = []
		for dividInp in divideLst:
			catlat1.append(dividInp @ V)
		catlat2 = self.selfAttention(catlat1, number=self.intTypes, inpDim=args.latdim)
		catlat3 = list()
		self.memoAtt = []
		for i in range(self.intTypes):
			resCatlat = catlat2[i] + catlat1[i]
			memoatt = FC(resCatlat, args.memosize, activation='relu', reg=True, useBias=True)
			memoTrans = tf.reshape(FC(memoatt, args.latdim**2, reg=True, name='memoTrans'), [-1, args.latdim, args.latdim])
			self.memoAtt.append(memoatt)

			tem = tf.reshape(resCatlat, [-1, 1, args.latdim])
			transCatlat = tf.reshape(tem @ memoTrans, [-1, args.latdim])
			catlat3.append(transCatlat)

		stkCatlat3 = tf.stack(catlat3, axis=1)

		weights = defineParam('fuseAttWeight', [1, self.intTypes, 1], reg=True, initializer='zeros')
		sftW = tf.nn.softmax(weights*2, axis=1)
		fusedLat = tf.reduce_sum(sftW * stkCatlat3, axis=1)
		self.memoAtt = tf.stack(self.memoAtt, axis=1)

		lat = fusedLat
		for i in range(2):
			lat = FC(lat, args.latdim, useBias=True, reg=True, activation=activation) + lat
		return lat

	def prepareModel(self):
		self.intTypes = 4
		self.interaction = tf.placeholder(dtype=tf.int32, shape=[None, self.inpDim], name='interaction')
		self.posLabel = tf.placeholder(dtype=tf.int32, shape=[None, None], name='posLabel')
		self.negLabel = tf.placeholder(dtype=tf.int32, shape=[None, None], name='negLabel')
		intEmbed = tf.reshape(self.mine(self.interaction), [-1, 1, args.latdim])
		self.learnedEmbed = tf.reshape(intEmbed, [-1, args.latdim])

		W = defineParam('W', [self.inpDim, args.latdim], reg=True)
		posEmbeds = tf.transpose(tf.nn.embedding_lookup(W, self.posLabel), [0, 2, 1])
		negEmbeds = tf.transpose(tf.nn.embedding_lookup(W, self.negLabel), [0, 2, 1])
		sampnum = tf.shape(self.posLabel)[1]

		posPred = tf.reshape(intEmbed @ posEmbeds, [-1, sampnum])
		negPred = tf.reshape(intEmbed @ negEmbeds, [-1, sampnum])
		self.posPred = posPred

		self.preLoss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred)), axis=-1))
		self.regLoss = args.reg * Regularize(method='L2')
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def trainEpoch(self):
		trnMat = self.trnMat
		num = trnMat.shape[0]
		trnSfIds = np.random.permutation(num)[:args.trn_num]
		tstSfIds = self.tstUsrs
		sfIds = np.random.permutation(np.concatenate((trnSfIds, tstSfIds)))
		# sfIds = trnSfIds
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			curLst = list(np.random.permutation(self.inpDim))
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batchIds = sfIds[st: ed]

			temTrn = trnMat[batchIds].toarray()
			tembuy = self.buyMat[batchIds].toarray()

			temPos = [[None]*(args.posbat*args.negsamp) for i in range(len(batchIds))]
			temNeg = [[None]*(args.posbat*args.negsamp) for i in range(len(batchIds))]
			for ii in range(len(batchIds)):
				row = batchIds[ii]
				posset = np.reshape(np.argwhere(tembuy[ii]!=0), [-1])
				negset = negSamp(tembuy[ii], curLst)
				idx = 0
				# if len(posset) == 0:
				# 	posset = np.random.choice(list(range(args.item)), args.posbat)
				for j in np.random.choice(posset, args.posbat):
					for k in np.random.choice(negset, args.negsamp):
						temPos[ii][idx] = j
						temNeg[ii][idx] = k
						idx += 1
			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			res = self.sess.run(target, feed_dict={self.interaction: (temTrn).astype('int32'),
				self.posLabel: temPos, self.negLabel: temNeg
				}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f       ' %\
				(i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def testEpoch(self):
		trnMat = self.trnMat
		tstInt = self.tstInt
		epochHit, epochNdcg = [0] * 2
		ids = self.tstUsrs
		num = len(ids)
		testbatch = args.batch
		steps = int(np.ceil(num / testbatch))
		for i in range(steps):
			st = i * testbatch
			ed = min((i+1) * testbatch, num)
			batchIds = ids[st:ed]

			temTrn = trnMat[batchIds].toarray()
			temTst = tstInt[batchIds]
			tembuy = self.buyMat[batchIds].toarray()

			# get test locations
			tstLocs = [None] * len(batchIds)
			for j in range(len(batchIds)):
				negset = np.reshape(np.argwhere(tembuy[j]==0), [-1])
				rdnNegSet = np.random.permutation(negset)
				tstLocs[j] = list(rdnNegSet[:99])
				tem = ([rdnNegSet[99]] if temTst[j] in tstLocs[j] else [temTst[j]])
				tstLocs[j] = tstLocs[j] + tem

			preds = self.sess.run(self.posPred, feed_dict={self.interaction:temTrn.astype('int32'), self.posLabel: tstLocs}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			hit, ndcg = self.calcRes(preds, temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Step %d/%d: hit = %d, ndcg = %d      ' %\
				(i, steps, hit, ndcg), save=False, oneline=True)
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
		recom = Recommender(sess, datas, args.item)
		recom.run()
```
