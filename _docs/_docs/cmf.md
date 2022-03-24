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

<!-- #region id="Np5vPhJjEpNK" -->
# Collective Matrix Factorization on ML-1m
<!-- #endregion -->

<!-- #region id="_Klvaj0QEtkf" -->
## CMF on dummy dataset using PyCMF library
<!-- #endregion -->

```python id="k6CKGfEu_E_U"
!pip install git+https://github.com/smn-ailab/PyCMF
```

```python id="9y-m5IzJ_IK4"
import numpy as np                                                                                          
import pycmf

X = np.abs(np.random.randn(5, 4)); Y = np.abs(np.random.randn(4, 1))
model = pycmf.CMF(n_components=4)
U, V, Z = model.fit_transform(X, Y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qqEXG-3i_Mbc" executionInfo={"status": "ok", "timestamp": 1635744822247, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3a4cd6d3-2b3e-4531-cb30-e418ab00a84c"
np.linalg.norm(X - U @ V.T) / np.linalg.norm(X)
```

```python colab={"base_uri": "https://localhost:8080/"} id="q8a_IZUQ_Kr8" executionInfo={"status": "ok", "timestamp": 1635744822249, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1a62427d-37cc-435d-f266-7ee6ad9637ed"
np.linalg.norm(Y - V @ Z.T) / np.linalg.norm(Y)
```

<!-- #region id="TpShyMNvExcz" -->
## CMF on ML-1m
<!-- #endregion -->

```python id="aYmPfufUAvtm"
!git clone https://github.com/VincentLiu3/CMF.git
%cd CMF
```

```python id="XDXyS0lV_k1s"
import numpy
from functools import reduce

def logistic(vec):
	out_vec = 1.0 / (1.0 + numpy.exp(-1 * vec))
	return out_vec

def d_logistic(vec):
	log_vec = logistic(vec)
	out_vec = numpy.multiply(log_vec, 1-log_vec)
	return out_vec

'''
def loss_for_one_row(Xi, U, V, reg):
	Yi = numpy.dot(U, V.T)
	loss = sum( pow( Xi-Yi, 2) ) + reg * numpy.linalg.norm(U) / 2
	return loss
def Armijo_line_search(U, one_step, Xi, V, reg):
	prev_loss = 
	while True:
		U -= one_step
		loss = loss_for_one_row(Xi, U, V, reg)
		if prev_loss
'''

def newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t):
	nsize = Ns[t] # size for entity t t, e.g. number of user for type 0 (user) 
	U_t = Us[t] # random factor matrix for entity t

	A = numpy.zeros((K, K)) # place holders for hessian: q'(Ui)
	b = numpy.zeros(K) # place holders for gradient: q(Ui)

	for i in range(nsize): # randomly pick one instance in X
		A[:] = 0
		b[:] = 0
		for j in range(len(Xs)):  # for j = 1~number of relations

			if alphas[j] == 0:
				continue
			
			if rc_schema[j, 0] == t or rc_schema[j, 1] == t:
				# only need to update if type t is in relation j
				if rc_schema[j, 0] == t:
					# if type t = x-axis of relation j 
					X = Xts[j] # transpose X (n2, n1)
					U = U_t[i, :] # (1 * k)
					V = Us[rc_schema[j, 1]]  # (n2 * k)
				
					data = X.data # content of the matrix
					indptr = X.indptr 
					indices = X.indices 

					ind_i0, ind_i1 = (indptr[i], indptr[i+1]) 
					# Step 1: XiV
					if ind_i0 == ind_i1:
						if modes[j] == "sparse": # sparse -> no data on the i-th row of X -> no need to update
							continue
						else: # dense/log_dense -> 0 vector
						 	XiV = numpy.zeros(K) # (1 * k)
					else:
						inds_i = indices[ind_i0:ind_i1] # index to the non-zero elements on the i-th row of X
						data_i = data[ind_i0:ind_i1] # non-zero element on the i-th row of X (1 * x)
						XiV = numpy.dot(data_i, V[inds_i, :]) # (1 * x) (x * k) -> (1 * k)
	
					if modes[j] == "sparse":
						V = V[inds_i, :] # only need those column factors for non-zero element in the i-th row
						# Step 2: UVt
						UiVt = numpy.dot(U, V.T) # (1*k) (k*n2) -> (1 * x)
						# Step 3: UVtV
						UiVtV = numpy.dot(UiVt, V)  # (1 * k)
						# Step 4: VtDiV
						Hes = numpy.dot(numpy.multiply(V.T, UiVt), V)

					elif modes[j] == 'log_dense':
						UiVt = numpy.dot(U, V.T) # (1 * n2)
						UiVtV = numpy.dot(logistic(UiVt), V) # (1 * k)
						Hes = numpy.dot(numpy.multiply(V.T, d_logistic(UiVt)), V)
					
					elif modes[j] == 'dense':
						UiVt = numpy.dot(U, V.T) # (1 * n2)
						UiVtV = numpy.dot(UiVt, V)  # (1 * k)
						Hes = numpy.dot(numpy.multiply(V.T, UiVt), V)

					A += alphas[j] * Hes
					b += alphas[j] * (UiVtV - XiV)

				elif rc_schema[j, 1] == t:
					# if type t = x-axis of relation j 
					X = Xs[j] # (n1 * n2)
					U = Us[rc_schema[j, 0]] # (n1 * k)
					V = U_t[i, :] # (1 * k)

					data = X.data # content of the matrix
					indptr = X.indptr
					indices = X.indices

					ind_i0, ind_i1 = (indptr[i], indptr[i+1])
					if ind_i0 == ind_i1: 
						if modes[j] == "sparse": # no data on the i-th column of X -> no need to update
							continue
						else:
							XiU = numpy.zeros(K) # (1 * k)
					else:
						inds_i = indices[ind_i0:ind_i1] 
						data_i = data[ind_i0:ind_i1] # non-zero elements on the j-th column of X (1 * x)
						XiU = numpy.dot(data_i, U[inds_i, :]) # (1 * k)

					if modes[j] == "sparse":
						U = U[inds_i, :] # (x * k)
						UVt = numpy.dot(U, V.T) # (x * k) (k * 1) -> (x * 1)
						UVtU = numpy.dot(UVt.T, U) # (1 * k)
						Hes = numpy.dot(numpy.multiply(U.T, UVt), U)

					elif modes[j] == 'log_dense':
						UVt = numpy.dot(U, V.T) # (x * k) (k * 1) -> (x * 1)
						UVtU = numpy.dot(logistic(UVt).T, U) # (1 * k)
						Hes = numpy.dot(numpy.multiply(U.T, d_logistic(UVt)), U)

					elif modes[j] == 'dense':
						UVt = numpy.dot(U, V.T) # (x * k) (k * 1) -> (x * 1)
						UVtU = numpy.dot(UVt.T, U) # (1 * k)
						Hes = numpy.dot(numpy.multiply(U.T, UVt), U)

					A += alphas[j] * Hes
					b += alphas[j] * (UVtU - XiU)
			
		if numpy.all(b == 0):
			continue
			
		# regularizer
		A += reg * numpy.eye(K, K)
		b += reg * U_t[i, :].copy() # the previous factor for i-th data

		d = numpy.dot(b, numpy.linalg.inv(A))
		Us[t][i, :] -= learn_rate * d  

	# return change

def old_newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t):
	'''
	code from http://ihome.ust.hk/~zluab/code/
	'''
	assert(t <= len(Ns) and t >= 0)
	eyeK = reg * numpy.eye(K, K)
	N = Ns[t] # number of instances for type t
	V = Us[t] # U
	A = numpy.zeros((K, K)) # place holders for hessian
	b = numpy.zeros(K) # place holders for gradient
	UtUs = numpy.empty(len(Xs),object)
	# change = 0
	for j in range(len(Xs)):
		if modes[j] == 'dense':
			if rc_schema[j, 0] == t:		
				U = Us[rc_schema[j, 1]]
			else:
				U = Us[rc_schema[j, 0]] 
			UtUs[j] = numpy.dot(U.T,U) # UtUs = VtV
	for i in range(N):
		A[:] = 0
		b[:] = 0
		for j in range(len(Xs)):
			if alphas[j] == 0:
				continue
			if rc_schema[j, 0] == t or rc_schema[j, 1] == t:
				if rc_schema[j, 0] == t:
					X = Xts[j]
					U = Us[rc_schema[j, 1]] # V
				else:
					X = Xs[j]
					U = Us[rc_schema[j, 0]]
				data = X.data
				indptr = X.indptr
				indices = X.indices
				
				ind_i0, ind_i1 = (indptr[i], indptr[i+1])
				if ind_i0 == ind_i1:
					continue
				
				inds_i = indices[ind_i0:ind_i1] 
				data_i = data[ind_i0:ind_i1]
				
				if modes[j] == "dense": # square loss, dense binary representation
					UtU = UtUs[j]
					Utemp = U[inds_i, :]
					A += alphas[j] * UtU
					b += alphas[j] * (numpy.dot(UtU,V[i,:])-numpy.dot(data_i, Utemp))
				elif modes[j] == "log_dense": # logistic loss
					Xi = numpy.dot(U, V[i, :])
					Yi = - 1 * numpy.ones(U.shape[0])
					Yi[inds_i] = 1
					# (sigma(yx)-1)
					Wi = 1.0 / (1 + numpy.exp(-1 * numpy.multiply(Yi, Xi))) - 1 
					Wi = numpy.multiply(Wi, Yi)
					gv = numpy.dot(Wi, U)
					# compute sigmoid(x)
					Ai = 1 / (1 + numpy.exp(-Xi))
					Ai = numpy.multiply(Ai, 1 - Ai)
					Ai = Ai.reshape(Ai.size, 1)
					AiU = numpy.multiply(Ai, U)
					Hv = numpy.dot(AiU.T, U)
					A += alphas[j] * Hv
					b += alphas[j] * gv
					
				elif modes[j] == "sparse": # square loss
					Utemp = U[inds_i, :]
					UtU = numpy.dot(Utemp.T, Utemp)
					A += alphas[j] * UtU
					b += alphas[j] * (numpy.dot(UtU, V[i,:])-numpy.dot(data_i, Utemp))
					
		A += eyeK
		b += reg*V[i, :]
		d = numpy.dot(numpy.linalg.inv(A), b)
		vi = V[i,:].copy()
		V[i, :] -= learn_rate*d
	# return change

# http://sebastianruder.com/optimizing-gradient-descent/
```

```python id="rmtAVlsW_oNU"
import numpy
import scipy.sparse
import os.path

def read_dense_data(train_file, test_file, user_file, item_file, feature_mat_type):
    return 

def loadTripleData(filename):
    '''
    laod triple data (row, column, value) to csc_matrix format
    '''
    fData = numpy.loadtxt(filename, delimiter=',').T
    fData = fData.astype(int)
    fData = scipy.sparse.coo_matrix((fData[2],(fData[0],fData[1]))).tocsc()
    return(fData)

def read_triple_data(train, test, user, item, feature_mat_type):
    '''
    read data from three column format (row, column, value)
    '''
    assert( feature_mat_type in ['sparse', 'dense', 'log_dense'] ), 'Unrecognized link function'

    # need to make sure training & testing data with the same shapes as user and item features
    num_user = num_item = 0	
    if user != '':
        X_userFeat = loadTripleData(user)
        num_user = X_userFeat.shape[0]
    if item != '':
        X_itemFeat = loadTripleData(item)
        num_item = X_itemFeat.shape[0]

    Dtrain = numpy.loadtxt(train, delimiter = ',').T
    Dtest = numpy.loadtxt(test, delimiter = ',').T
    num_user = int( max(Dtrain[0].max(), Dtest[0].max(), num_user-1) ) + 1
    num_item = int( max(Dtrain[1].max(), Dtest[1].max(), num_item-1) ) + 1
    X_train = scipy.sparse.coo_matrix((Dtrain[2],(Dtrain[0],Dtrain[1])), shape=(num_user, num_item)).tocsc()
    X_test = scipy.sparse.coo_matrix((Dtest[2],(Dtest[0],Dtest[1])), shape=(num_user, num_item)).tocsc()
    # transform to csc format
    # X_train = scipy.sparse.csc_matrix(X_train)
    # X_test = scipy.sparse.csc_matrix(X_test)

    # user or item features
    if user != '' and item != '':
        Xs_trn = [X_train, X_userFeat, X_itemFeat]
        Xs_tst = [X_test, None, None]
        
        rc_schema = numpy.array([[0, 1], [0, 2], [1, 3]])
        # [row entity number, column entity number]
        # 0=user, 1=item, 2=userFeat, 3=itemFeat

        modes = ['sparse', feature_mat_type, feature_mat_type]
        # modes of each relation: sparse, dense or log_dense
        # dense if Wij = 1 for all ij 
        # sparse if Wij = 1 if Xij>0
        # log if link function = logistic

    elif user == '' and item != '':
        Xs_trn = [X_train, X_itemFeat]
        Xs_tst = [X_test, None]

        rc_schema = numpy.array([[0, 1], [1, 2]]) # 0=user, 1=item, 2=itemFeat
        modes = ['sparse', feature_mat_type]

    elif user != '' and item == '':
        Xs_trn = [X_train, X_userFeat]
        Xs_tst = [X_test, None]

        rc_schema = numpy.array([[0, 1], [0, 2]]) # 0=user, 1=item, 2=userFeat
        modes = ['sparse', feature_mat_type]

    elif user == '' and item == '':
        assert False, 'No user and item features.'
        Xs_trn = [X_train]
        Xs_tst = [X_test]

        rc_schema = numpy.array([[0, 1]])
        modes = ['sparse']

    return [Xs_trn, Xs_tst, rc_schema, modes] 

def get_config(Xs, rc_schema):
    '''
    get neccessary configurations of the given relation
    ---------------------
    S = number of entity
    Ns = number of instances for each entity
    '''
    assert(len(Xs)==len(rc_schema)), "rc_schema lenth must be the same as input data."

    S = rc_schema.max() + 1
    Ns = -1 * numpy.ones(S, int)
    for i in range(len(Xs)):
        ri = rc_schema[i, 0]
        ci = rc_schema[i, 1]
        
        [m, n] = Xs[i].shape
        
        if Ns[ri] < 0:
            Ns[ri] = m
        else:
            assert(Ns[ri] == m), "rc_schema does not match data."
                            
        if Ns[ci] < 0:
            Ns[ci] = n
        else:
            assert(Ns[ci] == n), "rc_schema does not match data."
    return [S, Ns]

def RMSE(X, Y):
    '''
    X is prediction, Y is ground truth
    Both X and Y should be scipy.sparse.csc_matrix
    '''
    assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr) and X.size > 0)
    return numpy.sqrt(sum(pow(X.data - Y.data, 2)) / X.size)

def MAE(X, Y):
    assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr) and X.size > 0)
    return sum(abs(X.data - Y.data)) / X.size

def check_modes(modes):
    for mode in modes:
        if mode != 'sparse' and mode != 'dense' and mode != 'log_dense':
            assert False, 'Unrecognized mode: {}'.format(mode)

def string2list(input_string, num, sep='-'):
    string_list = input_string.split(sep)
    assert( len(string_list) == num ), 'argument alphas must be the same length as numbers of relations.'
    return [float(x) for x in string_list]

def save_result(args, rmse):
    if args.user != '' and args.item != '':
        cmf_type = 'useritem'
    elif args.user == '' and args.item != '':
        cmf_type = 'item'
    elif args.user != '' and args.item == '':
        cmf_type = 'user'
    elif args.user == '' and args.item == '':
        cmf_type = 'none'

    if args.out != '':
        if os.path.exists(args.out) is False:
            with open(args.out, 'w') as fp:
                fp.write('type,k,reg,lr,tol,alphas,RMSE\n')
        with open(args.out, 'a') as fp:
            fp.write('{},{},{},{},{},{},{:.4f}\n'.format(cmf_type, args.k, args.reg, args.lr, args.tol, args.alphas, rmse))
```

```python colab={"base_uri": "https://localhost:8080/"} id="ETbsC3iu_aa5" executionInfo={"status": "ok", "timestamp": 1635745607975, "user_tz": -330, "elapsed": 171727, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f36c4fd4-48dc-42ef-f7d7-e095b8784051"
import numpy
import time
import logging
import scipy.sparse
import argparse

class Args:
    train = 'data/ml-1m/train.txt' # Training file
    test = 'data/ml-1m/test.txt' # Testing file
    user = 'data/ml-1m/user.txt' # User features file
    item = 'data/ml-1m/item.txt' # Item features file
    out = 'ml-1m.txt' # File where fianl result will be saved
    alphas = '0.5-0.5-0.5' # Alpha in [0, 1] weights the relative importance of relations
    link = 'log_dense' # link function for feature relations (dense or log_dense)
    k = 8 # Dimension of latent fectors
    reg = 0.1 # Regularization for latent facotrs
    lr = 0.1 # Initial learning rate for training
    iter = 50 # Max training iteration
    tol = 0 # Tolerant for change in training loss
    verbose = 1 # Verbose or not (1 for INFO, 0 for WARNING)

args = Args()


def learn(Xs, Xstst, rc_schema, modes, alphas, K, reg, learn_rate, max_iter, tol):
    assert(rc_schema.shape[0] == len(Xs) and rc_schema.shape[1] == 2) # schema match data
    assert(numpy.all(rc_schema[:, 0] != rc_schema[:, 1])) # should not have symmetric relations
    assert(rc_schema.shape[0] == len(alphas))
    assert(rc_schema.shape[0] == len(modes))
    check_modes(modes) 

    Xts = [None] * len(Xs)
    for i in range(len(Xs)):
        if Xs[i] is not None:
            Xts[i] = scipy.sparse.csc_matrix(Xs[i].T) # Transpose
            Xs[i] = scipy.sparse.csc_matrix(Xs[i]) # no Transpose
        if Xstst[i] is not None:
            Xstst[i] = scipy.sparse.csc_matrix(Xstst[i])

    [S, Ns] = get_config(Xs, rc_schema)

    # randomly initialize factor matrices with small values
    Us = [None] * S
    for i in range(S):
        Us[i] = numpy.random.rand(Ns[i], K) * numpy.sqrt(1/K)  # so initial prediction will be in [0, 5]

    Ys = predict(Us, Xs, rc_schema, modes)
    prev_loss = loss(Us, Xs, Ys, rc_schema, modes, alphas, reg, S)
    i = 0
    while i < max_iter:
        i += 1
        tic = time.time()

        # training        
        for t in range(S): # update factors for entity t
            newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t)
        
        # evaluation
        Ys = predict(Us, Xs, rc_schema, modes)
        training_loss = loss(Us, Xs, Ys, rc_schema, modes, alphas, reg, S)
        train_rmse = RMSE(Xs[0], Ys[0])
        change_rate = (training_loss-prev_loss)/prev_loss * 100
        prev_loss = training_loss
        
        Ystst = predict(Us, Xstst, rc_schema, modes)
        test_rmse = RMSE(Xstst[0], Ystst[0])

        toc = time.time()
        logger.info('Iter {}/{}. Time: {:.1f}'.format(i, max_iter, toc - tic))
        logger.info('Training Loss: {:.1f} (change {:.2f}%). Training RMSE: {:.2f}. Testing RMSE: {:.2f}'.format(training_loss, change_rate, train_rmse, test_rmse))
    
        # early stop
        if tol!=0 and i!=1 and change_rate > -tol :
            break

    return Us

def loss(Us, Xs, Ys, rc_schema, modes, alphas, reg, num_entities):
	'''
	Calculate objective loss
	See page 4: Generalizing to Arbitrary Schemas
	'''
	assert(rc_schema.shape[0] == len(Xs) and rc_schema.shape[1] == 2)

	res = 0
	num_relation = len(Xs)
	# computing regularization for each latent factor
	for i in range(num_entities):
		for j in range(num_relation):
			if rc_schema[j, 0]==i or rc_schema[j, 1]==i:
				res += alphas[j] * reg * numpy.linalg.norm(Us[i].flat) / 2 # l2 norm

	# computing loss for each relation
	for j in range(num_relation):     
		alpha_j = alphas[j]
		if Xs[j] is None or Ys[j] is None or alpha_j == 0:
			continue

		# X = scipy.sparse.csc_matrix(Xs[j])
		# Y = scipy.sparse.csc_matrix(Ys[j])
		X = Xs[j]
		Y = Ys[j]
		
		if modes[j] == 'sparse':
			assert( X.size == Y.size )
			res += alpha_j * numpy.sum(pow(X.data - Y.data, 2))

		elif modes[j] == 'dense' or modes[j] == 'log_dense':
			assert( numpy.all(Y.shape == X.shape) )
			res += alpha_j * numpy.sum(pow(X.toarray() - Y.toarray(), 2))   

	return res

def predict(Us, Xs, rc_schema, modes):
    '''
    see page 3: RELATIONAL SCHEMAS
    return a list of csc_matrix
    '''
    Ys = []
    for i in range(len(Xs)): # i = 1
        if Xs[i] is None:
        	# no need to predict Y
            Ys.append(None) 
            continue
        
        X = Xs[i]
        U = Us[rc_schema[i, 0]]
        V = Us[rc_schema[i, 1]]

        if modes[i] == 'sparse':
            # predict only for non-zero elements in X
            # X = scipy.sparse.csc_matrix(X)
            data = X.data.copy()
            indices = X.indices.copy()
            indptr = X.indptr.copy()
           
            for j in range(X.shape[1]): # for each column in X
                inds_j = indices[indptr[j]:indptr[j+1]]
                # indptr[j]:indptr[j+1] points to the data on j-th column of X
                if inds_j.size == 0:
                    continue
                data[indptr[j]:indptr[j+1]] = numpy.dot(U[inds_j, :], V[j, :])

            Y = scipy.sparse.csc_matrix((data, indices, indptr), X.shape)
            Ys.append(Y)

        elif modes[i] == 'dense':
            # predict for all elements in X
            Y = numpy.dot(U, V.T)
            Y = scipy.sparse.csc_matrix(Y)
            Ys.append(Y)

        elif modes[i] == 'log_dense':
            # predict for all elements in X
            Y = numpy.dot(U, V.T)
            Y = logistic(Y)
            Y = scipy.sparse.csc_matrix(Y)
            Ys.append(Y)

    return Ys

def run_cmf(Xs_trn, Xs_tst, rc_schema, modes, alphas, args):
    '''
    run cmf
    '''
    start_time = time.time()

    Us = learn(Xs_trn, Xs_tst, rc_schema, modes, alphas, args.k, args.reg, args.lr, args.iter, args.tol)
    Ys_tst = predict(Us, Xs_tst, rc_schema, modes)
    rmse = RMSE(Xs_tst[0], Ys_tst[0])

    end_time = time.time()
    logger.info('RMSE: {:.4f}'.format(rmse))
    logger.info('Total Time: {:.0f} s'.format(end_time - start_time) )
    
    save_result(args, rmse)

    return 


if __name__ == "__main__":
    
	[Xs_trn, Xs_tst, rc_schema, modes] = read_triple_data(args.train, args.test, args.user, args.item, args.link)

	if(args.verbose == 1):
		logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
	else:
		logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
	
	logger = logging.getLogger()
	[S, Ns] = get_config(Xs_trn, rc_schema)
	alphas = string2list(args.alphas, len(modes))

	logger.info('------------------- CMF -------------------')
	logger.info('Data: Number of instnace for each entity = {}'.format(list(Ns)))
	logger.info('Data: Training size = {}. Testing size = {}'.format(Xs_trn[0].size, Xs_tst[0].size))
	logger.info('Settings: k = {}. reg = {}. lr = {}. alpha = {}. modes = {}.'.format(args.k, args.reg, args.lr, alphas, modes))

	run_cmf(Xs_trn, Xs_tst, rc_schema, modes, alphas, args)
```

<!-- #region id="_U2FYVR0Ar6e" -->
## Citations

Relational Learning via Collective Matrix Factorization. Singh et. al.. 2008. KDD. [http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf](http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf)
<!-- #endregion -->
