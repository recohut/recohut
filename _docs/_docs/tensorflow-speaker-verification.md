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

# Speaker verification with GE2E loss (Text independent version)

Mainly based on https://github.com/Janghyun1230/Speaker_Verification

```python _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import os
import tensorflow as tf
import librosa
import time
import random
from tensorflow.contrib import rnn

audio_path = "../input/vctk-corpus/VCTK-Corpus/wav48"

class hyperparam:
    train_path = "train"
    test_path = "test"
    model_path = "model"
    
    # spectrogram params
    tisv_frame = 180 # max frame number of utterances of tdsv
    hop = 0.01 # hop size (s)
    window = 0.025 # window length (s)
    sr = 8000 # sampling rate
    
    # model params
    num_layer = 3 # number of lstm layers
    hidden = 128 # hidden state dimension of lstm
    proj = 64 # projection dimension of lstm
    
    # training params
    N = 4 # number of speakers of batch 
    M = 5 # number of utterances per speaker
    train = True
    optim = 'sgd' # optimizer type
    loss = 'softmax' # loss type (softmax or contrast)

    iteration = 10000
    lr = 1e-2
    
    # test params
    tdsv = False # text dependent or not
    model_num = 1 # number of ckpt file to load
    
config = hyperparam()

tf.test.gpu_device_name()
```

## 1. Data preparation

We use pack [librosa](http://librosa.github.io/librosa/) to preprocess the audio.

The functions we use:
* [librosa.effects.trim](https://librosa.github.io/librosa/generated/librosa.effects.trim.html?highlight=librosa%20effects%20trim#librosa.effects.trim)

    Trim leading and trailing silence from an audio signal.

* [librosa.core.load](https://librosa.github.io/librosa/generated/librosa.core.load.html?highlight=librosa%20core%20load#librosa.core.load)
    
    Load an audio file as a floating point time series.
    
    Audio will be automatically resampled to the given rate (default sr=22050).

    To preserve the native sampling rate of the file, use sr=None.

* [librosa.feature.mfcc](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html#librosa-feature-mfcc)

    Mel-frequency cepstral coefficients. Return a MFCC sequence with shape [n_mfcc, t].
    
For simplicity, we just use the first 30 files with its first 100 items in the datasets. You can adjust parameter `cutted` and `cut_utter` to change this setting.

```python _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
def save_spectrogram_tisv(audio_path, cutting = True, cutted = 30, cut_utter = 100):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    total_speaker_num = len(os.listdir(audio_path))
    
    if cutting:
        train_speaker_num= (cutted//10)*8 # for cutting set
    else:
        train_speaker_num= (total_speaker_num//10)*9  # split total data 90% train and 10% test
    
    
    print("total speaker number : %d"%total_speaker_num)
    if cutting:
        print("train : %d, test : %d"%(train_speaker_num, cutted+1-train_speaker_num))
    else:
        print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    print(os.listdir(audio_path))
    
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%d th speaker processing..." % i)
        utterances_spec = []
        k=0
        for j, utter_name in enumerate(os.listdir(speaker_path)):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
            # trim the silence in the audio. the interval of utter_trim corresponding to the non-silent region
            utter_trim, index = librosa.effects.trim(utter, top_db=20)
            
            cur_slide = 0
            mfcc_win_sample = int(config.sr*config.hop*config.tisv_frame) # the length of a mfcc window
            while(True):
                if(cur_slide + mfcc_win_sample > utter_trim.shape[0]):
                    break
                slide_win = utter_trim[cur_slide : cur_slide+mfcc_win_sample]

                S = librosa.feature.mfcc(y=slide_win, sr=config.sr, n_mfcc=40)
                utterances_spec.append(S)
                cur_slide += int(mfcc_win_sample/2)
            if cutting:
                if j > cut_utter:
                    break
                
        utterances_spec = np.array(utterances_spec)
        print('utterances_spec.shape = {}'.format(utterances_spec.shape))

        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(config.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(config.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)
        
        if cutting:
            if i > cutted-1:
                break
        
        
```

Organize batches.

```python
def random_batch(shuffle=True, noise_filenum=None, utter_start=0):
    """ Generate 1 batch.
        For TD-SV, noise is added to each utterance.
        For TI-SV, random frame length is applied to each batch of utterances (140-180 frames)
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
        shuffle : random sampling or not
        noise_filenum : specify noise file or not (TD-SV)
        utter_start : start point of slicing (TI-SV)
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """
    speaker_num = config.N
    utter_num = config.M 
    # data path
    if config.train:
        path = config.train_path
    else:
        path = config.test_path
    
    # TI-SV 本实验使用： 与文本无关说话人鉴别
    np_file_list = os.listdir(path)
    total_speaker = len(np_file_list)

    if shuffle:
        selected_files = random.sample(np_file_list, speaker_num)  # select random N speakers
    else:
        selected_files = np_file_list[:speaker_num]                # select first N speakers
            
    utter_batch = []
    for file in selected_files:
        utters  =  np.load(os.path.join(path, file))        # load utterance spectrogram of selected speaker
        if shuffle:
            utter_index = np.random.randint(0, utters.shape[0], utter_num)   # select M utterances per speaker
            utter_batch.append(utters[utter_index])       # each speaker's utterance [M, n_mels, frames] is appended
        else:
            utter_batch.append(utters[utter_start: utter_start+utter_num])

    utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), n_mels, frames]

    if config.train:
        frame_slice = np.random.randint(140,181)          # for train session, random slicing of input batch
        utter_batch = utter_batch[:,:,:frame_slice]
    else:
        utter_batch = utter_batch[:,:,:160]               # for train session, fixed length slicing of input batch

    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]

    return utter_batch
```

## 2. Define model and training session

```python
def train(path):
    tf.reset_default_graph()    # reset graph
    # draw graph
    batch = tf.placeholder(shape= [None, config.N*config.M, 40], dtype=tf.float32)  # input batch (time x batch x n_mel)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    print("embedded size: ", embedded.shape)
    # loss
    sim_matrix = similarity(embedded, w, b)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, type=config.loss)

    # optimizer operation
    trainable_vars= tf.trainable_variables()                # get variable list
    optimizer= optim(lr)                                    # get optimizer (type is determined by configuration)
    grads, vars= zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss
    grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)      # l2 norm clipping by 3
    grads_rescale= [0.01*grad for grad in grads_clip[:2]] + grads_clip[2:]   # smaller gradient scale for w, b
    train_op= optimizer.apply_gradients(zip(grads_rescale, vars), global_step= global_step)   # gradient update operation

    # check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # record loss
    loss_summary = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    # training session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

#        if(os.path.exists(path)):
#            print("Restore from {}".format(os.path.join(path, "Check_Point/model.ckpt-2")))
#            saver.restore(sess, os.path.join(path, "Check_Point/model.ckpt-2"))  # restore variables from selected ckpt file
#        else:
#            os.makedirs(os.path.join(path, "Check_Point"), exist_ok=True)  # make folder to save model
#            os.makedirs(os.path.join(path, "logs"), exist_ok=True)          # make folder to save log

        os.makedirs(os.path.join(path, "Check_Point"), exist_ok=True)  # make folder to save model
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)          # make folder to save log
        
        writer = tf.summary.FileWriter(os.path.join(path, "logs"), sess.graph)
        epoch = 0
        lr_factor = 1   # lr decay factor ( 1/2 per 10000 iteration)
        loss_acc = 0    # accumulated loss ( for running average of loss)

        for iter in range(config.iteration):
            # run forward and backward propagation and update parameters
            _, loss_cur, summary = sess.run([train_op, loss, merged],
                                  feed_dict={batch: random_batch(), lr: config.lr*lr_factor})
            loss_acc += loss_cur    # accumulated loss for each 100 iteration
            if iter % 10 == 0:
                writer.add_summary(summary, iter)   # write at tensorboard
            if (iter+1) % 100 == 0:
                print("(iter : %d) loss: %.4f" % ((iter+1),loss_acc/100))
                loss_acc = 0                        # reset accumulated loss
            if (iter+1) % 2000 == 0:
                lr_factor /= 2                      # lr decay
                print("learning rate is decayed! current lr : ", config.lr*lr_factor)
            if (iter+1) % 5000 == 0:
                saver.save(sess, os.path.join(path, "./Check_Point/model.ckpt"), global_step=iter//5000)
                print("model is saved!")
```

## 3. Define the loss
and other utilities.

```python
def similarity(embedded, w, b, N=4, M=5, P=64, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    embedded_split = tf.reshape(embedded, shape=[N, M, P])

    if center is None:
        center = normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keep_dims=True)
                                             - embedded_split, shape=[N*M,P]))  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keep_dims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exist, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    S = tf.abs(w)*S+b   # rescaling

    return S


def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)


def cossim(x,y, normalized=True):
    """ calculate similarity between tensors
    :return: cos similarity tf op node
    """
    if normalized:
        return tf.reduce_sum(x*y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x**2)+1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y**2)+1e-6)
        return tf.reduce_sum(x*y)/x_norm/y_norm
    
def loss_cal(S, type="softmax", N=config.N, M=config.M):
    """ calculate loss with similarity matrix(S) eq.(6) (7) 
    :type: "softmax" or "contrast"
    :return: loss
    """
    S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if type == "softmax":
        total = -tf.reduce_sum(S_correct-tf.log(tf.reduce_sum(tf.exp(S), axis=1, keep_dims=True) + 1e-6))
    elif type == "contrast":
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0*S_sig[i*M:(i+1)*M, j:(j+1)] if i==j
                              else S_sig[i*M:(i+1)*M, j:(j+1)] for j in range(N)], axis=1)
                             for i in range(N)], axis=0)
        total = tf.reduce_sum(1-tf.sigmoid(S_correct)+tf.reduce_max(S_sig, axis=1, keep_dims=True))
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total

def optim(lr):
    """ return optimizer determined by configuration
    :return: tf optimizer
    """
    if config.optim == "sgd":
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim == "rmsprop":
        return tf.train.RMSPropOptimizer(lr)
    elif config.optim == "adam":
        return tf.train.AdamOptimizer(lr, beta1=config.beta1, beta2=config.beta2)
    else:
        raise AssertionError("Wrong optimizer type!")
```

## 4. Test session

```python
# Test Session
def test(path):
    tf.reset_default_graph()

    # draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    verif = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.concat([enroll, verif], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
    # verification embedded vectors
    verif_embed = embedded[config.N*config.M:, :]

    similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        # load model
        print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        print("test file path : ", config.test_path)
        '''
            test speaker:p225--p243
        '''
        # return similarity matrix after enrollment and verification
        time1 = time.time() # for check inference time
        if config.tdsv:
            S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False, noise_filenum=1),
                                                       verif:random_batch(shuffle=False, noise_filenum=2)})
        else:
            S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False),
                                                       verif:random_batch(shuffle=False, utter_start=config.M)})
        S = S.reshape([config.N, config.M, -1])
        time2 = time.time()

        np.set_printoptions(precision=2)
        print("inference time for %d utterences : %0.2fs"%(2*config.M*config.N, time2-time1))
        print(S)    # print similarity matrix

        # calculating EER
        diff = 1; EER=0; EER_thres = 0; EER_FAR=0; EER_FRR=0

        # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
        for thres in [0.01*i+0.5 for i in range(50)]:
            S_thres = S>thres

            # False acceptance ratio = false acceptance / mismatched population (enroll speaker != verification speaker)
            FAR = sum([np.sum(S_thres[i])-np.sum(S_thres[i,:,i]) for i in range(config.N)])/(config.N-1)/config.M/config.N

            # False reject ratio = false reject / matched population (enroll speaker = verification speaker)
            FRR = sum([config.M-np.sum(S_thres[i][:,i]) for i in range(config.N)])/config.M/config.N

            # Save threshold when FAR = FRR (=EER)
            if diff> abs(FAR-FRR):
                diff = abs(FAR-FRR)
                EER = (FAR+FRR)/2
                EER_thres = thres
                EER_FAR = FAR
                EER_FRR = FRR

        print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thres,EER_FAR,EER_FRR))
```

## 5. Train model

```python
tf.reset_default_graph()

if __name__ == "__main__":
    save_spectrogram_tisv(audio_path, cutting = True)
    config.train = True
    # start training
    print("\nTraining Session")
    with tf.device("/gpu:0"):
        train(config.model_path)
            
    # start test
    config.train = False
    print("\nTest session")
    if os.path.isdir(config.model_path):
        with tf.device("/gpu:0"):
            test(config.model_path)
    else:
        raise AssertionError("model path doesn't exist!")
```
