#!/usr/bin/env python
from __future__ import print_function
# # 1. Import Libraries 
from time import time
t_start= time()
import sys
### command line parsers
import argparse
parser = argparse.ArgumentParser(description="Deap Learning cosmology classification")
parser.add_argument('--batch_size',type=int, help='batch size', default=64)
parser.add_argument('--sz', type=int, help='size of the figure in each dimension', default=224)
parser.add_argument('--PATH', help='root directory', default='/lus/theta-fs0/projects/mmaADSP/hzheng/new_DL_DES/')
parser.add_argument('--verbose', type=int, help='output level', default=2)
parser.add_argument('--num_intra', type=int, help='Number of intra threads', default=0)
parser.add_argument('--num_inter', type=int, help='Number of inter threads', default=2)
parser.add_argument('--horovod', action='store_true', help='Whether use horovod')
parser.add_argument('--model', default='Xception', help='Choose model between Inception3, Xception, ResNet, VGG16, VGG19')
parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup ephochs')
parser.add_argument('--intermediate_score', action='store_true', help='Whether output score for intermediate stages or not.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Base learning rate')
parser.add_argument('--early_stop', action='store_true', help='Whether to do early stop or not.')
parser.add_argument('--epochs_1', type=int, default=1, help='Number of epoch for the first stage')
parser.add_argument('--epochs_2', type=int, default=20, help='Number of epoch for the second stage')
parser.add_argument('--epochs_3', type=int, default=20, help='Number of epoch for the third stage')
parser.add_argument('--save_model', default='Model_Final.h5', help='Whether to split data or not')
parser.add_argument('--data_format', default='channels_last', help='dataformat: channels_first or channels_last')
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()
PATH = args.PATH
sz=args.sz
batch_size=args.batch_size
shuffle=args.shuffle
lr = args.learning_rate
# In[67]:

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50, Xception, InceptionV3, VGG16, VGG19
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import os
from os import listdir
from os.path import isfile, join, exists
import tensorflow as tf
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
#import seaborn as sn
import pandas as pd 
import numpy as np
K.set_image_data_format(args.data_format)
if args.data_format=='channels_last':
    input_shape=(sz, sz, 3)
else:
    input_shape=(3, sz, sz)



#### Check whether using Horovod or not. If yes, initialize hvd.
if (args.horovod):
    import horovod.keras as hvd
    hvd.init()
    print("Parallelization method: Horovod")
    print("HOROVOD: I am rank %s of %s" %(hvd.rank(), hvd.size()))
else:
    class hvd:
        def rank():
            return 0
        def size():
            return 1
        


#### Print command lines
if hvd.rank()==0:
    print("*** User input parameters *** ")
    for arg in vars(args):
        print("* %s: %s"%(arg, getattr(args, arg)))
    print("* Tensorflow version: %s"%tf.__version__)
    print("* Keras data format: %s"%K.image_data_format())
    print("*************************************************")
#### Setup session

config = tf.ConfigProto(intra_op_parallelism_threads=args.num_intra,
                        inter_op_parallelism_threads=args.num_inter)
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
K.set_session(tf.Session(config=config))

#### define output verbose level
if (hvd.rank()==0):
    verbose = args.verbose
else:
    verbose = 0


# In[71]:
step_rescale=hvd.size()
train_data_dir = PATH+'deeplearning/data/train/'
validation_data_dir = PATH+'deeplearning/data/valid/'

from hdf5_preprocessing import *

train_fh = h5py.File(PATH+'/deeplearning/data/train_%s.hdf5'%args.data_format, 'r')
ntrain = train_fh['data'].shape[0]
ntrain_loc = ntrain//hvd.size()
train_offset = ntrain_loc*hvd.rank()
valid_fh = h5py.File(PATH+'/deeplearning/data/valid_%s.hdf5'%args.data_format, 'r')
nvalid = valid_fh['data'].shape[0]
nvalid_loc = nvalid//hvd.size()
valid_offset = nvalid_loc*hvd.rank()
t0=time()
X_train = train_fh['data'][train_offset:train_offset+ntrain_loc]
Y_train = train_fh['labels'][train_offset:train_offset+ntrain_loc]

X_valid = valid_fh['data'][valid_offset:valid_offset+nvalid_loc]
Y_valid = valid_fh['labels'][valid_offset:valid_offset+nvalid_loc]
t1=time()
if hvd.rank()==0:
    print(" Loading data: %s seconds" %(t1 - t0))
num_of_classes = 2
if args.model == 'InceptionV3':
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
elif args.model =='Xception':
    base_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False)
elif args.model == 'ResNet50':
    base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
elif args.model=='VGG16':
    base_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
elif args.model=='VGG19':
    base_model = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
def AddLayers(base_model, model='InceptionV3'):
    x = base_model.output
    if (model=="VGG16" or model=='VGG19'):
        x = Flatten()(x)
    else:
        x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(1024, activation="relu", name='second_last_layer')(x)
    predictions = Dense(num_of_classes, activation="softmax")(x)
    model_final = Model(inputs=base_model.input, outputs=predictions)
    return model_final
if args.model=='AlexNet':
    from alexnet import AlexNet
    model_final=AlexNet(input_shape, num_of_classes)
else:    
    model_final = AddLayers(base_model, model=args.model)
    for layer in base_model.layers:
        layer.trainable = False

if (hvd.rank()==0):
    model_final.summary()

opt = optimizers.Adam(lr=lr*hvd.size())# Scale learning rate
if args.horovod:
    opt = hvd.DistributedOptimizer(opt)
model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])


# # 4. Train

# ## 1. Training Data 
t0 = time()
if (args.horovod):
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), 
                 hvd.callbacks.MetricAverageCallback(),
                 hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose)]
else:
    callbacks = []
history_1 = model_final.fit(X_train, Y_train, validation_data = (X_valid, Y_valid), 
                            batch_size = batch_size, 
                            shuffle = shuffle,
                            epochs = args.epochs_1,
                            verbose=verbose, callbacks=callbacks) 
t1 = time()

if args.intermediate_score:
    train_score = hvd.allreduce(model_final.evaluate_generator(train_generator, train_generator.n//train_generator.batch_size//step_rescale, workers=num_workers, verbose=verbose))
    val_score = hvd.allreduce(model_final.evaluate_generator(validation_generator, validation_generator.n//validation_generator.batch_size//step_rescale, workers=num_workers, verbose=verbose))
    t2 = time()
    if (hvd.rank()==0):
        print('**AllReduce  loss: %s  -  acc: %s  -  val_loss: %s  -  val_acc: %s\n Total time for Stage 2: %s seconds' 
              %(train_score[0], train_score[1], val_score[0], val_score[1], t1-t0))
        print('**Evaluation time: %s' %(t2-t1))
else:
    if (hvd.rank()==0):
        print('**Total time for Stage 1: %s seconds'%(t1-t0))
if (hvd.rank()==0):
    model_final.save(PATH + 'deeplearning/weights/%sNew_freeze.h5'%args.model)

#len(model_final.layers)

# In[18]:
split_at = 40
if args.model=='VGG16' or args.model=='VGG19':
    split_at=20
for layer in model_final.layers[:split_at]: layer.trainable = False
for layer in model_final.layers[split_at:]: layer.trainable = True  

model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

#Select Call Backs

Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8, verbose=verbose, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(PATH + 'deeplearning/weights/%s_new_UnFreeze.h5'%args.model, monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=False, mode='min', period=1)
if args.horovod:
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), 
                 hvd.callbacks.MetricAverageCallback(),
                 hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),]
else:
    callbacks = []
if args.early_stop:
    callbacks = callbacks + [Early_Stop, reduce_lr]
else:
    callbacks = callbacks + [Early_Stop, reduce_lr]


if hvd.rank()==0:
    callbacks.append(checkpoint)


t0 = time()
history2 = model_final.fit(X_train, Y_train, validation_data = (X_valid, Y_valid), 
                            batch_size = batch_size, 
                            shuffle = shuffle,
                            epochs = args.epochs_2,
                            verbose=verbose, callbacks=callbacks) 
t1 = time()

# #### (iii) Unfreeze [at Layer 2] 

if args.intermediate_score:
    train_score = hvd.allreduce(model_final.evaluate_generator(train_generator, train_generator.n//train_generator.batch_size//step_rescale, workers=num_workers, verbose=verbose))
    val_score = hvd.allreduce(model_final.evaluate_generator(validation_generator, validation_generator.n//validation_generator.batch_size//step_rescale, workers=num_workers, verbose=verbose))
    t2 = time()
    if (hvd.rank()==0):
        print('**AllReduce  loss: %s  -  acc: %s  -  val_loss: %s  -  val_acc: %s\n Total time for Stage 2: %s seconds' 
              %(train_score[0], train_score[1], val_score[0], val_score[1], t1-t0))
        print('**Evaluation time: %s' %(t2-t1))
else:
    if (hvd.rank()==0):
        print('**Total time for Stage 2: %s seconds'%(t1-t0))

split_at = 2
for layer in model_final.layers[:split_at]: layer.trainable = False
for layer in model_final.layers[split_at:]: layer.trainable = True  

model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

# In[15]:

#Select Call Backs

Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8, verbose=verbose, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(PATH + 'deeplearning/weights/%s_new_UnFreeze_2.h5'%args.model, monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=False, mode='min', period=1)
if args.horovod:
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), 
                 hvd.callbacks.MetricAverageCallback(),
                 hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),
             ]
else:
    callbacks=[]
if hvd.rank()==0:
    callbacks.append(checkpoint)
if args.early_stop:
    callbacks=callbacks+[Early_Stop, reduce_lr]
else:
    callbacks=callbacks+[reduce_lr]

# In[16]:

#get_ipython().run_cell_magic('time', '', '
t0 = time()
history3 = model_final.fit(X_train, Y_train, validation_data = (X_valid, Y_valid), 
                            batch_size = batch_size, 
                            shuffle = shuffle,
                            epochs = args.epochs_3,
                            verbose=verbose, callbacks=callbacks) 

t1 = time()
train_score = hvd.allreduce(model_final.evaluate(X_train, Y_train, verbose=verbose))
train_score = hvd.allreduce(model_final.evaluate(X_valid, Y_valid, verbose=verbose))
t2 = time()
if (hvd.rank()==0):
    print('**AllReduce  loss: %s  -  acc: %s  -  val_loss: %s  -  val_acc: %s\n Total time for Stage 3: %s seconds' 
          %(train_score[0], train_score[1], val_score[0], val_score[1], t1-t0))
    print('**Evaluation time: %s' %(t2-t1))
    print("Saving model")
    model_final.save(PATH + 'deeplearning/weights/%s_Final.h5'%args.model)
    model_final.save(PATH + 'deeplearning/weights/%s'%args.save_model)

t_end = time()
if (hvd.rank()==0):
    print("-----------------------------")
    print("** Total Wall Time: %s seconds" %(t_end - t_start))