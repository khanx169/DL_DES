#!/usr/bin/env python

# # 1. Import Libraries 

# In[65]:

import sys
### command line parsers
import argparse
parser = argparse.ArgumentParser(description="Xception input")
parser.add_argument('--batch_size',type=int, help='batch size', default=64)
parser.add_argument('--sz', type=int, help='size of the figure in each dimension', default=224)
parser.add_argument('--PATH', help='root directory', default='../')
parser.add_argument('--num_gpus_per_node', type=int, help='number of GPUs per node', default=2)
parser.add_argument('--device', help="device type: cpu or gpu", default='gpu')
parser.add_argument('--verbose', type=int, help='output level', default=2)
parser.add_argument('--num_intra', help='Number of intra threads', default=0)
parser.add_argument('--num_inter', help='Number of inter threads', default=2)
parser.add_argument('--num_workers', help='Number of workers in reading data', default=1)
args = parser.parse_args()
num_workers=args.num_workers
PATH = args.PATH
sz=args.sz
batch_size=args.batch_size
num_gpus_per_node=args.num_gpus_per_node
device=args.device
# In[67]:

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50, Xception
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import os
from os import listdir
from os.path import isfile, join, exists

#import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
import seaborn as sn
import pandas as pd 
import numpy as np
import horovod.keras as hvd
import tensorflow as tf
from time import time
hvd.init()
print("Parallelization method: Horovod")
print("HOROVOD: I am rank %s of %s" %(hvd.rank(), hvd.size()))

if device=='gpu':
    if hvd.rank()==0:
        print("Using GPUs")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank()%num_gpus_per_node)
else:
    config = tf.ConfigProto(intra_op_parallelism_threads=args.num_intra,
                            inter_op_parallelism_threads=args.num_inter)
    os.environ["CUDA_VISIBLE_DEVICES"]='-1'

K.set_session(tf.Session(config=config))


# # 2. Load Data / Create data_generators

# In[70]:

train_df = pd.read_csv(PATH + 'deeplearning/data/training_set.csv')
val_df = pd.read_csv(PATH + 'deeplearning/data/validation_set.csv')
HP_crossmatch_df = pd.read_csv(PATH + 'deeplearning/data/high_prob_crossmatch_test_set.csv')
FO_crossmatch_df = pd.read_csv(PATH + 'deeplearning/data/full_overlap_crossmatch_test_set.csv')


# #### flow_from_dir

# In[71]:

train_data_dir = PATH+'deeplearning/data/train/'
validation_data_dir = PATH+'deeplearning/data/valid/'

HP_SDSS_test_data_dir = PATH+'deeplearning/data/HP_crossmatch_test/sdss/'
HP_DES_test_data_dir = PATH+'deeplearning/data/HP_crossmatch_test/des/'

FO_SDSS_test_data_dir = PATH+'deeplearning/data/FO_crossmatch_test/sdss/'
FO_DES_test_data_dir = PATH+'deeplearning/data/FO_crossmatch_test/des/'


# In[72]:

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=45)

valid_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=45)



train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (sz, sz),
    batch_size = batch_size, 
    class_mode = "categorical",
    shuffle = True,
    interpolation = 'nearest')

validation_generator = valid_datagen.flow_from_directory(
validation_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = "categorical",
shuffle = False,
interpolation = 'nearest')
'''
gen = validation_generator
m=0
for i in gen:
    m=m+1
    idx = (gen.batch_index - 1) * gen.batch_size
    print("Rank %s: %s %s" %(hvd.rank(), gen.batch_index, gen.filenames[idx : idx + gen.batch_size]))
    if(m>4):
        exit()
'''
HP_SDSS_test_generator = test_datagen.flow_from_directory(
HP_SDSS_test_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = None,
shuffle = False)

HP_DES_test_generator = test_datagen.flow_from_directory(
HP_DES_test_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = None,
shuffle = False)


FO_SDSS_test_generator = test_datagen.flow_from_directory(
FO_SDSS_test_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = None,
shuffle = False)

FO_DES_test_generator = test_datagen.flow_from_directory(
FO_DES_test_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = None,
shuffle = False)


# # 3. Define Model 

# In[73]:

num_of_classes = 2

base_model = Xception(input_shape=(sz,sz,3), weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.7)(x)
x = Dense(1024, activation="relu", name='second_last_layer')(x)
predictions = Dense(num_of_classes, activation="softmax")(x)

model_final = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
if (hvd.rank()==0):
    model_final.summary()

opt = optimizers.Adam(lr=0.0001*hvd.size())
opt = hvd.DistributedOptimizer(opt)
model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])


# # 4. Train

# ## 1. Training Data 

# In[74]:

#files_s = os.listdir(PATH+'deeplearning/data/train/spiral')[:20]
#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.axis('Off')
#    plt.subplots_adjust(wspace=0.1,hspace=0.05)
#    plt.suptitle('SDSS spirals', fontsize=16)
#    img = plt.imread(PATH+'deeplearning/data/train/spiral/'+files_s[i])
#    plt.imshow(img)
    
#plt.savefig(PATH+'paper/pictures/SDSS_spirals.pdf', bbox_inches='tight')


# In[64]:

#files_e = os.listdir(PATH+'deeplearning/data/train/elliptical')[:20]
#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.axis('Off')
#    plt.subplots_adjust(wspace=0.1,hspace=0.05)
#    plt.suptitle('SDSS ellipticals', fontsize=16)
#    img = plt.imread(PATH+'deeplearning/data/train/elliptical/'+files_e[i+12])
#    plt.imshow(img)
    
#plt.savefig(PATH+'paper/pictures/SDSS_elliptical.pdf', bbox_inches='tight')


# In[75]:

# ## 2. Data Augmentation 

# In[76]:

# Creat a datagen similar to train_datagen [except for rescaling, in order to visualize]

#datagen = ImageDataGenerator(
#    #rescale = 1./255,
#    horizontal_flip = True,
#    vertical_flip = True,
#    fill_mode = "nearest",
#    zoom_range = 0.3,
#    width_shift_range = 0.3,
#    height_shift_range=0.3,
#    rotation_range=45)


# In[77]:

# Original Picture

#files_s = os.listdir(PATH+'deeplearning/data/train/spiral')[:20]
#x = plt.imread(PATH+'deeplearning/data/train/spiral/'+files_s[5])

#plt.figure( figsize=(3,2) )
#plt.axis('Off')
#plt.title('Original')
#plt.imshow(x)

#plt.savefig(PATH + 'paper/pictures/Augmentations_a.pdf')


# In[32]:

# Augmentations

#x = x.reshape((1,) + x.shape)
#i = 0

#for batch in datagen.flow(x, batch_size=1):
#    i += 1
#    plt.subplot(3,4,i)
#    plt.axis('Off')
#    plt.suptitle('Augmentations')
#    plt.imshow(batch[0])
#
#    if i > 11:
#        break  # otherwise the generator would loop indefinitely
#        
#plt.savefig(PATH + 'paper/pictures/Augmentations_b.pdf')


# ## 3. Learning 

# In[78]:

#K.clear_session()


# In[86]:

#os.environ["CUDA_VISIBLE_DEVICES"]='0'


# In[80]:
if (hvd.rank()==0):
    verbose = args.verbose
else:
    verbose = 0

t0 = time()

history_1 = model_final.fit_generator(
    train_generator,
    steps_per_epoch = (train_generator.n // train_generator.batch_size)//hvd.size(),
    epochs = 1,
    workers=num_workers, 
    validation_data = validation_generator,
    validation_steps = (validation_generator.n // validation_generator.batch_size)//hvd.size(), 
    verbose=verbose
) 
t1 = time()
if (hvd.rank()==0):
    print("Total time for history_1: %s seconds" %(t1 - t0))
    model_final.save(PATH + 'deeplearning/weights/XceptionNew_freeze.h5')
#model_final = load_model(PATH + 'deeplearning/weights/XceptionNew_freeze.h5',compile=False)
#len(model_final.layers)

# In[18]:

split_at = 40
for layer in model_final.layers[:split_at]: layer.trainable = False
for layer in model_final.layers[split_at:]: layer.trainable = True  

model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

#Select Call Backs

Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8, verbose=verbose, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(PATH + 'deeplearning/weights/Xception_new_UnFreeze.h5', monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=False, mode='min', period=1)

callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), Early_Stop, reduce_lr]
if hvd.rank()==0:
    callbacks.append(checkpoint)
# In[ ]:
t0 = time()
history2 = model_final.fit_generator(train_generator, steps_per_epoch=(train_generator.n // train_generator.batch_size)//hvd.size(), 
                                     epochs=20, workers=num_workers,
                                     validation_data=validation_generator, 
                                     validation_steps=validation_generator.n // (validation_generator.batch_size)//hvd.size(), 
                                     callbacks=callbacks, 
                                     verbose=verbose)
t1 = time()

# #### (iii) Unfreeze [at Layer 2] 

# In[9]:

if (hvd.rank()==0):
    print("Total time for history_2: %s " %(t1-t0))
hvd.allreduce([0])


split_at = 2
for layer in model_final.layers[:split_at]: layer.trainable = False
for layer in model_final.layers[split_at:]: layer.trainable = True  

model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

# In[15]:

#Select Call Backs

Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8, verbose=verbose, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(PATH + 'deeplearning/weights/Xception_new_UnFreeze_2.h5', monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=False, mode='min', period=1)
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
if hvd.rank()==0:
    callbacks.append(checkpoint)

callbacks=callbacks+[Early_Stop, reduce_lr]

# In[16]:

#get_ipython().run_cell_magic('time', '', '
t0 = time()
history3 = model_final.fit_generator(train_generator, 
                                     steps_per_epoch=(train_generator.n // train_generator.batch_size)//hvd.size(), 
                                     epochs=20, workers=num_workers,   
                                     validation_data=validation_generator, 
                                     validation_steps=(validation_generator.n // validation_generator.batch_size//hvd.size()), 
                                     callbacks=callbacks, 
                                     verbose=verbose)
t1 = time()
if (hvd.rank()==0):
    print('Total time for history 3: %s' %(t1 - t0))
# In[28]:

#Training Accuracy/Loss

#plt.figure( figsize=(11,4) )

# Plot training & validation accuracy values
#plt.subplot(1,2,1)
#plt.plot(history3.history['acc'])
#plt.plot(history3.history['val_acc'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

# Plot training & validation loss values
#plt.subplot(1,2,2)
#plt.plot(history3.history['loss'])
#plt.plot(history3.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
if (hvd.rank()==0):
    model_final.save(PATH + 'deeplearning/weights/Xception_Final.h5')
exit()
# ## 4. Save Final Model with Weights 

# In[17]:


# In[9]:

model_final = load_model(PATH + 'deeplearning/weights/Xception_Final.h5')


# # 5. HP Crossmatch Test 

# ### Load sklearn / Def metrics 

# In[89]:

import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[90]:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figure_size = (11, 6),
                          save=0,
                          save_path='/home/khan74'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if normalize:
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.figure(figsize=figure_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path)
        #plt.savefig('confusion_matrix.png')


# In[91]:

# Define probability threshold mask

def threshold_mask(pred_array, prob_threshold = 0.9999):
    ''' 
    returns the positions in the array where the probability for each class is greater than prob_threshold
    
    pred_array: A one-hot encoded array of softmax probability outputs 
    prob_threshold: Float b/w 0 and 1 to use as a threshold mask
    '''
    pred_class_indices = np.argmax(pred_array, axis=1)
    pos = np.where( pred_array[ range( pred_array.shape[0] ), pred_class_indices ] > prob_threshold )[0]
    return pos


# In[92]:

# Define top n predictions mask

def top_pred_mask(pred_array, n_top_predictions = 5000 ):
    '''
    reversedurn the positions of top n most confident predictions
    
    pred_array: A one-reversedencoded array of softmax probability outputs
    n_top_predictions: Num of top predictions
    '''
    pred_class_indices = np.argmax(pred_array, axis=1)
    pos = np.argsort( pred_array[ range( pred_array.shape[0] ), pred_class_indices ] ) 
    mask = np.flip( pos )
    return mask[: n_top_predictions]


# ### (i) SDSS 

# In[29]:

HP_SDSS_predictions = model_final.predict_generator(HP_SDSS_test_generator,  verbose=1)


# In[30]:

HP_SDSS_predicted_class_indices=np.argmax(HP_SDSS_predictions,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in HP_SDSS_predicted_class_indices] )

filenames= np.array( HP_SDSS_test_generator.filenames )
#filenames = [x.split('/')[1].split('.')[0] for x in filenames]

results=pd.DataFrame({"Filename":filenames,
                      "Pred Labels":predictions,
                     "Predictions":HP_SDSS_predicted_class_indices})


# In[31]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, HP_crossmatch_df, left_on='OBJID', right_on='SDSS_OBJID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[32]:

final[['OBJID','Predictions','Actuals', 'Pred Labels', 'P_EL_DEBIASED', 'P_CS_DEBIASED']].head()


# In[33]:

print( "Accuracy: ", metrics.accuracy_score(final['Actuals'], final['Predictions']) )
print( "f1_score: ", metrics.f1_score(final['Actuals'], final['Predictions']))


# In[34]:

cm = confusion_matrix(final['Actuals'], final['Predictions'])
plot_confusion_matrix(cm, classes=['Elliptical', 'Spiral'], title="HP SDSS Test Set", figure_size=(4,3.4), save=1, save_path=PATH+'paper/pictures/HP_SDSS_cm.pdf')


# #### Looking at some examples of correctly and incorrectly classified pictures 

# In[21]:

#Seperate into Correct/InCorrect

Correctly_classified = final[ final.Predictions == final.Actuals ]
Incorrectly_classified = final[ final.Predictions != final.Actuals ]


# In[22]:

#Seperate into Spiral/Elliptical

cc_spirals = Correctly_classified[ Correctly_classified.P_CS_DEBIASED > Correctly_classified.P_EL_DEBIASED ]
cc_elliptical = Correctly_classified[ Correctly_classified.P_CS_DEBIASED < Correctly_classified.P_EL_DEBIASED ]

ic_spirals = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED > Incorrectly_classified.P_EL_DEBIASED ]
ic_elliptical = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED < Incorrectly_classified.P_EL_DEBIASED ]


# In[23]:

# Correct Spirals

for i, file in enumerate( np.random.choice(cc_spirals.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified spirals')
    img = plt.imread(HP_SDSS_test_data_dir + file)
    plt.imshow(img)


# In[24]:

# Correct Ellipticals

for i, file in enumerate( np.random.choice(cc_elliptical.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified ellipticals')
    img = plt.imread(HP_SDSS_test_data_dir + file)
    plt.imshow(img)


# In[25]:

# InCorrect Spirals

for i, file in enumerate( np.random.choice(ic_spirals.Filename, len(ic_spirals), replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified spirals')
    img = plt.imread(HP_SDSS_test_data_dir + file)
    plt.imshow(img)


# In[25]:

# InCorrect Ellipticals

for i, file in enumerate( np.random.choice(ic_elliptical.Filename, len(ic_elliptical), replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified ellipticals')
    img = plt.imread(HP_SDSS_test_data_dir + file)
    plt.imshow(img)


# ### (ii) DES 

# In[23]:

HP_DES_predictions = model_final.predict_generator(HP_DES_test_generator,  verbose=1)


# In[24]:

HP_DES_predicted_class_indices=np.argmax(HP_DES_predictions,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in HP_DES_predicted_class_indices] )

filenames= np.array( HP_DES_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames,
                      "Pred Labels":predictions,
                     "Predictions":HP_DES_predicted_class_indices})


# In[25]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, HP_crossmatch_df, left_on='OBJID', right_on='DES_COADD_OBJECT_ID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[26]:

final[['OBJID','Predictions','Actuals', 'Pred Labels', 'P_EL_DEBIASED', 'P_CS_DEBIASED']].head()


# In[27]:

print( "Accuracy: ", metrics.accuracy_score(final['Actuals'], final['Predictions']) )
print( "f1_score: ", metrics.f1_score(final['Actuals'], final['Predictions']))


# In[28]:

cm = confusion_matrix(final['Actuals'], final['Predictions'])
plot_confusion_matrix(cm, classes=['Elliptical', 'Spirals'], title="HP DES Test Set", figure_size=(4,3.4), save=1, save_path=PATH+'paper/pictures/HP_DES_cm.pdf')


# #### Looking at some examples of correctly and incorrectly classified pictures 

# In[49]:

#Seperate into Correct/InCorrect

Correctly_classified = final[ final.Predictions == final.Actuals ]
Incorrectly_classified = final[ final.Predictions != final.Actuals ]


# In[50]:

#Seperate into Spiral/Elliptical

cc_spirals = Correctly_classified[ Correctly_classified.P_CS_DEBIASED > Correctly_classified.P_EL_DEBIASED ]
cc_elliptical = Correctly_classified[ Correctly_classified.P_CS_DEBIASED < Correctly_classified.P_EL_DEBIASED ]

ic_spirals = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED > Incorrectly_classified.P_EL_DEBIASED ]
ic_elliptical = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED < Incorrectly_classified.P_EL_DEBIASED ]


# In[51]:

# Correct Spirals

for i, file in enumerate( np.random.choice(cc_spirals.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified spirals')
    img = plt.imread(HP_DES_test_data_dir + file)
    plt.imshow(img)


# In[52]:

# Correct Ellipticals

for i, file in enumerate( np.random.choice(cc_elliptical.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified ellipticals')
    img = plt.imread(HP_DES_test_data_dir + file)
    plt.imshow(img)


# In[53]:

# InCorrect Spirals

for i, file in enumerate( np.random.choice(ic_spirals.Filename, len(ic_spirals), replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified spirals')
    img = plt.imread(HP_DES_test_data_dir + file)
    plt.imshow(img)


# In[54]:

# InCorrect Elliptical

for i, file in enumerate( np.random.choice(ic_elliptical.Filename, len(ic_elliptical), replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified ellipticals')
    img = plt.imread(HP_DES_test_data_dir + file)
    plt.imshow(img)


# # 6. FO Crossmatch Test  

# ### (i) SDSS 

# In[35]:

FO_SDSS_predictions = model_final.predict_generator(FO_SDSS_test_generator,  verbose=1)


# In[60]:

#mask = threshold_mask(FO_SDSS_predictions, prob_threshold=0.9999)
#mask = threshold_mask(FO_SDSS_predictions, prob_threshold=0.999999)
mask = top_pred_mask(FO_SDSS_predictions, 6300)


# In[61]:

FO_SDSS_predicted_class_indices=np.argmax(FO_SDSS_predictions,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in FO_SDSS_predicted_class_indices] )

filenames= np.array( FO_SDSS_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":FO_SDSS_predicted_class_indices[mask]})


# In[62]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, FO_crossmatch_df, left_on='OBJID', right_on='SDSS_OBJID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[63]:

final[['OBJID','Predictions','Actuals', 'Pred Labels', 'P_EL_DEBIASED', 'P_CS_DEBIASED']].head()


# In[64]:

final.shape


# In[65]:

print( "Accuracy: ", metrics.accuracy_score(final['Actuals'], final['Predictions']) )
print( "f1_score: ", metrics.f1_score(final['Actuals'], final['Predictions']))


# In[66]:

cm = confusion_matrix(final['Actuals'], final['Predictions'])
plot_confusion_matrix(cm, classes=['Elliptical', 'Spiral'], title="FO SDSS Test Set", figure_size=(4,3.4), save=1, save_path=PATH+'paper/pictures/FO_SDSS_cm.pdf')


# #### Looking at some examples of correctly and incorrectly classified pictures 

# In[131]:

#Seperate into Correct/InCorrect

Correctly_classified = final[ final.Predictions == final.Actuals ]
Incorrectly_classified = final[ final.Predictions != final.Actuals ]


# In[132]:

#Seperate into Spiral/Elliptical

cc_spirals = Correctly_classified[ Correctly_classified.P_CS_DEBIASED > Correctly_classified.P_EL_DEBIASED ]
cc_elliptical = Correctly_classified[ Correctly_classified.P_CS_DEBIASED < Correctly_classified.P_EL_DEBIASED ]

ic_spirals = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED > Incorrectly_classified.P_EL_DEBIASED ]
ic_elliptical = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED < Incorrectly_classified.P_EL_DEBIASED ]


# In[133]:

# Correct Spirals

for i, file in enumerate( np.random.choice(cc_spirals.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified spirals')
    img = plt.imread(FO_SDSS_test_data_dir + file)
    plt.imshow(img)


# In[134]:

# Correct Ellipticals

for i, file in enumerate( np.random.choice(cc_elliptical.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified ellipticals')
    img = plt.imread(FO_SDSS_test_data_dir + file)
    plt.imshow(img)


# In[137]:

# InCorrect Spirals

for i, file in enumerate( np.random.choice(ic_spirals.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified spirals (i.e classified as ellip)')
    img = plt.imread(FO_SDSS_test_data_dir + file)
    plt.title('P_CS: '+ str(round(float(ic_spirals[ ic_spirals.Filename==file ]['P_CS_DEBIASED']),2)), fontsize=8)
    print('P_EL: '+ str(round(float(ic_spirals[ ic_spirals.Filename==file ]['P_EL_DEBIASED']),2)))
    plt.imshow(img)


# In[140]:

# InCorrect Elliptical

for i, file in enumerate( np.random.choice(ic_elliptical.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified ellipticals (i.e classified as spirals)')
    img = plt.imread(FO_SDSS_test_data_dir + file)
    plt.title('P_EL: '+ str(round(float(ic_elliptical[ ic_elliptical.Filename==file ]['P_EL_DEBIASED']),2)), fontsize=8)
    print('P_CS: '+ str(round(float(ic_elliptical[ ic_elliptical.Filename==file ]['P_CS_DEBIASED']),2)))
    plt.imshow(img)


# ### (ii) DES 

# In[67]:

FO_DES_predictions = model_final.predict_generator(FO_DES_test_generator,  verbose=1)


# In[124]:

#mask = threshold_mask(FO_DES_predictions, prob_threshold=0.9999)
mask = top_pred_mask(FO_DES_predictions, 6300)


# In[125]:

FO_DES_predicted_class_indices=np.argmax(FO_DES_predictions,axis=1)

#FO_SDSS_predicted_class_indices = FO_SDSS_predicted_class_indices[pos]

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in FO_DES_predicted_class_indices] )

filenames= np.array( FO_DES_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":FO_DES_predicted_class_indices[mask]})


# In[126]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, FO_crossmatch_df, left_on='OBJID', right_on='DES_COADD_OBJECT_ID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[127]:

final[['OBJID','Predictions','Actuals', 'Pred Labels', 'P_EL_DEBIASED', 'P_CS_DEBIASED']].head()


# In[128]:

final.shape


# In[129]:

print( "Accuracy: ", metrics.accuracy_score(final['Actuals'], final['Predictions']) )
print( "f1_score: ", metrics.f1_score(final['Actuals'], final['Predictions']))


# In[130]:

cm = confusion_matrix(final['Actuals'], final['Predictions'])
plot_confusion_matrix(cm, classes=['Elliptical', 'Spiral'], title="FO DES Test Set", figure_size=(4,3.4), save=1, save_path=PATH+'paper/pictures/FO_DES_cm.pdf')


# #### Looking at some examples of correctly and incorrectly classified pictures 

# In[78]:

#Seperate into Correct/InCorrect

Correctly_classified = final[ final.Predictions == final.Actuals ]
Incorrectly_classified = final[ final.Predictions != final.Actuals ]


# In[79]:

#Seperate into Spiral/Elliptical

cc_spirals = Correctly_classified[ Correctly_classified.P_CS_DEBIASED > Correctly_classified.P_EL_DEBIASED ]
cc_elliptical = Correctly_classified[ Correctly_classified.P_CS_DEBIASED < Correctly_classified.P_EL_DEBIASED ]

ic_spirals = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED > Incorrectly_classified.P_EL_DEBIASED ]
ic_elliptical = Incorrectly_classified[ Incorrectly_classified.P_CS_DEBIASED < Incorrectly_classified.P_EL_DEBIASED ]


# In[80]:

# Correct Spirals

for i, file in enumerate( np.random.choice(cc_spirals.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified spirals')
    img = plt.imread(FO_DES_test_data_dir + file)
    plt.imshow(img)


# In[81]:

# Correct Ellipticals

for i, file in enumerate( np.random.choice(cc_elliptical.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('Correctly classified ellipticals')
    img = plt.imread(FO_DES_test_data_dir + file)
    plt.imshow(img)


# In[116]:

# InCorrect Spirals

for i, file in enumerate( np.random.choice(ic_spirals.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified spirals (i.e classified as ellip)')
    img = plt.imread(FO_DES_test_data_dir + file)
    plt.title('P_CS: '+ str(round(float(ic_spirals[ ic_spirals.Filename==file ]['P_CS_DEBIASED']),2)), fontsize=8)
    print('P_EL: '+ str(round(float(ic_spirals[ ic_spirals.Filename==file ]['P_EL_DEBIASED']),2)))
    plt.imshow(img)


# In[120]:

# InCorrect Elliptical

for i, file in enumerate( np.random.choice(ic_elliptical.Filename, 6, replace=False) ): 
    plt.subplot(2,3,i+1)
    plt.axis('Off')
    plt.subplots_adjust(wspace=0.1,hspace=0.05)
    plt.suptitle('InCorrectly classified ellipticals (i.e classified as spirals)')
    img = plt.imread(FO_DES_test_data_dir + file)
    plt.title('P_EL: '+ str(round(float(ic_elliptical[ ic_elliptical.Filename==file ]['P_EL_DEBIASED']),2)), fontsize=8)
    print('P_CS: '+ str(round(float(ic_elliptical[ ic_elliptical.Filename==file ]['P_CS_DEBIASED']),2)))
    plt.imshow(img)


# # 7. Clustering 

# ### Load tSNE  / Define Clustering Model

# In[10]:

from sklearn.manifold import TSNE
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


# In[11]:

clustering_model = Model(input = model_final.input, output = model_final.get_layer('second_last_layer').output)


# ### 1. HP Crossmatch Test 

# #### (i) SDSS

# In[12]:

HP_SDSS_out = clustering_model.predict_generator(HP_SDSS_test_generator, verbose=1)


# In[13]:

tsne_HP_SDSS = TSNE(n_components=3, verbose=1, n_iter=10000, learning_rate=10).fit_transform(HP_SDSS_out)


# In[143]:

filenames= np.array( HP_SDSS_test_generator.filenames )
results=pd.DataFrame({"Filename":filenames})


# In[144]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, HP_crossmatch_df, left_on='OBJID', right_on='SDSS_OBJID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[145]:

labels = np.array( final['Actuals'] )


# In[149]:

#Plot tSNE

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_HP_SDSS[:,0][labels==0], tsne_HP_SDSS[:,2][labels==0], tsne_HP_SDSS[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_HP_SDSS[:,0][labels==1], tsne_HP_SDSS[:,2][labels==1], tsne_HP_SDSS[:,1][labels==1], c='black', label='Spiral')

ax.legend(loc=[0.72,0.85], fontsize=14)
ax.view_init(elev=12, azim=-88)

font = {'fontname':'Times New Roman'}
plt.title('HP SDSS Test Set', fontsize=24, fontweight="bold", pad=-40, **font)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_zticklabels(), visible=False)
ax.tick_params(axis='both', which='both', length=0)



#Set Ticks
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.zaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.savefig(PATH + 'paper/pictures/tSNE_SDSS_HP.pdf')


# #### (ii) DES

# In[61]:

HP_DES_out = clustering_model.predict_generator(HP_DES_test_generator, verbose=1)


# In[124]:

HP_DES_predictions = model_final.predict_generator(HP_DES_test_generator, verbose=1)


# In[66]:

tsne_HP_DES = TSNE(n_components=3, verbose=1, n_iter=10000, learning_rate=10).fit_transform(HP_DES_out)


# In[137]:

filenames= np.array( HP_DES_test_generator.filenames )
results=pd.DataFrame({"Filename":filenames})


# In[138]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, HP_crossmatch_df, left_on='OBJID', right_on='DES_COADD_OBJECT_ID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[139]:

labels = np.array( final['Actuals'] )


# In[141]:

#Plot tSNE

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_HP_DES[:,0][labels==0], tsne_HP_DES[:,2][labels==0], tsne_HP_DES[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_HP_DES[:,0][labels==1], tsne_HP_DES[:,2][labels==1], tsne_HP_DES[:,1][labels==1], c='black', label='Spiral')

#ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
#ax.view_init(elev=22, azim=158)
ax.view_init(elev=12, azim=-88)

font = {'fontname':'Times New Roman'}
plt.title('HP DES Test Set', fontsize=24, fontweight="bold", pad=-40, **font)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_zticklabels(), visible=False)
ax.tick_params(axis='both', which='both', length=0)


#Set Ticks
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.zaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.savefig(PATH + 'paper/pictures/tSNE_DES_HP.pdf')


# ### 2. FO Crossmatch Test 

# #### (i) SDSS

# In[33]:

FO_SDSS_out = clustering_model.predict_generator(FO_SDSS_test_generator, verbose=1)


# In[34]:

FO_SDSS_predictions = model_final.predict_generator(FO_SDSS_test_generator, verbose=1)


# In[75]:

mask = threshold_mask(FO_SDSS_predictions, prob_threshold=0.999999)


# In[76]:

tsne_FO_SDSS = TSNE(n_components=3, verbose=1, n_iter=5000, learning_rate=10).fit_transform(FO_SDSS_out[mask])


# In[77]:

FO_SDSS_predicted_class_indices=np.argmax(FO_SDSS_predictions,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in FO_SDSS_predicted_class_indices] )

filenames= np.array( FO_SDSS_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":FO_SDSS_predicted_class_indices[mask]})


# In[78]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, FO_crossmatch_df, left_on='OBJID', right_on='SDSS_OBJID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[92]:

labels = np.array( final['Actuals'] )


# In[81]:

labels = np.array( final['Predictions'] )


# In[93]:

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_FO_SDSS[:,0][labels==0], tsne_FO_SDSS[:,2][labels==0], tsne_FO_SDSS[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_FO_SDSS[:,0][labels==1], tsne_FO_SDSS[:,2][labels==1], tsne_FO_SDSS[:,1][labels==1], c='black', label='Spiral')

ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
ax.view_init(elev=30, azim=-71)

font = {'fontname':'Times New Roman'}
plt.title('FO SDSS test set', fontsize=12, fontweight="bold", pad=15, **font)

#plt.savefig('tSNE_SDSS.pdf')


# #### (ii) DES

# In[94]:

FO_DES_out = clustering_model.predict_generator(FO_DES_test_generator, verbose=1)


# In[95]:

FO_DES_predictions = model_final.predict_generator(FO_DES_test_generator, verbose=1)


# ###### Mask by Prob threshold

# In[96]:

mask = threshold_mask(FO_DES_predictions, prob_threshold=0.999999)


# In[97]:

tsne_FO_DES = TSNE(n_components=3, verbose=1, n_iter=5000, learning_rate=10).fit_transform(FO_DES_out[mask])


# In[109]:

FO_DES_predicted_class_indices=np.argmax(FO_DES_predictions,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in FO_DES_predicted_class_indices] )

filenames= np.array( FO_DES_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":FO_DES_predicted_class_indices[mask]})


# In[110]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, FO_crossmatch_df, left_on='OBJID', right_on='DES_COADD_OBJECT_ID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[112]:

labels = np.array( final['Actuals'] )


# In[114]:

labels = np.array( final['Predictions'] )


# In[115]:

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_FO_DES[:,0][labels==0], tsne_FO_DES[:,2][labels==0], tsne_FO_DES[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_FO_DES[:,0][labels==1], tsne_FO_DES[:,2][labels==1], tsne_FO_DES[:,1][labels==1], c='black', label='Spiral')

ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
ax.view_init(elev=30, azim=-71)

font = {'fontname':'Times New Roman'}
plt.title('FO DES test set', fontsize=12, fontweight="bold", pad=15, **font)

#plt.savefig('tSNE_SDSS.pdf')


# ###### Mask by top pred 

# In[144]:

mask = top_pred_mask(FO_DES_predictions, 3000)


# In[145]:

tsne_FO_DES = TSNE(n_components=3, verbose=1, n_iter=5000, learning_rate=10).fit_transform(FO_DES_out[mask])


# In[146]:

FO_DES_predicted_class_indices=np.argmax(FO_DES_predictions,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in FO_DES_predicted_class_indices] )

filenames= np.array( FO_DES_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":FO_DES_predicted_class_indices[mask]})


# In[147]:

results['OBJID'] = results['Filename'].apply(lambda x: x.split('/')[-1][:-4]).astype('int64') 
final = pd.merge(results, FO_crossmatch_df, left_on='OBJID', right_on='DES_COADD_OBJECT_ID')
final['Actuals'] = final['P_CS_DEBIASED'] > final['P_EL_DEBIASED']


# In[148]:

labels = np.array( final['Actuals'] )


# In[156]:

labels = np.array( final['Predictions'] )


# In[157]:

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_FO_DES[:,0][labels==0], tsne_FO_DES[:,2][labels==0], tsne_FO_DES[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_FO_DES[:,0][labels==1], tsne_FO_DES[:,2][labels==1], tsne_FO_DES[:,1][labels==1], c='black', label='Spiral')

ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
ax.view_init(elev=30, azim=-71)

font = {'fontname':'Times New Roman'}
plt.title('FO DES test set', fontsize=12, fontweight="bold", pad=15, **font)

#plt.savefig('tSNE_SDSS.pdf')


# ### 3. Unlabelled DES 

# In[97]:

UL_test_data_dir = PATH+'deeplearning/data/unlabelled/'


# In[98]:

UL_test_generator = test_datagen.flow_from_directory(
UL_test_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = None,
shuffle = False)


# In[99]:

UL_out = clustering_model.predict_generator(UL_test_generator, verbose=1)


# In[87]:

UL_predictions = model_final.predict_generator(UL_test_generator, verbose=1)


# ###### Mask by Prob threshold 

# In[98]:

mask = threshold_mask(UL_predictions, prob_threshold=0.999999)


# In[22]:

tsne_UL = TSNE(n_components=3, verbose=1, n_iter=5000, learning_rate=10).fit_transform(UL_out[mask])


# In[25]:

labels = ( np.argmax(UL_predictions, axis=1) )[mask]


# In[26]:

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_UL[:,0][labels==0], tsne_UL[:,2][labels==0], tsne_UL[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_UL[:,0][labels==1], tsne_UL[:,2][labels==1], tsne_UL[:,1][labels==1], c='black', label='Spiral')

ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
ax.view_init(elev=30, azim=-71)

font = {'fontname':'Times New Roman'}
plt.title('FO DES test set', fontsize=12, fontweight="bold", pad=15, **font)

#plt.savefig('tSNE_SDSS.pdf')


# ###### Mask by top pred 

# In[100]:

mask = top_pred_mask(UL_predictions, 4500)


# In[101]:

tsne_UL = TSNE(n_components=3, verbose=1, n_iter=10000, learning_rate=10).fit_transform(UL_out[mask])


# In[102]:

labels = ( np.argmax(UL_predictions, axis=1) )[mask]


# In[122]:

#Plot tSNE

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_UL[:,0][labels==0], tsne_UL[:,2][labels==0], tsne_UL[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_UL[:,0][labels==1], tsne_UL[:,2][labels==1], tsne_UL[:,1][labels==1], c='black', label='Spiral')

#ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
ax.view_init(elev=10, azim=-174)

font = {'fontname':'Times New Roman'}
plt.title('Unlabelled DES', fontsize=24, fontweight="bold", pad=-40, **font)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_zticklabels(), visible=False)
ax.tick_params(axis='both', which='both', length=0)



#Set Ticks
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.zaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.savefig(PATH + 'paper/pictures/tSNE_Unlabelled_DES.pdf')


# ###### Looking at Pictures

# In[ ]:

mask = threshold_mask(UL_predictions, prob_threshold=0.999999)


# In[160]:

mask = top_pred_mask(UL_predictions, 4500)


# In[161]:

UL_predicted_class_indices=np.argmax(UL_predictions,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in UL_predicted_class_indices] )

filenames= np.array( UL_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":UL_predicted_class_indices[mask]})


# In[162]:

spirals = results[ results.Predictions==1 ]


# In[163]:

spirals.shape


# In[164]:

elliptical = results[ results.Predictions==0 ]


# In[165]:

elliptical.shape


# In[170]:

def plot_pred(df, start_index = 0, savefig=False, savepath='/home/khan74/'):
    
    for i, index in enumerate( df.index[start_index:start_index+50] ):
    
        row = results.loc[index]
        filename = row.Filename
        type_ = row['Pred Labels']

        img_path = UL_test_data_dir + str(filename)

        img = plt.imread(img_path )

        plt.subplot(5,10,i+1)
        plt.axis('Off')
        plt.subplots_adjust(wspace=0.1,hspace=0.05)
        plt.imshow(img)
        plt.suptitle('Predicted ' + str.upper(type_[0]) + type_[1:] + 's')
        
    plt.show()
    
    if savefig:
        plt.savefig( savepath )


# In[174]:

plot_pred(spirals, 100, savefig=False, savepath=PATH+'paper/pictures/des_predicted_spirals.pdf')


# In[185]:

plot_pred(elliptical, 0, savefig=False, savepath=PATH+'paper/pictures/des_predicted_ellipticals.pdf')


# ### 4. _tmp_ Old Unlabelled DES 

# In[16]:

OUL_test_data_dir = "/home/khan74/scratch/data/SDSS/test2/"


# In[17]:

OUL_test_generator = test_datagen.flow_from_directory(
OUL_test_data_dir,
target_size = (sz, sz),
batch_size = 1,
class_mode = None,
shuffle = False)


# In[18]:

OUL_out = clustering_model.predict_generator(OUL_test_generator, verbose=1)


# In[19]:

OUL_predictions = model_final.predict_generator(OUL_test_generator, verbose=1)


# ###### Mask by top pred 

# In[83]:

mask = top_pred_mask(OUL_predictions, 500)


# In[84]:

tsne_OUL = TSNE(n_components=3, verbose=1, n_iter=5000, learning_rate=10).fit_transform(OUL_out[mask])


# In[85]:

labels = ( np.argmax(OUL_predictions, axis=1) )[mask]


# In[86]:

fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)

ax.scatter3D(tsne_OUL[:,0][labels==0], tsne_OUL[:,2][labels==0], tsne_OUL[:,1][labels==0], c='red', label='Elliptical')
ax.scatter3D(tsne_OUL[:,0][labels==1], tsne_OUL[:,2][labels==1], tsne_OUL[:,1][labels==1], c='black', label='Spiral')

ax.legend(loc='upper right')
#ax.view_init(elev=30, azim=-170)
ax.view_init(elev=30, azim=-71)

font = {'fontname':'Times New Roman'}
plt.title('Old Unlabelled DES test set', fontsize=12, fontweight="bold", pad=15, **font)

#plt.savefig('tSNE_SDSS.pdf')


# ###### Looking at Pictures

# In[59]:

mask = threshold_mask(OUL_predictions, prob_threshold=0.9999)


# In[68]:

mask = top_pred_mask(OUL_predictions, 450)


# In[69]:

OUL_predicted_class_indices=np.argmax(OUL_predictions,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.array( [labels[k] for k in OUL_predicted_class_indices] )

filenames= np.array( OUL_test_generator.filenames )

results=pd.DataFrame({"Filename":filenames[mask],
                      "Pred Labels":predictions[mask],
                     "Predictions":OUL_predicted_class_indices[mask]})


# In[70]:

spirals = results[ results.Predictions==1 ]


# In[71]:

spirals.shape


# In[72]:

elliptical = results[ results.Predictions==0 ]


# In[73]:

elliptical.shape


# In[80]:

def plot_pred(df, start_index = 0):
    
    for i, index in enumerate( df.index[start_index:start_index+100] ):
    
        row = results.loc[index]
        filename = row.Filename
        type_ = row['Pred Labels']

        img_path = OUL_test_data_dir + str(filename)

        img = plt.imread(img_path )

        plt.subplot(10,10,i+1)
        plt.axis('Off')
        plt.subplots_adjust(wspace=0.1,hspace=0.05)
        plt.imshow(img)
        plt.suptitle('Predicted ' + type_)
        
    plt.show()


# In[81]:

plot_pred(spirals, 0)


# In[82]:

plot_pred(elliptical, 0)


# In[ ]:



