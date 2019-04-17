import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator
from hdf5_preprocessing import *
import h5py
from keras import backend as K


from tqdm import tqdm
from time import time
from mpi4py import MPI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_batches", default=20, type=int, help='Number of batches to read')
parser.add_argument("--batch_size", default=16, type=int, help='Batch size')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--PATH", default='/lus/theta-fs0/projects/mmaADSP/hzheng/new_DL_DES/')
parser.add_argument("--data_format", default='channels_last', )
args = parser.parse_args()
K.set_image_data_format(args.data_format)
if args.data_format=='channels_last':
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')
print(K.image_data_format())
comm = MPI.COMM_WORLD
print("MPI: %s of %s" %(comm.size, comm.rank))
sz = 224
PATH=args.PATH
nbatch=args.num_batches
batch_size=args.batch_size
shuffle=args.shuffle
data_format=args.data_format
if comm.rank==0:
    print("h5py version: %s" %h5py.__version__)
    print("Testing flow from directory performance")

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    data_format=data_format, 
    rotation_range=45)
train_generator = train_datagen.flow_from_directory(PATH+"/deeplearning/data/train",
    target_size = (sz, sz),
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = shuffle,
    interpolation = 'nearest')
t0 = time()
for i in tqdm(range(nbatch)):
    next(train_generator)
t1 = time()
if comm.rank==0:
    print("Throughput(flow_from_directory): %s images per second" %(nbatch*batch_size/(t1-t0)))

t0 = time()
print("Creating HDF5 file for train data: ")
datagen = ImageDataGenerator(rescale = 1./255, data_format=data_format)    
hdf5_from_directory("./data/train_%s.hdf5"%data_format,
                    './data/train', datagen,
                    target_size=(sz, sz),
                    batch_size=batch_size,
                    class_mode="categorical",
                    interpolation='nearest', mpi=False, verbose=1)
t1 = time()
print(" time: %s second" %(t1-t0))

print("Creating HDF5 file for validation data: ")
hdf5_from_directory("./data/valid_%s.hdf5"%data_format,
                    './data/valid', datagen,
                    target_size=(sz, sz),
                    batch_size=1,
                    class_mode="categorical",
                    interpolation='nearest', mpi=False, verbose=1)
t2 = time()
print(" time: %s second" %(t2-t1))
