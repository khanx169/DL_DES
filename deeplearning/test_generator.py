import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator
from hdf5_preprocessing import *
import h5py

from tqdm import tqdm
from time import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
print("MPI: %s of %s" %(comm.size, comm.rank))
sz = 224
nbatch=20
batch_size=16
rank = comm.rank
size = comm.size

if comm.rank==0:
    print("h5py version: %s" %h5py.__version__)
    print("Testing flow from directory performance")
PATH='/lus/theta-fs0/projects/mmaADSP/hzheng/new_DL_DES/'
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=45)
train_generator = train_datagen.flow_from_directory(PATH+"/deeplearning/data/train",
    target_size = (sz, sz),
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True,
    interpolation = 'nearest')
t0 = time()
for i in tqdm(range(nbatch)):
    next(train_generator)
t1 = time()
if comm.rank==0:
    print("Throughput(flow_from_directory): %s images per second" %(nbatch*batch_size/(t1-t0)))

'''
t0 = time()
print("Creating HDF5 file for train data: ")
datagen = ImageDataGenerator(rescale = 1./255)    
hdf5_from_directory("./data/train.hdf5",
                    './data/train', datagen,
                    target_size=(sz, sz),
                    batch_size=batch_size,
                    class_mode="categorical",
                    interpolation='nearest', mpi=False)
t1 = time()
print(" time: %s second" %(t1-t0))

print("Creating HDF5 file for validation data: ")
hdf5_from_directory("./data/valid.hdf5",
                    './data/valid', datagen,
                    target_size=(sz, sz),
                    batch_size=1,
                    class_mode="categorical",
                    interpolation='nearest', mpi=False)
t2 = time()
print(" time: %s second" %(t2-t1))
'''
gen = HDF5ImageGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         fill_mode = "nearest",
                         zoom_range = 0.3,
                         width_shift_range = 0.3,
                         height_shift_range = 0.3,
                         rotation_range=45)

fh = h5py.File(PATH+'/deeplearning/data/train_save.hdf5', 'r')
gen.fit(fh['data'][rank:rank+1])
flow = gen.flow(fh['data'][0:nbatch*batch_size], fh['labels'][0:nbatch*batch_size], batch_size=batch_size)

t3 = time()
print(flow.n)
for i in tqdm(range(nbatch)):
    x, y = next(flow)
t4 = time()
if comm.rank==0:
    print("Throughput(flow): %s images per second" %(nbatch*batch_size/(t4-t3)))
#df = gen.flow_from_hdf5(fh, shuffle=True, batch_size=batch_size, nsample=nbatch*batch_size)
#print("Testing flow from HDF5 performance")
#t3 = time()
#print(df.n)
#for i in tqdm(range(nbatch)):
#    x, y = next(df)
#t4 = time()
#print("Throughput(flow_from_hdf5): %s images per second" %(nbatch*batch_size/(t4-t3)))
