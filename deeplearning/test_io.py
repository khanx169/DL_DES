#!/usr/bin/env python
# Testing I/O throughput 
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator
from hdf5_preprocessing import *
from keras import backend as K
import h5py
from tqdm import tqdm
from time import time
from mpi4py import MPI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_batches", default=20, type=int, help='Number of batches to read')
parser.add_argument("--batch_size", default=16, type=int, help='Batch size')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--PATH", default='/lus/theta-fs0/projects/mmaADSP/hzheng/new_DL_DES/')
parser.add_argument("--hdf5_file", default='train_channels_first.hdf5')
args = parser.parse_args()
fh = h5py.File(args.PATH+'/deeplearning/data/'+args.hdf5_file, 'r')
data_format=fh['data'].attrs['data_format']
print(data_format)
print(fh['data'].shape)
K.set_image_data_format(data_format)
if data_format=='channels_last':
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')



comm = MPI.COMM_WORLD
print("MPI: %s of %s" %(comm.size, comm.rank))
sz = 224
PATH=args.PATH
nbatch=args.num_batches
batch_size=args.batch_size
shuffle=args.shuffle
rank = comm.rank
size = comm.size
# Flow from directory method
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
it = range(nbatch)
if rank==0:
    it = tqdm(it)
for i in it:
    next(train_generator)
t1 = time()
dtinv = 1./(t1 - t0)
dtinv_avg = comm.allreduce(dtinv,op=MPI.SUM)
if comm.rank==0:
    print("*Throughput(flow_from_directory): %s imgs/s    --  %s MB/s(estimate)" %(nbatch*batch_size*dtinv_avg, nbatch*batch_size*dtinv_avg*40000/1024/1024*comm.size))
### Flow method
gen = HDF5ImageGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         fill_mode = "nearest",
                         zoom_range = 0.3,
                         width_shift_range = 0.3,
                         height_shift_range = 0.3,
                         rotation_range=45, data_format=data_format)

print(fh['data'].shape)
nsample = fh['data'].shape[0] // size
offset = nsample*rank

df = gen.flow_from_hdf5(fh, shuffle=shuffle, batch_size=batch_size, nsample=nsample, offset=offset)
if rank==0:
    print("Testing flow from HDF5 performance")
t3 = time()

it = range(nbatch)
if rank==0:
    it = tqdm(it)

for i in it:
    x, y = next(df)
t4 = time()
dtinv = 1./(t4 - t3)
dtinv_avg = comm.allreduce(dtinv,op=MPI.SUM)
if comm.rank==0:
    print("*Throughput(flow_from_hdf5): %s imgs/s    --  %s MB/s" %(nbatch*batch_size*dtinv_avg, nbatch*batch_size*dtinv_avg*sz*sz*3*16/1024/1024))
if (data_format=='channels_first'):
    X=np.zeros((nsample, 3, sz, sz))
else:
    X=np.zeros((nsample, sz, sz, 3))

Y=np.zeros((nsample,)+ fh['labels'].shape[1:])
gen.fit(X[0:1])
flow = gen.flow(X, Y,  # Y
                batch_size=batch_size, 
                shuffle = shuffle,)
it = range(nbatch)
if rank==0:
    it = tqdm(it)
t3 = time()
for i in it:
    x, y = next(flow)
t4 = time()
dtinv = 1./(t4 - t3)
dtinv_avg = comm.allreduce(dtinv,op=MPI.SUM)
if comm.rank==0:
    print("*Throughput(flow_from_mem): %s imgs/s --  %s MB/s" %(nbatch*batch_size*dtinv_avg, nbatch*batch_size*dtinv_avg*sz*sz*3*16/1024/1024))

gen.fit(fh['data'][rank:rank+1])

flow = gen.flow(fh['data'][offset:offset+nsample],  # X
                fh['labels'][offset:offset+nsample],  # Y
                batch_size=batch_size,
                shuffle = shuffle,)
it = range(nbatch)
if rank==0:
    it = tqdm(it)
t3 = time()
for i in it:
    x, y = next(flow)
t4 = time()
dtinv = 1./(t4 - t3)
dtinv_avg = comm.allreduce(dtinv,op=MPI.SUM)
if comm.rank==0:
    print("*Throughput(flow): %s imgs/s --  %s MB/s" %(nbatch*batch_size*dtinv_avg, nbatch*batch_size*dtinv_avg*sz*sz*3*16/1024/1024))
fh.close()



try:
    fh = h5py.File('/scratch/%s'%args.hdf5_file, 'r')
    gen.fit(fh['data'][rank:rank+1])

    flow = gen.flow(fh['data'][offset:offset+nsample],  # X
                    fh['labels'][offset:offset+nsample],  # Y
                    batch_size=batch_size, 
                    shuffle = shuffle,)
    t3=time()
    it = range(nbatch)
    if rank==0:
        it = tqdm(it)

    for i in it:
        x, y = next(flow)
    t4 = time()
    dtinv = 1./(t4 - t3)
    dtinv_avg = comm.allreduce(dtinv,op=MPI.SUM)
    if comm.rank==0:
        print("*Throughput(flow, SSD): %s imgs/s  --  %s MB/s" %(nbatch*batch_size*dtinv_avg, nbatch*batch_size*dtinv_avg*sz*sz*3*16/1024/1024))

    nsample = fh['data'].shape[0] // size
    offset = nsample*rank

    df = gen.flow_from_hdf5(fh, shuffle=shuffle, batch_size=batch_size, nsample=nsample, offset=offset)
    if rank==0:
        print("Testing flow from HDF5 performance (SSD)")
    t3 = time()

    it = range(nbatch)
    if rank==0:
        it = tqdm(it)
        
    for i in it:
        x, y = next(df)
    t4 = time()
    dtinv = 1./(t4 - t3)
    dtinv_avg = comm.allreduce(dtinv,op=MPI.SUM)
    if comm.rank==0:
        print("*Throughput(flow_from_hdf5, SSD): %s imgs/s    --  %s MB/s" %(nbatch*batch_size*dtinv_avg, nbatch*batch_size*dtinv_avg*sz*sz*3*16/1024/1024))

except:
    if comm.rank==0:
        print("I could not do SSD test for some reason")
