#!/usr/bin/env python
# Testing I/O throughput 
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator
from hdf5_preprocessing import *
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
parser.add_argument("--hdf5_file", default='train_nochunk.hdf5')

args = parser.parse_args()

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
                         rotation_range=45)

fh = h5py.File(PATH+'/deeplearning/data/'+args.hdf5_file, 'r')
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


X=np.zeros((nsample, sz, sz, 3))
Y=np.zeros((nsample,)+ fh['labels'].shape[1:])
gen.fit(X)

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

try:
    fh = h5py.File('/scratch/train_nochunk.hdf5', 'r')
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
