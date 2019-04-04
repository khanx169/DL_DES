import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator
from hdf5_preprocessing import *
import h5py
from time import time
sz = 224
batch_size=16
datagen = ImageDataGenerator(rescale = 1./255)    
t0 = time()
print("Creating HDF5 file for train data: ")
hdf5_from_directory("./data/train.hdf5",
                    './data/train', datagen,
                    target_size=(sz, sz),
                    batch_size=batch_size,
                    class_mode="categorical",
                    interpolation='nearest')
t1 = time()
print(" time: %s second" %(t1-t0))

print("Creating HDF5 file for validation data: ")
hdf5_from_directory("val.hdf5",
                    './data/val', datagen,
                    target_size=(sz, sz),
                    batch_size=1,
                    class_mode="categorical",
                    interpolation='nearest')
t2 = time()
print(" time: %s second" %(t2-t1))
gen = HDF5ImageGenerator(horizontal_flip=True)
fh = h5py.File('./data/train.hdf5', 'r')
df = gen.flow_from_hdf5(fh)
t3 = time()
while i<df.n//batch_size:
    a = next(df)
    x, y = a
    i=i+1
t4 = time()
print(" time for flow from HDF5: %s " %(t4-t3))
