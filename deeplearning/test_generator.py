import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator
from hdf5_preprocessing import *
import h5py

sz = 24
batch_size=16
datagen = ImageDataGenerator(rescale = 1./255)    
hdf5_from_directory("train.hdf5",
                    './train', datagen,
                    target_size=(sz, sz),
                    batch_size=batch_size,
                    class_mode="categorical",
                    interpolation='nearest')
hdf5_from_directory("val.hdf5",
                    './val', datagen,
                    target_size=(sz, sz),
                    batch_size=batch_size,
                    class_mode="categorical",
                    interpolation='nearest')
gen = HDF5ImageGenerator(horizontal_flip=True)
fh = h5py.File('train.hdf5', 'r')
df = gen.flow_from_hdf5(fh)
i=0
while i<4:
    a = next(df)
    x, y = a
    print(a)
    i=i+1
