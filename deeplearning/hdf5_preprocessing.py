import keras
from keras.preprocessing.image import ImageDataGenerator, Iterator

from keras_preprocessing import image

import h5py
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
            
def hdf5_from_directory(fname, directory, datagen,
                        target_size=(256, 256),
                        color_mode='rgb',
                        classes=None,
                        shuffle=True,
                        class_mode='categorical',
                        batch_size=32,
                        data_format='channels_last',
                        interpolation='nearest',
                        dtype=None, nsample=None, verbose=1, mpi=False, chunks=False):
    if (verbose!=0):
        print('-----------------------------------')
        print("Creating HDF5 from %s: " %directory)
    dataflow = image.DirectoryIterator(
        directory,datagen,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_links=False,
        interpolation=interpolation)
    if nsample==None:
        nsample = dataflow.n
    rank=0;size=1
    if (mpi):
        comm = MPI.COMM_WORLD
        f=h5py.File(fname, 'w', driver='mpio', comm=comm)
        rank = comm.rank
        size = comm.size
        if (rank==0):
            print("Writing HDF5 files using %s processors"%size)
    else:
        f=h5py.File(fname, 'w')
    x, y = dataflow[rank]
    #    f.create_dataset('filenames', shape=(dataflow.n, 1), dtype='S32')
    x_shape = (nsample, ) + x.shape[1:]
    y_shape = (nsample, ) + y.shape[1:]
    if chunks:
        ds = f.create_dataset('data', shape=x_shape, dtype=dtype, chunks=chunks)
    else:
        ds = f.create_dataset('data', shape=x_shape, dtype=dtype)
    ds.attrs['data_format'] = data_format
    ds.attrs['shape'] = x_shape
    ds.attrs['image_shape'] = target_size
    if chunks:
        ys = f.create_dataset('labels', shape=y_shape, dtype=np.uint8, chunks=True)
    else:
        ys = f.create_dataset('labels', shape=y_shape, dtype=np.uint8)
    ys.attrs['shape'] = y_shape
    nb_global = nsample//batch_size
    nb_local = nb_global//size
    nb_offset = nb_local*rank
    if (rank<nb_global%size):
        nb_local+=1
        nb_offset+=rank
    it = range(nb_offset, nb_offset+nb_local)
    if (verbose!=0):
        it = tqdm(it)
    for i in it:
        x, y = dataflow[i]
        #       f['filenames'][i*batch_size:(i+1)*batch_size, 0]=dataflow.filenames[i*batch_size:(i+1)*batch_size]
        f['data'][i*batch_size:(i+1)*batch_size] = x
        f['labels'][i*batch_size:(i+1)*batch_size] = y
    if (nsample%batch_size!=0 and rank==size-1):
        f['data'][nsample-nsample%batch_size:nsample] = x[:nsample%batch_size]
        f['labels'][nsample-nsample%batch_size:nsample] = y[:nsample%batch_size]
    f.close()
    print('-----------------------------------')

class HDF5ArrayIterator(Iterator):
    def __init__(self, fh,
                 image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 dtype=float,
                 offset=0,
                 nsample=None,
                 class_mode='categorical',
                 
    ):
        self.dtype=dtype
        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight)
        else:
            self.sample_weight = None
        self.image_data_generator = image_data_generator
        self.data_format = fh['data'].attrs['data_format']
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.class_mode = class_mode
        if nsample==None:
            nsample = fh['data'].shape[0]
        self.offset = offset
        self.hdf5 = fh
        self.image_shape = fh['data'].attrs['image_shape']
        self.x_shape=tuple(fh['data'].attrs['shape'])
        self.y_shape=tuple(fh['labels'].attrs['shape'])
        super().__init__(nsample,
                         batch_size,
                         shuffle,
                         seed)

        x_shape = (self.n,)+self.x_shape[1:]
        y_shape = (self.n,)+self.y_shape[1:]
        self.batch_x = np.zeros(x_shape, dtype=self.dtype)
        self.batch_y = np.zeros(y_shape, dtype=np.uint8)

    def _set_index_array(self):
        ''' Overwrite the _set_index_array_ in array generator to make it to use multiple rank'''
        self.index_array = np.arange(self.offset, self.offset + self.n, dtype=int)
        if self.shuffle:
            np.random.shuffle(self.index_array)
    
    def _get_batches_of_transformed_samples(self, index_array):
        for i, j in enumerate(index_array):
            self.batch_x[i] = self.hdf5['data'][j]
            self.batch_y[i] = self.hdf5['labels'][j]
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(self.batch_x[i].shape)
                x = self.image_data_generator.apply_transform(self.batch_x[i], params)
                x = self.image_data_generator.standardize(self.batch_x[i])
            self.batch_x[i] = x
#            self.batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.sample_weight is None:
            return self.batch_x, self.batch_y
        else:
            return self.batch_x, self.batch_y, self.sample_weight[index_array]

class HDF5ImageGenerator(ImageDataGenerator):
    def __init__(self,
	         featurewise_center=False,
	         samplewise_center=False,
	         featurewise_std_normalization=False,
	         samplewise_std_normalization=False,
	         zca_whitening=False,
	         zca_epsilon=1e-6,
	         rotation_range=0,
	         width_shift_range=0.,
	         height_shift_range=0.,
	         brightness_range=None,
	         shear_range=0.,
	         zoom_range=0.,
	         channel_shift_range=0.,
	         fill_mode='nearest',
	         cval=0.,
	         horizontal_flip=False,
	         vertical_flip=False,
	         rescale=None,
	         preprocessing_function=None,
	         data_format='channels_last',
	         validation_split=0.0,
#	         interpolation_order=1,
	         dtype=None):
        super().__init__(featurewise_center=featurewise_center,
		         samplewise_center=samplewise_center,
		         featurewise_std_normalization=featurewise_std_normalization,
		         samplewise_std_normalization=samplewise_std_normalization,
		         zca_whitening=zca_whitening,
		         zca_epsilon=zca_epsilon,
		         rotation_range=rotation_range,
		         width_shift_range=width_shift_range,
		         height_shift_range=height_shift_range,
		         brightness_range=brightness_range,
		         shear_range=shear_range,
		         zoom_range=zoom_range,
		         channel_shift_range=channel_shift_range,
		         fill_mode=fill_mode,
		         cval=cval,
		         horizontal_flip=horizontal_flip,
		         vertical_flip=vertical_flip,
		         rescale=rescale,
		         preprocessing_function=preprocessing_function,
		         data_format=data_format,
		         validation_split=validation_split,
#		         interpolation_order=interpolation_order,
		         dtype=dtype)
    def flow_from_hdf5(self, fh, batch_size=32,
                       shuffle=False,
                       sample_weight=None,
                       seed=None,
                       save_to_dir=None,
                       save_prefix='',
                       save_format='png',
                       subset=None,
                       dtype=None, offset=0, nsample=None):
        return HDF5ArrayIterator(fh, self,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sample_weight=sample_weight,
                                 seed=seed,
                                 save_to_dir=save_to_dir,
                                 save_prefix=save_prefix,
                                 save_format=save_format,
                                 subset=subset,
                                 dtype=dtype,
                                 offset=offset,
                                 nsample=nsample)
class DirectoryIteratorOffset(image.DirectoryIterator):
    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32', offset=0, nsample=None):
        self.offset=offset
        self.nsample=nsample
        super().__init__(directory,
                         image_data_generator=image_data_generator,
                         target_size=target_size,
                         color_mode=color_mode,
                         classes=classes,
                         class_mode=class_mode,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         seed=seed,
                         data_format=data_format,
                         save_to_dir=save_to_dir,
                         save_prefix=save_prefix,
                         save_format=save_format,
                         follow_links=follow_links,
                         subset=subset,
                         interpolation=interpolation,
                         dtype=dtype)
        if (nsample==None):
            self.nsample=self.n
    def _set_index_array(self):
        ''' Overwrite the _set_index_array_ in array generator to make it to use multiple rank'''
        self.index_array = np.arange(self.offset, self.offset + self.nsample)
        if self.shuffle:
            np.random.shuffle(self.index_array)
