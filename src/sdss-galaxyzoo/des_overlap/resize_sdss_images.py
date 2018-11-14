import numpy as np
from skimage.transform import resize
from os import path
from glob import glob
from task_pool import *


## Configs
indir = '../../../data/sdss-galaxyzoo/des_overlap/images/clipped/'
outdir = '../../../data/sdss-galaxyzoo/des_overlap/images/resized/'
tgt_dim = (200, 200)


## Fetch tasks
files_all = glob(path.join(indir, '*.npz'))
#files_todo = [file for file in files_all 
#              if not path.exists(path.join(outdir, path.basename(file)))]
files_todo = files_all
files_todo.sort()


## Work
def work(file):
    in_ds = np.load(file)
    try:
        out_ds = {
            k: resize(in_ds[k].astype('float'), tgt_dim).astype('uint16') 
                for k in in_ds
        }
        out_path = path.join(outdir, path.basename(file))
        np.savez(out_path, **out_ds)
    except Exception as e:
        print('FAILED: %s %s' % (file, str(e)))

exe = MPITaskPool()
exe.run(files_todo, work, log_freq=100)
