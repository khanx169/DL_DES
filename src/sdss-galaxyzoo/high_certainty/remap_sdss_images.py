import numpy as np
from skimage.io import imsave
from os import path, makedirs
from glob import glob


def rescale(img_in, filters=('g', 'r', 'i', 'z')):
    out = {}
    for f in filters:
        img = img_in[f].astype('float')
        img_bg_sorted = np.sort(img)[:img.size // 2]
        img_bg_std = img_bg_sorted.std()
        img_bg_mean = img_bg_sorted.mean()
        img_min = - img_bg_std + img_bg_mean
        img_max = img_bg_std * 5 + img_bg_mean
        img_scaled = (img - img_min) / (img_max - img_min)
        img_scaled[img_scaled < 0] = 0
        img_scaled[img_scaled > 1] = 1
        out[f] = img_scaled
    return out


def merge_pseudocolor(ds, filters):
    nrows, ncols = ds[list(ds.keys())[0]].shape
    rgb = np.empty((nrows, ncols, 3))
    rgb[:, :, 0] = ds[filters[0]]
    rgb[:, :, 1] = ds[filters[1]]
    rgb[:, :, 2] = ds[filters[2]]
    rgb = (rgb * 255).astype('uint8')
    return rgb


def process_image(npz_in, png_out):
    ds = np.load(npz_in)
    scaled = rescale(ds, ('z', 'r', 'g'))
    rgb = merge_pseudocolor(scaled, ('z', 'r', 'g'))
    imsave(png_out, rgb)


if __name__ == '__main__':
    in_list = glob('../../../data/sdss-galaxyzoo/high_certainty/images/resized/*.npz')
    out_dir = '../../../data/sdss-galaxyzoo/high_certainty/images/remapped'
    if not path.exists(out_dir):
        makedirs(out_dir)
    in_list.sort()

    from task_pool import *

    def work(in_file):
        objid = path.basename(in_file).split('.')[0]
        out_file = path.join(out_dir, '%s.png' % objid)
        process_image(in_file, out_file)

    exe = MPITaskPool()
    exe.run(in_list, work, log_freq=1000)
