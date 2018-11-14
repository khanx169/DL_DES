import pandas as pd
import numpy as np
from os import path, makedirs
from glob import glob
from astropy.io import fits
from astropy.wcs import WCS
from task_pool import *


def clip_img_from_fits(fits_path, ra, dec, rad, pxl_size=0.396, dim=2):
    """
    Extract a small square image containing the target galaxy from a large telescope image.
    
    Parameters:
    fits_path -- path to the input FITS file. This assumes the image data is contained in
                 the 0th entry of the FITS file.
    ra, dec   -- position of the galaxy in the sky.
    rad       -- angular radius of the galaxy in arcseconds.
    pxl_size  -- the size of each pixel in arcseconds. The SDSS FITS files come in 0.396"
                 (see http://classic.sdss.org/dr7/instruments/imager/).
    dim       -- the size of the bounding box in unit of `side_length / galaxy_diameter`.
    
    Returns:
    The clipped galaxy image as a 2d numpy array.
    """
    ds = fits.open(fits_path)
    w = WCS(ds[0].header, ds)
    rad_pxl = rad / pxl_size
    # index naming: r/c: row/col; c/s/e: center/start/end
    cc, rc = w.all_world2pix([[ra, dec]], 1)[0]
    rs, re = [rc + sign * dim * rad_pxl for sign in (-1, 1)]
    cs, ce = [cc + sign * dim * rad_pxl for sign in (-1, 1)]
    rs, re, cs, ce = [int(round(x)) for x in (rs, re, cs, ce)]
    clip = ds[0].data[rs:re, cs:ce]
    return clip


def clip_galaxy_imgs(md_entry, in_fmt, filters=('u', 'g', 'r', 'i', 'z'), pxl_size=0.396, dim=2):
    """
    Extract the image of a galaxy from a larger telescope image.

    Parameters:
    md_entry  -- the row entry of the galaxy in the merged GalaxyZoo-SDSS dataset.
    in_fmt    -- a formating string containing `{run}`, `{rerun}`, `{camcol}`, `{field}`, and
                 `{filter}` such that when replaced properly it should be the path to the 
                 corresponding FITS file.
    filters   -- an iterable containing the bands to extract.
    pxl_size  -- the size of each pixel in arcseconds. The SDSS FITS files come in 0.396"
                 (see http://classic.sdss.org/dr7/instruments/imager/).
    dim       -- the size of the bounding box in unit of `side_length / galaxy_diameter`.
    
    Retruns:
    A set of band_name-image_array pairs as a dictionary.
    """
    data = {
        filter: clip_img_from_fits(
            in_fmt.format(run=md_entry.run, rerun=md_entry.rerun, camcol=md_entry.camcol,
                          field=md_entry.field, filter=filter),
            md_entry.ra, md_entry.dec, md_entry.petroRad_r
        )
        for filter in filters
    }
    return data


if __name__ == '__main__':
    ## Configs
    in_format = '../../../data/sdss-galaxyzoo/high_certainty/images/raw/' \
              + 'run{run}-rerun{rerun}-camrol{camcol}-field{field}-{filter}.fits'
    out_dir = '../../../data/sdss-galaxyzoo/high_certainty/images/clipped/'
    galaxy_ds_path = '../../../data/sdss-galaxyzoo/high_certainty/merged_dataset.csv'

    ## Read galaxy dataset
    galaxy_ds = pd.read_csv(galaxy_ds_path).set_index('OBJID')
    objids = list(galaxy_ds.index)

    ## Parallelizing
    exe = MPITaskPool()
    
    def wrapper(objid):
        entry = galaxy_ds.loc[objid]
        try:
            data = clip_galaxy_imgs(entry, in_format)
            out_path = path.join(out_dir, '%d.npz' % objid)
            np.savez(out_path, **data)
        except:
            print('FAILED:', objid)

    exe.run(objids, wrapper, log_freq=100)

