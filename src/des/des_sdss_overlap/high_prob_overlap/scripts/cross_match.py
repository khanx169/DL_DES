import pandas as pd
import numpy as np
from os import path, makedirs, listdir
from os.path import isfile, join
import glob

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


def CrossMatch(df_sdss, df_des):
    
    df_des.rename(columns={'COADD_OBJECT_ID': 'DES_COADD_OBJECT_ID'}, inplace=True)
    
    coo_sdss = SkyCoord(np.array(df_sdss['ra'])*u.deg, np.array(df_sdss['dec'])*u.deg)
    coo_des = SkyCoord(np.array(df_des['RA'])*u.deg, np.array(df_des['DEC'])*u.deg)
    
    idx_des, d2d_des, d3d_des = coo_sdss.match_to_catalog_sky(coo_des)
    
    match_idx_des = idx_des[ d2d_des.arcsec <= 1 ]
    match_idx_sdss = np.argwhere( d2d_des.arcsec <= 1 )
    match_idx_sdss = np.squeeze(match_idx_sdss)
    
    match_d2d = d2d_des[ d2d_des.arcsec <= 1 ].arcsec
    
    sdss_tmp = sdss.loc[match_idx_sdss].reset_index(drop=True)
    des_tmp = des.loc[match_idx_des].reset_index(drop=True)
    
    merged = pd.merge(sdss_tmp, des_tmp, how='inner', left_index=True, right_index=True, suffixes=['_sdss', '_des'])
    
    merged['d2d_des' ] = match_d2d
    
    return merged




if __name__ == '__main__':
    
    sdss_high_prob_dataset_path = 'data/sdss-galaxyzoo/high_certainty/merged_dataset.csv'
    des_ret_sql_query_dataset_path = 'src/des/des_sdss_overlap/high_prob_overlap/metadata/ret_sql_query/'
    out_dir = 'src/des/des_sdss_overlap/high_prob_overlap/metadata/crossmatch/'
    
    des_overlap_dbs = [f for f in listdir(des_ret_sql_query_dataset_path) if isfile(join(des_ret_sql_query_dataset_path, f))]
    
    sdss = pd.read_csv(sdss_high_prob_dataset_path)
    sdss.rename(columns={'OBJID': 'SDSS_OBJID'}, inplace=True)
    
    
    for dfs in des_overlap_dbs:
        
        out_path  = path.join(out_dir, 'cross_match_%s' % dfs.split('_')[-1])
        if path.exists(out_path):
            continue
       
    
        des = pd.read_csv(des_ret_sql_query_dataset_path + dfs)        
        try:
            merged = CrossMatch(sdss, des)
        except:
            print('FAILED:', dfs)
            continue
            
        merged.to_csv(out_path)