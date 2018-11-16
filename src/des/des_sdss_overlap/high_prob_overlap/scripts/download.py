import numpy as np
import pandas as pd
from os import path, makedirs,listdir
from os.path import isfile, join
from subprocess import call
from sys import argv, exit


PATH = '/home/khan74/project/priv/new_DL_DES/'

if __name__ == '__main__':
    
    outpath = PATH + '/data/des/des_sdss_overlap/high_prob_overlap/raw/'
    
    high_prob_crossmatch_test_set = pd.read_csv( PATH + 'deeplearning/data/high_prob_crossmatch_test_set.csv')
    tilenames = np.array( high_prob_crossmatch_test_set.TILENAME.unique() )
    
    
    print('len tilenames: ', len(tilenames))
    n = 0
    for tilename in tilenames:
        req_url = 'http://desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/%s/' % (tilename)
        retval = call(['wget', '-r', '-nd', 'robots=off', '-np', '-R', "index.html*", req_url, '--directory-prefix='+outpath, '-q'])
        if retval != 0:
            print('FAILED:', tilename, ', retval:', retval)
        n += 1
        
    print('downloaded number of tiles: ', n)
