import numpy as np
import pandas as pd
from os import path, makedirs,listdir
from os.path import isfile, join
from subprocess import call
from sys import argv, exit


PATH = '/home/khan74/project/priv/new_DL_DES/'


if __name__ == '__main__':

    outpath = PATH + '/data/des/unlabelled/raw/'

    unlabelled_test_set = pd.read_csv( PATH + 'deeplearning/data/unlabelled_test_set.csv')
    tilenames = sorted( np.array( unlabelled_test_set.TILENAME.unique() ) )


    print('len tilenames: ', len(tilenames))
    n = 0
    for tilename in tilenames[4000:5000]:
        req_url = 'http://desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/%s/' % (tilename)
        retval = call(['wget', '-r', '-nd', 'robots=off', '-np', '-R', "index.html*", req_url, '--directory-prefix='+outpath, '-q'])
        if retval != 0:
            print('FAILED:', tilename, ', retval:', retval)
        n += 1

    print('downloaded number of tiles: ', n)
    print('last tile: ', tilename)
