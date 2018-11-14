import pandas as pd
from task_pool import *
from os import path, makedirs
from subprocess import call


filters = ['u', 'g', 'r', 'i', 'z']
out_dir = '../../../data/sdss-galaxyzoo/high_certainty/images/raw/'
ds_path = '../../../data/sdss-galaxyzoo/high_certainty/merged_dataset.csv'


def work(config):
    objid, filter_, entry = config
    run, rerun, camcol, field = entry.run, entry.rerun, entry.camcol, entry.field
    out_path = path.join(out_dir,
                'run%d-rerun%d-camrol%d-field%d-%s.fits'
                    % (run, rerun, camcol, field, filter_))
    req_url = ('http://das.sdss.org/cgi-bin/drC?RUN=%d&RERUN=%d&CAMCOL=%d&FIELD=%d&FILTER=%s'
                    % (run, rerun, camcol, field, filter_))
    # if path.exists(out_path):
    #     print('skipping', objid, filter_)
    #     continue
    # print(objid, filter_)
    retval = call(['wget', req_url, '-O', out_path, '-q'])
    if retval != 0:
        print('FAILED:', objid, req_url, out_path, retval)

exe = MPITaskPool()

if exe.is_parent():
    df = pd.read_csv(ds_path).set_index('OBJID')
    objids = sorted(list(set(df.index.data)))
    configs = [(objid, filter_, df.loc[objid])
                    for objid in objids for filter_ in filters]
else:
    configs = None

exe.run(configs, work, log_freq=100)
