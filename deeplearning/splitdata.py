#!/usr/bin/env python
import os, sys
def SplitData(source, dest, hvd):
    rank=hvd.rank()
    size=hvd.size()
    try:
        if (rank==0):
            os.mkdir(dest)
    except:
        print(dest+"/hvd-*/does not exist. Will create")
    hvd.allreduce([0])
    try:
        os.mkdir(dest+'/hvd-%s/'%rank)
    except:
        print("Could not create hvd-%s"%rank)
    dir_list = next(os.walk(source))[1]
    for subdir in dir_list:
        if (rank==0):
            print("Creating symlinks to files from %s"%source)
        files = os.listdir(source+"/"+subdir)
        if (rank==0):
            print("%s/%s: %s files"%(source, subdir, len(files)))
        destdir=dest+'/hvd-%s/%s'%(rank, subdir)
        try:
            os.mkdir(dest+'/hvd-%s/%s'%(rank, subdir))
        except:
            print("hvd-%s exists"%rank)
        RemoveSplitData(dest, subdir, hvd)
        for i in range(rank, len(files), size):
            f = files[i]
            os.symlink(source+"/"+subdir+"/%s"%f, destdir+"/%s"%f)
    hvd.allreduce([0])
    return dest+"/hvd-%s/"%rank

def RemoveSplitData(dest, subdir, hvd):
    destdir = dest+'/hvd-%s/%s/'%(hvd.rank(), subdir)
    for f in os.listdir(destdir):
        os.unlink(destdir+f)

