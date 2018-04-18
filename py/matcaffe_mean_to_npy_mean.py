#!/usr/bin/env python2
# coding: utf-8
"""
--------------------------------------------------------
"Deep Learning of Binary Hash Codes for Fast Image Retrieval"
PyCaffe implementation
Written by Zhuo Zhang (https://github.com/zchrissirhcz)
--------------------------------------------------------
"""

import h5py

import scipy.io as scio
import cPickle, os


f_pt = '/opt/work/caffe-cvprw15/matlab/caffe/ilsvrc_2012_mean.mat'

#data = h5py.File(f_pt, 'r')
data = scio.loadmat(f_pt)
mean = data['image_mean']

save_file = os.path.join('ilsvrc2012_mat_mean.pkl')
fid = open(save_file, 'wb')  
cPickle.dump(mean, fid)
fid.close()

print('\nDone')