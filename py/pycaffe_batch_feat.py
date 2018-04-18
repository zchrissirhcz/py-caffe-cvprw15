#!/usr/bin/env python2
# coding: utf-8

"""
--------------------------------------------------------
"Deep Learning of Binary Hash Codes for Fast Image Retrieval"
PyCaffe implementation
Written by Zhuo Zhang (https://github.com/zchrissirhcz)
--------------------------------------------------------

description:
按batch提取图片的特征(only forward)
"""
from __future__ import print_function
caffe_dir = '/home/chris/work/caffe-BVLC'
#caffe_dir = '/opt/work/caffe-BVLC-cvprw15'
import sys, os
sys.path.insert(0, os.path.join(caffe_dir, 'python'))
import caffe
import time
import numpy as np
from prepare_batch import prepare_batch
import cPickle

def pycaffe_batch_feat(file_list, use_gpu, feat_len, model_def_file, model_file, root_dir):
    # Assume it is a file contaning the list of images
    fin = open(file_list)
    im_name_list = []
    for line in fin.readlines():
        t = line.rstrip()
        if t[0]=='/':
            t = t[1:]
        im_name = os.path.join(root_dir, t)
        im_name_list.append(im_name)
    fin.close()
    
    # Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
    batch_size = 10
    dim = feat_len
    num_im = len(im_name_list)

    if num_im % batch_size != 0:
        print('Assuming batches of {:d} images, rest({:d}) will be filled with zeros'.format(batch_size, num_im%batch_size))

    # init caffe network (spews logging info)
    if use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # load mean file
    mean_file_pth = os.path.join(root_dir, 'py', 'ilsvrc2012_mat_mean.pkl')
    fid = open(mean_file_pth, 'rb')
    mean = cPickle.load(fid).astype(np.float32)
    fid.close()
    # 执行四舍五入
    mean = np.round(mean.astype(float), 4)


    # prepare input
    scores = np.zeros((num_im, dim), dtype=float)
    num_batches = int(num_im/batch_size+0.5) # should be the same as ceiling
    initic=time.time()
    for bb in range(num_batches):
        rg = np.arange(batch_size*bb, min(num_im, batch_size*(bb+1)))  # range
        batch_im_name_list = []
        for i in rg:
            im_name = im_name_list[i]
            batch_im_name_list.append(im_name)

        input_data = prepare_batch(batch_im_name_list, mean, batch_size)

        t_consume = time.time() - initic
        print('Batch {:d} out of {:d} {:.2f}% Complete ETA {:.2f} seconds'.format( \
            bb, num_batches, (bb+1)*1.0/num_batches*10, t_consume/(bb+1)*(num_batches-bb) \
        ))
        net.blobs['data'].data[...] = input_data
        output_data = net.forward()

        raw_score = output_data['fc8_kevin_encode']
        ok_score = np.round(raw_score.astype(float), 4)
        scores[rg, :] = ok_score

    return scores