#!/usr/bin/env python2
# coding: utf-8
"""
--------------------------------------------------------
"Deep Learning of Binary Hash Codes for Fast Image Retrieval"
PyCaffe implementation
Written by Zhuo Zhang (https://github.com/zchrissirhcz)
--------------------------------------------------------
"""

from __future__ import print_function
import os, sys
import h5py  # use h5py to load v7.3 .mat files
from pprint import pprint
from precision import precision
from pycaffe_batch_feat import pycaffe_batch_feat
import matplotlib.pyplot as plt
import numpy as np
import cPickle

def load_label(label_file):
    fin = open(label_file)
    label = [float(_.rstrip()) for _ in fin.readlines()]
    fin.close()
    return label

def run():
    use_gpu = 1

    # top K returned images
    top_k = 1000
    feat_len = 48  # number of activation output on hidden layer

    # Absolute path of this project
    root_dir = '/opt/work/caffe-cvprw15'

    # set result folder
    result_folder = os.path.join(root_dir, 'analysis')

    # models
    model_file = os.path.join(root_dir, 'examples/cvprw15-cifar10/KevinNet_CIFAR10_48.caffemodel')
    model_def_file = os.path.join(root_dir, 'examples/cvprw15-cifar10/KevinNet_CIFAR10_48_deploy.prototxt')

    # train-test
    test_file_list = os.path.join(root_dir, 'examples/cvprw15-cifar10/dataset/test-file-list.txt')
    test_label_file = os.path.join(root_dir, 'examples/cvprw15-cifar10/dataset/test-label.txt')
    train_file_list = os.path.join(root_dir, 'examples/cvprw15-cifar10/dataset/train-file-list.txt')
    train_label_file = os.path.join(root_dir, 'examples/cvprw15-cifar10/dataset/train-label.txt')

    # outputs
    #feat_test_file = '{:s}/feat-test.mat'.format(result_folder)
    #feat_train_file = '{:s}/feat-train.mat'.format(result_folder)
    binary_test_file = '{:s}/binary-test.mat'.format(result_folder)
    binary_train_file = '{:s}/binary-train.mat'.format(result_folder)

    py_feat_test_file = '{:s}/feat-test.pkl'.format(result_folder)
    py_feat_train_file = '{:s}/feat-train.pkl'.format(result_folder)
    py_binary_test_file = '{:s}/binary-test.pkl'.format(result_folder)
    py_binary_train_file = '{:s}/binary-train.pkl'.format(result_folder)

    USE_MAT_FEAT = False # whether load MATLAB code generated and saved feature files, or not

    # feature extraction- test set
    if USE_MAT_FEAT and os.path.exists(binary_test_file): # load .mat file
        data = h5py.File(binary_test_file)
        binary_test = data['binary_test'][:]  # 10000 x 48
        #binary_test = binary_test.T  # transpose　转置 => 48 x 1000
    elif os.path.exists(py_binary_test_file):
        fid = open(py_binary_test_file, 'rb')
        binary_test = cPickle.load(fid)
        fid.close()
    else:
        feat_test = pycaffe_batch_feat(test_file_list, use_gpu, feat_len, model_def_file, model_file, root_dir)
        binary_test = (feat_test>0.5).astype(int)
        # save extracted features to disk
        with open(py_feat_test_file, 'wb') as fid:
            cPickle.dump(feat_test, fid)
        with open(py_binary_test_file, 'wb') as fid:
            cPickle.dump(binary_test, fid)

    # feature extraction- train set
    if USE_MAT_FEAT and os.path.exists(binary_train_file): # load .mat file
        data = h5py.File(binary_train_file)
        binary_train = data['binary_train'][:]  # 10000 x 48
        #binary_train = binary_train.T  # transpose　转置 => 48 x 1000
    elif os.path.exists(py_binary_train_file):
        fid = open(py_binary_train_file, 'rb')
        binary_train = cPickle.load(fid)
        fid.close()
    else:
        feat_train = pycaffe_batch_feat(train_file_list, use_gpu, feat_len, model_def_file, model_file, root_dir)
        binary_train = (feat_train>0.5).astype(int)
        # save extracted features to disk
        with open(py_feat_train_file, 'wb') as fid:
            cPickle.dump(feat_train, fid)
        with open(py_binary_train_file, 'wb') as fid:
            cPickle.dump(binary_train, fid)

    # load training images' labels. double type
    trn_label = load_label(train_label_file)
    tst_label = load_label(test_label_file)

    mAP, precision_at_k = precision(trn_label, binary_train, tst_label, binary_test, top_k, 1)

    print('mAP = {:f}'.format(mAP))
    saved_result = {
        'mAP': mAP,
        'precision_at_k': precision_at_k
    }

    save_file = os.path.join(root_dir, 'saved_result.pkl')
    fid = open(save_file, 'wb')  
    cPickle.dump(saved_result,fid)
    fid.close()

def plot():
    fid = open('saved_result.pkl', 'rb')
    saved_result = cPickle.load(fid)
    fid.close()
    mAP = saved_result['mAP']
    precision_at_k = saved_result['precision_at_k']
    x = np.arange(len(precision_at_k))
    plt.plot(x, precision_at_k, label='Ours', linewidth=2)
    plt.title('Image Retrieval Precision of CIFAR-10')
    plt.xlabel('# of Top Images Retrieved')
    plt.ylabel('Precision')

    alg_name_list = ['BRE', 'CNNH+', 'CNNH', 'ITQ-CCA', 'ITQ', 'KSH', 'MLH', 'SH']
    mk = ['.', 'o', '^', 'D', '*', 'v', 'p', '+' ]
    ls = [':', '-', '-.', '--', ':', '-', '-.', '--']
    for i, alg_name in enumerate(alg_name_list):
        alg_precion_file = 'analysis/{:s}-p-at-k.txt'.format(alg_name)
        x = []
        y = []
        with open(alg_precion_file) as fin:
            for line in fin.readlines():
                t = line.rstrip().split(' ')
                x.append(int(t[0]))
                y.append(float(t[1]))
        plt.plot(x, y, label=alg_name, linewidth=2, marker=mk[i], linestyle=ls[i])
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0.1, 1.1, 0.1))
    plt.show()

if __name__ == '__main__':
    #run()
    plot()
