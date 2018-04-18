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
import numpy as np
import time
from pprint import pprint

def pdist2(v1, v2, typ):
	"""
	v1: m x h    m: number of example images   h: dimension of hidden layer (e.g. 48)
	v2: 1 x h
	typ: type of distance. supported: 'hamming', 'euclidean'

	在信息理论中，Hamming Distance 表示两个等长字符串在对应位置上不同字符的数目
	"""
	if typ == 'euclidean':
		return abs(v1 - v2)
	if typ == 'hamming':
		tt = 1.0 * np.sum(v1!=v2, axis=1) / v2.shape[0]
		return tt


def precision(trn_label, trn_binary, tst_label, tst_binary, top_k, mode):
	"""
	trn_binary: h x m   h: dimension of hidden layer (e.g. 48)   m: number of example images (e.g. 50000)
	"""
	K = top_k
	QueryTimes = tst_binary.shape[1]  # number of query images

	AP = np.zeros((QueryTimes, 1), dtype=float).squeeze()

	# Ns = 1:1:K  # from 1, to K (include), step is 1
	Ns = np.arange(1, K+1, 1)

	sum_tp = np.zeros((1, len(Ns)), dtype=float).squeeze()

	for i in range(0, QueryTimes):
		query_label = tst_label[i]
		print('query {:d}\n'.format(i+1))  # in python all index from 0. but for print we should plus 1
		query_binary = tst_binary[:,i]
		if mode==1:
			t_start = time.time()
			similarity = pdist2(trn_binary.T, query_binary.T, 'hamming')
			t_consume = time.time() - t_start
			print('Complete Query [Hamming] {:.2f} seconds'.format(t_consume))
		elif mode ==2:
			t_start = time.time()
			similarity = pdist2(trn_binary.T, query_binary.T, 'euclidean')
			t_consume = time.time() - t_start
			print('Complete Query [Euclidean] {:.2f} seconds'.format(t_consume))

		y2=np.argsort(similarity, kind='mergesort')
		
		buffer_yes = np.zeros((K,1)).squeeze()
		buffer_total = np.zeros((K,1))
		total_relevant = 0
		
		for j in range(0, K):
			retrieval_label = trn_label[y2[j]]
			if (query_label==retrieval_label):
				buffer_yes[j] = 1
				total_relevant = total_relevant + 1
			buffer_total[j] = 1
		
		# compute precision
		P = np.cumsum(buffer_yes) / Ns.T
		
		if (buffer_yes.sum() == 0):
			AP[i] = 0
		else:
			AP[i] = np.sum(P*buffer_yes) / buffer_yes.sum()
		
		sum_tp = sum_tp + np.cumsum(buffer_yes)

	precision_at_k = sum_tp / (Ns * QueryTimes)
	mAP = AP.mean()

	return mAP, precision_at_k


