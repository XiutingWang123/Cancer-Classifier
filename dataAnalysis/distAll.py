# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:55:17 2017

@author: yyc
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn import preprocessing
import seaborn.apionly as sns

"""
Compute standard deviation of CDN and SN method
"""

def std(cn, ct, sn, st):
	cn = np.array(cn)
	ct = np.array(ct)
	sn = np.array(sn)
	st = np.array(st)
	
	cnStd = []
	ctStd = []
	snStd = []
	stStd = []
	
	for i in range(217):
		cn_s =  np.std(cn[0:, i])
		ct_s =  np.std(ct[0:, i])
		sn_s =  np.std(sn[0:, i])
		st_s =  np.std(st[0:, i])
		
		#print("{0} CDN: t: {1:.4f}, p: {2:.4f}   SN: t:{3:.4f}, p: {4:.4f}".format(i+1, t_c, p_c, t_s, p_s))
		cnStd.append(cn_s)
		ctStd.append(ct_s)
		snStd.append(sn_s)
		stStd.append(st_s)
	
	
		
	#graph
	
	fig, ax = plt.subplots(figsize=(20,5))
	x = np.arange(217)
	bar_width = 0.35
	ax.set_axis_bgcolor('white')
	rect1 = plt.bar(x, ctStd, bar_width, color='c',label='CDN', alpha=.6)
	rect2 = plt.bar(x+bar_width, stStd, bar_width,  color='violet', label='SN', alpha=.6)
	
	plt.legend(loc='upper right')
	plt.xlim(0,217)
	plt.tight_layout()


"""
Compute t statistic and p value of CDN and SN method
"""

def ttest(cn, ct, sn, st):
	cn = np.array(cn)
	ct = np.array(ct)
	sn = np.array(sn)
	st = np.array(st)
	
	
	t_CDN = []
	t_SN = []
	p_CDN = []
	p_SN = []
	
	CDN_small = 0; SN_small = 0; CDN_SN = 0; CDN_05 = 0; SN_05 = 0
	
	
	for i in range(217):
		t_c, p_c =  stats.ttest_ind(cn[0:, i], ct[0:, i], equal_var=False)
		t_s, p_s =  stats.ttest_ind(sn[0:, i], st[0:, i], equal_var=False)
		np.seterr(divide='ignore', invalid='ignore')	
		
		#string = ""		
		if p_c < p_s:
			CDN_small += 1
		elif p_c > p_s:
			SN_small +=1
		else:
			CDN_SN +=1
		
		if p_c <= 0.05:
			CDN_05 += 1
			#string += "case {0}".format(i)
		if p_s <= 0.05:
			SN_05 += 1
			#string += "stand {0}".format(i)
		
		#print(string)		
		#print("{0} CDN: t: {1:.4f}, p: {2:.4f}   SN: t:{3:.4f}, p: {4:.4f}".format(i+1, t_c, p_c, t_s, p_s))
		t_CDN.append(t_c)
		t_SN.append(t_s)
		p_CDN.append(p_c)
		p_SN.append(p_s)
	
	#print("CDN<SN: {0}, SN<CDN: {1}  CDN=SN: {2}, CDN<=0.05: {3}, SN<=0.05: {4} ".format(CDN_small, SN_small, CDN_SN, CDN_05, SN_05))
	
	mpl.rc('font', family='Helvetica')
	fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,4))
	data_t = np.vstack([t_CDN, t_SN]).T
	bins_t = np.linspace(-10, 15, 10)
	data_p = np.vstack([p_CDN, p_SN]).T
	bins_p = np.linspace(-0.5, 1.5, 20)
	
	colors = ['lightcoral', 'skyblue']	
	
	axes[0].hist(data_t, bins_t, alpha=0.7,color=colors, label=['CDN','SN'])
	axes[0].legend(prop={'size': 14})
	axes[0].set_xlabel('t statistic', fontsize=20)
	axes[0].set_ylabel('Number of miRNAs', fontsize=20)
	axes[1].hist(data_p, bins_p, alpha=0.7, color=colors, label=['CDN','SN'])
	axes[1].legend(prop={'size': 14})
	axes[1].set_xlabel('p value',fontsize=20)
	
		
	
	for ax in axes:
		ax.get_yaxis().set_tick_params(which='both', direction='out', right='off', labelsize=14)
		ax.get_xaxis().set_tick_params(which='both', direction='out',top='off', labelsize=14)
		
		
	
	fig.tight_layout()
	fig.savefig('hist_pt.jpg', dpi=1200)
	
	
	



def main():
	df = pd.read_csv('/Users/yyc/Desktop/classifier/miRNA_train.txt')
	df = df.as_matrix()
	df = df[0:,1:]	
	CDN_N = []
	CDN_T = []
	
	for item in df:
		if item[-1] == 0:
			CDN_N.append(item)
		else:
			CDN_T.append(item)
	
	
		
	df2 = pd.read_csv('/Users/yyc/Desktop/classifier/miRNA_combine_raw.txt')
	X_all = df2.ix[0:120,2:219]
	minmax_scaler = preprocessing.MinMaxScaler()
	X_minmax = minmax_scaler.fit_transform(X_all)
	X_train = X_minmax[0:94,:]
	y = df2.ix[0:93,'target']
	y = y.as_matrix()
	
	X_all = np.hstack((X_train, np.atleast_2d(y).T))
		
	SN_N = []
	SN_T = []
	
	for item in X_all:
		if item[-1] == 0:
			SN_N.append(item)
		else:
			SN_T.append(item)
	
	ttest(CDN_N, CDN_T, SN_N, SN_T)	
	
	
if __name__ == '__main__':
	main()
	
	
	