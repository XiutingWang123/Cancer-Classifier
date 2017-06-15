# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:11:31 2017

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
Draw boxplot to see data distribution
"""
def boxPlot(CDN_N, CDN_T, SN_N, SN_T):
	CDN_N_15 = list(CDN_N[:,13])
	CDN_N_20 = list(CDN_N[:,18])
	CDN_N_25 = list(CDN_N[:,23])
	CDN_N_42 = list(CDN_N[:,40])	
	CDN_N_49 = list(CDN_N[:,47])
	CDN_N_59 = list(CDN_N[:,57])
	CDN_N_65 = list(CDN_N[:,63])
	CDN_N_73 = list(CDN_N[:,71])
	CDN_N_90 = list(CDN_N[:,88])
	CDN_N_106 = list(CDN_N[:,104])
	CDN_N_157 = list(CDN_N[:,155])
	CDN_N_178 = list(CDN_N[:,176])
	CDN_N_191 = list(CDN_N[:,189])
	CDN_N_194 = list(CDN_N[:,192])
	CDN_N_198 = list(CDN_N[:,196])
	CDN_N_207 = list(CDN_N[:,205])
	CDN_N_211 = list(CDN_N[:,209])
	CDN_T_15 = list(CDN_T[:,13])
	CDN_T_20 = list(CDN_T[:,18])
	CDN_T_25 = list(CDN_T[:,23])
	CDN_T_42 = list(CDN_T[:,40])
	CDN_T_49 = list(CDN_T[:,47])
	CDN_T_59 = list(CDN_T[:,57])
	CDN_T_65 = list(CDN_T[:,63])
	CDN_T_73 = list(CDN_T[:,71])
	CDN_T_90 = list(CDN_T[:,88])
	CDN_T_106 = list(CDN_T[:,104])
	CDN_T_157 = list(CDN_T[:,155])
	CDN_T_178 = list(CDN_T[:,176])
	CDN_T_191 = list(CDN_T[:,189])
	CDN_T_194 = list(CDN_T[:,192])
	CDN_T_198 = list(CDN_T[:,196])
	CDN_T_207 = list(CDN_T[:,205])
	CDN_T_211 = list(CDN_T[:,209])

	SN_N_15 = SN_N[:,13]
	SN_N_20 = SN_N[:,18]
	SN_N_25 = SN_N[:,23]
	SN_N_42 = SN_N[:,40]	
	SN_N_49 = SN_N[:,47]
	SN_N_59 = SN_N[:,57]
	SN_N_65 = SN_N[:,63]
	SN_N_73 = SN_N[:,71]
	SN_N_90 = SN_N[:,88]
	SN_N_106 = SN_N[:,104]
	SN_N_157 = SN_N[:,155]
	SN_N_178 = SN_N[:,176]
	SN_N_191 = SN_N[:,189]
	SN_N_194 = SN_N[:,192]
	SN_N_198 = SN_N[:,196]
	SN_N_207 = SN_N[:,205]
	SN_N_211 = SN_N[:,209]
	SN_T_15 = SN_T[:,13]
	SN_T_20 = SN_T[:,18]
	SN_T_25 = SN_T[:,23]
	SN_T_42 = SN_T[:,40]	
	SN_T_49 = SN_T[:,47]
	SN_T_59 = SN_T[:,57]
	SN_T_65 = SN_T[:,63]
	SN_T_73 = SN_T[:,71]
	SN_T_90 = SN_T[:,88]
	SN_T_106 = SN_T[:,104]
	SN_T_157 = SN_T[:,155]
	SN_T_178 = SN_T[:,176]
	SN_T_191 = SN_T[:,189]
	SN_T_194 = SN_T[:,192]
	SN_T_198 = SN_T[:,196]
	SN_T_207 = SN_T[:,205]
	SN_T_211 = SN_T[:,209]
	
			
	data1 = [CDN_N_15, CDN_T_15,SN_N_15, SN_T_15]
	data2 = [CDN_N_20, CDN_T_20,SN_N_20, SN_T_20]
	data3	= [CDN_N_25, CDN_T_25,SN_N_25, SN_T_25] 
	data4 = [CDN_N_42, CDN_T_42,SN_N_42, SN_T_42]
	data5 = [CDN_N_49, CDN_T_49,SN_N_49, SN_T_49]
	data6 = [CDN_N_59, CDN_T_59,SN_N_59, SN_T_59]
	data7 = [CDN_N_65, CDN_T_65,SN_N_65, SN_T_65]
	data8 = [CDN_N_73, CDN_T_73,SN_N_73, SN_T_73]
	data9 = [CDN_N_90, CDN_T_90,SN_N_90, SN_T_90]
	data10 = [CDN_N_106, CDN_T_106,SN_N_106, SN_T_106]
	data11 = [CDN_N_157, CDN_T_157,SN_N_157, SN_T_157]
	data12 = [CDN_N_178, CDN_T_178,SN_N_178, SN_T_178]
	data13 = [CDN_N_191, CDN_T_191,SN_N_191, SN_T_191]
	data14 = [CDN_N_194, CDN_T_194,SN_N_194, SN_T_194]
	data15 = [CDN_N_198, CDN_T_198,SN_N_198, SN_T_198]
	data16 = [CDN_N_207, CDN_T_207,SN_N_207, SN_T_207]
	data17 = [CDN_N_211, CDN_T_211,SN_N_211, SN_T_211]
	
	plt.figure(1)
	mpl.rc('font', family='Helvetica')
	colors = ['lightcoral','lightcoral', 'skyblue','skyblue']
	labels = ['N','T','N','T']
		
	fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True, figsize=(8,12))	
	
	bp1 = axes[0,0].boxplot(data1 ,patch_artist=True)
	axes[0,0].set_title('hsa-miR-130a', fontsize=20)
	
	
	bp2 = axes[0, 1].boxplot(data2,patch_artist=True)
	axes[0, 1].set_title('hsa-miR-106a', fontsize=20)
	
	bp3 = axes[0, 2].boxplot(data3,patch_artist=True)
	axes[0, 2].set_title('hsa-miR-134', fontsize=20)
	
	bp4 = axes[1, 0].boxplot(data4,patch_artist=True)
	axes[1, 0].set_title('hsa-miR-221', fontsize=20)
	
	bp5 = axes[1, 1].boxplot(data5,patch_artist=True)
	axes[1, 1].set_title('hsa-miR-198', fontsize=20)
	
	bp6 = axes[1, 2].boxplot(data6,patch_artist=True)
	axes[1, 2].set_title('hsa-miR-34a', fontsize=20)
	
	bp7 = axes[2, 0].boxplot(data7,patch_artist=True)
	axes[2, 0].set_title('hsa-miR-197', fontsize=20)
	
	bp8 = axes[2, 1].boxplot(data8,patch_artist=True)
	axes[2, 1].set_title('hsa-miR-29c', fontsize=20)
	
	bp9 = axes[2, 2].boxplot(data9,patch_artist=True)
	axes[2, 2].set_title('hsa-miR-19b', fontsize=20)
	
	bp10 = axes[3, 0].boxplot(data10,patch_artist=True)
	axes[3, 0].set_title('hsa-miR-144', fontsize=20)
	
	bp11 = axes[3, 1].boxplot(data11,patch_artist=True)
	axes[3, 1].set_title('hsa-miR-321', fontsize=20)
	
	bp12 = axes[3, 2].boxplot(data12,patch_artist=True)
	axes[3, 2].set_title('mmu-miR-298', fontsize=20)
	
	bp13 = axes[4, 0].boxplot(data13,patch_artist=True)
	axes[4, 0].set_title('mmu-miR-339', fontsize=20)
	
	bp14 = axes[4, 1].boxplot(data14,patch_artist=True)
	axes[4, 1].set_title('mmu-miR-342', fontsize=20)
	
	bp15 = axes[4, 2].boxplot(data15,patch_artist=True)
	axes[4, 2].set_title('mmu-miR-34b', fontsize=20)
	
	bp16 = axes[5, 0].boxplot(data16,patch_artist=True)
	axes[5, 0].set_title('rno-miR-333', fontsize=20)
	
	bp17 = axes[5, 1].boxplot(data17,patch_artist=True)
	axes[5, 1].set_title('rno-miR-344', fontsize=20)
	
	
	fig.delaxes(axes[5,2])	
	#fig.text(0.01, 0.5, 'miRNA Expressions', ha='center', va='center',rotation='vertical',fontsize=20)

	for ax in axes.flatten():
		ax.get_yaxis().set_tick_params(which='both', direction='out', right='off',labelsize=14)
		ax.get_xaxis().set_tick_params(which='both', direction='out',top='off',labelsize=12)
		ax.set_xticklabels(labels)
		
	#for ax in axes.flatten():
		#ax.set_yscale('log')
		
	for bp in [bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8, bp9, bp10, bp11, bp12, bp13, bp14, bp15, bp16, bp17]:
		for whisker in bp['whiskers']:
			whisker.set(color='k',linestyle='-')
		
		for box, color in zip(bp['boxes'], colors):
			box.set(color='k')
			box.set(facecolor = color)			
			
		for flier in bp['fliers']:
			flier.set(marker='o', markersize=4, color='k', alpha=0.5)
	
	bbox_props = dict(boxstyle='square,pad=0.3', fc='lightcoral', ec='k')
	bbox_props2 = dict(boxstyle='square,pad=0.3', fc='skyblue', ec='k')	
	fig.text(0.8, 0.12, 'CDN', ha='right', va='center', size=12, bbox=bbox_props)
	fig.text(0.8, 0.09, ' SN  ', ha='right', va='center', size=12, bbox=bbox_props2)

		
	
	fig.subplots_adjust(hspace=0.4)
	fig.tight_layout()
		
	fig.savefig('boxplot_sap.jpg', dpi=1200)	









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
	
	CDN_N = np.array(CDN_N)
	CDN_T = np.array(CDN_T)	
	SN_N = np.array(SN_N)
	SN_T = np.array(SN_T)		
	
	boxPlot(CDN_N, CDN_T, SN_N, SN_T)
	
if __name__ == '__main__':
	main()
	
	
	