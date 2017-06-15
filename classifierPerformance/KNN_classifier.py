#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 00:24:32 2016

@author: yyc
"""

import pandas as pd
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score

"""
 Use KNN to test cancer prediction accuracy with different range of k
	
"""


def KNN(X, y):
	# Try different range of k value
	k_range = range(1, 31)
	kscores_acc = []
	kscores_recall =[]
	kscores_precision = []
	kscores_f1 = []
	for k in k_range:
		knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
		acc = cross_val_score(knn, X, y, cv=10,scoring='accuracy')
		kscores_acc.append(acc.mean())
		recall = cross_val_score(knn, X, y, cv=10,scoring='recall')
		kscores_recall.append(recall.mean())
		precision = cross_val_score(knn, X, y, cv=10,scoring='precision')
		kscores_precision.append(precision.mean())
		f1 = cross_val_score(knn, X, y, cv=10,scoring='f1')
		kscores_f1.append(f1.mean())
    
		print('k = %d Acc: %0.5f, recall: %0.5f, precision: %0.5f, f1: %0.5f' \
	              % (k, kscores_acc[k-1], kscores_recall[k-1], kscores_precision[k-1], kscores_f1[k-1]))


def main():
	# Import training data
	df = pd.read_csv('/Users/yyc/Desktop/classifier/miRNA_train.txt')
	X = df.ix[0:,1:218]
	y = df.ix[0:,'target']	
	KNN(X, y)


	
if __name__ == '__main__':
	main()	

 