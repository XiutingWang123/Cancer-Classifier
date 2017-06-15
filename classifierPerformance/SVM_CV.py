# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 23:02:11 2016

@author: yyc
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV


"""
	Do 10 fold cross validation to compute confusion matrix
"""


def SVMcrossValidation(X, y, C_l, C_rbf, gamma):

	svcl = SVC(kernel="linear", C=C_l, class_weight='balanced')
	acc = cross_val_score(svcl, X, y, cv=10,scoring='accuracy')
	recall = cross_val_score(svcl, X, y, cv=10,scoring='recall')
	precision = cross_val_score(svcl, X, y, cv=10,scoring='precision')
	f1 = cross_val_score(svcl, X, y, cv=10,scoring='f1')
    
	print('svcl: Acc: %0.5f, recall: %0.5f, precision: %0.5f, f1: %0.5f' \
              % (acc.mean(), recall.mean(), precision.mean(), f1.mean()))

	svcr = SVC(kernel="rbf", C=C_rbf, gamma=gamma)
	acc = cross_val_score(svcr, X, y, cv=10,scoring='accuracy')
	recall = cross_val_score(svcr, X, y, cv=10,scoring='recall')
	precision = cross_val_score(svcr, X, y, cv=10,scoring='precision')
	f1 = cross_val_score(svcr, X, y, cv=10,scoring='f1')
	print('svcr: Acc: %0.5f, recall: %0.5f, precision: %0.5f, f1: %0.5f' \
              % (acc.mean(), recall.mean(), precision.mean(), f1.mean()))


"""
  Do grid search with cross-validation to get best parameters
  return 
	C for linear SVM
	C for rbf SVM
	gamma for rbf SVM 
"""


def getParameters(X, y):
			
	C_range = np.logspace(-2, 10, 13, base=2)
	gamma_range = np.logspace(-9, 3, 13, base=2)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
	grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
	grid.fit(X, y)

	C_l = grid.best_params_	
	
	param_2 = {'C':C_range}
	grid = GridSearchCV(SVC(kernel='linear',class_weight='balanced'), param_grid=param_2, cv=cv)
	grid.fit(X, y)

	C_rbf = grid.best_params_
	gamma = grid.best_score_	
	
	return C_l, C_rbf, gamma


def main():
	# Import training data
	df = pd.read_csv('/Users/yyc/Desktop/classifier/miRNA_train.txt')
	X = df.ix[0:,1:218]
	y = df.ix[0:,'target']	
	
	C_l, C_rbf, gamma = getParameters(X, y)
	SVMcrossValidation(X, y, C_l, C_rbf, gamma)


	
if __name__ == '__main__':
	main()	