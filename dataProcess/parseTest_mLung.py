#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 21:52:38 2016

@author: yyc
"""

import pandas as pd



def normalize_lung(data, minCol, maxCol):

    new_df = pd.DataFrame()
    
    
    for i, column in zip(range(0,12), data):
        s = []
        for row, j, k in zip(data[column], minCol, maxCol):
            if k - j == 0 : nol = 0
            else:           nol = (row - j)/(k - j)
            s.append(nol)     
        new_df['mlung_test%d' % (i+1)] = s
    return new_df 
    
    
        
def label(data):
    header = list(data.columns.values)
    target = []
    for i in range(0,12):
        if header[i][0] == 'N': target.append(0)
        else:                   target.append(1)
    return target





def main():
	df = pd.read_csv('/Users/yyc/Desktop/subset.txt',sep=',')
	df_lung = df.iloc[:,76:86]
	
	minCol = df_lung.min(axis=1)
	maxCol = df_lung.max(axis=1)
	
	df_test = pd.read_csv('/Users/yyc/Desktop/miRNA-classifier/parse_txt/mLung.txt',sep=',')
	df_test_lung = df.iloc[:,0:12]
	result = normalize_lung(df_test_lung, minCol, maxCol)
	result = result.T
	result['target'] = label(df)
	result.to_csv('miRNA_test_mlung.txt')
	

if __name__ == '__main__':
	main()   