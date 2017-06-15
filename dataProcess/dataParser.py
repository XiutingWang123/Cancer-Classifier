#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:59:32 2016

@author: yyc
"""

import pandas as pd




def label(data):
    header = list(data.columns.values)
    target = []
    for i in range(1,95):
        if header[i][0] == 'N': target.append(0)
        else:                   target.append(1)
    return target





def normalize(data):
    minCol = data.min(axis=1)
    maxCol = data.max(axis=1)
    new_df = pd.DataFrame()
    
    
    for i, column in zip(range(len(data.columns)), data):
        s = []
        for row, j, k in zip(data[column], minCol, maxCol):
            if k - j == 0 : nol = 0
            else:           nol = (row - j)/(k - j)
            s.append(nol)     
        new_df['brst_S%d' % (i+1)] = s
    return new_df     


def main():
	df = pd.read_csv('/Users/yyc/Desktop/subset.txt',sep=',')

	df_colon = df.iloc[:,1:16]
	df_pan = df.iloc[:,16:26]
	df_kid = df.iloc[:,26:34]
	df_bldr = df.iloc[:,34:43]
	df_prost = df.iloc[:,43:57]
	df_ut = df.iloc[:,57:76]
	df_lung = df.iloc[:,76:86]
	df_brst = df.iloc[:,86:95]	
	
	
	# normalized training set
	d1 = normalize(df_colon)
	d2 = normalize(df_pan)
	d3 = normalize(df_kid)
	d4 = normalize(df_bldr)
	d5 = normalize(df_prost)
	d6 = normalize(df_ut)
	d7 = normalize(df_lung)
	d8 = normalize(df_brst)
	target = label(df)

	result = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8], axis=1)
	result = result.T
	result['target'] = target
	result.to_csv('miRNA_train.txt')
	



if __name__ == '__main__':
	main()
