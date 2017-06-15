#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:06:24 2016

@author: yyc
"""


"""
This script parse excel data and save as txt file

"""


import openpyxl

wb = openpyxl.load_workbook('/Users/yyc/Desktop/mLung_test.xlsx')
sheet = wb.get_sheet_by_name('工作表1')

f = open('mLung.txt','a')
for rowOfCellObjects in sheet['C3':'N220']:
    for cellObj in rowOfCellObjects:
        f.write(format(str(cellObj.value)) + ',' )
    f.write('\n')
    
f.close()