#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:05:17 2024

@author: john
"""

import pandas as pd
import numpy as np
import re
def readData(fpath = 'MatchingData.txt'):
    dfile = open(fpath, mode = 'r')
    instring = '#'
    wftype = ''
    nolines = 0
    pw = []
    pf = []
    while (instring != ''):
        instring = dfile.readline()
        if instring == '': continue
        nolines += 1
        #decode the type of data line: data, firm indicator, worker indicator or comment
        if instring[:5] == 'firms' :
            wftype = 'F'
            nf = int(instring.split('=')[1])
            print(f"Number of firms = {nf}")
            pf = list(range(0,nf + 1))
            continue
        elif instring[:7] == 'workers':
            wftype = 'W'
            nw = int(instring.split('=')[1])
            pw = list(range(0,nw + 1))
            print(f"Number of workers = {nw}")
            continue
        elif instring[:1] == '#':
            continue


        #it's a data line
        if (wftype == 'F'):
            firmno = int(instring.split(':')[0])
            outiter = re.finditer(r'\{.*?\}',instring.split(':')[1])
            setlist = []
            for item in outiter:
                itemstring = item.group(0).replace('{','').replace('}','')
                itemset = set([int(itx) for itx in itemstring.split(',')])
                setlist.append(itemset)  
            pf[firmno] = setlist
            print(f"pf[{firmno}] = {pf[firmno]}")
        elif (wftype == 'W'):
            temp = instring.split(':')
            workerno = int(temp[0])
            firmlist = [itx.strip() for itx in temp[1].split(',')]
            firmlist = [int(itx) for itx in firmlist]
            pw[workerno] = firmlist
            print(f"pw[{workerno}] = {firmlist}")
    print(f"Lines read: {nolines}")    
    return(nw,nf,pw,pf)
        
if __name__ == "__main__":  
    nw, nf, pw, pf = readData(fpath = '/Volumes/Working/Ministore/MiniMini/Rsync/JMBOX/Research/Matching/MatchData.txt')