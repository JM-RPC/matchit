#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:05:17 2024

@author: john
"""

import pandas as pd
import numpy as np
import re
def readFile(fpath = 'MatchingData.txt'):
    dfile = open(fpath, mode = 'r')
    instring = '#'
    outstr = ''
    outstring = ''
    nolines = 0
    while (instring != ''):
        instring = dfile.readline()
        #instring = instring.replace(' ','') #remove spaces
        if instring == '': continue
        nolines += 1
        outstring += instring
    #outstr = "\n".join(outstring)
    #print(f"readFile done: lines read {nolines}")
    return(outstring)

def readData(inlist = []):
    instring = '#'
    wftype = ''
    nolines = 0
    nf = 0
    nw = 0
    pw = []
    pf = []
    outstringlist = []
    if len(inlist) == 0: return
    for instring in inlist:
        #first pull out comments, remove extra spaces
        instring = instring.replace(' ','')
        instring = instring.split('#')[0] #  if the comment sign # is is in the first column, then split returns '' for instring
        if instring == '': continue
        nolines += 1
        outstringlist.append(instring)

        #decode the type of data line: data, firm indicator, worker indicator or comment
        if instring[0:5] == 'firms' :
            wftype = 'F'
            nf = int(instring.split('=')[1])
            pf = [[] for itx in range(0,nf + 1)]
            continue
        elif instring[0:7] == 'workers':
            wftype = 'W'
            nw = int(instring.split('=')[1])
            pw = [[] for itx in range(0,nw +1)]
            continue
        elif instring[:1] == '#': #this should never happen
            print(f'*****>>>>>>>>>{instring}')
            continue
        elif (instring[:1] not in "123456789"):
            return(0,0,[],[],f"Typo? {instring} => {instring[:1]}")

        #it's a data line
        if (wftype == 'F'):
            #check syntax...sort of
            if sum([0 if item in "][}{:0123456789," else 1 for item in instring])>0:
                return(0,0,[],[],f"Typo? {instring} => {instring[:1]}")
            #now process it
            temp = instring.split(':')
            if len(temp)<=1: return(0,0,[],[],f"Missing ':' in {instring}?")
            firmno = int(temp[0])
            if firmno> nf:
                return(0,0,[],[],f"firm number out of range at: {instring}")
            outiter = re.finditer(r'\{.*?\}',instring.split(':')[1])
            setlist = []
            for item in outiter:
                itemstring = item.group(0).replace('{','').replace('}','')
                itemset = set([int(itx) for itx in itemstring.split(',')])
                if (itemset in setlist):
                    return(0,0,[],[],f"Firm {firmno} has intransitive preferences. Set: {itemstring}")
                setlist.append(itemset)  
            pf[firmno] = setlist
        elif (wftype == 'W'):
            #check syntax...sort of
            if sum([0 if item in ":0123456789," else 1 for item in instring])>0:
                return(0,0,[],[],f"Typo? {instring}")
            #now process it
            temp = instring.split(':')
            if len(temp)<=1: return(0,0,[],[],f"Missing ':' in {instring}?")
            workerno = int(temp[0])
            if workerno > nw: 
                return(0,0,[],[],f"worker number {workerno} out of range at:{instring}")
            firmlist = [itx.strip() for itx in temp[1].split(',')]
            firmlist = [int(itx) for itx in firmlist if itx != '']
            if (len(set(firmlist))!= len(firmlist)):
                return(0,0,[],[],f"Worker {workerno} preferences are intransitive")
            pw[workerno] = firmlist
    #check numbering consistency
    for setx in pf[1:]:
        if setx != []:
            if (max([max(item) for item in setx]) > nw):
                return(0,0,[],[],f"worker number out of range in firm prefs: {setx} workers indicated {nw}")
    for firmx in pw[1:]:
        if max(firmx) > nf:
            return(0,0,[],[],f"firm number {max(firmx)} out of range in worker prefs: {firmx} firms indicated = {nf}")
    tempstring = '\n'.join(outstringlist)
    #print(f"readData completed data : {tempstring}")
    return(nw,nf,pw,pf,tempstring)
        
if __name__ == "__main__":  
    nw, nf, pw, pf, dataset= readData(fpath = '/Volumes/Working/Ministore/MiniMini/Rsync/JMBOX/Research/Matching/MatchData.txt')