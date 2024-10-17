 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 8 13:33:04 2024

@author: john
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

def nodeCoordinates(n):
    #independent graph columns numbered 0 to n-1
    a = [(round(np.cos(2*i*np.pi/n),2),round(np.sin(2*i*np.pi/n),2)) for i in range(0,n)]
    return(a)

def makeSegs(imat,nodes):
    nr,nc = imat.shape
    n = nr
    if len(nodes) != n : return []
    lines=[[]]
    for ix in range(0,n):
        for jx in range(0,n):
            if (imat[ix,jx] == 1):
                lines = lines + [[nodes[ix],nodes[jx]]]
                #print(f"line from node({ix}) to node({jx}).")
    lines = lines[1:]
    return lines
        

if __name__ == "__main__":  


    n=5
    node = nodeCoordinates(n)

    #make some dots
    dotx = [item[0] for item in node[1:]]
    doty = [item[1] for item in node[1:]]


    imat = np.zeros((n,n))
    imat[0,1] = 1
    imat[1,0] = 1
    imat[2,3] = 1
    imat[3,2] = 1
    imat[3,4] = 1
    imat[4,3] = 1

    lines = makeSegs(imat)

    # n=5
    # node = nodeCoordinates(n)

    # lines = [[]]
    # for ix in range(1,n+1):
    #     for jx in range(1,ix):
    #         lines = lines + [[node[ix],node[jx]]]
    # lines = lines[1:]
    # #lines2 = [[node[1],node[2]],[node[3],node[2]],[node[3],node[4]],[node[4],node[5]]],[node[5],node[6]],[node[6],node[7]]]
    # #graph the segment from (0,.25) to (.75,.5) and from (-.75,-.25) to (0,-.5)
    # lines2 =  [[(0,.25),(.75,.5)],[(-.75,-.25),(0,-.75)]]
    # lines = [[(0,.75),(.25,.5)],[(-.75,0),(-.25,-.75)]]  

    fig, ax = plt.subplots(figsize =(8,8))
    #fig = plt.figure(figsize = (12,9), tight_layout = False)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    lc = LineCollection(lines,linewidths = 2)
    ax.add_collection(lc)
    ax.plot(dotx,doty,'bo-')
    plt.show()
