#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:12:33 2024

@author: john
"""
import pandas as pd
import numpy as np
import itertools
from scipy.optimize import linprog
from pandas import option_context

def solveIt(CONSTRAINTS =  [], RHS = [], OBJ = [],Verbose = False):
    linprogstat = ["Optimization Nominal", "Iteration Limit Reached", "Infeasible", "Unbounded","Numerical Problems, call a professional."]
    if Verbose:
        print("#####################")
        print("### Solving ##########")
        print("#####################")
    if (len(CONSTRAINTS) == 0): 
        print("No constraint matrix specified. Quitting.")
        return
    
    nr, nc = CONSTRAINTS.shape
    bds = [(0,1) for ix in np.arange(0,nc)]
    if len(RHS) == 0:
        RHS = np.ones((nr,1))
    if len(OBJ) ==0:
        OBJ = np.ones((nc,1))*-1
    lp_results = linprog(c = OBJ, A_ub = CONSTRAINTS, b_ub = RHS, bounds = bds)
    if Verbose:
        if lp_results.status == 0:
            print("Optimization successful")
        else:
            print(f"optimization less than successful. Status: {linprogstat[lp_results.status]}")
            return([],lp_results.status)
        print(f'Objective : {lp_results.fun}')
        print(f'Solution: {lp_results.x}')
        ISINT = [min(np.abs(item-0),np.abs(item-1)) for item in lp_results.x]
        MXI = max(ISINT)
        if (MXI > 0.000005): print("Solution not integer!") 
        else: print(f"Solution integer. Tolerance: {MXI}")
    return(lp_results.x,lp_results.status)
    
def checkTU(mat,Verbose = False, Tol = 1e-10):
    if Verbose:
        print("#####################")
        print("### Checking TU #####")
        print("#####################")
    (nrow,ncol) = mat.shape
    maxsize = min(nrow,ncol)
    cursize = 2
    count = 0
    outstring = ''
    ISTU = True
    while cursize <= maxsize :
        #choose all cursize x cursize sub-matrices 
        rowsets = list(itertools.combinations(set(np.arange(0,nrow)),cursize))
        colsets = list(itertools.combinations(set(np.arange(0,ncol)),cursize))
        for ix in rowsets:
            for jx in colsets:
                a = mat[ix,:]
                b = a[:,jx]
                d = np.linalg.det(b)
                print("\nTesting")
                print(np.array_str(b))
                print(f"Determinant: {d}")
                count = count +1
                if (np.abs(d)>Tol) & (np.abs(d-1)>Tol) & (np.abs(d+1)>Tol): 
                    ISTU = False
                    if Verbose:
                        return ISTU, f">>Is NOT TU<< \n Determinant: {d} \n iteration: {count}, \n Submatrix: \n {np.array_str(b)}"
                    else: 
                        return ISTU
        cursize += 1 
    if Verbose:
        return ISTU, f">>IS TU<< Number of determinants tested: {count}"
    else:
        return ISTU


def firmPref(ifirm, preflist, set1, set2):
    prefers = False
    if set2 in preflist[ifirm]:
        if preflist[ifirm].index(set1)  <  preflist[ifirm].index(set2):
            prefers = True
        else:
            prefers = False
    else:
        prefers = False
    return(prefers)

def workerPref(iworker, preflist, firm1, firm2):
    prefers = False
    if firm1 not in preflist[iworker]:
        prefers = False
    else:
        if firm2 in preflist[iworker]:
            if preflist[iworker].index(firm1) < preflist[iworker].index(firm2):
                prefers = True
            else:
                prefers = False
        else:
            prefers = False
    return(prefers)    
    
    
    
def doLP(nw, nf, pw = [], p=[], DoOneSet = True, DoBounds = False, StabilityConstraints = True, Dual = False, Verbose = False):
    if Verbose:
        print("#############################")
        print("### Generating LP Model #####")
        print("#############################")
    if len(p) == 0:
        print("No firm preferences specified, quitting.")
        return([],[],[],[])
    firm_no = list() #firm for each column
    set_assgn = list() #set for each column
    set_ind = np.zeros((nw+1,1))
    #*******************************************
    #construct the incidence matrix column-wise
    #*******************************************
    for ix in range(1,nf+1,1): #for each firm 1,...,nf
        for jx in range(0,len(p[ix]),1): #and each set in that firm's preference list
            #create a characteristic or indicator vector (0 if a worker is not in the set 1 if worker is)
            #a and b are 0 offset lists, pw and p start at 1 --the 0 entry is empty
            a = [1 if item in p[ix][jx] else 0 for item in range(1,nw+1,1)] #the assigned team's characteristic vector
            #a[item] = 1 if worker # item = 1 is in the jx-th subset of firm i's preference list
            b = [1 if ((a[itx]==1) & (ix not in pw[itx+1])) else 0 for itx in range(0,nw,1) ] #b[itx]=1 if worker itx+1 will not work for firm ix
            #check that non-individually rational team assignments that have been dropped
            #for ixq in range(0,len(b),1): if (b[ixq] == 1): print(f"Worker {ixq+1} will not work for firm {ix}") 
            if (sum(b) > 0): continue #at least one worker in this set will not work for firm ix this team is not individually rational for the workers drop it
            a = [0]+ a  #row entries start at row 1 not row 0
            set_ind = np.column_stack((set_ind,a))
            firm_no.append(ix)
            set_assgn.append(p[ix][jx])
    set_ind = np.delete(set_ind,0,1)
    set_ind = np.delete(set_ind,0,0)
    nr, nc = set_ind.shape
    #print(f">>>>>>>>  After team incidence matrix rows = {nr} columns = {nc}  <<<<<<")
    rhs = np.ones((nr,1))
    rowlabels = list(range(1,nw+1))
    #columns are now set, construct objective 
    obj = np.zeros(nc)
    #*******************************************
    #now add the side constraints of one subset per firm
    #*******************************************
    if DoOneSet == True:
        for ix in range(0,nf):
            c = [1 if firm_no[kx] == ix+1 else 0 for kx in range(0,nc)]
            set_ind = np.row_stack((set_ind,c))
            rhs = np.append(rhs,1)
            rowlabels.append("One set to a firm.")
    #righthand side up to this point is 1
#    (nrow,ncol) = set_ind.shape
#    rhs = np.ones((nrow + nf + len(firm_no),1))
#    rhs[nc:len(rhs),0] = -1
    
    #*******************************************
    #now add the variable bounds (not needed may tighten LP relaxation)
    #*******************************************
    if DoBounds ==  True:
        for ix in range(0,nc):
            newrowu = np.zeros((1,nc))
            newrowu[0,ix] = 1
            set_ind = np.row_stack((set_ind,newrowu))
            rhs = np.append(rhs,1)
            rowlabels.append('Variable UB')
            newrowl = np.zeros((1,nc))
            newrowl[0,ix] = -1
            set_ind = np.row_stack((set_ind, newrowl))
            rhs = np.append(rhs,0)
            rowlabels.append('Variable LB')
    #*******************************************
    #  now add stability constraints
    #*******************************************
    colset = [tuple(item) for item in set_assgn]
    colname = list(zip(firm_no,colset))
    stab_columns = np.ones((1,len(colname))) * 0
    rowlabels_stab = []
    for item in colname:
        ifirm = item[0]
        iset =  item[1]
        #start our new row
        newrow = np.zeros(len(colname))
        #first put a -1 in column corresponding to current firm/team pair
        newrow[colname.index((ifirm,iset))] = -1
        #now scan across all of the columns and put -1's in for all teams that ifirm prefers to iset (has to be a set in ifirms preference relation then ifirm has to prefer it)
        for setitem in colname:
            colno = colname.index(setitem)
            if (setitem[0] != ifirm):
                continue
            if (firmPref(ifirm,p,set(setitem[1]),set(iset))):
                newrow[colno] = -1
        #now the hard part: for every worker in iset scan all firm/set pairs put a 1 in the column if worker j prefers that firm and is in that set.
        for iwk in iset:
            for item_w in colname:
                colno_w = colname.index(item_w)
                ifirm_w = item_w[0]
                iset_w = item_w[1]
                if workerPref(iwk,pw,ifirm_w,ifirm):
                    if (iwk in iset_w):
                        newrow[colno_w] = -1
        if Dual:
            obj = obj + newrow*(1)
        else:
            #set_ind = np.row_stack((set_ind,newrow))
            stab_columns = np.row_stack((stab_columns,newrow))
            rhs_stab = np.append(rhs,-1)
            rowlabels_stab = rowlabels_stab + ['Stability' ]
    stab_columns = stab_columns[1:,:] #adjust out the initial column of all zeros
    if  not Dual:
        obj = obj - 1
    if StabilityConstraints:
        nrs,ncs = stab_columns.shape
        rhs_stab = -1*np.ones((nrs,1))
        rhs = np.append(rhs, rhs_stab)
        rowlabels = rowlabels + rowlabels_stab
        #set_ind = np.row_stack((set_ind,stab_columns[1:,:]))
        set_ind = np.row_stack((set_ind,stab_columns))

    return(set_ind, rhs, obj, firm_no, set_assgn,rowlabels,stab_columns)
    
#format, display and save the constraint matrix
#constraints:  each column is a firm/team assignment first nw rows are the team indicators
#              second nf rows are the one-subset-to-a-firm constraints
#              next set of rows (one for each firm/team pair) are the stability constraints
#              optionally can generate upper and lower bounds for each variable
#              default objective function is all 1's
#teams = the team for each column
#firms = the firm # for each column


def displayLP(constraints = [], rhs = [], obj = [], teams = [], firms = [],rowlabels = []):
    colset = [str(item) for item in teams]
    colname = list(zip(firms,colset))
    #colname = [(item[0],set(item[1])) for item in colname]
    constraints = pd.DataFrame(data = constraints, columns=colname)
    constraints['RHS'] = rhs
    constraints['RowLabels'] = rowlabels
    constraints.index = [item+1 for item in constraints.index]
    constraints.loc[constraints.index.max() + 1] = list(obj) + [' ','OBJ']
    constraints.to_csv("LPmodel_2.csv")
    return(constraints)

def decodeSolution(firms = [], teams = [],  solution = []):
    if (len(firms)== 0): 
            msgtxt = " Input: three vectors of length equal to number of columns in LP.\n"
            + " firms= afirm for each column, teams = a set of workers for each column, \n"
            +   " solution = solution vector of length = # columns"
            print (msgtxt)
    outstring = ''
    for zx in range(0,len(solution)):
        if solution[zx] > 0.0:
            outstring = outstring + f"Firm: {firms[zx]}, Assigned Set: {teams[zx]} weight:{solution[zx]} \n"
    return(outstring)

def doIntersectionGraph(constraint_mat):
    #mat is a binary array
    nr, nc = constraint_mat.shape
    intersection_mat = np.zeros((nc,nc))
    #for each pair of distinct columns check to see if they have 1's in the same row (inner product > 0)
    for ir in range(nc):
        for ic in range(nc):
            if (sum(constraint_mat[:,ic]*constraint_mat[:,ir]) > 0) & (ir != ic):
                intersection_mat[ic,ir] = 1
    return(intersection_mat)
    
def doIndependentSets(inmat,teams,firms, StabConst = [],Verbose = False):
    #very inefficient brute force enumeration
    #enumerate all of subsets of columns
    #inmat is the incidence matrix for the intersection graph of the constraint matrix
    #it is n by n with n= number of columns (and in the same order) as the constraint matrix
    nr,nc = inmat.shape

    # first create a list that contains the power set of the set of columns (this is the brute force part)
    colset = list(range(nc))
    colsubsets = [[]]
    for colitem in colset:
        colsubsets += [item + [colitem] for item in colsubsets]

    #print(f" Cardinality of power set: {len(colsubsets)}")  #just checking

    #now test every subset is_inedependent 
    #is an indicator for whether or not the given subset is an independent set
    
    is_independent = np.ones((1,len(colsubsets))) #assume a subset of columns is independent until proven dependent
    for indx, item in enumerate(colsubsets): #for each subset of columns
        #item is the current subset of columns we are working on
        #indx is the position in colsubsets where that item resides
        if len(item) == 0: is_independent[0,indx] = 0
        if len(item) == 1: continue
        test_list = list(itertools.combinations(tuple(item),2)) #all sets of size 2 from the set of columns given by item
        for xtpl in test_list:
            if inmat[xtpl[1],xtpl[0]] == 1: #xptl[1] and xptl[2] have an arc between them (inmat is symmetric as arcs are undirected only need to check one of (i,j) and (j,i))
                is_independent[0,indx] = 0
                break;
    # the list of independent sets is complete. Now reconstruct the worker subsets and firms that go with them
    indcol = np.zeros((nc,1))
    solution_count = 0
    stringout = ''
    for indx,item in enumerate(is_independent[0,:]):
        if (item == 1): #this entry corresponds to an independent set
            solution_count += 1
            newstring = ''
            newstring = newstring + f"\n #{solution_count}  Independent set of columns: {colsubsets[indx]}" + "\n"
            cols = colsubsets[indx] #find the correspoinding set of workers
            newindcol = np.zeros((nc,1)) #create a column length vector to record which contraint columns are active.
            for item in cols:
                newstring = newstring + f"Firm: {firms[item]} Subset: {teams[item]}" + "\n"
                newindcol[item,0] = 1
            indcol = np.column_stack((indcol,newindcol))    
            if (len(StabConst) != 0 ):
                stabtest = np.matmul(StabConst,newindcol)
                #newstring += f" stability calculation :{np.array_str(stabtest[:,0])} \n"
                if max(stabtest[:,0]) > -1.0 :
                    newstring += "----------------- Not Stable*  ---------------\n"
                    if (Verbose): 
                        stringout += newstring #verbose mode:  report all matchings
                        #stringout += f"{np.array_str(stabtest)}"
                else:
                    newstring += "+++++++++++++++++  Stable*    ++++++++++++++++\n "
                    stringout = stringout + newstring
            else:
                stringout = stringout + newstring
    indcol = indcol[:,1:]
    return(indcol,stringout)
            
    
if __name__ == "__main__":    
    
###################################################################    
###################################################################    
###################################################################    
###################################################################    
###################################################################    
    
     
    
#### Test Examples H4, H8, EO19, B1
    Example = 'B1'
    Vbose = True
    TestTu = False
    if Example == 'H1':
##############Huang Example 1
        #no stable solution
        nw = 3  #of workers
        nf = 3  #of firms
    
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
    
        # Enter firm preferences over subsets here
        pf[1] = [{1,2,3}]
        pf[2] = [{1}, {2}]
        pf[3] = [{2,3}]
        
        #enter worker preferences over firms
        pw[1] = [1,2]
        pw[2] = [2,1,3]
        pw[3] = [1,3]
#################################### 
    elif Example == 'H4':
##############Huang Example 4
        #stable matching: firm 1: {1,2}, firm 2: {}, Firm 3: {3}
        nw = 3  #of workers
        nf = 2  #of firms
    
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
    
        # Enter firm preferences over subsets here
        pf[1] = [{1,2}, {3}]
        pf[2] = [{1,2}]
        
        #enter worker preferences over firms
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [1]
#################################### 
##############Huang Example 8
    elif Example == 'H8':
        nw = 4  #of workers
        nf = 2  #of firms
    
        pf = list(range(0,nf+1))
        pw = list(range(0,nw + 1))

        # Enter firm preferences over subsets here
        pf[1] = [{1,2}, {3}]
        pf[2] = [{3,4}, {1,2}]
        
        #worker preferences over firms here
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [1,2]
        pw[4] = [2]
        
    elif Example == 'EO19':
##################################### 
###Echenique and Oviedo example 19
        nw = 3
        nf = 3
        pf = list(range(0,nf+1))
        pw = list(range(0,nw + 1))

        pf[1] = [{1,3}, {1,2}, {2,3}, {1}, {2}, {3}]
        pf[2] = [{1,3}, {2,3}, {1,2}, {3}, {2}, {1}]
        pf[3] = [{1,3}, {1,2}, {2,3}, {1}, {2}, {3}]  
        
        pw[1] = [1,2,3]
        pw[2] = [3,2,1]
        pw[3] = [1,3,2]
        
    elif Example == 'EO12':
##################################### 
###Echenique and Oviedo example 12 no core
        nw = 3
        nf = 3
        pf = list(range(0,nf+1))
        pw = list(range(0,nw + 1))

        pf[1] = [{1,2}, {3}]
        pf[2] = [{2,3}, {1}]
        pf[3] = [{1,3}, {2}]  
        
        pw[1] = [1,3,2]
        pw[2] = [2,1,3]
        pw[3] = [3,2,1]
    elif Example == 'EO11':
##################################### 
###Echenique and Oviedo example 11 
        nw = 4
        nf = 2
        pf = list(range(0,nf+1))
        pw = list(range(0,nw + 1))

        pf[1] = [{1,2},{3,4}, {1,3}, {2,4}, {1},{2},{3},{4}]
        pf[2] = [{1,2},{1,3}, {2,4}, {3,4}, {1},{2},{3},{4}]
 
        
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [1,2]
        pw[4] = [1,2]
    elif Example == 'B1':
##################################### 
### Bikhchandani substitutes/complements example
        nw = 4
        nf = 2
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
        
        #pf[1] = [{1,3}, {2,4},{3}]
        #pf[2] = [{1,3}, {2,4},{4}]   
        pf[1] = [{1,3}, {2,4}]
        pf[2] = [{1,3}, {2,4}]   
        
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [2,1]
        pw[4] = [1,2]
    elif Example == 'B2':
##################################### 
### Bikhchandani substitutes/complements example
### with extra sets
        nw = 4
        nf = 2
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
        
        pf[1] = [{1,3}, {2,4},{3}]
        pf[2] = [{1,3}, {2,4},{4}]   
        
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [2,1]
        pw[4] = [1,2]
        
        
    
# Try an example 
    print("########################")
    print(f" Example: {Example}")
    print("########################")
    
    ####Create the constraint matrix
    const_mat,rhs,obj,firms,teams,rowlab,stab_constr = doLP(nw, nf, pw, pf, DoOneSet = True, DoBounds = False, StabilityConstraints = False, Dual = False, Verbose = Vbose)
    
    dfLP = displayLP(constraints = const_mat, rhs = rhs, obj = obj, teams = teams, firms = firms, rowlabels = rowlab)
    if Vbose:
        with pd.option_context('display.max_rows', 50,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'display.precision', 0,
                               'display.colheader_justify', 'right'):
                  
            print(dfLP)
    
    
    
    if TestTu:
        IS_TU = checkTU(const_mat, Verbose = Vbose)
        if (IS_TU):
            print(" Is TU")
        else:
            print(" is NOT TU")
        
        
    solution,stat = solveIt(CONSTRAINTS = const_mat, RHS = rhs, OBJ = obj, Verbose = Vbose)  
    if stat == 0:    
        decodeSolution(firms, teams, solution)
    else: 
        print("Uh oh!")
        
    imat = doIntersectionGraph(const_mat)
    
    with pd.option_context('display.max_rows', 50,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'display.precision', 0,
                               'display.colheader_justify', 'right'):
        print("constraint matrix:")      
        print(const_mat)
        print("intersection graph incidence matrix")
        print(imat)
        independent_columns,txt_out = doIndependentSets(imat, teams, firms, StabConst = stab_constr)
        print(txt_out)
        print("Extremal (integral) feasible solutions")
        print(independent_columns)
        print("Are any of the integer solutions stable?")
        c = np.matmul(stab_constr,independent_columns)
        print(c)

    