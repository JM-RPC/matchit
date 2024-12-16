#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:12:33 2024

@author: john
"""
import pandas as pd
import numpy as np
import random
import itertools as it
from scipy.optimize import linprog
from pandas import option_context
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import time



rng = np.random.default_rng(1158490)


def solveIt(CONSTRAINTS =  [], RHS = [], OBJ = [],Verbose = False, method = "highs", bds = (0,1), AllEq = False):#, method = "revised simplex"):
    linprogstat = ["Optimization Nominal", "Iteration Limit Reached", "Infeasible", "Unbounded","Numerical Problems, call a professional."]
    if Verbose:
        print("#####################")
        print("### Solving ##########")
        print("#####################")
    #if (len(CONSTRAINTS) == 0): 
    #    print("No constraint matrix specified. Quitting.")
    #    return
    nr, nc = CONSTRAINTS.shape
    if len(RHS) == 0:
        RHS = np.ones((nr,1))
    if len(OBJ) ==0:
        OBJ = np.ones((nc,1))*-1
    if AllEq:
        lp_results = linprog(c = OBJ, A_eq = CONSTRAINTS, b_ub = RHS, bounds = bds)
    else:
        lp_results = linprog(c = OBJ, A_ub = CONSTRAINTS, b_ub = RHS, bounds = bds)
    if lp_results.status == 0:
        if Verbose:            
            print("Optimization successful")
        return(lp_results)
    else:
        if Verbose:
            print(f"optimization less than successful. Status: {linprogstat[lp_results.status]}")
        return(lp_results)
    
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
        rowsets = list(it.combinations(set(np.arange(0,nrow)),cursize))
        colsets = list(it.combinations(set(np.arange(0,ncol)),cursize))
        for ix in rowsets:
            for jx in colsets:
                a = mat[ix,:]
                b = a[:,jx]
                d = np.linalg.det(b)
                if Verbose:
                    print("\nTesting")
                    print(np.array_str(b))
                    print(f"Determinant: {d}")
                count = count +1
                if (np.abs(d)>Tol) & (np.abs(d-1)>Tol) & (np.abs(d+1)>Tol): 
                    ISTU = False
                    return ISTU, f">>Is NOT TU<< \n Determinant: {d} \n iteration: {count}, \n Submatrix: \n {np.array_str(b)} \n rows: {ix}\n cols: {jx}"
        cursize += 1 
    return ISTU, f">>IS TU<< Number of determinants tested: {count}"


def firmPref(ifirm, preflist, set1, set2):
    prefers = False
#    if (set2 not in preflist[ifirm]) & (set1 not in preflist[ifirm]):
#        return("error")#under current config, this should not happen
    if set1 not in preflist[ifirm]:
        prefers = False
    else:
        if set2 in preflist[ifirm]:
            if preflist[ifirm].index(set1)  <  preflist[ifirm].index(set2):
                prefers = True
            else:
                prefers = False
        else: #in particular if set2 is the empty set (set()) any set in it's preference ordering is preferred
            prefers = True
    return(prefers)

def workerPref(iworker, preflist, firm1, firm2):
    #note: an unassigned worker prefers remaining unassigned to working at a firm not in his or her preference list
    #so when we encounter a firm1 value that is not in iworker's preference list we return False always.

    prefers = False
    #if (firm1 not in preflist[iworker]) & (firm2 not in preflist[iworker]) :
        #print(f"worker preference error. Neither firm1 #{firm1} nor firm2 #{firm2} in preference list for worker {iworker}.")
        #return("error") # this happens when an a firm not in the worker's firm preference list is compared to no assignment.
    if firm1 not in preflist[iworker]: #any firm in list is preferred to nothing
        prefers = False
    else:
        if firm2 in preflist[iworker]:
            if preflist[iworker].index(firm1) < preflist[iworker].index(firm2):
                prefers = True
            else:
                prefers = False
        else: # firm2 is not in pref list, firm1 is
            prefers = True
    return(prefers)    
    
    
    
def doLP(nw, nf, pw = [], p=[], DoOneSet = False, DoWorkers = True,  StabilityConstraints = False, Mxmn = False, Dual = False, OFcard = False, AllEq = False, Verbose = False):
    #Note: workers and firms are indexed 1-offset: workers 1,...,m and firms 1,...,n.  Python arrays/lists are 0-offset, an
    #array with n components is indexed 0,...,n-1.  Sooo... 
    #p is a list of lists that holds firm preferences: p[1] is a list of sets of workers (teams) in preference order for firm 1. 
    #For example: p[1][0] is the favorite team for firm 1, p[1][2] the next most favored team, etc.
    #the list of teams starts at 0 the list of firm preference lists starts at 1.   
    #Same deal for worker preferences but pw[i][0] is the firm number that worker i likes best, p[i][1] second best, etc...
    #Confusing? Yes, I know...
    #At this point, no preprocessing is done to remove non firm individually rational teams from firm preferences
    #non worker individually rational team assignments (teams containin a worker who prefers unemployment to the assigned firm)
    # are screened out below.
    if Verbose:
        print("#############################")
        print("### Generating LP Model #####")
        print("#############################")
    if len(p) == 0:
        print("No firm preferences specified, quitting.")
        return([],[],[],[])
    firm_no = list() #firm for each column
    set_assgn = list() #set for each column
    worker_ind = np.zeros((nw+1,1))
    set_ind = np.zeros((nw+1,1))
    #*******************************************
    #construct the incidence matrix column-wise
    #*******************************************
    for ix in range(1,nf+1,1): #for each firm 1,...,nf
        for jx in range(0,len(p[ix]),1): #and each for set in that firm's preference list
            #create a characteristic or indicator vector (0 if a worker is not in the set 1 if worker is)
            #a and b are 0 offset lists, pw and p start at 1 --the 0 entry is empty

            a = [1 if item in p[ix][jx] else 0 for item in range(1,nw+1,1)] #the assigned team's characteristic vector
            #a[item] = 1 if worker # item = 1 is in the jx-th subset of firm i's preference list

            b = [1 if ((a[itx]==1) & (ix not in pw[itx+1])) else 0 for itx in range(0,nw,1) ] #b[itx]=1 if worker itx+1 will not work for firm ix
            #check that non- worker individually rational team assignments that have been dropped
            #for ixq in range(0,len(b),1): if (b[ixq] == 1): print(f"Worker {ixq+1} will not work for firm {ix}") 
            if (sum(b) > 0): continue #at least one worker in this set will not work for firm ix this team is not individually rational for the workers drop it

            a = [0]+ a  #row entries start at row 1 not row 0
            worker_ind = np.column_stack((worker_ind,a))
            firm_no.append(ix)
            set_assgn.append(p[ix][jx])
    worker_ind = np.delete(worker_ind,0,1)
    worker_ind = np.delete(worker_ind,0,0)
    nr, nc = worker_ind.shape
    obj = np.zeros(nc)
    #print(f">>>>>>>>  After team incidence matrix rows = {nr} columns = {nc}  <<<<<<")
    if DoWorkers:
        rhs = np.ones((nr,1))
        rowlabels = list(range(1,nw+1))
        set_ind = worker_ind
    else: 
        nrtemp , nctemp = worker_ind.shape
        set_ind = np.empty([0,nctemp])
        rowlabels = list([])
        rhs = np.empty([0,1])

    #*******************************************
    #now add the side constraints of one subset per firm
    #*******************************************
    if DoOneSet:
        for ix in range(0,nf):
            c = [1 if firm_no[kx] == ix+1 else 0 for kx in range(0,nc)]
            set_ind = np.vstack((set_ind,c))
            rhs = np.append(rhs,1)
            rowlabels.append("F_" + str(ix+1))
    #righthand side up to this point is 1
#    (nrow,ncol) = set_ind.shape
#    rhs = np.ones((nrow + nf + len(firm_no),1))
#    rhs[nc:len(rhs),0] = -1
    
    #*******************************************
    #  now create stability constraints
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
        #now the hard part: for every worker in iset, scan all firm/set pairs. 
        #For sets in which worker iwk resides, put a 1 in the column if iwk prefers that firm
        for iwk in iset: #for each worker in iset
            for item_w in colname: #look through all columns (possible firm-team matches)
                colno_w = colname.index(item_w)
                ifirm_w = item_w[0]
                iset_w = item_w[1]
                if (ifirm_w == ifirm) & (set(iset_w).issubset(set(iset))): #do not apply this to the column being processed (ifirm iset) it's already -1
                    continue
                if (iwk in iset_w) & (workerPref(iwk,pw,ifirm_w,ifirm)):  #if our worker iwk is in the set iset_w and prefers ifirm_w
                    newrow[colno_w] = -1
        #always generate stability constraints, include in LP formulation only if asked(below)
        stab_columns = np.vstack((stab_columns,newrow))
        #rhs_stab = np.append(rhs,-1)
        rowlabels_stab = rowlabels_stab + [f'St_{ifirm}:{iset}' ]
        if Dual:
            obj = obj + newrow*(1)
    #*******************************************
    #now put the constraint matrix together
    #*******************************************

    #objective
    stab_columns = stab_columns[1:,:] #adjust out the initial column of all zeros
    if  not Dual: #if we're not dualizing out the stability constraint just put in -1's
        obj = obj - 1
        if (OFcard): #unless we're trying to match as many workers as possible, then use set cardinality as objective coef.
            obj = np.array([len(item) for item in set_assgn])
            obj = obj * (-1)

    #Stability constraints
    if StabilityConstraints:
        crows, ccols = set_ind.shape
        nrs,ncs = stab_columns.shape
        nucolmm = 0*np.zeros((crows,1))
        if (Mxmn):
            rhs_stab = 0*np.zeros((nrs,1))
            nucolmm = np.append(nucolmm,np.ones((nrs,1)))
        else:
            rhs_stab = -1*np.ones((nrs,1))
        rhs = np.append(rhs, rhs_stab)
        rowlabels = rowlabels + rowlabels_stab
        #set_ind = np.row_stack((set_ind,stab_columns[1:,:]))
        set_ind = np.vstack((set_ind,stab_columns))
        if Mxmn:
            set_ind = np.column_stack((set_ind,nucolmm))
            obj = 0* obj
            obj = np.append(obj,-1)
            firm_no.append(firm_no[-1]+1)
            set_assgn.append(set())

    #explicit slack variables
    if AllEq:  
        #now stick a big identity matrix on the end of the constraint matrix
        nr, nc = set_ind.shape
        set_ind = np.column_stack((set_ind,np.identity(nr)))
        #now fix up the column labels
        columnfirms = list(range(nc,nc+nr))
        #columnsets = [set(["slack"])]*nr
        columnsets = [set(["S", str(item)]) for item in rowlabels]
        firm_no = firm_no + columnfirms
        set_assgn = set_assgn + columnsets
        obj = np.append(obj, np.array([0]*nr))

    return(set_ind, rhs, obj, firm_no, set_assgn,rowlabels,stab_columns)
    
#format, display and save the constraint matrix
#constraints:  each column is a firm/team assignment first nw rows are the team indicators
#              second nf rows are the one-subset-to-a-firm constraints
#              next set of rows (one for each firm/team pair) are the stability constraints
#              optionally can generate upper and lower bounds for each variable
#              default objective function is all 1's
#teams = the team for each column
#firms = the firm # for each column

def isStable(pw, pf, firms, teams , icol):
    #nw = #workers
    #nf = #firms
    #pf = firm preferences (maybe not needed) list of lists of workers use entries 1 though nf
    #pw = worker prefs list of firms use entries 1 through nw
    #firms = column list of firms
    #teams = column list of team assigned (list of tuples)
    #icol = a proposed independent set (i.e.matching) column indicators 1 in the set 0 not)
    #return: 0 is stable, 1 is not firm individually rational (IR),  2 is not worker IR, 3 is not stable
    #check firm individual rationality (if assgned a single set is IR because all possible sets for a firm are in that firms choice set)
    nw = len(pw)-1 # pw, pf are one longer than corresponding number of firms/workers
    nf = len(pf)-1
    # #check stability* by hand, one firm and a subset of workers who can improve their
    # match by deviating.
    firm_match = [set() for ix in range(nf+1)]#the set matched to each firm (0 for none), index in list is firm number (1 to nf)
    #worker_match = np.zeros((1,nw+1)) #index in list if worker number 
    worker_match = [0]*(nw+1) #creates a list of 0 of length nw+1. worker_match[iworker] = 0  means iworker is unmatched
    for ixt in range(0,len(icol)): #iterate through the matching determined by icol
        if icol[ixt] == 0: continue
        ixf = firms[ixt]
        xw = teams[ixt]
        if (firm_match[ixf] != set()): # if there is a non zero index in firm_match then that firm has already been assigned a team
            return(1,'Firm matched to two sets.**********************************')
        else:
            firm_match[ixf] =  xw # firms[ixt] has been matched to team xw
            for ixw in xw:
                if (worker_match[ixw] != 0):
                    return(4,'Worker matched to more than one firm*******************************') #multiply assigned worker
                else:
                    worker_match[ixw]=ixf
    outstr = ""
    #outstr = "Implied Matching: \n"
    #firmlst = [f"Firm: {item} Subset: {firm_match[item]}" for item in range(1,nf+1)]
    #worklst = [f"Worker: {item} Firm: {worker_match[item]}" for item in range(1,nw+1)]
    #outstr += "| ".join(firmlst) + "\n"
    #outstr += "| ".join(worklst) + "\n"
    #outstr += "\n" + f"Independent Set: {icol[0]}"
    #outstr +=  "\n" + f"Firm Assignments: {firm_match}" + "\n"
    #outstr +=  "\n" + f"Worker Assignments: {worker_match}" + "\n"

    #now look for a blocking coalition
    subset_preferring = [set() for ix in range(nf+1)]
    isBlocked = False
    for ixf in range(1,nf+1):
        #for firm ixf find the set of workers that prefers ixf to their current match
        for ixw in range(1,nw+1):
            if (workerPref(ixw,pw,ixf,worker_match[ixw])):
                subset_preferring[ixf].add(ixw)
        outstr += f"Workers preferring firm {ixf}: {subset_preferring[ixf]}\n"
    #outstr = outstr + "\n" +f"Workers preferring each firm {subset_preferring}" + "\n"
    #now look for a blocking coalition: one firm and a set of workers
    #for each firm add take the union of the set of workers that prefer that firm to their matched firm
    #check to see if this set contains a set that the firm likes better than it's matched set
    bfirm = 0
    bset = set()
    for ixf in range(1,nf+1):
        if (subset_preferring[ixf] == set()): continue
        matched_set = firm_match[ixf]
        test_set = matched_set.union(subset_preferring[ixf]) #test set is union of the set assigned to ixf and proposed blocking worker set
        for ixs in pf[ixf]: #check each set in ixf's preference list
            if (ixs.issubset(test_set)) & (firmPref(ixf,pf,ixs,matched_set)): #found a blocking firm and set of workers
                if (len(bset) == 0) | (len(ixs) < len(bset)): 
                    bset = ixs - matched_set
                    bfirm = ixf
                outstr += f"Blocked by Firm:{ixf} Subset:{ixs}\n"
                #return(3,f"Blocked by Firm:{ixf} Subset:{ixs}")
                isBlocked = True
    #outstr += "Stable"
    if isBlocked:
        outstr += f"Smallest block,  Firm:{bfirm} Subset:{bset}\n"
        return(3,outstr)
    return(0,outstr)


def displayLP(constraints = [], rhs = [], obj = [], teams = [], firms = [],rowlabels = [], results = []):
    colset = [str(item) for item in teams]
    colname = list(zip(firms,colset))
    #colname = [(item[0],set(item[1])) for item in colname]
    constraints = pd.DataFrame(data = constraints, columns=colname)
    constraints['RHS'] = rhs
    constraints['RowLabels'] = rowlabels
    constraints.index = [item+1 for item in constraints.index]
    constraints.loc[constraints.index.max() + 1] = list(obj) + [' ','OBJ']
    if (len(results) != 0): 
        constraints['Slack'] = np.append(results.ineqlin.residual,0)
        constraints['Dual'] = np.append(results.ineqlin.marginals,0)
        x = results.x
#        constraints.loc[len(constraints)+1] = list(x) + ['X',' ',' ',' ']
        constraints.loc[len(constraints)+1] = list(x) + ['X','Primal',0,0]
    #constraints.to_csv("LPmodel_2.csv")
    return(constraints)

def decodeSolution(firms = [], teams = [],  RowLabels = [], lp_Result = []):
    if (len(firms)== 0): 
            msgtxt = " Input: three vectors of length equal to number of columns in LP.\n"
            msgtxt += " firms= afirm for each column, teams = a set of workers for each column, \n"
            msgtxt +=   " solution = solution vector of length = # columns"
            return(msgtxt)
    outstring = ''
    if  (len(lp_Result)==0): 
        msgtxt = 'No solution to decode.'
        return(msgtxt)
    if (lp_Result.status != 0): 
        outstring += "Optimization misunderstanding status = {lp_Result.status}"
        return(outstring)
    solution = lp_Result.x
    dual_solution = lp_Result.ineqlin.marginals
    if len(RowLabels) == 0 :
        Rowlabels = [zx for zx in range(len(dual_solution))]
    for zx in range(0,len(solution)):
        if solution[zx] > 0.0:
            outstring = outstring + f"Firm: {firms[zx]}, Assigned Set: {teams[zx]} weight:{solution[zx]} \n"
    for zx in range(0,len(dual_solution)):
        if dual_solution[zx] != 0.0:
            outstring = outstring + f"Row {RowLabels[zx]}, Dual: {dual_solution[zx]}\n"
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
    return(intersection_mat) #this is the intersection matrix of the columns (dimension nc x nc)
    
def doIndependentSets(inmat,teams,firms, pw, pf, OneSet = False, StabConst = [],StabOnly = False, Verbose = False, StabCheck = False):
    #very inefficient brute force enumeration
    #enumerate all of subsets of columns
    #inmat is the incidence matrix for the intersection graph of the constraint matrix
    #it is n by n with n= number of columns (and in the same order) as the constraint matrix
    nr,nc = inmat.shape
###################
    # the brute force part:
    #if the one-set-to-a-firm constraints have been triggered, then find feasible extreme points by 
    #enumerating the power set of the set of columns and saving just the feasible extreme points
    #if the o-s-t-a-f contraints have not been triggered generate candidate extreme points
    #by enumerating only those subsets of columns that obey the o-s-t-a-f constraints.  This generates 
    #many fewer columns

    colsubsets = [[]]
    #if (OneSet ):#always use the more efficient enumeration scheme
    if (1 == 0):
        colset = list(range(nc))
        colsubsets = [[]]
        for colitem in colset:
            colsubsets += [item + [colitem] for item in colsubsets]
    else: # this code alters the extreme point enumeration to consider only extreme points that satisfy the one-set-to-a-firm constraints
        ftemp = [[indx for indx in range(len(firms)) if firms[indx] == tmp] for tmp in np.unique(firms)]
        ftemp = [[-1] + fitem for fitem in ftemp]
        colsubsetsnu = [list(item) for item in it.product(*ftemp)]
        colsubsets = [[item for item in colitem if item != -1] for colitem in colsubsetsnu]


    #now test every subset record results in is_independent 
    #is an indicator for whether or not the given subset is an independent set
    
    is_independent = np.ones((1,len(colsubsets))) #assume a subset of columns is independent until proven dependent
    for indx, item in enumerate(colsubsets): #for each subset of columns
        #item is the current subset of columns we are working on
        #indx is the position in colsubsets where that item resides
        #print(f">>>>>>>>Testing column set: {item}")
        if len(item) == 0: #empty set of columns is independent
            is_independent[0,indx] = 1
            continue
        if len(item) == 1: #any set of columns consisting of one column is independent
            is_independent[0,indx] = 1
            continue
        test_list = list(it.combinations(tuple(item),2)) #all sets of size 2 from the set of columns given by item
        #check each pair of columns in the current member of colsubsets for common 1's using the intersection matrix
        for xtpl in test_list:
            if inmat[xtpl[1],xtpl[0]] == 1: #xptl[1] and xptl[2] have an arc between them (inmat is symmetric as arcs are undirected only need to check one of (i,j) and (j,i))
                is_independent[0,indx] = 0
                break;
    # the list of independent sets is complete. is_independent[ix] = 1 iff the ix'th subset of colsubsets is independent and 0 otherwise
    # 
    # Now reconstruct the worker subsets and firms that go with them and test each one for stability
    indcol = np.zeros((nc,1))#record the independent set as a column of column # indicators; #columns = # indep. sets.
    stability_index = [0] # record the stability index for each independent column
    solution_count = 0
    stab_count = 0
    stringout = ''
    newstring = ''
    stringout = f"Number of column subsets enumerated: {len(colsubsets)} \n"
    stringout += f"Number of feasible solutions (independent sets) found: {sum(is_independent[0,:])} \n"
    warnstring = '-'
    for indx,item in enumerate(is_independent[0,:]):
        sstatus = 0
        if (item == 1): #this entry corresponds to an independent set
            newstring = '\n\n*************************************\n'
            newstring += f"set #: {indx}, solution count: {solution_count}" + "\n"
            newstring += f"Independent set of columns: {colsubsets[indx]}" + "\n"
            newstring += '************************************* \n'
            cols = colsubsets[indx] #find the corresponding set of workers
            newindcol = np.zeros((nc,1)) #create a column length vector to record which columns are active.
            newstring += "Independent Set Details:" + "\n"
            for itemx in cols:
                newstring += f"Firm: {firms[itemx]} Subset: {teams[itemx]}" + "\n"
                newindcol[itemx,0] = 1
            if sum(newindcol[:,0])==0: newstring += '\n'
            indcol = np.column_stack((indcol,newindcol))    #add the column to the array of columns
            #colindicator = [item for item in newindcol[:,0].tolist()] #use to print out a description of the current extreme point
            #check for stability using the stability constraint
            stind = 0
            cstatus = ' '
            if (len(StabConst) != 0 ):
                stabtest = np.matmul(StabConst,newindcol[:StabConst.shape[1],0])  #take just the columns of primal variables involved in the stability constraints
                #the second argument is a one dimensional array, matmul will return a one dimensional array (access as stabtest[ix], not stabtest[ix,0])
                stind = sum(stabtest >= 0) #matmul seems to return a one dimensional array
                stability_index.append(stind)
                if Verbose:
                    newstring += f" stability calculation :{np.array_str(stabtest)} \n"
                temp =max(stabtest)
                if (temp < 0): 
                    cstatus = f"Stable* stability index = {stind}" 
                else: cstatus = f"Not Stable* stability index = {stind}"
            else:
                stind = 0
                stability_index.append(stind)
                cstatus = f"Empty stability constraint matrix, stable."
            #check stability from scratch by checking all possible blocking coalitions
            if (StabCheck):
                sstatus,stabstr = isStable(pw,pf,firms,teams,newindcol)
            else:
                sstatus = stind
                stabstr = 'Stability constraints only, external stability check off.'
                
            if ((sstatus == 0) & (stind !=0)) | ((sstatus >0) & (stind == 0)):
                newstring += "Stability checks disagree!! at ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!!\n"
                warnstring = f"\n Stability check inconsistency at solution {solution_count} indepset {indx} LP columns {cols}\n"
            if ((sstatus > 0) | (stind >0)):
                if (not StabOnly): #we are printing details on the non-stable matchings
                    newstring += f"{stabstr}\n --------- Not Stable*  ------\n sstatus = {sstatus}--cstatus = {cstatus}\n"
                    if Verbose:
                        newstring +=  stabstr #verbose mode:  report all matchings
                        #stringout += f"{np.array_str(stabtest)}"
                else: #we are not printing anything for non-stable matchings
                    newstring = ' '
            else:
                stab_count += 1
                #temp = ','.join(colindicator)
                #print(f"Stable match, col indicator: ", colindicator)
                if Verbose:
                    newstring += f"{stabstr} \n###### *****Stable***  ####### \n sstatus  = {sstatus}--cstatus = {cstatus} \n "
                else:
                    newstring += f" ###### *****Stable***  ######\n sstatus = {sstatus}--cstatus = {cstatus}  \n "
                
            stringout += newstring
            solution_count += 1

    #print(f">>>>>>>>>>>>>>>>>>> number of stable solutions: {stab_count}")
    indcol = indcol[:,1:] #remove the first column of 0's
    stability_index = stability_index[1:]
    if stringout == '': 
        stringout = 'Nada'
    else:
        stringout += f"\n Number of stable* matchings found: {stab_count}"
        if warnstring != '-': stringout+= warnstring
    return(indcol,stability_index,stab_count,stringout) #indcol is the matrix whose columns are the independent sets, stringout is the printable results

              
def extremeAdjacency(idCols, indep_mat):  #indep_mat is the intersection graph (adjacency matrix) of the columns                                  
    arclist = []
    nonarclist = []
    nr,nc = idCols.shape #idcols is the matrix of independent sets,
    #print("::::::::::::::  intersection graph ::::::::::::::::::::")
    grph = nx.from_numpy_array(indep_mat,create_using=nx.Graph)
    A = nx.adjacency_matrix(grph)
    #print(A.toarray())
    for ix in range(nc):
        for iy in range(nc):
            if iy >= ix: break
            #print(" ")
            #print("================================++++++++++++++++++++++++++++++++++++++++")
            #print(f"Extreme PT 1(red): (#{ix}), cols:{np.where(np.array(idCols[:,ix]))[0]}, Extreme PT 2(green): (#{iy}), cols: {np.where(np.array(idCols[:,iy]))[0]} ")
            #ix is and independent set and iy is another independent set
            active1 = idCols[:,ix]
            active2 = idCols[:,iy]
            if (sum(active1) + sum(active2) == 0): continue
            #if (sum(active1)  == 0): continue
            #if (sum(active2)  == 0): continue
         #now calculate the symmetric difference. Note active1 and active2 have the same length equal to the number of columns
            active = [ 1 if (((active1[ix] == 1) & (active2[ix] == 0)) | ((active1[ix] == 0) & (active2[ix] == 1))) else 0 for ix in list(range(0,len(active1)))]
            #now find the subgraph of the intersection graph
            nri,nci = indep_mat.shape  #should be nxn so nri == nci = True
            if (nri != nci) : 
                print(f"intersection graph incidence matrix not square! shape: {indep_mat.shape}")
                continue
            #remove arcs not between nodes of the symmetric diff
            imatnew = indep_mat.copy()
            for ixi in range(0,nri):
                for jxi in range(0,nci):
                    if ((active[ixi] == 0) | (active[jxi] == 0)):
                        imatnew[ixi,jxi] =0
            #print(f"symmetric diff adjacency matrix sum = {sum(sum(imatnew))}")
            #print(imatnew)
            #retain rows and columns corresponding to nodes in the symmetric difference only
            #create the subgraph of nodes in the symmetric difference
            sym_dif_nodes = [ix for ix in list(range(0,len(active))) if active[ix]==1]
            #now create adjacency matrix of just the subgraph of nodes in the symmetric difference
            imatcols = [ixl for ixl in sym_dif_nodes]
            imatrows = imatcols
            imatnew = imatnew[:,imatcols]
            imatnew = imatnew[imatrows,:]
            #print("symmetric diff adjacency sub-matrix")
            #print(imatnew)
            cres = isConnected_Imat(imatnew)
            if cres > 0:  
                arclist.append((ix,iy))
                #print(f"Connected: arc: {(ix,iy)}")
            else:
                nonarclist.append((ix,iy))
            #if cres == 0:
            #    #print("NOT Connected")
            if cres == -1:
                print(f"Dimension failure.  imat.shape = {imatnew.shape}") 
            elif cres == -2:
                print(f"imat has no nonzero entries.  {imatnew}")   
            elif cres == -3:
                print(f"No edges recovered from imat. \n {imatnew}") 
            elif cres == -4: 
                print(f"Edge list empty")     
    return arclist, nonarclist

def isConnected_Imat(imat):
    x,y = imat.shape

    #print(f"@@@@@  Entering isConnected_Imat dimensions: ({x},{y}) ")
    #print(imat)

    #if (x == 0) | (y == 0): return -1
    #if (x != y): return(-1)
    #if (sum(sum(imat)) == 0): return -2

    grph = nx.from_numpy_array(imat,create_using=nx.Graph)
    #print("checking: adjacency matrix of symmetric diff graph")
    #print(nx.adjacency_matrix(grph))
    if nx.is_connected(grph):
        return 1
    else:
        return 0



if __name__ == "__main__":    
    
###################################################################    
###################################################################    
###################################################################    
###################################################################    
###################################################################    
    
     


#### Test Examples H4, H8, EO19, B1
    Example = 'SUBCOMP_12-1'
    Vbose = False
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
        
        #pf[1] = [{1,3}, {2,4},{3}]  
        #pf[2] = [{1,3}, {2,4},{4}]   
        pf[1] = [{1,3}, {2,4}]  
        pf[2] = [{1,3}, {2,4}]   
        
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [1,2]
        pw[4] = [1,2]
#######################################
    elif Example == 'B3':
##################################### 
### Bikhchandani substitutes/complements example
### with extra sets
        nw = 4
        nf = 2
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
        
        #pf[1] = [{1,3}, {2,4},{3}]  
        #pf[2] = [{1,3}, {2,4},{4}]   
        pf[1] = [{1,3}, {2,4}]  
        pf[2] = [{2,4}, {1,3}]   
        
        pw[1] = [1,2]
        pw[2] = [2,1]
        pw[3] = [1,2]
        pw[4] = [1,2]
#######################################

    elif Example =='TU':
        nw = 6
        nf = 4
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
        firms = 4
        pf[1]=[{1,2,3,4}, {2,3}, {3,4}]
        pf[2]=[{3,4,5,6},{5,6},{6}]
        pf[3]=[{2,3,4,5},{2,3,4},{3,4}]
        pf[4]=[{3,4},{2,3,4}, {3,4,5}]
        workers = 6
        pw[1]=[1,4,2,3]
        pw[2]=[4,1,2,3]
        pw[3]=[2,3,4,1]
        pw[4]=[3,2,1,4]
        pw[5]=[1,2,3,4]
        pw[6]=[1,4,2,3]

    elif Example =='SUBCOMP':
        nw = 6  #of workers
        nf = 3  #of firms
    
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))


        pf[1]= [{1,4,6},{1,5,6},{2,5,6}]
        pf[2]= [{2,4,6}]
        pf[3]= [{2,5}]

        pw[1]=[1,2,3]
        pw[2]=[1,2,3]
        pw[3]=[1,2,3]
        pw[4]=[2,3,0]
        pw[5]=[2,3,1]
        pw[6]=[1,2,3]

    elif Example =='BIG':
        nw = 9  #of workers
        nf = 6  #of firms
    
        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
        pf[1] = [{1, 2, 3, 4}, {2, 3}, {1, 4}, {1}, {2}] 
        pf[2] = [{8, 9, 7}, {8, 7}, {9, 7}, {8}, {9}] 
        pf[3] = [{1, 3, 5, 7, 9}, {8, 2, 4, 6}] 
        pf[4] = [{8, 9, 6, 7}, {8, 9, 7}, {9, 7}, {8, 9}] 
        pf[5] = [{1}, {2}, {3}, {4}, {5}] 
        pf[6] = [{9}, {8}, {7}, {6}] 
        pw[1] = [1, 2, 3, 4, 5, 6]
        pw[2] = [4, 2, 6]
        pw[3] = [6, 5, 4, 3, 2, 1]
        pw[4] = [1, 2, 3, 4, 5, 6]
        pw[5] = [4, 2, 6]
        pw[6] = [6, 5, 4, 3, 2, 1]
        pw[7] = [1, 2, 3, 4, 5, 6]
        pw[8] = [4, 2, 6]
        pw[9] = [6, 5, 4, 3, 2, 1] 

    elif Example =='HUPS':
        nw = 3  #of workers
        nf = 3  #of firms

        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))

        pf[1] = [{1, 2}] 
        pf[2] = [{2, 3}] 
        pf[3] = [{1, 3}] 
        pw[1] = [1, 3]
        pw[2] = [2, 1]
        pw[3] = [3, 2]

    elif Example =='HUPS_EX1': 

        nw = 6  #of workers
        nf = 5  #of firms

        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))

        pf[1] = [{1, 2, 3}] 
        pf[2] = [{1, 2}, {1}] 
        pf[3] = [{2, 3}, {2}] 
        pf[4] = [{3, 4}, {3}] 
        pf[5] = [{4, 5, 6}] 
        
        pw[1] = [1, 2, 3, 4, 5]
        pw[2] = [1, 2, 3, 4, 5]
        pw[3] = [1, 2, 3, 4, 5]
        pw[4] = [1, 2, 3, 4, 5]
        pw[5] = [1, 2, 3, 4, 5]
        pw[6] = [1, 2, 3, 4, 5]

    elif Example =='SUBCOMP_12-1': 

        nw = 4  #of workers
        nf = 2  #of firms

        pf = list(range(0,nf + 1))
        pw = list(range(0,nw + 1))
        
        pf[1] = [{1,2},{1,4},{3,4},{3,2}]
        pf[2] = [{3,4},{3,2},{1,4},{1,2}]
        pw[1] = [1, 2, 3]
        pw[2] = [1, 2, 3]
        pw[3] = [1, 2, 3]
        pw[4] = [3, 2, 1]
    
# Exhaustively enumerate worker preferences 
    print("########################")
    print(f" Example: {Example}")
    print("########################")
    

    Solve_one = False

    Find_nostab = True

    if (Solve_one):
        maxminS = False #set up a max min variable and put stability into the objective
        ####Create the constraint matrix
        #const_mat,rhs,obj,firms,teams,rowlab,stab_constr = doLP(nw, nf, pw, pf, DoOneSet = False, StabilityConstraints = True, Dual = False, Verbose = Vbose)

        print(f"At start: nf= {nf}, nw = {nw} \n pw = {pw} pf={pf}")
        const_mat,rhs,obj,firms,teams,rowlab,stab_constr = doLP(nw, nf, pw, pf, DoWorkers = True, DoOneSet = True, StabilityConstraints = True, Dual = True, Mxmn = maxminS, Verbose = Vbose)
        #imat = doIntersectionGraph(const_mat)
        #independent_columns,stab_index,stab_count,txt_out = doIndependentSets(imat, teams, firms,pw, pf,  StabConst = stab_constr,StabOnly = True)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print(txt_out)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        dfLP0 = displayLP(constraints = const_mat, rhs = rhs, obj = obj, teams = teams, firms = firms, rowlabels = rowlab)
        bounds = [(0,1) for item in obj]
        bounds[-1] = (0,None)
        solution = solveIt(CONSTRAINTS = const_mat, RHS = rhs, OBJ = obj, Verbose = True) 
        dfLP = displayLP(constraints = const_mat, rhs = rhs, obj = obj, teams = teams, firms = firms, rowlabels = rowlab, results = solution)
        decodeSolution(firms, teams, solution)

        with pd.option_context('display.max_rows', 50,
                                'display.max_columns', None,
                                'display.width', 1000,
                                'display.precision', 0,
                                'display.colheader_justify', 'right'):
            print(dfLP0)
            print(dfLP)
            print("pw: ")
            print(pw)
            print("pf: ")
            print(pf)
 
    if Find_nostab:
        count = 0
        maxstabcount = 0
        firmlist = list(range(1,nf+1)) + [0]*(nf)
        print(f"Firm list: {firmlist}")
        firmperm = list(it.permutations(firmlist,nf))
        workerperm = it.product(range(len(firmperm)),repeat = nw)
        UNSTABLE = False
        time0 = time.time()
        time1 = time0
        for item in workerperm:
            pw = [0] + [list(firmperm[itemx]) for itemx in item]
            const_mat,rhs,obj,firms,teams,rowlab,stab_constr = doLP(nw, nf, pw, pf, DoWorkers = True, DoOneSet = True, StabilityConstraints = False, Mxmn = False, Dual = False, Verbose = Vbose)
            dfLP = displayLP(constraints = const_mat, rhs = rhs, obj = obj, teams = teams, firms = firms, rowlabels = rowlab)
            count += 1
            if (const_mat.size == 0):
                #count = count + 1
                #print("Null constraint matrix.")
                continue
            if (count <=1):
                with pd.option_context('display.max_rows', 50,
                                'display.max_columns', None,
                                'display.width', 1000,
                                'display.precision', 0,
                                'display.colheader_justify', 'right'):
                    print(const_mat)
                    print(dfLP)
                    print("pw: ")
                    print(pw)
                    print("pf: ")
                    print(pf)

            imat = doIntersectionGraph(const_mat)
            independent_columns,stab_index,stab_count,txt_out = doIndependentSets(imat, teams, firms,pw, pf,  StabConst = stab_constr,StabOnly = True)
            if len(stab_index) > 0:
                if min([item[0]  for item in stab_index]) > 0:
                    print(f"No Stable* Solution Found at iteration {count}. Time = {time.time()-time0}")
                    print(f"Preferences: {pw}")
                    UNSTABLE = True
                    break
                #print(f"Number of stable solutions: {stab_count} iteration: {count}")
            #count = count + 1
            if stab_count > maxstabcount: maxstabcount = stab_count
            #if stab_count > 1: print(f"Mulitple Stable Matchings found: number= {stab_count}, iteration = {count} preferences = {pw}")
            if count % 5000 == 0:
                print(f"Progress: count = {count}")
            if count % 10000 == 0:
                time2 = time.time()
                print(f"Avg iteration per second, last 10,000: {10000/(time2-time1)}, since beginning: {count/(time2-time0)}")
                time1 = time2
                print(f"Worker preferences: {pw}")
        #print(f"count: {count}")
        time2 = time.time()
        print(f"Iterations per second: {count/(time2-time0)}")
        print(f"maximal number of stable matchings found in any worker preference instance : {maxstabcount}")
        if not(UNSTABLE):       
            print(f"All models contained stable solutions count = {count} max # of stable solutions found in any example {maxstabcount}")








    # if Vbose:
    #     with pd.option_context('display.max_rows', 50,
    #                            'display.max_columns', None,
    #                            'display.width', 1000,
    #                            'display.precision', 0,
    #                            'display.colheader_justify', 'right'):
                  
    #         print(dfLP)
    
    
 
    # if TestTu:
    #     IS_TU = checkTU(const_mat, Verbose = False)
    #     if (IS_TU):
    #         print(" Is TU")
    #     else:
    #         print(" is NOT TU")
        
        
    # solution = solveIt(CONSTRAINTS = const_mat, RHS = rhs, OBJ = obj, Verbose = Vbose) 

    # #dfLP = displayLP(constraints = const_mat, rhs = rhs, obj = obj, teams = teams, firms = firms, rowlabels = rowlab, results = solution )

    # decodeSolution(firms, teams, solution)
    # if solution.status != 0:
    #     print("Uh oh!")
    # else:
    #     dfLP = displayLP(constraints = const_mat, rhs = rhs, obj = obj, teams = teams, firms = firms, rowlabels = rowlab, results = solution )
    #     if Vbose:
    #         with pd.option_context('display.max_rows', 50,
    #                            'display.max_columns', None,
    #                            'display.width', 1000,
    #                            'display.precision', 0,
    #                            'display.colheader_justify', 'right'):                 
    #             print(dfLP)

    # imat = doIntersectionGraph(const_mat)
    
    # with pd.option_context('display.max_rows', 50,
    #                            'display.max_columns', None,
    #                            'display.width', 1000,
    #                            'display.precision', 0,
    #                            'display.colheader_justify', 'right'):
    #     print("constraint matrix:")      
    #     print(const_mat)
    #     print("intersection graph incidence matrix")
    #     print(imat)
    # independent_columns,stab_index,txt_out = doIndependentSets(imat, teams, firms,pw, pf,  StabConst = stab_constr,StabOnly = False)
    # print(txt_out)
    # print("Extremal (integral) feasible solutions")
    # print(independent_columns)
    # print("stability test:")
    # print(stab_index)
    # print("Are any of the integer solutions stable?")
    # c = np.matmul(stab_constr,independent_columns)
    # nr,nc = independent_columns.shape
    # for ixc in range(0,nc):
    #     col = independent_columns[:,ixc]
    #     res,stab_status = isStable(pw,pf,firms,teams,col)
    #     print(f"{ixc}: stability status = {res}")
    # print("Stability constraint value:")
    # print(c)
    # print("Adjacency arc data for extreme point solutions:")
    # extreme_arcs,nonextreme_arcs = extremeAdjacency(independent_columns, imat)
    # #arcst = [(a[0],a[1]) for a in extreme_arcs]
    # #nonarcst = [(a[0],a[1]) for a in nonextreme_arcs]
    # G = nx.Graph()
    # G.add_edges_from(extreme_arcs)
    # nx.draw_networkx(G)
    # plt.show()

    # nonG = nx.Graph()
    # nonG.add_edges_from(nonextreme_arcs)
    # nx.draw_networkx(nonG)
    # plt.show()



    
