#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:33:04 2024

@author: john
"""

import pandas as pd
import numpy as np
import itertools
from scipy.optimize import linprog
from pandas import option_context
import Matching
import ReadMatchData
import re
import idGraph
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import shinywidgets
from shinywidgets import render_widget, output_widget

import os
import signal
from datetime import datetime

import networkx as nx

lpchoices = ["add worker constraint","add 1 set per firm","add stability const.","dualize stab. constr.","card. match"]
cmap = cm.get_cmap('viridis')

app_ui = ui.page_navbar( 
    ui.nav_panel("Input", 
        ui.row(
            ui.HTML("<p>Either choose file or type data in below, then click read data when ready to proceed.</p>"),
            ),
        ui.row(
                ui.column(4,offset = 0,*[ui.input_file("file1", "Choose .txt File", accept=[".csv", ".CSV", ".dta", ".DTA"," .txt", ".TXT"], multiple=False, placeholder = '', width = "600px")]),
            ),
        ui.row(
                ui.input_text_area("inputdatalog","Input Data (editable):",value = '-', width = "600px", height = "400px")
            ),        
        ui.row(
                #ui.column(3,offset=0,*[ui.input_action_button("datalogGo","Show Data (after reading)",width = '600px')]),
            ),
        ui.row(
                ui.column(2,offset = 0,*[ui.input_action_button("doUpdate","Read Input Data",width = '200px')]),
                #ui.column(1,offset = 0),
                ui.column(2,offset = 0,*[ui.download_button("download_data","Save Input Data",width = "200px")]),
                #ui.column(1,offset = 0),
                ui.column(2,offset = 0,*[ui.input_action_button("doReset","Reset Everything",width = '200px')]),
                ui.column(6,offset = 0),

                #ui.column(3,offset=0,*[ui.input_action_button("datalogGo","Show Data",width = '300px')]),
                #ui.column(3,offset=0,*[ui.input_action_button("datalogUpdate","Show Data",width = '300px')])
            ),
        ui.row(ui.HTML("<p> </p>"),
            ),
        ui.row( 
                ui.HTML("<p>Worker and Firm preferences:</p>"),
                ui.output_text_verbatim("datalog"),
            ),        
        ),
    ui.nav_panel("Linear/Integer Program",
        #ui.row(
                #ui.column(4,offset = 0,*[ui.input_file("file1", "Choose .txt File", accept=[".csv", ".CSV", ".dta", ".DTA"," .txt", ".TXT"], multiple=False, placeholder = '', width = "600px")]),
            #),
        ui.row(
                ui.column(3, offset = 0,*[ui.input_action_button('generateLP',"Generate LP")]),
                ui.column(3, offset = 0,*[ui.input_action_button('solveLP',"Solve LP")]),
                ui.column(6, offset = 0,*[ui.input_checkbox_group("genoptions","Options: ",choices = lpchoices,selected = ["add worker constraint","add 1 set per firm"],
                          width = "500px",inline = True)]),
            ),
        ui.row(
                #ui.output_data_frame("LPOut"),
                ui.output_text_verbatim("LPOut"),
                ui.output_text_verbatim("LPSolvedOut"),
            ),
        ui.row(
             ui.column(3, offset = 0,*[ui.input_action_button('testTU',"Test TU: (this can take time)")])
            ),
        ui.row(
             ui.output_text_verbatim("TUreport")
            ),
        ui.row(
             #ui.column(3, offset = 0,*[ui.input_action_button("solveIt","Solve LP")])
            ),
        ),
    ui.nav_panel("Enumeration",
        ui.row(
                ui.column(3,offset=0,*[ui.input_action_button("goextreme","Enumerate Extreme Points",width = '300px')]),
                ui.column(6,offset=0,*[ui.input_radio_buttons("stype","Show: ",choices = ['All Extreme Points','Stable Extreme Points Only'],inline = True)]),
                ui.column(3,offset=0,*[ui.input_radio_buttons("vmode","Output: ",choices = ['Terse','Verbose'],selected = 'Terse',inline = True)]),
            ),
        ui.row(
                ui.output_text_verbatim("extremelog"),
            ),
        ui.row(
                ui.column(3,offset=0,*[ui.input_action_button("gointersection","Show Intersection Graph",width = '300px')]),
            ),
        ui.row(
                ui.column(6,offset=0,*[ui.output_text_verbatim("intgraph")]),
                
            ),
        ui.row(
            ui.column(6,offset=0, *[ui.input_numeric("iset","Show Independent Set",value = -1,min=-2,max=-1,step=1)]),
            ),
        ui.row( 
                ui.column(6, offset = 0,*[ui.output_plot("intgraphPic",width = "800px",height="800px")]),
                #ui.column(6, offset = 0,*[ui.output_plot("subgraphPic",width = "800px",height="800px")]),
                height = "1200px", 
            ),
        ),
    ui.nav_panel("PairwiseAdjacency",
        ui.row(
            ui.column(4,offset = 0,*[ui.input_numeric("iset1","First Indep. Set (red):", value = -1, min = -2, max = -1)]),
            ui.column(4,offset = 0,*[ui.input_numeric("iset2","Second Indep. Set (green):", value = -1, min = -2, max = -1)]),
            ),
        ui.row(
            ui.column(4,offset = 0,*[ui.output_text_verbatim("aset1")]),
            ui.column(4,offset = 0,*[ui.output_text_verbatim("aset2")]),
            ),
        ui.row(
            ui.column(6, offset = 0, *[ui.output_plot("idSetsPic", width = "800px", height="800px")]),
            ui.column(6, offset = 0, *[ui.output_plot("adjacencyPic", width = "800px", height="800px")]),
            ),
        ),
    ui.nav_panel("Adjacency",
        ui.input_action_button("doEPAdjacency","Show Extreme Point Adjacency Graph"),
        ui.output_plot("EPAdjacency", width = "1000px", height = "900px"),
    ),
underline = True, title = "Stable Matcher 3.0 ", position = "fixed_top")
                 
def server(input: Inputs, output: Outputs, session: Session):
    
    nw = reactive.value(0) # number of workers 1,..,nw
    nf = reactive.value(0) # number of firms 1,...,fn
    pw = reactive.value([]) #worker preferences 
    pf = reactive.value([]) # firm preferences
    cmat = reactive.value([]) #constraint matrix (for TU checking)
    crhs = reactive.value([]) #righthand sides
    cobj = reactive.value({}) #objective function
    df_LP = reactive.value(pd.DataFrame()) # Full generated LP with row and column labels for display mostly
    res_LP = reactive.value(None)
    solution_LP = reactive.value('') #solution to the LP as a string
    row_labels_LP = reactive.value([]) #labels to go with LP formulation rows
    imat_G = reactive.value([]) #incidence matrix of the intersection graph
    teams_LP = reactive.value([]) #the team that goes with each column of cmat or df_LP
    firms_LP = reactive.value([]) #the worker that goes with each column of cmat or df_LP
    stab_constr_LP = reactive.value([]) # just the stability constraint coefficients
    input_data = reactive.value([]) #this is the cleaned up input data: a list of lines, no spaces, no comments
    output_data = reactive.value([]) # the "compiled" version of the input model for display
    indep_cols = reactive.value([]) #the characteristic vectors for the independent sets of columns of imat_G (rows=#columns in the LP, #cols = # indep sets)
    indep_cols_stab = reactive.value([]) #Stability index for each of the sets of independent columns
    extreme_points = reactive.value('') #text output from extreme point enumeration.
    TU_msg = reactive.value('')

    @reactive.effect
    @reactive.event(input.doReset)
    def doReset():
        resetIt()
        # nw.set(0)
        # nf.set(0)
        # cmat.set([])
        # df_LP.set('')
        # solution_LP.set('')
        # output_data.set('')
        # extreme_points.set('')
        # TU_msg.set('')
        # imat_G.set([])
        # ui.update_numeric("iset",value = -2, min = -2)

    def resetIt():
        nw.set(0)
        nf.set(0)
        cmat.set([])
        df_LP.set('')
        solution_LP.set('')
        output_data.set('')
        extreme_points.set('')
        TU_msg.set('')
        imat_G.set([])
        ui.update_numeric("iset",value = -2, min = -2)
        res_LP.set(None)
        df_LP.set(pd.DataFrame())
        indep_cols.set([])
  

    @render.download(filename="MatchData_"+str(datetime.now())+".txt")
    def download_data():
        yield input.inputdatalog()



    #@reactive.calc
    @reactive.effect
    def get_file():
        #print(f"$$$$$$$$$$$$$$$$$$Path: {str(input.file1()[0]['datapath'])}")
        if input.file1() is None:
            return('-')
        else: #Note the ui passes only paths ending in  .csv, .CSV, .dta, .DTA and .txt
            fpath = str(input.file1()[0]['datapath'])
            #print(f"$$$$$$$$$$$$$$$$$$Path: {fpath}")
            if (fpath[-4:] == '.txt') or (fpath[-4:] == '.TXT'):
                data_in = ReadMatchData.readFile(fpath)
                #temp = "\n".join(data_in)
                ui.update_text_area("inputdatalog", value = data_in)
                resetIt()
                output_data.set('Click Read Input Data')
                input_data.set(data_in)
                return(data_in)
    
    @reactive.effect
    @reactive.event(input.doUpdate)
    def recompile():
        datalist = []
        data = input.inputdatalog()
        if data == '-':
            return
            #data = get_file()
            #if data == '-': return
        #datalist = input.inputdatalog().split("\n")
        datalist = data.split("\n")
        nwt, nft, pwt, pft, dataset = ReadMatchData.readData(datalist)
        nf.set(nft)
        nw.set(nwt)
        pf.set(pft)
        pw.set(pwt)
        input_data.set(data)
        if nft == 0:
            output_data.set(dataset)
            return
        outstr = ''
        outstr = f"Number of workers = {nwt}. Number of firms = {nft} \n"
        for ix in range(1,nft+1):
            outstr = outstr + f"pf[{ix}] = {pft[ix]} \n"
        for ix in range(1,nwt+1):
            outstr = outstr + f"pw[{ix}] = {pwt[ix]}\n"
        output_data.set(outstr)
        #print("leaving recompile")
        #print(outstr)
        return

    @render.text
    #@reactive.event(input.doUpdate)
    def datalog(): 
        data = output_data()
        if len(data) == 0: 
            return('Click Read Data after choosing a file or entering data.')
        else:
            return(data)


    #@render.text
    @reactive.effect
    @reactive.event(input.goextreme)
    def extremist():
        nft = nf()
        nwt = nw()
        pft = pf()
        pwt = pw()

        if nft == 0: return('')
        if cmat() == [] : return('')

        #const_mat,rhs,obj,firms,teams,rowlab,stab_constr = Matching.doLP(nwt, nft, pwt, pft, DoOneSet = oneper, DoBounds = False, StabilityConstraints = dostab, Dual = False, Verbose = False)
        imat = Matching.doIntersectionGraph(cmat())
        imat_G.set(imat)
        if (input.stype() == "All Extreme Points"): 
            stonly = False
        else:
            stonly = True
        if "add stability const." in input.genoptions():
            outstring = "The enumeration process for extreme points requires a non-negative binary constraint matrix.\n  Remove the stability constraints and try again!"
        else:
            if input.vmode() == 'Verbose':
                vm = True
            else:
                vm = False
            independent_columns, stability_index,outstring = Matching.doIndependentSets(imat, teams_LP() , firms_LP(), pw(), pf(), StabConst = stab_constr_LP(), Verbose = vm, StabOnly = stonly)
            indep_cols.set(independent_columns)
            indep_cols_stab.set(stability_index)
            nr,nc = independent_columns.shape
            ui.update_numeric("iset", min=0, max=nc-1)
            ui.update_numeric("iset1",min=0, max=nc-1)
            ui.update_numeric("iset2",min=0, max=nc-1)
            extreme_points.set(outstring)
        return
        #return(outstring)

    @render.text
    def extremelog():
        if extreme_points() == '': return
        return(extreme_points())

    @reactive.effect
    @reactive.event(input.generateLP)
    def formulate_LP():
        #nft = parsed_file()
        solution_LP.set('')
        res_LP.set(None)
        nwt = nw()
        nft = nf()
        oneper = False
        dostab = False
        #print(f"  formulate_LP :: #workers: {nwt}, #firms: {nft}, oneper: {oneper}, dostab: {dostab}")
        if nw() == 0: 
            return        
        oneper = False
        dostab = False
        dodual = False
        cdopt = False
        doworkers = False
        if ("add 1 set per firm" in input.genoptions()):
            oneper = True
        if ("add stability const." in input.genoptions()):
            dostab = True
        if ("dualize stab. constr." in input.genoptions()):
            dodual = True
        if ("card. match" in input.genoptions()):
            dodual = False
            cdopt = True
        if ("add worker constraint" in input.genoptions()):
            doworkers = True
        cols, rhs, obj, firm_no, set_assgn, rowlabels, stab_columns = Matching.doLP(nw(), nf(),pw(),pf(),DoWorkers = doworkers,DoOneSet = oneper, StabilityConstraints=dostab, Dual = dodual, OFcard = cdopt)
        dfout = Matching.displayLP(constraints = cols, rhs = rhs, obj = obj, teams = set_assgn, firms = firm_no, rowlabels = rowlabels, results = res_LP())
        df_LP.set(dfout)
        cmat.set(cols)
        cobj.set(obj)
        crhs.set(rhs)
        teams_LP.set(set_assgn)
        firms_LP.set(firm_no)
        stab_constr_LP.set(stab_columns)
        row_labels_LP.set(rowlabels)

    #@render.data_frame
    @render.text
    @reactive.event(input.generateLP,input.solveLP,input.doReset)
    def LPOut():
        dflocal = df_LP()
        if len(dflocal) == 0:
            return "Make sure to choose a file and then click 'Read Input Data' on Input panel before clicking on 'Generate LP' "
        return dflocal.to_string() + '\n'# + solution_LP()
        #return dflocal



    @reactive.effect
    @reactive.event(input.solveLP)
    def goSolve():
        linprogstat = ["Optimization Nominal", "Iteration Limit Reached", "Infeasible", "Unbounded","Numerical Problems, call a professional."]

        #now solve it
        if len(df_LP()) == 0:
            solution_LP.set("Nothing to solve here.  Forgot to GENERATE LP?")
            return
        lp_res = Matching.solveIt(cmat(), crhs(), cobj())
        status = lp_res.status
        if status == 0:
            #outstring = Matching.decodeSolution(firms = firms_LP(), teams = teams_LP(),  RowLabels = row_labels_LP(), lp_Result = lp_res)
            #dfout = Matching.displayLP(constraints = cmat(), rhs = crhs(), obj= cobj(), teams = teams_LP(), firms = firms_LP(), rowlabels = row_labels_LP(), results = res_LP())
            #df_LP.set(dfout)
            res_LP.set(lp_res)
        else:
            outstring = f"Status: {status}, {linprogstat[status]}" 
            res_LP.set(None)
            solution_LP.set(outstring)
        #solution_LP.set(outstring)
        return


    #@render.data_frame
    @render.text
    def LPSolvedOut():
        if len(firms_LP()) == 0: return
        if res_LP() == None : 
            outstring = "You need to solve the problem before displaying the solution."
            return outstring
        outstring = Matching.decodeSolution(firms = firms_LP(), teams = teams_LP(),  RowLabels = row_labels_LP(), lp_Result = res_LP())
        dfout = Matching.displayLP(constraints = cmat(), rhs = crhs(), obj= cobj(), teams = teams_LP(), firms = firms_LP(), rowlabels = row_labels_LP(), results = res_LP())
        return outstring + '\n' +dfout.to_string()



    @render.text
    @reactive.event(input.gointersection, input.iset)    
    def intgraph():
        if (imat_G() == []): return(' ')
        ISTU, outstring = Matching.checkTU(imat_G())
        temp = np.array_str(imat_G())
        temp += '\n' + outstring
        #return(np.array_str(imat_G()))
        return(temp)
    
    @reactive.effect
    #@render.text
    @reactive.event(input.testTU)
    def e_TU():
        if cmat() == []: return("No model formulated.")
        ISTU, outstring = Matching.checkTU(cmat(), Tol = 1e-10)
        TU_msg.set(outstring)
        return
        #return(outstring)

    @render.text
    def TUreport():
        if (TU_msg() == ''): return
        return(TU_msg())

    #@reactive.effect
    #@reactive.event(input.gointersection)
    
    @render.plot
    @reactive.event(input.gointersection,input.iset)
    def intgraphPic():
        if (input.iset() == -2): 
            ui.update_numeric("iset", value = -1,min = -2)
            fig, ax = plt.subplots(1,1)
            ax.axis('off')
            return(plt.draw())
            #return(' ')
        if imat_G() == []: 
            #print("No incidence matrix, generate extreme points first.")
            fig, ax = plt.subplots(1,1)
            ax.axis('off')
            return(plt.draw())
            #return(' ')
        imat = np.array(imat_G())
        nr,nc = imat.shape  #should be nxn
        node = idGraph.nodeCoordinates(nr)
        dotx = [item[0] for item in node]
        doty = [item[1] for item in node]
        lines = idGraph.makeSegs(imat,node)
        fig, ax = plt.subplots(figsize =(8,12))
        #fig = plt.figure(figsize = (12,9), tight_layout = False)
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        if lines != []:
            lc = LineCollection(lines,linewidths = 1,colors = 'black')
            ax.add_collection(lc)
        ax.plot(dotx,doty,'o',ms = 8, markeredgecolor = 'k', markerfacecolor = 'none')
        for ix in range(0,nr):
            pfac = 1.10
            nfac = 1.5
            vfac = 1.05
            if (dotx[ix] < 0 ): 
                fac = nfac
            else:
                fac = pfac
            plt.text(dotx[ix]*fac, doty[ix]*vfac, f"{ix}={firms_LP()[ix]}: {teams_LP()[ix]}", fontsize = 10)
        plt.title("Intersection Graph of the Constraint Matrix\n node/column = firm# :  assigned team")
        #add subgraph if appropriate
        if (input.iset() < 0) :
            return(plt.draw())
        active = indep_cols()[:,input.iset()]
        imatnew = imat.copy()
        for ix in range(0,nr):
            for jx in range(0,nc):
                if ((active[ix] == 0) & (active[jx] == 0)):
                    imatnew[ix,jx] =0              
        lines2 = idGraph.makeSegs(imatnew,node)
        if lines2 != []:
            lc2 = LineCollection(lines2,linewidths = 2,colors = 'red')
            ax.add_collection(lc2)
        for ix in range(0,len(active)):
            fclstr = 'none'
            eclstr = 'k'
            if active[ix] == 1 : 
                eclstr = 'r'
                fclstr = 'r'
            ax.plot(dotx[ix],doty[ix],'o',ms = 8, markerfacecolor = fclstr,markeredgecolor = eclstr)
        return(plt.draw())

    # @render.plot
    # @reactive.event(input.iset)
    # def subgraphPic():
    #     if (input.iset() < 0): return
    #     active = indep_cols()[:,input.iset()]
    #     #print(f">>>>>>>>set:{input.iset()} active: {active}")
    #     imat = np.array(imat_G())
    #     if imat == []: 
    #         print("No incidence matrix, generate extreme points first.")
    #         fig, ax = plt.subplots(1,1)
    #         return(plt.draw())
    #     nr,nc = imat.shape  #should be nxn
    #     node = idGraph.nodeCoordinates(nr)
    #     dotx = [item[0] for item in node]
    #     doty = [item[1] for item in node]
    #     #mask out the columns not in the current independent set
    #     imatnew = imat
    #     for ix in range(0,nr):
    #         for jx in range(0,nc):
    #             if ((active[ix] == 0) & (active[jx] == 0)):
    #                 imatnew[ix,jx] =0              
    #     lines = idGraph.makeSegs(imatnew,node)
    #     fig, ax = plt.subplots(figsize =(8,12))
    #     #fig = plt.figure(figsize = (12,9), tight_layout = False)
    #     ax.set_xlim(-1.5,1.5)
    #     ax.set_ylim(-1.5,1.5)
    #     if lines != []:
    #         lc = LineCollection(lines,linewidths = 1)
    #         ax.add_collection(lc)
    #     for ix in range(0,len(active)):
    #         fclstr = 'none'
    #         eclstr = 'k'
    #         if active[ix] == 1 : 
    #             eclstr = 'r'
    #             fclstr = 'r'
    #         ax.plot(dotx[ix],doty[ix],'o',ms = 8, markerfacecolor = fclstr,markeredgecolor = eclstr)
    #     #ax.plot(dotx,doty,'bo',color = clr2)
    #     #offsets for the node labels
    #     for ix in range(0,nr):
    #         pfac = 1.10
    #         nfac = 1.5
    #         vfac = 1.05
    #         if (dotx[ix] < 0 ): 
    #             fac = nfac
    #         else:
    #             fac = pfac
    #         plt.text(dotx[ix]*fac, doty[ix]*vfac, f"{ix}={firms_LP()[ix]}: {teams_LP()[ix]}", fontsize = 10)
    #     plt.title(f"Intersection Graph of independent set {input.iset()}\n columns: {active}")
    #     return(plt.draw())

    @render.text
    def aset1():
        if imat_G() == []: 
            ui.update_numeric("iset1", value =-1)
            return('No incidence matrix, generate extreme points first. ')

        if input.iset1() < 0: return(' ') #column set number in indep_cols
        if input.iset2() < 0: return(' ')
        active = indep_cols()[:,input.iset1()] #0: column not in iset1, 1: column is in iset1
        outstr = ''
        for ix in range(len(active)): #for each active column list the firm and team
            if active[ix] == 1:
                outstr += f"column {ix}  firm: {firms_LP()[ix]} team: {teams_LP()[ix]} \n"
        return(outstr)

    @render.text
    def aset2():
        if imat_G() == []: 
            ui.update_numeric("iset2", value = -1)
            return('No incidence matrix, generate extreme points first. ')
        if input.iset1() < 0: return(' ') #column set number in indep_cols
        if input.iset2() < 0: return(' ')
        active = indep_cols()[:,input.iset2()] #0: column not in iset2, 1: column is in iset1
        outstr = ''
        for ix in range(len(active)): #for each active column list the firm and team
            if active[ix] == 1:
                outstr += f"column: {ix} firm: {firms_LP()[ix]} team: {teams_LP()[ix]} \n"
        return(outstr)

    @render.plot
    #@reactive.event(input.iset1, input.iset2)
    def idSetsPic():
        if (input.iset1() < 0) | (input.iset2() < 0):
            fig, ax = plt.subplots(1,1)
            ax.axis('off')
            return(plt.draw())
        active1 = indep_cols()[:,input.iset1()]
        active2 = indep_cols()[:,input.iset2()]
        #now calculate the symmetric difference
        active = [ 1 if (((active1[ix] == 1) | (active2[ix] == 1))) else 0 for ix in list(range(0,len(active1)))]
        #print(f">>>>>>>>set:{input.iset()} active: {active}")
        if imat_G() == []: 
            fig, ax = plt.subplots(1,1)
            ax.axis['off']
            return(plt.draw())
        imat = np.array(imat_G())
        nr,nc = imat.shape  #should be nxn
        node = idGraph.nodeCoordinates(nr)
        dotx = [item[0] for item in node]
        doty = [item[1] for item in node]
        #mask out the columns not in the current independent set
        imatnew = imat
        for ix in range(0,nr):
            for jx in range(0,nc):#retain an arc from imat if and only if both incident nodes are in the symmetric difference
                if ((active[ix] == 0) & (active[jx] == 0)):
                    imatnew[ix,jx] =0              
        lines = idGraph.makeSegs(imatnew,node)
        fig, ax = plt.subplots(figsize =(8,12))
        #fig = plt.figure(figsize = (12,9), tight_layout = False)
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        if lines != []:
            lc = LineCollection(lines,linewidths = 1)
            ax.add_collection(lc)
        for ix in range(0,len(active)):
            eclstr = 'k'
            fclstr = 'none'
            if (active1[ix]*active2[ix] == 1):
                eclstr = 'r'
                fclstr = 'g'
            elif active1[ix] == 1 : 
                eclstr = 'r'
                fclstr = 'r'
            elif active2[ix] == 1 : 
                eclstr = 'g'
                fclstr = 'g'
            ax.plot(dotx[ix],doty[ix],'o',ms=10, markeredgecolor = eclstr,markerfacecolor = fclstr)
        #ax.plot(dotx,doty,'bo',color = clr2)
        #offsets for the node labels
        for ix in range(0,nr):
            pfac = 1.10
            nfac = 1.5
            vfac = 1.05
            if (dotx[ix] < 0 ): 
                fac = nfac
            else:
                fac = pfac
            plt.text(dotx[ix]*fac, doty[ix]*vfac, f"{ix}={firms_LP()[ix]}: {teams_LP()[ix]}", fontsize = 10)
        a1 = [ix  for ix in list(range(0,len(active1))) if active1[ix]==1]
        a2 = [ix  for ix in list(range(0,len(active2))) if active2[ix]==1]
        plt.title(f"Intersection Graph\n set: {input.iset1()} columns: {a1} and set:  {input.iset2()} columns: {a2}")
        return(plt.draw())

    @render.plot
    @reactive.event(input.iset1, input.iset2)
    def adjacencyPic():
        if (input.iset1() < 0): return
        if (input.iset2() < 0): return
        active1 = indep_cols()[:,input.iset1()]
        active2 = indep_cols()[:,input.iset2()]
        #now calculate the symmetric difference
        active = [ 1 if (((active1[ix] == 1) & (active2[ix] == 0)) | ((active1[ix] == 0) & (active2[ix] == 1))) else 0 for ix in list(range(0,len(active1)))]
        if sum(active) == 0:
            fig, ax = plt.subplots(1,1)
            ax.axis('off')
            return(plt.draw())
        #print(f">>>>>>>>set:{input.iset()} active: {active}")
        if imat_G() == []: 
            fig, ax = plt.subplots(1,1)
            #print("$$$$$$$$$$$$$ imat_G fails")
            return(plt.draw())
        imat = np.array(imat_G())
        nr,nc = imat.shape  #should be nxn
        node = idGraph.nodeCoordinates(nr)
        dotx = [item[0] for item in node]
        doty = [item[1] for item in node]
        #remove arcs that do not connect nodes in the symmetric diff of the independent sets (nodes where active[ix] == 1)
        imatnew = imat.copy()
        for ix in range(0,nr):
            for jx in range(0,nc):#retain an arc from imat if and only if both incident nodes are in the symmetric difference
                if ((active[ix] == 0) | (active[jx] == 0)):
                    imatnew[ix,jx] =0              
        lines = idGraph.makeSegs(imatnew,node)
        fig, ax = plt.subplots(figsize =(8,12))
        #fig = plt.figure(figsize = (12,9), tight_layout = False)
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        if lines != []:
            lc = LineCollection(lines,linewidths = 1)
            ax.add_collection(lc)
        for ix in range(0,len(active)):
            eclstr = 'k'
            fclstr = 'none'
            if (active[ix] == 1):
                eclstr = 'k'
                fclstr = 'k'
            # elif (active1[ix]*active2[ix] == 1):
            #     eclstr = 'g'
            #     fclstr = 'r'
            # elif active1[ix] == 1 : 
            #     eclstr = 'r'
            #     fclstr = 'r'
            # elif active2[ix] == 1 : 
            #     eclstr = 'g'
            #     fclstr = 'g'
            ax.plot(dotx[ix],doty[ix],'o',ms = 10, markeredgecolor = eclstr,markerfacecolor = fclstr)
        #ax.plot(dotx,doty,'bo',color = clr2)
        #offsets for the node labels
        for ix in range(0,nr):
            pfac = 1.10
            nfac = 1.5
            vfac = 1.05
            if (dotx[ix] < 0 ): 
                fac = nfac
            else:
                fac = pfac
            plt.text(dotx[ix]*fac, doty[ix]*vfac, f"{ix}={firms_LP()[ix]}: {teams_LP()[ix]}", fontsize = 10)
        #if sum(sum(imat))==0: return

        a1 = [ix  for ix in list(range(0,len(active1))) if active1[ix]==1]
        a2 = [ix  for ix in list(range(0,len(active2))) if active2[ix]==1]
        #create the subgrah of nodes in the symmetric difference
        sym_dif_nodes = [ix for ix in list(range(0,len(active))) if active[ix]==1]
        #now create adjacency matrix of just the subgraph of nodes in the symmetric difference
        imatcols = [ix for ix in sym_dif_nodes]
        imatnew = imat[:,imatcols]
        imatnew = imatnew[imatcols,:]
        #xn,yn = imatnew.shape
        #if (xn == 0) | (yn == 0): return -2
        plt.title(f"Symmetric Diff. \n set: {input.iset1()} columns: {a1}, set:  {input.iset2()} columns: {a2} \n Adjacent extreme pts? {Matching.isConnected_Imat(imatnew)}")
        return(plt.draw())

    @render.plot
    @reactive.event(input.doEPAdjacency)
    def EPAdjacency():
        imt = imat_G()
        #print(f"EPAdjacency(1): {imt}")
        arcs,nonarcs = Matching.extremeAdjacency(indep_cols(),imt)
        #arcst = [(a[0],a[1]) for a in arcs]
        #nonarcst = [(a[0],a[1]) for a in nonarcs]
        idCols = indep_cols()
        idCols_stab = indep_cols_stab()
        nr,nc = idCols.shape
        G = nx.Graph()
        nodes = list(range(nc))
        col_map = ['green' if item[0] <=0 else 'red' for item in indep_cols_stab() ]
        #col_map = [cmap(item[0]/nr) for item in idCols_stab]
        G.add_nodes_from(nodes)
        G.add_edges_from(arcs)
        pos = nx.shell_layout(G)
        squidge = [1.1, 1.05]
        shifted_pos ={item: node_pos * squidge for item, node_pos in pos.items()}
        labels = {item: idCols_stab[item] for item in list(range(nc))}
        fig, axg = plt.subplots(figsize =(8,12))
        #print(f"EPAdjacency(2): {imt}")
        #return(nx.draw_spring(G, ax=axg, with_labels=True))
        #return(nx.draw_circular(G, ax=axg, with_labels=True))
        #return(nx.draw_random(G, ax=axg, with_labels=True))
        nx.draw_shell(G, ax=axg, with_labels=True,node_color = col_map)
        nx.draw_networkx_labels(G, shifted_pos, labels=labels)
        plt.title("Independent set Adjacency Graph with number of stability constraints not satistfied for each independent set.")
        return(plt.draw())
        #return(nx.draw_spectral(G, ax=axg, with_labels=True))
        #return(nx.draw_planar(G, ax=axg, with_labels=True))
        #return(nx.draw_networkx(G))

app = App(app_ui, server,debug=False)


