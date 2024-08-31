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


from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import shinywidgets
from shinywidgets import render_widget, output_widget

import os
import signal
from datetime import datetime


app_ui = ui.page_navbar( 
    # ui.nav_panel("Input",
    #     ui.row(
    #             ui.column(4,offset = 0,*[ui.input_file("file1", "Choose .txt File", accept=[".csv", ".CSV", ".dta", ".DTA"," .txt", ".TXT"], multiple=False, placeholder = '', width = "600px")]),
    #         ),

    # ),
    ui.nav_panel("Input/Linear Program",
        ui.row(
                ui.column(4,offset = 0,*[ui.input_file("file1", "Choose .txt File", accept=[".csv", ".CSV", ".dta", ".DTA"," .txt", ".TXT"], multiple=False, placeholder = '', width = "600px")]),
            ),
        ui.row(
                ui.column(3, offset = 0,*[ui.input_action_button('generateLP',"Generate LP")]),
                ui.column(3, offset = 0,*[ui.input_action_button('solveLP',"Solve LP")]),
                ui.column(6, offset = 0,*[ui.input_checkbox_group("genoptions","Options: ",choices = ["add 1 set per firm","add stability const.","dualize stab. constr."],
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
                ui.column(3,offset=0,*[ui.input_action_button("datalogGo","Show Data",width = '300px')]),
            ),
        ui.row(
                ui.output_text_verbatim("datalog"),
            ),
        ui.row(
                ui.column(3,offset=0,*[ui.input_action_button("goextreme","Enumerate Extreme Points",width = '300px')]),
                ui.column(6,offset=0,*[ui.input_radio_buttons("stype","Show: ",choices = ['All Extreme Points','Stable Extreme Points Only'],inline = True)]),
            ),
        ui.row(
                ui.output_text_verbatim("extremelog"),
            ),
        ui.row(
                ui.column(3,offset=0,*[ui.input_action_button("gointersection","Show Intersection Graph",width = '300px')]),
            ),
        ui.row(
                ui.output_text_verbatim("intgraph"),
            ),
        ),
#     ui.nav_panel("Optimization",
#                  ),
underline = True, title = "Stable Matcher 2.0 ")
                 
def server(input: Inputs, output: Outputs, session: Session):
    
    nw = reactive.value(0) # number of workers 1,..,nw
    nf = reactive.value(0) # number of firms 1,...,fn
    pw = reactive.value([]) #worker preferences 
    pf = reactive.value([]) # firm preferences
    cmat = reactive.value([]) #constraint matrix (for TU checking)
    crhs = reactive.value([]) #righthand sides
    cobj = reactive.value({}) #objective function
    df_LP = reactive.value(pd.DataFrame()) # Full generated LP with row and column labels for display mostly
    solution_LP = reactive.value('') #solution to the LP as a string
    imat_G = reactive.value([]) #incidence matrix of the intersection graph
    teams_LP = reactive.value([]) #the team that goes with each column of cmat or df_LP
    firms_LP = reactive.value([]) #the worker that goes with each column of cmat or df_LP
    stab_constr_LP = reactive.value([]) # just the stability constraint coefficients
    
    @reactive.calc
    def parsed_file():
        print(f"$$$$$$$$$$$$$$$$$$Path: {str(input.file1()[0]['datapath'])}")
        if input.file1() is None:
            return('')
        else: #Note the ui passes only paths ending in  .csv, .CSV, .dta, .DTA and .txt
            fpath = str(input.file1()[0]['datapath'])
            print(f"$$$$$$$$$$$$$$$$$$Path: {fpath}")
            if (fpath[-4:] == '.txt') or (fpath[-4:] == '.TXT'):
                nwt,nft,pwt,pft = ReadMatchData.readData(fpath)
            #print(f'@@@@@@@@@@@@@@@@nw = {nwt}')
            nw.set(nwt)
            nf.set(nft)
            pw.set(pwt)
            pf.set(pft)
            outstr = ''
            #print(f"***************pwt[1] = {pwt[1]}")
            for ix in range(1,nf()+1):
                outstr = outstr + f"pf[{ix}] = {pf()[ix]} \n"
            for ix in range(1,nw()+1):
                outstr = outstr + f"pw[{ix}] = {pw()[ix]}\n"
            #print(outstr)
            return( nft)


    @render.text
    @reactive.event(input.datalogGo)
    def datalog(): 
        nft = parsed_file()
        nwt = nw()
        nft = nf()
        pft =  pf()
        pwt = pw()

        if (nwt == 0): 
            print("**********quitting no data*************")
            return
        outstr = ''
        for ix in range(1,nft+1):
            outstr = outstr + f"pf[{ix}] = {pft[ix]} \n"
        for ix in range(1,nwt+1):
            outstr = outstr + f"pw[{ix}] = {pwt[ix]}\n"
        return(outstr)
  

    @render.text
    @reactive.event(input.goextreme)
    def extremelog():
        nft = nf()
        nwt = nw()
        pft = pf()
        pwt = pw()

        #const_mat,rhs,obj,firms,teams,rowlab,stab_constr = Matching.doLP(nwt, nft, pwt, pft, DoOneSet = oneper, DoBounds = False, StabilityConstraints = dostab, Dual = False, Verbose = False)
        imat = Matching.doIntersectionGraph(cmat())
        imat_G.set(imat)
        if (input.stype() == "All Extreme Points"): 
            vbs = True
        else:
            vbs = False
        if "add stability const." in input.genoptions():
            outstring = "The enumeration process for extreme points requires a non-negative binary constraint matrix.\n  Remove the stability constraints and try again!"
        else:
            independent_columns, outstring = Matching.doIndependentSets(imat,teams_LP() , firms_LP(), StabConst = stab_constr_LP(), Verbose = vbs)
        return(outstring)

    @reactive.effect
    @reactive.event(input.generateLP)
    def formulate_LP():
        nft = parsed_file()
        solution_LP.set('')
        nwt = nw()
        nft = nf()
        oneper = False
        dostab = False
        print(f"  formulate_LP :: #workers: {nwt}, #firms: {nft}, oneper: {oneper}, dostab: {dostab}")
        if nw() == 0: 
            return        
        oneper = False
        dostab = False
        dodual = False
        if ("add 1 set per firm" in input.genoptions()):
            oneper = True
        if ("add stability const." in input.genoptions()):
            dostab = True
        if ("dualize stab. constr." in input.genoptions()):
            dodual = True
        cols, rhs, obj, firm_no, set_assgn, rowlabels, stab_columns = Matching.doLP(nw(), nf(),pw(),pf(),DoOneSet = oneper, DoBounds = False, StabilityConstraints=dostab, Dual = dodual)
        dfout = Matching.displayLP(constraints = cols, rhs = rhs, obj = obj, teams = set_assgn, firms = firm_no, rowlabels = rowlabels)
        df_LP.set(dfout)

        cmat.set(cols)
        cobj.set(obj)
        crhs.set(rhs)
        teams_LP.set(set_assgn)
        firms_LP.set(firm_no)
        stab_constr_LP.set(stab_columns)

    #@render.data_frame
    @render.text
    @reactive.event(input.generateLP)
    def LPOut():
        dflocal = df_LP()
        if len(dflocal) == 0:
            return
        return dflocal.to_string() + '\n' + solution_LP()
        #return dflocal

    @reactive.effect
    @reactive.event(input.solveLP)
    def goSolve():
        #now solve it
        results,status = Matching.solveIt(cmat(), crhs(), cobj())
        outstring = Matching.decodeSolution(firms = firms_LP(), teams = teams_LP(),  solution = results)
        solution_LP.set(outstring)


    #@render.data_frame
    @render.text
    def LPSolvedOut():
        return solution_LP()


    @render.text
    @reactive.event(input.gointersection)    
    def intgraph():
        return(np.array_str(imat_G()))
    
    @render.text
    @reactive.event(input.testTU)
    def TUreport():
        ISTU, outstring = Matching.checkTU(cmat(), Tol = 1e-10)
        return outstring


app = App(app_ui, server,debug=True)


