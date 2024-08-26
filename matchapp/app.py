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
    ui.nav_panel("Plotting",
                ui.row(
                     ui.column(4,offset = 0,*[ui.input_file("file1", "Choose .txt File", accept=[".csv", ".CSV", ".dta", ".DTA"," .txt", ".TXT"], multiple=False, placeholder = '', width = "600px")]),
                     ),
                ui.row(
                     ui.input_action_button("datalogGo","Show Data",width = '300px'),
                     ui.output_text_verbatim("datalog"),
                     ),
                ui.row(
                     ui.input_action_button("goextreme","Enumerate Extreme Points",width = '300px'),
                     ui.column(6,offset=0,*[ui.input_radio_buttons("stype","Show: ",choices = ['All Extreme Points','Stable Extreme Points Only'],inline = True)]),
                     ui.output_text_verbatim("extremelog"),
                     ),
                ),
    ui.nav_panel("Matching",
                 ),
    ui.nav_panel("Optimization",
                 ),
underline = True, title = "Stable Matcher 1.0 ")
                 
def server(input: Inputs, output: Outputs, session: Session):
    
    nw = reactive.value(0)
    nf = reactive.value(0)
    pw = reactive.value([])
    pf = reactive.value([])
    
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
            print(f'@@@@@@@@@@@@@@@@nw = {nwt}')
            nw.set(nwt)
            nf.set(nft)
            pw.set(pwt)
            pf.set(pft)
            outstr = ''
            print(f"***************pwt[1] = {pwt[1]}")
            for ix in range(1,nf()+1):
                outstr = outstr + f"pf[{ix}] = {pf()[ix]} \n"
            for ix in range(1,nw()+1):
                outstr = outstr + f"pw[{ix}] = {pw()[ix]}\n"
            print(outstr)
            return( nft)

            # pushlog("************************************************")
            # pushlog("File read: "  + input.file1()[0]['name'])
            # pushlog(f"....Number of rows: {len(df)}")
            # pushlog("************************************************")

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
        print(f'@@@@@@@@@@@@@@@@@@@@@@@@ Starting enumeration goextreme = {input.goextreme()}')
        nft = nf()
        nwt = nw()
        pft = pf()
        pwt = pw()
        const_mat,rhs,obj,firms,teams,rowlab,stab_constr = Matching.doLP(nwt, nft, pwt, pft, DoOneSet = True, DoBounds = False, StabilityConstraints = False, Dual = False, Verbose = False)
        imat = Matching.doIntersectionGraph(const_mat)
        if (input.stype() == "All Extreme Points"): 
            vbs = True
        else:
            vbs = False
        independent_columns, outstring = Matching.doIndependentSets(imat, teams, firms, StabConst = stab_constr, Verbose = vbs)
        #print(outstring)
        return(outstring)
        
app = App(app_ui, server,debug=True)


