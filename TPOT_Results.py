# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:51:37 2019

@author: casper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

loc = "C:\OneDrive - Building Blocks\Thesis\Data\TestData\Test"
loc = "C:\OneDrive - Building Blocks\Thesis\Results\FinalResults"
resultsorig = glob.glob(os.path.join(loc, '*orig.pkl'))
resultsunary = glob.glob(os.path.join(loc, '*unary.pkl'))
resultsub2 = glob.glob(os.path.join(loc, '*ub2.pkl'))
resultsub1 = glob.glob(os.path.join(loc, '*ub1.pkl'))
resultsselect = glob.glob(os.path.join(loc, '*select_fixed.pkl'))
resultsselectall = glob.glob(os.path.join(loc, '*select_full.pkl'))

for k in range(0,len(resultsorig)):
    print(resultsorig[k])
    df = pd.read_pickle(resultsorig[k])
    df2 = pd.read_pickle(resultsunary[k])
    df3 = pd.read_pickle(resultsub2[k])
    df4 = pd.read_pickle(resultsub1[k])
    df5 = pd.read_pickle(resultsselect[k])
    df6 = pd.read_pickle(resultsselectall[k])
    df = df.sort_values(by=["Internal time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    df2 = df2.sort_values(by=["Internal time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    df3 = df3.sort_values(by=["Internal time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    df4 = df4.sort_values(by=["Internal time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    df5 = df5.sort_values(by=["Internal time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    df6 = df6.sort_values(by=["Internal time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    
    maxtime = np.ceil(max(max(df["Internal_time_plus_FE"]),max(df2["Internal_time_plus_FE"]),max(df3["Internal_time_plus_FE"]),max(df4["Internal_time_plus_FE"]),max(df5["Internal_time_plus_FE"]),max(df6["Internal_time_plus_FE"])))
    
    avg_test_ranking = pd.DataFrame(columns={'TPOTorig_test_score','TPOTunary_test_score','TPOTub_test_score','TPOTub1_test_score','TPOTselect_test_score','TPOTselectall_test_score'}, index=range(1, int(maxtime)+1))
    avg_cv = pd.DataFrame(columns={'TPOTorig_cv_score','TPOTunary_cv_score','TPOTub_cv_score','TPOTub1_cv_score','TPOTselect_cv_score','TPOTselectall_cv_score'}, index=range(1, int(maxtime)+1))
    bestdictcv = {key: 0.0 for key in range(1,int(maxtime)+1)}
    bestdict = {key: 0.0 for key in range(1,int(maxtime)+1)}
    for ind, row in df.iterrows():
        if row["Cross-Validation accuracy"] >= bestdictcv[np.ceil(row["Internal_time_plus_FE"])]:
            for key in bestdictcv.keys():
                if key >= np.ceil(row["Internal_time_plus_FE"]):
                    bestdictcv[key] = row["Cross-Validation accuracy"]
                    bestdict[key] = row["Test accuracy"]
    for i in bestdict.keys():
        avg_test_ranking["TPOTorig_test_score"][i] = bestdict[i]
        avg_cv["TPOTorig_cv_score"][i] = bestdictcv[i]
      
    bestdictcv = {key: 0.0 for key in range(1,int(maxtime)+1)}
    bestdict = {key: 0.0 for key in range(1,int(maxtime)+1)}
    for ind, row in df2.iterrows():
        if row["Cross-Validation accuracy"] >= bestdictcv[np.ceil(row["Internal_time_plus_FE"])]:
            for key in bestdictcv.keys():
                if key >= np.ceil(row["Internal_time_plus_FE"]):
                    bestdictcv[key] = row["Cross-Validation accuracy"]
                    bestdict[key] = row["Test accuracy"]
    for i in bestdict.keys():
        avg_test_ranking["TPOTunary_test_score"][i] = bestdict[i]
        avg_cv["TPOTunary_cv_score"][i] = bestdictcv[i]    
    
    bestdictcv = {key: 0.0 for key in range(1,int(maxtime)+1)}
    bestdict = {key: 0.0 for key in range(1,int(maxtime)+1)}
    for ind, row in df3.iterrows():
        if row["Cross-Validation accuracy"] >= bestdictcv[np.ceil(row["Internal_time_plus_FE"])]:
            for key in bestdictcv.keys():
                if key >= np.ceil(row["Internal_time_plus_FE"]):
                    bestdictcv[key] = row["Cross-Validation accuracy"]
                    bestdict[key] = row["Test accuracy"]
    for i in bestdict.keys():
        avg_test_ranking["TPOTub_test_score"][i] = bestdict[i]
        avg_cv["TPOTub_cv_score"][i] = bestdictcv[i]
        
    bestdictcv = {key: 0.0 for key in range(1,int(maxtime)+1)}
    bestdict = {key: 0.0 for key in range(1,int(maxtime)+1)}
    for ind, row in df4.iterrows():
        if row["Cross-Validation accuracy"] >= bestdictcv[np.ceil(row["Internal_time_plus_FE"])]:
            for key in bestdictcv.keys():
                if key >= np.ceil(row["Internal_time_plus_FE"]):
                    bestdictcv[key] = row["Cross-Validation accuracy"]
                    bestdict[key] = row["Test accuracy"]
    for i in bestdict.keys():
        avg_test_ranking["TPOTub1_test_score"][i] = bestdict[i]
        avg_cv["TPOTub1_cv_score"][i] = bestdictcv[i]
        
    bestdictcv = {key: 0.0 for key in range(1,int(maxtime)+1)}
    bestdict = {key: 0.0 for key in range(1,int(maxtime)+1)}
    for ind, row in df5.iterrows():
        if row["Cross-Validation accuracy"] >= bestdictcv[np.ceil(row["Internal_time_plus_FE"])]:
            for key in bestdictcv.keys():
                if key >= np.ceil(row["Internal_time_plus_FE"]):
                    bestdictcv[key] = row["Cross-Validation accuracy"]
                    bestdict[key] = row["Test accuracy"]
    for i in bestdict.keys():
        avg_test_ranking["TPOTselect_test_score"][i] = bestdict[i]
        avg_cv["TPOTselect_cv_score"][i] = bestdictcv[i]
        
    bestdictcv = {key: 0.0 for key in range(1,int(maxtime)+1)}
    bestdict = {key: 0.0 for key in range(1,int(maxtime)+1)}
    for ind, row in df6.iterrows():
        if row["Cross-Validation accuracy"] >= bestdictcv[np.ceil(row["Internal_time_plus_FE"])]:
            for key in bestdictcv.keys():
                if key >= np.ceil(row["Internal_time_plus_FE"]):
                    bestdictcv[key] = row["Cross-Validation accuracy"]
                    bestdict[key] = row["Test accuracy"]
    for i in bestdict.keys():
        avg_test_ranking["TPOTselectall_test_score"][i] = bestdict[i]
        avg_cv["TPOTselectall_cv_score"][i] = bestdictcv[i]
        
    ymin = np.min([np.min(np.amin(avg_cv[avg_cv != 0])) - 0.01,np.min(np.amin(avg_test_ranking[avg_test_ranking != 0])) - 0.01])
    ymax = np.max([np.min([np.max(np.max(avg_cv)) + 0.01,1]),np.min([np.max(np.max(avg_test_ranking)) + 0.01,1])])
    saveloc = r'C:\OneDrive - Building Blocks\Thesis\Report\Figures'
    
    plt.figure(figsize=[8,8])
    plt.plot(avg_test_ranking["TPOTorig_test_score"].values, label="A",linewidth=2.0)
    plt.plot(avg_test_ranking["TPOTunary_test_score"].values, label="B",linewidth=2.0)
    plt.plot(avg_test_ranking["TPOTub_test_score"].values, label="C",linewidth=2.0)
    plt.plot(avg_test_ranking["TPOTub1_test_score"].values, label="D",linewidth=2.0)
    plt.plot(avg_test_ranking["TPOTselect_test_score"].values, label="E",linewidth=2.0)
    plt.plot(avg_test_ranking["TPOTselectall_test_score"].values, label="F",linewidth=2.0)
    plt.xlim([0,185])
    plt.xlabel("Minutes")
    plt.ylim([ymin,ymax])
    plt.ylabel("Balanced Test Accuracy")
    plt.title("Balanced test accuracy of "+str(resultsorig[k][78:-8])+ " over time")
    plt.legend(loc="lower right", prop={'size': 15})
    plt.savefig(saveloc + '\Test_'+ str(resultsorig[k][78:-8]) + '.png', bbox_inches='tight')
    
    plt.figure(figsize=[8,8])
    plt.plot(avg_cv["TPOTorig_cv_score"].values, label="A",linewidth=2.0)
    plt.plot(avg_cv["TPOTunary_cv_score"].values, label="B",linewidth=2.0)
    plt.plot(avg_cv["TPOTub_cv_score"].values, label="C",linewidth=2.0)
    plt.plot(avg_cv["TPOTub1_cv_score"].values, label="D",linewidth=2.0)
    plt.plot(avg_cv["TPOTselect_cv_score"].values, label="E",linewidth=2.0)
    plt.plot(avg_cv["TPOTselectall_cv_score"].values, label="F",linewidth=2.0)
    plt.xlim([0,185])
    plt.xlabel("Minutes")
    plt.ylim([ymin,ymax])
    plt.ylabel("Balanced Cross-Validation Accuracy")
    plt.title("Balanced cross-validation accuracy of "+str(resultsorig[k][78:-8])+" over time")
    plt.legend(loc="lower right", prop={'size': 15})
    plt.savefig(saveloc + '\CV_'+ str(resultsorig[k][78:-8]) + '.png', bbox_inches='tight')
