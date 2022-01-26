# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:30:46 2019

@author: casper
"""

from __future__ import division
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def distance_calc(df):
    #distances = pd.DataFrame(np.nan, index=df.index, columns=df.index)
    XCat_f = df.loc[:, df.columns.str.endswith('True')]#.values
    XNum_f = df.loc[:, df.columns.str.endswith('False')]#.values
    if XCat_f.shape[1] == 0:
        XNum_f = (XNum_f - XNum_f.mean()) / (XNum_f.max() - XNum_f.min())
        XNum_f.fillna(XNum_f.mean(),inplace = True)
        distances = cdist(XNum_f,XNum_f,metric='euclidean')
    elif XNum_f.shape[1] == 0:
        for x in XCat_f:
            XCat_f[x].fillna(XCat_f[x].mode()[0], inplace=True)
        distances = cdist(XCat_f,XCat_f,metric='hamming')
    else:
        n_num = XNum_f.shape[1]
        n_cat = XCat_f.shape[1]
        XNum_f = (XNum_f - XNum_f.mean()) / (XNum_f.max() - XNum_f.min())
        XNum_f.fillna(XNum_f.mean(),inplace = True)
        for x in XCat_f:
            XCat_f[x].fillna(XCat_f[x].mode()[0], inplace=True)
        distNum = cdist(XNum_f,XNum_f,metric='euclidean')
        distCat = cdist(XCat_f,XCat_f,metric='hamming')
        #distances = np.array([[1.0*(cdist(XNum_f,XNum_f,metric='euclidean')[i, j] * n_num + cdist(XCat_f,XCat_f,metric='hamming')[i, j] * n_cat) / 
        #                       (n_num + n_cat) for j in range(XNum_f.shape[0])] for i in range(XNum_f.shape[0])])
        distances = np.array([[1.0*(distNum[i, j] * n_num + distCat[i, j] * n_cat) / 
                               (n_num + n_cat) for j in range(XNum_f.shape[0])] for i in range(XNum_f.shape[0])])
    
    np.fill_diagonal(distances, np.nan)
    return pd.DataFrame(distances)

def kNNImpute(df,neighbors = 5):
    #
    obs = np.where(np.asanyarray(np.isnan(df)))[0]
    feat = np.where(np.asanyarray(np.isnan(df)))[1]
    obsdict = {key: [] for key in obs}
    featdict = {key: [] for key in feat}
    for i in zip(feat,obs):
        featdict[i[0]].append(i[1])
        obsdict[i[1]].append(i[0])
    
    #y -> obs verwijderen
    if 0 in featdict.keys():
        print("Remove rows due to missing target from [%s]" % ', '.join(map(str, featdict[0])))
        #remove row
        df = df.drop(df.index[featdict[0]])
        #update obs,feat,obsdict,featdict
        obs = np.where(np.asanyarray(np.isnan(df)))[0]
        feat = np.where(np.asanyarray(np.isnan(df)))[1]
        obsdict = {key: [] for key in obs}
        featdict = {key: [] for key in feat}
        for i in zip(feat,obs):
            featdict[i[0]].append(i[1])
            obsdict[i[1]].append(i[0])
    
    #meer dan een kwart -> feature verwijderen
    colstodelete = []
    for i in featdict.keys():
        if len(featdict[i]) / df.shape[0] > 0.25:
            colstodelete.append(i)
    if len(colstodelete) > 0:
        print('Remove features from [%s]' % ', '.join(map(str, colstodelete)))
        df = df.drop(df.columns[colstodelete],axis=1)
        #update obs,feat,obsdict,featdict
        obs = np.where(np.asanyarray(np.isnan(df)))[0]
        feat = np.where(np.asanyarray(np.isnan(df)))[1]
        obsdict = {key: [] for key in obs}
        featdict = {key: [] for key in feat}
        for i in zip(feat,obs):
            featdict[i[0]].append(i[1])
            obsdict[i[1]].append(i[0])        
    
    #meer dan een kwart -> obs verwijderen
    rowstodelete = []
    for i in obsdict.keys():
        if len(obsdict[i]) / df.shape[1] > 0.25:
            rowstodelete.append(i)
    if len(rowstodelete) > 0:
        print('Remove rows from [%s]' % ', '.join(map(str, rowstodelete)))
        df = df.drop(df.index[rowstodelete])
        #update obs,feat,obsdict,featdict
        obs = np.where(np.asanyarray(np.isnan(df)))[0]
        feat = np.where(np.asanyarray(np.isnan(df)))[1]
        obsdict = {key: [] for key in obs}
        featdict = {key: [] for key in feat}
        for i in zip(feat,obs):
            featdict[i[0]].append(i[1])
            obsdict[i[1]].append(i[0])
    
    #de rest: True -> most, False -> median
    try:
        distances = distance_calc(df)
    except MemoryError:
        print("Simple med/mode impute")
        for i in obsdict.keys():
            for j in obsdict[i]:
                if list(df.columns.values)[j].endswith('False'):
                    insval = df.median()[j]
                else:
                    insval = df.mode().get_value(0,j,True)
                df.set_value(i,j,insval,True)
        return df
    
    for i in obsdict.keys():
        order = distances.iloc[i,:].values.argsort()
        for j in obsdict[i]:
            imputearray = []
            index = 0
            #add values from indexes, if not nan, to imputearray
            while len(imputearray) < 5:
                val = df.get_value(order[index],j,True)
                if np.isnan(val):
                    index=index+1
                else:
                    imputearray.append(val)
                    index=index+1
            if list(df.columns.values)[j].endswith('False'):
                insval = sum(imputearray)/len(imputearray)
            else:
                insval = stats.mode(imputearray)[0][0]
            df.set_value(i,j,insval,True) 
    return df