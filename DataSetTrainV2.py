# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:14:41 2019

@author: casper
"""

from OperatorsV2 import Numerical
from OperatorsV2 import Categorical
from OperatorsV2 import NumNum
from OperatorsV2 import NumCat
from MetaFeature import MetaNum
from MetaFeature import MetaCat
from MetaFeature import MetaNumNum
from MetaFeature import MetaNumCat
from OperatorsV2 import Bench

#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import numpy as np
import scipy as sp
import pandas as pd
from time import time
from copy import deepcopy
import warnings
import itertools
warnings.filterwarnings("ignore")

def Train(df,dicts):
    y = df['y'].values
    X = df.drop('y', 1).values
    XCatind = np.where(df.drop('y', 1).columns.str.endswith('True'))[0]
    XNumind = np.where(df.drop('y', 1).columns.str.endswith('False'))[0]
    
    for cat in XCatind:
        X[:,cat] = LabelEncoder().fit_transform(X[:,cat])
    
    Num_dict = dicts['Num'] ; NumNum_dict = dicts['NumNum']
    Cat_dict = dicts['Cat'] ; NumCat_dict = dicts['NumCat']
    
    Num_names = Num_dict.keys() ; NumNum_names = NumNum_dict.keys()
    Cat_names = Cat_dict.keys() ; NumCat_names = NumCat_dict.keys()
    
    #Find the raw score of data set
    rfc = RandomForestClassifier(random_state=8520)
    lrc = LogisticRegression(random_state=8520)
    svm = LinearSVC(random_state=8520)
    knn = KNeighborsClassifier()
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
    BenchScores = [0] * 4
    clf_pipe = make_pipeline(Bench(XCatind,XNumind),rfc)
    BenchScores[0] = cross_val_score(clf_pipe, deepcopy(X), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
    clf_pipe = make_pipeline(Bench(XCatind,XNumind),lrc)
    BenchScores[1] = cross_val_score(clf_pipe, deepcopy(X), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
    clf_pipe = make_pipeline(Bench(XCatind,XNumind),svm)
    BenchScores[2] = cross_val_score(clf_pipe, deepcopy(X), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
    clf_pipe = make_pipeline(Bench(XCatind,XNumind),knn)
    BenchScores[3] = cross_val_score(clf_pipe, deepcopy(X), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()    
    print(BenchScores)
    #All Numerial operators 
    for op in Num_names:
        #print(op)
        for i in XNumind:
            X2=deepcopy(X)
            XNumind2 = deepcopy(XNumind)
            XCatind2 = deepcopy(XCatind)
            
            prep = getattr(Numerical(), op)(i, XCatind2, XNumind2)
            if not prep.issuitable(X2):
                continue
            rfc = RandomForestClassifier(random_state=8520)
            lrc = LogisticRegression(random_state=8520)
            svm = LinearSVC(random_state=8520)
            knn = KNeighborsClassifier()
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
            NewScores = [0] * 4
            clf_pipe = make_pipeline(prep,rfc)
            NewScores[0] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,lrc)
            NewScores[1] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,svm)
            NewScores[2] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,knn)
            NewScores[3] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            #print(NewScore)
            MetaTarget = 0
            NImprov = sum(np.array(NewScores) > BenchScores + np.subtract([1]*len(BenchScores),BenchScores)*0.05)
            if NImprov >= 1:
                MetaTarget = 1
            #elif NImprov == 1:
            #    continue
            MetaFeature = MetaNum(X[:,i],MetaTarget,X,y,XCatind)
            Num_dict[op] = Num_dict[op].append(MetaFeature)
            
    #All Categorical operators
    for op in Cat_names:
        #print(op)
        for i in XCatind:
            #print(i)
            X2=deepcopy(X)
            XNumind2 = deepcopy(XNumind)
            XCatind2 = deepcopy(XCatind)
            
            prep = getattr(Categorical(), op)(i, XCatind2, XNumind2)
            if not prep.issuitable(X2):
                continue
            rfc = RandomForestClassifier(random_state=8520)
            lrc = LogisticRegression(random_state=8520)
            svm = LinearSVC(random_state=8520)
            knn = KNeighborsClassifier()
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
            NewScores = [0] * 4
            clf_pipe = make_pipeline(prep,rfc)
            NewScores[0] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,lrc)
            NewScores[1] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,svm)
            NewScores[2] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,knn)
            NewScores[3] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            #print(NewScore)
            MetaTarget = 0
            NImprov = sum(np.array(NewScores) > BenchScores + np.subtract([1]*len(BenchScores),BenchScores)*0.05)
            if NImprov >= 1:
                MetaTarget = 1
            #elif NImprov == 1:
            #    continue
            MetaFeature = MetaCat(X[:,i],MetaTarget,X,y,XCatind)
            Cat_dict[op] = Cat_dict[op].append(MetaFeature)
    '''        
    #All NumNum operators
    for op in NumNum_names:
        #print(op)
        for subset in itertools.combinations(XNumind, 2):
            i = subset[0] ; j = subset[1]
            X2=deepcopy(X)
            XNumind2 = deepcopy(XNumind)
            XCatind2 = deepcopy(XCatind)
            
            prep = getattr(NumNum(), op)(i,j, XCatind2, XNumind2)
            if not prep.issuitable(X2):
                continue
            rfc = RandomForestClassifier(random_state=8520)
            lrc = LogisticRegression(random_state=8520)
            svm = LinearSVC(random_state=8520)
            knn = KNeighborsClassifier()
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
            NewScores = [0] * 4
            clf_pipe = make_pipeline(prep,rfc)
            NewScores[0] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,lrc)
            NewScores[1] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,svm)
            NewScores[2] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
            clf_pipe = make_pipeline(prep,knn)
            NewScores[3] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()      
            #print(NewScore)
            MetaTarget = 0
            NImprov = sum(np.array(NewScores) > BenchScores + np.subtract([1]*len(BenchScores),BenchScores)*0.05)
            if NImprov >= 1:
                MetaTarget = 1
            #elif NImprov == 1:
            #    continue
            MetaFeature = MetaNumNum(X[:,i],X[:,j],MetaTarget,X,y,XCatind)
            NumNum_dict[op] = NumNum_dict[op].append(MetaFeature)
            if not prep.bothways():
                MetaFeature = MetaNumNum(X[:,j],X[:,i],MetaTarget,X,y,XCatind)
                NumNum_dict[op] = NumNum_dict[op].append(MetaFeature)
            else:
                prep = getattr(NumNum(), op)(j,i, XCatind2, XNumind2)
                rfc = RandomForestClassifier(random_state=8520)
                lrc = LogisticRegression(random_state=8520)
                svm = LinearSVC(random_state=8520)
                knn = KNeighborsClassifier()
                skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
                NewScores2 = [0] * 4
                clf_pipe = make_pipeline(prep,rfc)
                NewScores2[0] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                clf_pipe = make_pipeline(prep,lrc)
                NewScores2[1] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                clf_pipe = make_pipeline(prep,svm)
                NewScores2[2] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                clf_pipe = make_pipeline(prep,knn)
                NewScores2[3] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                MetaTarget = 0
                NImprov = sum(np.array(NewScores2) > BenchScores + np.subtract([1]*len(BenchScores),BenchScores)*0.05)
                if NImprov >= 1:
                    MetaTarget = 1
                #elif NImprov == 1:
                #    continue
                MetaFeature = MetaNumNum(X[:,j],X[:,i],MetaTarget,X,y,XCatind)
                NumNum_dict[op] = NumNum_dict[op].append(MetaFeature)
    
    #All NumCat operators
    for op in NumCat_names:
        #print(op)
        for i in XNumind:
            for j in XCatind:
                X2=deepcopy(X)
                XNumind2 = deepcopy(XNumind)
                XCatind2 = deepcopy(XCatind)
                prep = getattr(NumCat(), op)(i,j, XCatind2, XNumind2)
                if not prep.issuitable(X2):
                    continue
                rfc = RandomForestClassifier(random_state=8520)
                lrc = LogisticRegression(random_state=8520)
                svm = LinearSVC(random_state=8520)
                knn = KNeighborsClassifier()
                skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
                NewScores = [0] * 4
                clf_pipe = make_pipeline(prep,rfc)
                NewScores[0] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                clf_pipe = make_pipeline(prep,lrc)
                NewScores[1] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                clf_pipe = make_pipeline(prep,svm)
                NewScores[2] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()
                clf_pipe = make_pipeline(prep,knn)
                NewScores[3] = cross_val_score(clf_pipe, deepcopy(X2), y, scoring='balanced_accuracy',cv=skf,n_jobs = 1).mean()      
                #print(NewScore)
                MetaTarget = 0
                NImprov = sum(np.array(NewScores) > BenchScores + np.subtract([1]*len(BenchScores),BenchScores)*0.05)
                if NImprov >= 1:
                    MetaTarget = 1
                #elif NImprov == 1:
                #    continue
                MetaFeature = MetaNumCat(X[:,i],X[:,j],MetaTarget,X,y,XCatind)
                NumCat_dict[op] = NumCat_dict[op].append(MetaFeature)
    '''
    dicts_new = {'Num': Num_dict, 'Cat': Cat_dict, 'NumNum': NumNum_dict, 'NumCat': NumCat_dict}
    return dicts_new