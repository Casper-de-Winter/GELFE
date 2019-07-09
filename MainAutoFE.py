# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:08:51 2019

@author: casper
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from time import time
from MetaFeature import MetaNum
from MetaFeature import MetaCat
from MetaFeature import MetaNumNum
from MetaFeature import MetaNumCat
from OperatorsFE import Numerical
from OperatorsFE import Categorical
from OperatorsFE import NumNum
from OperatorsFE import NumCat
from OperatorsFE import Bench
from copy import deepcopy
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import bisect
import random
from CMI import cife

def AutoFE(df, test_size = 0.40, original = True, method = "BestK", binary = False, NofEval = 2, 
           NofUnary = 10, NofBinary = 15, binaryselect = False,selectfixed = True, NofBinaryTotal = 15):
    #load a data set and models
    metamodels = np.load('meta_models.npy').item()
    Num_models = metamodels['Num']; NumNum_models = metamodels['NumNum']
    Cat_models = metamodels['Cat']; NumCat_models = metamodels['NumCat']
    Extra=1
    if NofEval == 1:
        Extra=0
    #loc = r'C:\OneDrive - Building Blocks\Thesis\Data\TestData'
    #writeloc = r'C:\OneDrive - Building Blocks\Thesis\Data\TestFiles'
    #csvfiles = glob.glob(os.path.join(loc, '*.csv'))
    #data = csvfiles[1]
    #df = pd.read_csv(data,index_col='Unnamed: 0')
    #create all new data sets
    y = df['y'].values
    X = df.drop('y', 1).values
    XCatind = np.where(df.drop('y', 1).columns.str.endswith('True'))[0]
    XNumind = np.where(df.drop('y', 1).columns.str.endswith('False'))[0]
    XAll = np.where(df.drop('y', 1).columns.str.endswith('e'))[0]
    for cat in XCatind:
        X[:,cat] = LabelEncoder().fit_transform(X[:,cat])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state = 1995)
    X_tr_new = deepcopy(X_train)
    X_te_new = deepcopy(X_test)
    AllOps = set(Num_models.keys()).union(set(Cat_models.keys())).union(set(NumNum_models.keys())).union(set(NumCat_models.keys()))
    Counter = {key: 0 for key in AllOps}
    FeatureList = {key: [] for key in AllOps}
    
    if original:
        prep = Bench(XCatind,XNumind)
        X_train_o = prep.fit_transform(deepcopy(X_train))
        X_test_o = prep.transform(deepcopy(X_test))
        return X_train_o, X_test_o, y_train, y_test, Counter
        #dforig = pd.DataFrame(Xorig)
        #dforig = pd.DataFrame(X)
        #dforig.insert(loc=0, column='y', value=y)
        #dforig.to_csv(os.path.join(writeloc,data[51:-4] + '_orig' + r'.csv'))
    #UNARY
    if method == "Best":
        for i in XAll:
            if i in XNumind:
                f = X_train[:,i]
                MetaFeature = MetaNum(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                MetaFeature = MetaFeature.replace({'npeaks': {0: 1}})
                bestprob = 0
                bestop = None
                for op in Num_models.keys():
                    prep = getattr(Numerical(), op)(i, XCatind, XNumind)
                    if not prep.issuitable(deepcopy(X_train)):
                        continue
                    prob = Num_models[op].predict_proba(MetaFeature)[0][1]
                    if prob > bestprob:
                        bestprob = prob
                        bestop = op
                #print(bestprob, bestop)
                if bestprob > 0.5:
                    prep = getattr(Numerical(), bestop)(i, XCatind, XNumind)
                    X_tr_new = prep.fit_transform(X_tr_new)
                    X_te_new = prep.transform(X_te_new)
                    FeatureList[bestop].extend(list(set(prep.XNumind).difference(XNumind)))
                    FeatureList[bestop].extend(list(set(prep.XCatind).difference(XCatind)))
                    XCatind = prep.XCatind
                    XNumind = prep.XNumind
                    Counter[bestop] = Counter[bestop]+1
            else:
                f = X_train[:,i]
                MetaFeature = MetaCat(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                bestprob = 0
                bestop = None
                for op in Cat_models.keys():
                    if op != "BackwEnc" and op != "LargestVs" and op != "GroupSim" and op != "GroupLowFreq":
                        prep = getattr(Categorical(), op)(i, XCatind, XNumind)
                        if not prep.issuitable(deepcopy(X_train)):
                            continue
                        prob = Cat_models[op].predict_proba(MetaFeature)[0][1]
                        if prob > bestprob:
                            bestprob = prob
                            bestop = op
                #print(bestprob,bestop)
                if bestprob > 0.5:
                    prep = getattr(Categorical(), bestop)(i, XCatind, XNumind)
                    X_tr_new = prep.fit_transform(X_tr_new,y_train)
                    X_te_new = prep.transform(X_te_new)
                    if bestop in ["BackwEnc","GroupSim","GroupLowFreq","LargestVs"]:
                        FeatureList[bestop].extend([i])
                        Counter[bestop] = Counter[bestop]+1
                    FeatureList[bestop].extend(list(set(prep.XNumind).difference(XNumind)))
                    Counter[bestop] = Counter[bestop] + len(set(prep.XNumind).difference(XNumind))
                    FeatureList[bestop].extend(list(set(prep.XCatind).difference(XCatind)))
                    Counter[bestop] = Counter[bestop] + len(set(prep.XCatind).difference(XCatind))
                    XCatind = prep.XCatind
                    XNumind = prep.XNumind
    elif method == "BestK":
        bestpreps = {}
        bestprobs = list(np.zeros(NofUnary))
        maxeval = np.floor(NofUnary/2)
        thres = min(bestprobs)
        for i in XAll:
            bestpreps_i = {}
            bestprobs_i = list(np.zeros(NofEval+Extra))
            thres_i = min(bestprobs_i)
            if i in XNumind:
                f = X_train[:,i]
                MetaFeature = MetaNum(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                MetaFeature = MetaFeature.replace({'npeaks': {0: 1}})
                for op in Num_models.keys():
                    prep = getattr(Numerical(), op)(i, XCatind, XNumind)
                    if not prep.issuitable(deepcopy(X_train)):
                        continue
                    prob = Num_models[op].predict_proba(MetaFeature)[0][1]
                    if prob > thres_i:
                        bestprobs_i.append(prob)
                        bestprobs_i.sort(reverse = True)
                        bestprobs_i=bestprobs_i[:NofEval+Extra]
                        thres_i = min(bestprobs_i)
                        bestpreps_i[prep] = prob
                        bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
                        if all([l.startswith("Equ") for l in [j.__class__.__name__ for j in bestpreps_i.keys()]]):
                            bestprobs_i=bestprobs_i[:1]
                            thres_i = min(bestprobs_i)
                            bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            else:
                f = X_train[:,i]
                MetaFeature = MetaCat(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                for op in Cat_models.keys():
                    if op != "BackwEnc" and op != "LargestVs" and op != "GroupSim" and op != "GroupLowFreq":
                        prep = getattr(Categorical(), op)(i, XCatind, XNumind)
                        if not prep.issuitable(deepcopy(X_train)):
                            continue
                        prob = Cat_models[op].predict_proba(MetaFeature)[0][1]
                        if prob > thres_i:
                            bestprobs_i.append(prob)
                            bestprobs_i.sort(reverse = True)
                            bestprobs_i=bestprobs_i[:NofEval+Extra]
                            thres_i = min(bestprobs_i)
                            bestpreps_i[prep] = prob
                            bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            sorted_preps = sorted(bestpreps_i.items(), key=lambda kv: kv[1],reverse=True)
            Toadd=NofEval
            for prep,prob in sorted_preps:
                if Toadd == 0:
                    continue
                if prep.__class__.__name__ in [i.__class__.__name__ for i in bestpreps.keys()] and pd.value_counts([i.__class__.__name__ for i in bestpreps.keys()])[prep.__class__.__name__]>=maxeval:
                    temp = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ == prep.__class__.__name__}
                    bestpreps = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ != prep.__class__.__name__}
                    tempmin=min(temp.values())
                    if prob > tempmin:
                        temp={k:v for k,v in temp.items() if v != tempmin}
                        temp[prep] = prob
                        Toadd-=1
                    bestpreps.update(temp)
                    bestprobs = list(bestpreps.values())
                    bestprobs.sort(reverse = True)
                    thres = min(bestprobs)
                else:
                    if prob > thres and prob > 0.5:
                        bestprobs.append(prob)
                        bestprobs.sort(reverse = True)
                        bestprobs=bestprobs[:NofUnary]
                        bestpreps[prep] = prob
                        thres = min(bestprobs)
                        bestpreps = {key:val for key, val in bestpreps.items() if val >= thres}
                        Toadd-=1
        for prep in bestpreps.keys():
            #print(bestpreps[prep],prep)
            prep.XCatind = XCatind ; prep.XNumind = XNumind
            X_tr_new = prep.fit_transform(X_tr_new, y_train)
            X_te_new = prep.transform(X_te_new)
            if prep.__class__.__name__ in ["BackwEnc","GroupSim","GroupLowFreq","LargestVs"]:
                FeatureList[prep.__class__.__name__].extend([prep.col])
                Counter[prep.__class__.__name__] += 1
            extra = set(prep.XCatind) - set(XCatind)
            for i in extra:
                XCatind = np.append(XCatind,i)
                Counter[prep.__class__.__name__] += 1
            FeatureList[prep.__class__.__name__].extend(list(extra))
            extra = set(prep.XNumind) - set(XNumind)
            for i in extra:
                XNumind = np.append(XNumind,i)
                Counter[prep.__class__.__name__] += 1
            FeatureList[prep.__class__.__name__].extend(list(extra))
    elif method == "Selection":
        fixed = X_tr_new.shape[1]
        bestpreps = {}
        bestprobs = list(np.zeros(50))
        maxeval = np.floor(10)
        thres = min(bestprobs)
        for i in XAll:
            bestpreps_i = {}
            bestprobs_i = list(np.zeros(NofEval+Extra))
            thres_i = min(bestprobs_i)
            if i in XNumind:
                f = X_train[:,i]
                MetaFeature = MetaNum(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                MetaFeature = MetaFeature.replace({'npeaks': {0: 1}})
                for op in Num_models.keys():
                    prep = getattr(Numerical(), op)(i, XCatind, XNumind)
                    if not prep.issuitable(deepcopy(X_train)):
                        continue
                    prob = Num_models[op].predict_proba(MetaFeature)[0][1]
                    if prob > thres_i:
                        bestprobs_i.append(prob)
                        bestprobs_i.sort(reverse = True)
                        bestprobs_i=bestprobs_i[:NofEval+Extra]
                        thres_i = min(bestprobs_i)
                        bestpreps_i[prep] = prob
                        bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
                        if all([l.startswith("Equ") for l in [j.__class__.__name__ for j in bestpreps_i.keys()]]):
                            bestprobs_i=bestprobs_i[:1]
                            thres_i = min(bestprobs_i)
                            bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            else:
                f = X_train[:,i]
                MetaFeature = MetaCat(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                for op in Cat_models.keys():
                    if op != "BackwEnc" and op != "LargestVs" and op != "GroupSim" and op != "GroupLowFreq":
                        prep = getattr(Categorical(), op)(i, XCatind, XNumind)
                        if not prep.issuitable(deepcopy(X_train)):
                            continue
                        prob = Cat_models[op].predict_proba(MetaFeature)[0][1]
                        if prob > thres_i:
                            bestprobs_i.append(prob)
                            bestprobs_i.sort(reverse = True)
                            bestprobs_i=bestprobs_i[:NofEval+Extra]
                            thres_i = min(bestprobs_i)
                            bestpreps_i[prep] = prob
                            bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            sorted_preps = sorted(bestpreps_i.items(), key=lambda kv: kv[1],reverse=True)
            Toadd=NofEval
            for prep,prob in sorted_preps:
                if Toadd == 0:
                    continue
                if prep.__class__.__name__ in [i.__class__.__name__ for i in bestpreps.keys()] and pd.value_counts([i.__class__.__name__ for i in bestpreps.keys()])[prep.__class__.__name__]>=maxeval:
                    temp = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ == prep.__class__.__name__}
                    bestpreps = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ != prep.__class__.__name__}
                    tempmin=min(temp.values())
                    if prob > tempmin:
                        temp={k:v for k,v in temp.items() if v != tempmin}
                        temp[prep] = prob
                        Toadd-=1
                    bestpreps.update(temp)
                    bestprobs = list(bestpreps.values())
                    bestprobs.sort(reverse = True)
                    thres = min(bestprobs)
                else:
                    if prob > thres and prob > 0.5:
                        bestprobs.append(prob)
                        bestprobs.sort(reverse = True)
                        bestprobs=bestprobs[:50]
                        bestpreps[prep] = prob
                        thres = min(bestprobs)
                        bestpreps = {key:val for key, val in bestpreps.items() if val >= thres}
                        Toadd-=1
        for prep in bestpreps.keys():
            #print(bestpreps[prep],prep)
            prep.XCatind = XCatind ; prep.XNumind = XNumind
            X_tr_new = prep.fit_transform(X_tr_new, y_train)
            X_te_new = prep.transform(X_te_new)
            if prep.__class__.__name__ in ["BackwEnc","GroupSim","GroupLowFreq","LargestVs"]:
                FeatureList[prep.__class__.__name__].extend([prep.col])
                Counter[prep.__class__.__name__] += 1
            extra = set(prep.XCatind) - set(XCatind)
            for i in extra:
                XCatind = np.append(XCatind,i)
                Counter[prep.__class__.__name__] += 1
            FeatureList[prep.__class__.__name__].extend(list(extra))
            extra = set(prep.XNumind) - set(XNumind)
            for i in extra:
                XNumind = np.append(XNumind,i)
                Counter[prep.__class__.__name__] += 1
            FeatureList[prep.__class__.__name__].extend(list(extra))
        '''if X_tr_new.shape[1] > fixed + NofUnary:
            OUT1, OUT2, OUT3 = cife(X_tr_new, y_train, XNumind, fixed = fixed,n_selected_features = fixed + NofUnary)
            X_tr_new = X_tr_new[ : , sorted(OUT1)]
            X_te_new = X_te_new[ : , sorted(OUT1)]
            Xset = set(XCatind).union(set(XNumind))
            for i in sorted([i for i in Xset - set(OUT1)],reverse = True):
                if i in XCatind:
                    XCatind = np.setdiff1d(XCatind, i)
                else:
                    XNumind = np.setdiff1d(XNumind, i)
                for j in XCatind:
                    if j > i:
                        XCatind = np.where(XCatind==j, j-1, XCatind)
                for j in XNumind:
                    if j > i:
                        XNumind = np.where(XNumind==j, j-1, XNumind)
                for key,val in FeatureList.items():
                    if i in val:
                        val.remove(i)
                        Counter[key] -= 1
                    if len(val) > 0 and np.max(val) > i:
                        for j in range(0,len(val)):
                            if val[j] > i:
                                val[j] -=1'''
    elif method == "SelectionOrig":
        fixed = X_tr_new.shape[1]
        for i in XAll:
            bestpreps_i = {}
            bestprobs_i = list(np.zeros(NofEval))
            thres_i = 0.5
            if i in XNumind:
                f = X_train[:,i]
                MetaFeature = MetaNum(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                MetaFeature = MetaFeature.replace({'npeaks': {0: 1}})
                for op in Num_models.keys():
                    prep = getattr(Numerical(), op)(i, XCatind, XNumind)
                    if not prep.issuitable(deepcopy(X_train)):
                        continue
                    prob = Num_models[op].predict_proba(MetaFeature)[0][1]
                    if prob > thres_i:
                        bestprobs_i.append(prob)
                        bestprobs_i.sort(reverse = True)
                        bestprobs_i=bestprobs_i[:NofEval]
                        thres_i = np.max([min(bestprobs_i),0.5])
                        bestpreps_i[prep] = prob
                        bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
                        if all([l.startswith("Equ") for l in [j.__class__.__name__ for j in bestpreps_i.keys()]]):
                            bestprobs_i=bestprobs_i[:1]
                            thres_i = min(bestprobs_i)
                            bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            else:
                f = X_train[:,i]
                MetaFeature = MetaCat(f,1,X_train,y_train,XCatind)
                MetaFeature = MetaFeature.drop('y',1)
                MetaFeature = MetaFeature.drop('yfreq',1)
                MetaFeature = MetaFeature.drop('Xtotal',1)
                MetaFeature = MetaFeature.drop('XNum',1)
                MetaFeature = MetaFeature.drop('XCat',1)
                for op in Cat_models.keys():
                    if op != "BackwEnc" and op != "LargestVs" and op != "GroupSim" and op != "GroupLowFreq":
                        prep = getattr(Categorical(), op)(i, XCatind, XNumind)
                        if not prep.issuitable(deepcopy(X_train)):
                            continue
                        prob = Cat_models[op].predict_proba(MetaFeature)[0][1]
                        if prob > thres_i:
                            bestprobs_i.append(prob)
                            bestprobs_i.sort(reverse = True)
                            bestprobs_i=bestprobs_i[:NofEval]
                            thres_i = np.max([min(bestprobs_i),0.5])
                            bestpreps_i[prep] = prob
                            bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            for prep in bestpreps_i.keys():
                prep.XCatind = XCatind ; prep.XNumind = XNumind
                X_tr_new = prep.fit_transform(X_tr_new, y_train)
                X_te_new = prep.transform(X_te_new)
                if prep.__class__.__name__ in ["BackwEnc","GroupSim","GroupLowFreq","LargestVs"]:
                    FeatureList[prep.__class__.__name__].extend([prep.col])
                    Counter[prep.__class__.__name__] += 1
                extra = set(prep.XCatind) - set(XCatind)
                for i in extra:
                    XCatind = np.append(XCatind,i)
                    Counter[prep.__class__.__name__] += 1
                FeatureList[prep.__class__.__name__].extend(list(extra))
                extra = set(prep.XNumind) - set(XNumind)
                for i in extra:
                    XNumind = np.append(XNumind,i)
                    Counter[prep.__class__.__name__] += 1
                FeatureList[prep.__class__.__name__].extend(list(extra))
        if X_tr_new.shape[1] > fixed + NofUnary:
            OUT1, OUT2, OUT3 = cife(X_tr_new, y_train, XNumind, fixed = fixed,n_selected_features = fixed + NofUnary)
            X_tr_new = X_tr_new[ : , sorted(OUT1)]
            X_te_new = X_te_new[ : , sorted(OUT1)]
            Xset = set(XCatind).union(set(XNumind))
            for i in sorted([i for i in Xset - set(OUT1)],reverse = True):
                if i in XCatind:
                    XCatind = np.setdiff1d(XCatind, i)
                else:
                    XNumind = np.setdiff1d(XNumind, i)
                for j in XCatind:
                    if j > i:
                        XCatind = np.where(XCatind==j, j-1, XCatind)
                for j in XNumind:
                    if j > i:
                        XNumind = np.where(XNumind==j, j-1, XNumind)
                for key,val in FeatureList.items():
                    if i in val:
                        val.remove(i)
                        Counter[key] -= 1
                    if len(val) > 0 and np.max(val) > i:
                        for j in range(0,len(val)):
                            if val[j] > i:
                                val[j] -=1
    else:
        print("Wrong Method")
        5/0
            
    print("Unary done")
    #RETURN IF ONLY UNARY
    if not binary:
        prep = Bench(XCatind,XNumind)
        X_train_u_b = prep.fit_transform(deepcopy(X_tr_new))
        X_test_u_b = prep.transform(deepcopy(X_te_new))
        return X_train_u_b, X_test_u_b, y_train, y_test, Counter
    
    for cat in XCatind:
        le = LabelEncoder()
        X_tr_new[:,cat] = le.fit_transform(X_tr_new[:,cat])
        le_classes = le.classes_.tolist()
        bisect.insort_left(le_classes, 999999.)
        le.classes_ = le_classes
        X_te_new[:,cat] = np.asarray(pd.DataFrame(X_te_new[:,cat])[0].map(lambda s: 999999. if s not in le.classes_ else s))
        X_te_new[:,cat] = le.transform(X_te_new[:,cat])
    
    #START BINARY
    bestpreps = {}
    bestprobs = list(np.zeros(NofBinary))
    thres = min(bestprobs)
    pairs = list(itertools.combinations(XNumind, 2))
    random.shuffle(pairs)
    maxeval = np.ceil(NofBinary/3)
    MaxTime = 800
    TotalTime = ((len(XNumind)*(len(XNumind) - 1) / 2) + len(XNumind)*len(XCatind))*0.9
    if TotalTime > MaxTime:
        NecessaryTimeNC = len(XNumind)*len(XCatind)*0.9
        if NecessaryTimeNC > 0:
            TimeNC = np.max([100,NecessaryTimeNC / TotalTime * 750])
            TimeNN = MaxTime - TimeNC
        else: 
            TimeNC = 0
            TimeNN = MaxTime
    else:
        TimeNC = 750
        TimeNN = 750
    start = time()
    for subset in pairs:
        if time() - start > TimeNN:
            break
        bestpreps_i = {}
        bestprobs_i = list(np.zeros(NofEval+Extra))
        thres_i = min(bestprobs_i)
        i = subset[0] ; j = subset[1]
        f = X_tr_new[:,i] ; g = X_tr_new[:,j]
        MetaFeature = MetaNumNum(f,g,1,X_tr_new,y_train,XCatind)
        MetaFeature = MetaFeature.drop('y',1)
        MetaFeature = MetaFeature.drop('yfreq',1)
        MetaFeature = MetaFeature.drop('Xtotal',1)
        MetaFeature = MetaFeature.drop('XNum',1)
        MetaFeature = MetaFeature.drop('XCat',1)
        cols = list(MetaFeature.columns)
        cols = cols[17:34] + cols[:17] + cols[34:39] + [cols[40]] + [cols[39]]
        MetaFeature2 = MetaFeature[cols]
        for op in NumNum_models.keys():
            prep = getattr(NumNum(), op)(i, j, XCatind, XNumind)
            if not prep.issuitable(deepcopy(X_tr_new)):
                continue
            prob = NumNum_models[op].predict_proba(MetaFeature)[0][1]
            prob2 = NumNum_models[op].predict_proba(MetaFeature2)[0][1]
            if not prep.bothways():
                prob3 = np.max([prob,prob2])
                if prob3 > thres_i and prob3 > 0.5:
                    bestprobs_i.append(prob3)
                    bestprobs_i.sort(reverse = True)
                    bestprobs_i=bestprobs_i[:NofEval+Extra]
                    bestpreps_i[prep] = prob3
                    thres_i = min(bestprobs_i)
                    bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            if prep.bothways():
                if prob > thres_i and prob > 0.5:
                    bestprobs_i.append(prob)
                    bestprobs_i.sort(reverse = True)
                    bestprobs_i=bestprobs_i[:NofEval+Extra]
                    bestpreps_i[prep] = prob
                    thres_i = min(bestprobs_i)
                    bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
                if prob2 > thres_i and prob2 > 0.5:
                    bestprobs_i.append(prob2)
                    bestprobs_i.sort(reverse = True)
                    bestprobs_i=bestprobs_i[:NofEval+Extra]
                    prep2 = getattr(NumNum(), op)(j, i, XCatind, XNumind)
                    bestpreps_i[prep2] = prob2
                    thres_i = min(bestprobs_i)
                    bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
        sorted_preps = sorted(bestpreps_i.items(), key=lambda kv: kv[1],reverse=True)
        Toadd=NofEval
        for prep,prob in sorted_preps:
            if Toadd == 0:
                continue
            if prep.__class__.__name__ in [i.__class__.__name__ for i in bestpreps.keys()] and pd.value_counts([i.__class__.__name__ for i in bestpreps.keys()])[prep.__class__.__name__]>=maxeval:
                temp = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ == prep.__class__.__name__}
                bestpreps = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ != prep.__class__.__name__}
                tempmin=min(temp.values())
                if prob > tempmin:
                    temp={k:v for k,v in temp.items() if v != tempmin}
                    temp[prep] = prob
                    Toadd-=1
                bestpreps.update(temp)
                bestprobs = list(bestpreps.values())
                bestprobs.sort(reverse = True)
                thres = min(bestprobs)
            else:
                if prob > thres and prob > 0.5:
                    bestprobs.append(prob)
                    bestprobs.sort(reverse = True)
                    bestprobs=bestprobs[:NofBinary]
                    bestpreps[prep] = prob
                    thres = min(bestprobs)
                    bestpreps = {key:val for key, val in bestpreps.items() if val >= thres}
                    Toadd-=1
    start = time()
    for j in XCatind:
        if time() - start > TimeNC:
            break
        for i in XNumind:
            bestpreps_i = {}
            bestprobs_i = list(np.zeros(NofEval+Extra))
            thres_i = min(bestprobs_i)  
            f = X_tr_new[:,i] ; g = X_tr_new[:,j]
            MetaFeature = MetaNumCat(f,g,1,X_tr_new,y_train,XCatind)
            MetaFeature = MetaFeature.drop('y',1)
            MetaFeature = MetaFeature.drop('yfreq',1)
            MetaFeature = MetaFeature.drop('Xtotal',1)
            MetaFeature = MetaFeature.drop('XNum',1)
            MetaFeature = MetaFeature.drop('XCat',1)
            for op in NumCat_models.keys():
                prep = getattr(NumCat(), op)(i, j, XCatind, XNumind)
                if not prep.issuitable(deepcopy(X_tr_new)):
                    continue
                prob = NumCat_models[op].predict_proba(MetaFeature)[0][1]
                if prob > thres_i and prob > 0.5:
                    bestprobs_i.append(prob)
                    bestprobs_i.sort(reverse = True)
                    bestprobs_i=bestprobs_i[:NofEval+Extra]
                    thres_i = min(bestprobs_i)
                    bestpreps_i[prep] = prob
                    bestpreps_i = {key:val for key, val in bestpreps_i.items() if val >= thres_i}
            sorted_preps = sorted(bestpreps_i.items(), key=lambda kv: kv[1],reverse=True)
            Toadd=NofEval
            for prep,prob in sorted_preps:
                if Toadd == 0:
                    continue
                if prep.__class__.__name__ in [i.__class__.__name__ for i in bestpreps.keys()] and pd.value_counts([i.__class__.__name__ for i in bestpreps.keys()])[prep.__class__.__name__]>=maxeval:
                    temp = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ == prep.__class__.__name__}
                    bestpreps = {i: bestpreps[i] for i in bestpreps.keys() if i.__class__.__name__ != prep.__class__.__name__}
                    tempmin=min(temp.values())
                    if prob > tempmin:
                        temp={k:v for k,v in temp.items() if v != tempmin}
                        temp[prep] = prob
                        Toadd-=1
                    bestpreps.update(temp)
                    bestprobs = list(bestpreps.values())
                    bestprobs.sort(reverse = True)
                    thres = min(bestprobs)
                else:
                    if prob > thres and prob > 0.5:
                        bestprobs.append(prob)
                        bestprobs.sort(reverse = True)
                        bestprobs=bestprobs[:NofBinary]
                        bestpreps[prep] = prob
                        thres = min(bestprobs)
                        bestpreps = {key:val for key, val in bestpreps.items() if val >= thres}
                        Toadd-=1
    
    X_tr_new2 = deepcopy(X_tr_new)
    X_te_new2 = deepcopy(X_te_new)
    
    for prep in bestpreps.keys():
        #print(bestpreps[prep],prep)
        X_tr_new2 = prep.fit_transform(X_tr_new2)
        X_te_new2 = prep.transform(X_te_new2)
        extra = set(prep.XCatind) - set(XCatind)
        for i in extra:
            XCatind = np.append(XCatind,i)
            Counter[prep.__class__.__name__] += 1
        FeatureList[prep.__class__.__name__].extend(list(extra))
        extra = set(prep.XNumind) - set(XNumind)
        for i in extra:
            XNumind = np.append(XNumind,i)
            Counter[prep.__class__.__name__] += 1
        FeatureList[prep.__class__.__name__].extend(list(extra))
    if binaryselect:
        fixed = X_train.shape[1]
        if selectfixed:
            OUT1, _, _ = cife(X_tr_new2, y_train, XNumind, fixed = fixed,n_selected_features = fixed + NofBinaryTotal)
        else:
            OUT1, _, _ = cife(X_tr_new2, y_train, XNumind, fixed = 0,n_selected_features = fixed + NofBinaryTotal)
        X_tr_new2 = X_tr_new2[ : , sorted(OUT1)]
        X_te_new2 = X_te_new2[ : , sorted(OUT1)]
        Xset = set(XCatind).union(set(XNumind))
        for i in sorted([i for i in Xset - set(OUT1)],reverse = True):
            if i in XCatind:
                XCatind = np.setdiff1d(XCatind, i)
            else:
                XNumind = np.setdiff1d(XNumind, i)
            for j in XCatind:
                if j > i:
                    XCatind = np.where(XCatind==j, j-1, XCatind)
            for j in XNumind:
                if j > i:
                    XNumind = np.where(XNumind==j, j-1, XNumind)
            for key,val in FeatureList.items():
                if i in val:
                    val.remove(i)
                    Counter[key] -= 1
                if len(val) > 0 and np.max(val) > i:
                    for j in range(0,len(val)):
                        if val[j] > i:
                            val[j] -=1
    print("Binary done")
    prep = Bench(XCatind,XNumind)
    X_train_ub_b = prep.fit_transform(deepcopy(X_tr_new2))
    X_test_ub_b = prep.transform(deepcopy(X_te_new2))
    return X_train_ub_b, X_test_ub_b, y_train, y_test, Counter