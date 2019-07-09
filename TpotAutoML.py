# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:56:37 2019

@author: casper
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from tpot import TPOTClassifier
from deap import creator
#import sklearn.model_selection
from sklearn.metrics import balanced_accuracy_score as bacc
from MainAutoFE import AutoFE
from sklearn.ensemble import RandomForestClassifier

import os
import glob
import pickle
from time import time

bestdict = {}

def TPOT(X_train, X_test, y_train, y_test, total_time, AutoFE_time): #TPOT_comparison
    tpot = TPOTClassifier(generations=40,
                          population_size=30,
                          verbosity=2,
                          scoring="balanced_accuracy",
                          max_time_mins=total_time,
                          max_eval_time_mins = 15,
                          n_jobs=-1,
                          random_state=2051920
                          )

    tpot.fit(X_train, y_train)
    print(tpot.score(X_test,y_test))

    test_all_pipelines = [test_performance_tpot(tpot, j, X_train, X_test, y_train, y_test, AutoFE_time)
                          for j in range(len(list(tpot.evaluated_individuals_.keys())))]

    test_all_pipelines_df = pd.DataFrame(test_all_pipelines)

    test_all_pipelines_df.columns = ["Cross-Validation accuracy", "Test accuracy", "Total time elapsed",
                                     "Internal time elapsed", "Internal_time_plus_FE", "Confidence Left", "Confidence Right"]#, "Fitted pipeline"]

    generation_count = test_all_pipelines_df.loc[:, "Total time elapsed"] \
        .sort_values() \
        .round(2) \
        .unique()
    
    generation_count = pd.DataFrame(generation_count)
    generation_count.reset_index(inplace=True)
    generation_count.columns = ["Generation", "Total time elapsed"]
    
    test_all_pipelines_df = test_all_pipelines_df.round({"Total time elapsed": 2})
    test_all_pipelines_df = test_all_pipelines_df.round({"Internal time elapsed": 2})
    test_all_pipelines_df = pd.merge(test_all_pipelines_df, generation_count,
                                     how='right',
                                     on='Total time elapsed')
    
    return test_all_pipelines_df

def test_performance_tpot(tpot, j, X_train, X_test, y_train, y_test, AutoFE_time):
    pipeline_str = list(tpot.evaluated_individuals_.keys())[j]  # retrieve string of arbitrary trained pipeline,

    # convert pipeline string to scikit-learn pipeline object
    optimized_pipeline = creator.Individual.from_string(pipeline_str, tpot._pset)  # obtain the correspoding DEAP object
    fitted_pipeline = tpot._toolbox.compile(expr=optimized_pipeline)  # convert DEAP object to scikit-learn pipeline object

    internal_cv_score = tpot.evaluated_individuals_[pipeline_str]['internal_cv_score']
    total_mins_elapsed = tpot.evaluated_individuals_[pipeline_str]['time_elapsed']
    #print('Calculated {} of {} pipelines of TPOT'.format(j + 1, len(list(tpot.evaluated_individuals_.keys()))))
    internal_time_elapsed = tpot.evaluated_individuals_[pipeline_str]['internal_time_elapsed']
    time_plus_FE = internal_time_elapsed + AutoFE_time
    
    if internal_cv_score >= bestdict[np.ceil(internal_time_elapsed)]:
        # Fit pipeline from scikit-learn
        try:
            fitted_pipeline.fit(X_train, y_train) # Fit the pipeline like an ordinary scikit-learn pipeline
            y_pred = fitted_pipeline.predict(X_test)
            test_score = bacc(y_test,y_pred)
            for key in bestdict.keys():
                if key >= np.ceil(internal_time_elapsed):
                    bestdict[key] = internal_cv_score
            R = 1000
            BACCs = list(np.zeros(R))
            for i in range(0,R):
                idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
                y_predi = y_pred[idx]
                y_testi = y_test[idx]
                BACCs[i] = bacc(y_testi,y_predi)
            bias = np.mean(BACCs) - test_score
            var = np.mean((BACCs - test_score) * (BACCs - test_score))
            conf_left = test_score - bias - np.sqrt(var)*1.96
            conf_right = test_score - bias + np.sqrt(var)*1.96
        except: 
            test_score = float('NaN')
            conf_left = float('NaN')
            conf_right = float('NaN')
    else:
        test_score = float('NaN')
        conf_left = float('NaN')
        conf_right = float('NaN')
    
    return internal_cv_score, test_score, total_mins_elapsed, internal_time_elapsed, time_plus_FE, conf_left, conf_right #, fitted_pipeline

total_time = 180
loc = r'C:\OneDrive - Building Blocks\Thesis\Data\TestData\Test'
loc = r'C:\Users\casper\OneDrive - Building Blocks\Thesis\Data\TestData\Test' 
testfiles = glob.glob(os.path.join(loc, '*.csv'))
test_size = 0.40
bestdict = {key: 0.0 for key in range(0,total_time+60)}
CounterBAll = {}; CounterCAll = {}; CounterDAll = {}; CounterEAll = {}; CounterFAll = {}
FEtimeA = {}; FEtimeB = {}; FEtimeC = {}; FEtimeD = {}; FEtimeE = {}; FEtimeF = {}
Counters = {}; Times = {}
#CountersSel = {}; TimesSel = {}

for data in testfiles:
    df = pd.read_csv(data,index_col='Unnamed: 0')
    NofUnary = 10; NofBinary = 10
    if df.shape[1] > 80:
        NofUnary = 20; NofBinary = 20
    elif df.shape[1] > 40:
        NofUnary = 15; NofBinary = 15
    if df.shape[1] < 10:
        NofUnary = 7; NofBinary = 7
    print(data)
    
    bestdict = {key: 0.0 for key in range(0,total_time+60)}
    start = time()
    X_train, X_test, y_train, y_test, Counter = AutoFE(df, test_size, original = True, method = "BestK", binary = False, NofEval = 2,
                                                       NofUnary=NofUnary, NofBinary=NofBinary, binaryselect = False,selectfixed = True, NofBinaryTotal = 15)
    seconds = time()-start
    FEtimeA[str(data[69:-4])] = seconds
    print(seconds)
    print(bacc(y_test,RandomForestClassifier(random_state=8520).fit(X_train, y_train).predict(X_test)))
    test_tpot = TPOT(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                total_time=total_time, AutoFE_time = seconds/60)
    
    results_tpot = test_tpot.sort_values(by=["Total time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    #f = open(loc + r"\\tpot_results_" + str(data[56:-4]) + "orig" + ".pkl", "wb")
    f = open(loc + r"\\tpot_results_" + str(data[69:-4]) + "orig" + ".pkl", "wb")
    pickle.dump(results_tpot, f)
    f.close()
    
    bestdict = {key: 0.0 for key in range(0,total_time+60)}
    start = time()
    X_train, X_test, y_train, y_test, CounterB = AutoFE(df, test_size, original = False, method = "BestK", binary = False, NofEval = 2,
                                                       NofUnary=NofUnary, NofBinary=NofBinary, binaryselect = False,selectfixed = True, NofBinaryTotal = 15)
    seconds = time()-start
    FEtimeB[str(data[69:-4])] = seconds
    CounterBAll[str(data[69:-4])] = CounterB
    print(seconds)
    print(bacc(y_test,RandomForestClassifier(random_state=8520).fit(X_train, y_train).predict(X_test)))
    test_tpot = TPOT(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                total_time=total_time, AutoFE_time = seconds/60)
    
    results_tpot = test_tpot.sort_values(by=["Total time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    #f = open(loc + r"\\tpot_results_" + str(data[56:-4]) + "unary" + ".pkl", "wb")
    f = open(loc + r"\\tpot_results_" + str(data[69:-4]) + "unary" + ".pkl", "wb")
    pickle.dump(results_tpot, f)
    f.close()
    
    bestdict = {key: 0.0 for key in range(0,total_time+60)}
    start = time()
    X_train, X_test, y_train, y_test, CounterC = AutoFE(df, test_size, original = False, method = "BestK", binary = True, NofEval = 2,
                                                       NofUnary=NofUnary, NofBinary=NofBinary, binaryselect = False,selectfixed = True, NofBinaryTotal = 15)
    seconds = time()-start
    FEtimeC[str(data[69:-4])] = seconds
    CounterCAll[str(data[69:-4])] = CounterC
    print(seconds)
    print(bacc(y_test,RandomForestClassifier(random_state=8520).fit(X_train, y_train).predict(X_test)))
    test_tpot = TPOT(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                total_time=total_time, AutoFE_time = seconds/60)
    
    results_tpot = test_tpot.sort_values(by=["Total time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    #f = open(loc + r"\\tpot_results_" + str(data[56:-4]) + "ub2" + ".pkl", "wb")
    f = open(loc + r"\\tpot_results_" + str(data[69:-4]) + "ub2" + ".pkl", "wb")
    pickle.dump(results_tpot, f)
    f.close()
    
    bestdict = {key: 0.0 for key in range(0,total_time+60)}
    start = time()
    X_train, X_test, y_train, y_test, CounterD = AutoFE(df, test_size, original = False, method = "BestK", binary = True, NofEval = 1,
                                                       NofUnary=NofUnary, NofBinary=NofBinary, binaryselect = False,selectfixed = True, NofBinaryTotal = 15)
    seconds = time()-start
    FEtimeD[str(data[69:-4])] = seconds
    CounterDAll[str(data[69:-4])] = CounterD
    print(seconds)
    print(bacc(y_test,RandomForestClassifier(random_state=8520).fit(X_train, y_train).predict(X_test)))
    test_tpot = TPOT(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                total_time=total_time, AutoFE_time = seconds/60)
    
    results_tpot = test_tpot.sort_values(by=["Total time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    #f = open(loc + r"\\tpot_results_" + str(data[56:-4]) + "ub1" + ".pkl", "wb")
    f = open(loc + r"\\tpot_results_" + str(data[69:-4]) + "ub1" + ".pkl", "wb")
    pickle.dump(results_tpot, f)
    f.close()

    bestdict = {key: 0.0 for key in range(0,total_time+60)}
    start = time()
    X_train, X_test, y_train, y_test, CounterE = AutoFE(df, test_size, original = False, method = "Selection", binary = True, NofEval = 2,
                                                       NofUnary=NofUnary, NofBinary=50, binaryselect = True,selectfixed = True, NofBinaryTotal = np.ceil(NofUnary*1.5))
    CounterEAll[str(data[69:-4])] = CounterE
    seconds = time()-start
    FEtimeE[str(data[69:-4])] = seconds
    print(seconds) 
    print(bacc(y_test,RandomForestClassifier(random_state=8520).fit(X_train, y_train).predict(X_test)))
    test_tpot = TPOT(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                total_time=total_time, AutoFE_time = seconds/60)
    
    results_tpot = test_tpot.sort_values(by=["Total time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    #f = open(loc + r"\\tpot_results_" + str(data[56:-4]) + "orig" + ".pkl", "wb")
    f = open(loc + r"\\tpot_results_" + str(data[69:-4]) + "select_fixed" + ".pkl", "wb")
    pickle.dump(results_tpot, f)
    f.close()
    
    bestdict = {key: 0.0 for key in range(0,total_time+60)}
    start = time()
    X_train, X_test, y_train, y_test, CounterF = AutoFE(df, test_size, original = False, method = "Selection", binary = True, NofEval = 2,
                                                       NofUnary=NofUnary, NofBinary=50, binaryselect = True,selectfixed = False, NofBinaryTotal = np.ceil(NofUnary*1.5))
    CounterFAll[str(data[69:-4])] = CounterF
    seconds = time()-start
    FEtimeF[str(data[69:-4])] = seconds
    print(seconds)
    print(bacc(y_test,RandomForestClassifier(random_state=8520).fit(X_train, y_train).predict(X_test)))
    test_tpot = TPOT(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                total_time=total_time, AutoFE_time = seconds/60)
    
    results_tpot = test_tpot.sort_values(by=["Total time elapsed","Cross-Validation accuracy"], ascending=[True,False])
    #f = open(loc + r"\\tpot_results_" + str(data[56:-4]) + "orig" + ".pkl", "wb")
    f = open(loc + r"\\tpot_results_" + str(data[69:-4]) + "select_full" + ".pkl", "wb")
    pickle.dump(results_tpot, f)
    f.close()
    
    #CountersSel['E'] = CounterEAll; CountersSel['F'] = CounterFAll
    #TimesSel['E'] = FEtimeE; TimesSel['F'] = FEtimeF
    #np.save('CountersSel.npy', CountersSel)
    #np.save('TimesSel.npy', TimesSel)
    
    Counters['B'] = CounterBAll; Counters['C'] = CounterCAll; Counters['D'] = CounterDAll; Counters['E'] = CounterEAll; Counters['F'] = CounterFAll
    Times['A'] = FEtimeA; Times['B'] = FEtimeB; Times['C'] = FEtimeC; Times['D'] = FEtimeD; Times['E'] = FEtimeE; Times['F'] = FEtimeF
    np.save('Counters.npy', Counters)
    np.save('Times.npy', Times)