# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:39 2019

@author: casper
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
from time import time

dicts = np.load('meta_data_total0905.npy').item()

'''
dictstot = deepcopy(dicts)
for i in dictstot:
    for j in dictstot[i]:
        dictstot[i][j] = dictstot[i][j].append(dicts2[i][j])
'''

Num_dict = dicts['Num'] ; NumNum_dict = dicts['NumNum']
Cat_dict = dicts['Cat'] ; NumCat_dict = dicts['NumCat']

Num_names = Num_dict.keys() ; NumNum_names = NumNum_dict.keys()
Cat_names = Cat_dict.keys() ; NumCat_names = NumCat_dict.keys()

Num_models = {key: pd.DataFrame() for key in Num_names}
Cat_models = {key: pd.DataFrame() for key in Cat_names}
NumNum_models = {key: pd.DataFrame() for key in NumNum_names}
NumCat_models = {key: pd.DataFrame() for key in NumCat_names}

def create_model(df):
    y = df['y'].values
    X = df.drop('y', 1).values
    print(np.mean(y))
    
    param_grid = {"max_depth": [3, 5, 10, None],
                  "max_features": [1, 5, 10, "sqrt", 0.5],
                  "min_samples_split": [2, 4, 7, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "min_samples_leaf": [1, 3, 5],
                  "n_estimators": [199,200,201]}
    clf = RandomForestClassifier(random_state=8520)
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
    
    #gs = GridSearchCV(clf, param_grid=param_grid, cv=skf, scoring='balanced_accuracy', n_jobs=-1, pre_dispatch=40)
    #gs.fit(X,y)
    rs = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter = 300, cv=skf, 
                            scoring='f1', n_jobs=-1, pre_dispatch=300, random_state=1995)
    rs.fit(X, y)
    print(rs.best_score_)
    '''
        Xtest = X[test]
        ytest = y[test]
        
        print(confusion_matrix(ytest, rs.best_estimator_.predict(Xtest)))
    '''
    return rs.best_estimator_

def create_model_bin(df):
    y = df['y'].values
    X = df.drop('y', 1).values
    print(np.mean(y))
    
    param_grid = {"max_depth": [3, 5, 10, None],
                  "max_features": [1, 5, 10, "sqrt", 0.5],
                  "min_samples_split": [2, 4, 7, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "min_samples_leaf": [1, 3, 5],
                  "n_estimators": [199,200,201]}
    clf = RandomForestClassifier(random_state=8520)
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 409)
    
    #gs = GridSearchCV(clf, param_grid=param_grid, cv=skf, scoring='balanced_accuracy', n_jobs=-1, pre_dispatch=40)
    #gs.fit(X,y)
    rs = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter = 50, cv=skf, 
                            scoring='f1', n_jobs=-1, pre_dispatch=50, random_state=1995)
    rs.fit(X, y)
    print(rs.best_score_)
    return rs.best_estimator_

for op in Num_names:
    print(op)
    df = Num_dict[op]
    df = df.drop('yfreq',axis=1)
    df = df.drop('Xtotal',axis=1)
    df = df.drop('XNum',axis=1)
    df = df.drop('XCat',axis=1)
    df = df.replace({'npeaks': {0: 1}})
    start = time()
    bestmodel = create_model(df)
    print(time() - start)
    Num_models[op] = bestmodel
    
for op in Cat_names:
    print(op)
    df = Cat_dict[op]
    df = df.fillna(0)
    df = df.drop('yfreq',axis=1)
    df = df.drop('Xtotal',axis=1)
    df = df.drop('XNum',axis=1)
    df = df.drop('XCat',axis=1)
    start = time()
    bestmodel = create_model(df)
    print(time() - start)
    Cat_models[op] = bestmodel
    
for op in NumNum_names:
    print(op)
    df = NumNum_dict[op]
    df = df.drop('yfreq',axis=1)
    df = df.drop('Xtotal',axis=1)
    df = df.drop('XNum',axis=1)
    df = df.drop('XCat',axis=1)
    start = time()
    bestmodel = create_model_bin(df)
    print(time() - start)
    NumNum_models[op] = bestmodel
    
for op in NumCat_names:
    print(op)
    df = NumCat_dict[op]
    df = df.drop('yfreq',axis=1)
    df = df.drop('Xtotal',axis=1)
    df = df.drop('XNum',axis=1)
    df = df.drop('XCat',axis=1)
    start = time()
    bestmodel = create_model_bin(df)
    print(time() - start)
    NumCat_models[op] = bestmodel
    
modeldicts = {'Num': Num_models, 'Cat': Cat_models, 'NumNum': NumNum_models, 'NumCat': NumCat_models}
np.save('meta_models.npy', modeldicts)
