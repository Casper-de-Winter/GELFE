# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:07 2019

@author: casper
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer, KBinsDiscretizer, LabelEncoder#, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy

#!pip install category_encoders
import category_encoders as ce

class Bench(BaseEstimator, TransformerMixin):
    def __init__(self, XCatind, XNumind):
        self.XCatind = XCatind
        self.XNumind = XNumind
        self.Cats = {key: [] for key in XCatind}
    
    def fit(self, X, y=None):
        for i in self.XCatind:
            self.Cats[i] = list(set(X[:,i]))
        return self
    
    def transform(self, X):
        X2 = deepcopy(X)
        todel = []
        for i in self.XCatind:
            if len(self.Cats[i]) > 2:
                todel.append(i)
                a=pd.get_dummies(X2[:,i])
                if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                    a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                    for j in range(0,len(self.Cats[i])):
                        if self.Cats[i][j] not in a.columns.tolist():
                            a.insert(loc = j, column = self.Cats[i][j], value = 0)
                a=a.values[:,1:]
                X2 = np.column_stack([X2,a])
        X2 = np.delete(X2, todel, axis=1)
        return X2

class Numerical():
    class Ln(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if f.min() < 0:
                Cont = False
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            if f.min() == 0:
                f = np.log(f+1)
            else:
                f = np.log(f)
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
        
    class Inverse(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if f.min() < 0 and f.max() > 0:
                Cont = False
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            if f.min() >= 0: 
                for j in range(len(f)):
                    if f[j] == 0:
                        f[j] += 0.0001
            else:
                for j in range(len(f)):
                    if f[j] == 0:
                        f[j] -= 0.0001
            f = 1/(f)
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class Sqrt(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if f.min() < 0:
                Cont = False
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            f = np.sqrt(f)
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class InverseSqrt(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if f.min() < 0:
                Cont = False
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            f = np.sqrt(f)
            for j in range(len(f)):
                if f[j] == 0:
                    f[j] += 0.0001
            f = 1/(f)
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class Squared(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            f = f*f
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
        
    class Cubed(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            f = f*f*f
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class RobStddz(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.rob = RobustScaler()
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            f = X[:,self.col]
            self.rob.fit(f.reshape(-1,1))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.rob.transform(f.reshape(-1,1))[:,0]
            X[:,self.col] = g
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
        
    class Arctan(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.rob = RobustScaler()
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            f = X[:,self.col]
            self.rob.fit(f.reshape(-1,1))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.rob.transform(f.reshape(-1,1))[:,0]
            g = np.arctan(g)
            X[:,self.col] = g
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class Quantile(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.quantile = QuantileTransformer()
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            f = X[:,self.col]
            self.quantile.fit(f.reshape(-1,1))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.quantile.transform(f.reshape(-1,1))[:,0]
            X[:,self.col] = g
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
        
    class EqualF5(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.kbins = KBinsDiscretizer(encode = 'ordinal', strategy = 'quantile', n_bins = 5)
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            self.kbins.fit(f.reshape(-1,1))
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = list(set(self.kbins.transform(f.reshape(-1,1))[:,0]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.kbins.transform(f.reshape(-1,1))[:,0]
            #if len(set(g)) < 5:
            #    g = LabelEncoder().fit_transform(g)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,g])
            XCatOrig = deepcopy(self.XCatind)
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
    
    class EqualF10(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.kbins = KBinsDiscretizer(encode = 'ordinal', strategy = 'quantile', n_bins = 10)
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            self.kbins.fit(f.reshape(-1,1))
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = list(set(self.kbins.transform(f.reshape(-1,1))[:,0]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.kbins.transform(f.reshape(-1,1))[:,0]
            #if len(set(g)) < 10:
            #    g = LabelEncoder().fit_transform(g)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,g])
            XCatOrig = deepcopy(self.XCatind)
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
        
    class EqualR5(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.kbins = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 5)
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            self.kbins.fit(f.reshape(-1,1))
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = list(set(self.kbins.transform(f.reshape(-1,1))[:,0]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.kbins.transform(f.reshape(-1,1))[:,0]
            #if len(set(g)) < 5:
            #    g = LabelEncoder().fit_transform(g)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,g])
            XCatOrig = deepcopy(self.XCatind)
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
        
    class EqualR10(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.kbins = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 10)
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            self.kbins.fit(f.reshape(-1,1))
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = list(set(self.kbins.transform(f.reshape(-1,1))[:,0]))
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            g = self.kbins.transform(f.reshape(-1,1))[:,0]
            #if len(set(g)) < 10:
            #    g = LabelEncoder().fit_transform(g)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,g])
            XCatOrig = deepcopy(self.XCatind)
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            #print(j, self.Cats[i][j], a.columns.tolist())
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
    
    class Bigger0(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            if f.min() > 0 or f.max() <= 0:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = [0.0,1.0]
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            f = f>0
            f = f.astype(float)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,f])
            XCatOrig = deepcopy(self.XCatind)
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
        
    class MedSplit(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.med = 0
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            if np.median(f) == max(f):
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            self.med = np.median(f)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = [0.0,1.0]
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            f = f>self.med
            f = f.astype(float)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,f])
            XCatOrig = deepcopy(self.XCatind)
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
        
class Categorical():
    class GroupLowFreq(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.colsgroup = set()
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
            self.new = 0
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) <= 2:
                Cont = False
                return Cont
            if sorted(pd.value_counts(f))[1] > len(f)*0.05: #At least two below 0.05
                Cont = False
            if sorted(pd.value_counts(f),reverse=True)[0] < len(f)*0.05: #At least two below 0.05
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            z = pd.value_counts(f)
            group = set()
            while z[z.idxmin()] < len(f)*0.05:
                group.add(z.idxmin())
                z=z.drop(z.idxmin())
            self.colsgroup = set(f) - group
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.new = max(f)+100
            self.Cats[self.col] = list(self.colsgroup)
            self.Cats[self.col].append(self.new)
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            for z in range(0,len(f)):
                if f[z] not in self.colsgroup:
                    f[z] = self.new
            #f = LabelEncoder().fit_transform(f)
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
        
    class GroupSim(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.colsgroup = []
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.fset = set()
            self.fmax = 0.0
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if sorted(pd.value_counts(f))[0] > len(f)*0.05:
                Cont = False
            if len(set(f)) <= 2:
                Cont = False
            return Cont
        
        def fit(self, X, y):
            f = X[:,self.col]
            self.fset = set(deepcopy(f))
            tab = pd.value_counts(f)
            self.fmax = tab.idxmax()
            gby = pd.DataFrame([y,f]).T.groupby(1)[0].mean()
            cut = len(f)*0.05
            while tab[tab.idxmin()] < cut:
                close = gby.iloc[(gby-gby[tab.idxmin()]).abs().argsort()].index.tolist()
                while close[0] == tab.idxmin():
                    close.pop(0)
                close = close[0]
                self.colsgroup.append([tab.idxmin(),close])
                for j in range(len(f)):
                    if f[j] == tab.idxmin():
                        f[j] = close
                tab = pd.value_counts(f)
                gby = pd.DataFrame([y,f]).T.groupby(1)[0].mean()
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[self.col] = list(tab.keys())
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            if len(set(f) - self.fset) > 0:
                cats = set(f) - self.fset
                for j in range(len(f)):
                    if f[j] in cats:
                        f[j] = self.fmax
                for pair in self.colsgroup:
                    for j in range(len(f)):
                        if f[j] == pair[0]:
                            f[j] = pair[1]
            else:
                for pair in self.colsgroup:
                    for j in range(len(f)):
                        if f[j] == pair[0]:
                            f[j] = pair[1]
            
            #f = LabelEncoder().fit_transform(f)
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class LargestVs(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.largest = 0.0
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) <= 2:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            z = pd.value_counts(f)
            self.largest = z.idxmax()
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[self.col] = list([0,1])
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            for j in range(0,len(f)):
                if f[j] == self.largest:
                    f[j] = 1
                else:
                    f[j] = 0
            X[:,self.col] = f
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            return X2
    
    class Cat2Num(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) <= 2:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats.pop(self.col)
            return self
        
        def transform(self, X):
            XCatOrig = deepcopy(self.XCatind)
            XNumOrig = deepcopy(self.XNumind)
            self.XCatind = np.setdiff1d(self.XCatind,self.col,True)
            self.XNumind = np.append(self.XNumind,self.col)
            self.XNumind = sorted(self.XNumind)
            X2 = deepcopy(X)
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            self.XNumind = XNumOrig
            return X2
    
    class FreqEnc(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.tab = pd.Series()
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) == 1:
                Cont = False
            ff = deepcopy(f)
            tab = pd.value_counts(ff)
            for j in range(len(ff)):
                ff[j] = tab[f[j]]/len(ff)
            if len(set(ff)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            self.tab = pd.value_counts(f)/len(f)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XCatOrig = deepcopy(self.XCatind)
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.col]
            for j in range(len(f)):
                if f[j] in self.tab.keys():
                    f[j] = self.tab[f[j]]
                else:
                    f[j] = 0.0            
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,f])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            self.XNumind = XNumOrig
            return X2
        
    class BinaryEnc(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.binary = ce.BinaryEncoder(cols=['col'])
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) <= 2:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            z = pd.DataFrame(f,columns = ['col'])
            self.binary.fit(z)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[self.col] = list([0,1])
            return self
        
        def transform(self, X):
            f = X[:,self.col]
            z = pd.DataFrame(f,columns = ['col'])
            g = self.binary.transform(z)
            g = np.asarray(g)
            XCatOrig = deepcopy(self.XCatind)
            X2 = deepcopy(X)
            X2[:,self.col] = g[:,0]
            X2 = np.column_stack([X2,g[:,1:]])
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            for ind in range(X.shape[1],X2.shape[1]):
                self.Cats[ind] = list([0,1])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
        
    class BackwEnc(BaseEstimator, TransformerMixin):
        def __init__(self,col, XCatind, XNumind):
            self.backw = ce.BackwardDifferenceEncoder(cols=['col'])
            self.col = col
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.col]
            if len(set(f)) <= 2:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.col]
            z = pd.DataFrame(f,columns = ['col'])
            self.backw.fit(z)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats.pop(self.col)
            return self
        
        def transform(self, X):
            XCatOrig = deepcopy(self.XCatind)
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.col]
            z = pd.DataFrame(f,columns = ['col'])
            g = self.backw.transform(z)
            g = np.asarray(g)[:,1:]
            X2 = deepcopy(X)
            X2[:,self.col] = g[:,0]
            self.XCatind = np.setdiff1d(self.XCatind,self.col,True)
            self.XNumind = np.append(self.XNumind,self.col)
            self.XNumind = sorted(self.XNumind)
            X2 = np.column_stack([X2,g[:,1:]])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            self.XNumind = XNumOrig
            return X2

class NumNum():
    class Plus(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            f = X[:,self.colf]
            g = X[:,self.colg]
            XNumOrig = deepcopy(self.XNumind)
            h=f+g
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
    
    class Minus(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h=f-g
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class Diff(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h=np.abs(f-g)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class Times(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            if len(set(f*g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def bothways(self):
            return False
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h=f*g
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
    
    class Division(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def bothways(self):
            return True
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            f = f.astype(float)
            h = np.divide(f, g, out=np.zeros_like(f), where=g!=0, dtype = float)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class Hypot(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            if len(set(np.sqrt(f*f + g*g))) == 1:
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h=np.sqrt(f*f + g*g)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class LnTimes(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            if f.min() < 0 or g.min() < 0:
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h=f*g
            if h.min() == 0:
                h = np.log(h+1)
            else:
                h = np.log(h)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class Larger(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            if f.min() > g.max() or g.min() > f.max():
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            self.Cats[X.shape[1]] = [0.0,1.0]
            return self
        
        def transform(self, X):
            XCatOrig = deepcopy(self.XCatind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = f>g
            h = h.astype(float)
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2
        
    class Quadrant(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.medf = 0.0
            self.medg = 0.0
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            if np.median(f) == f.max() and np.median(g) == g.max():
                Cont = False
            return Cont
        
        def bothways(self):
            return False
        
        def fit(self, X, y=None):
            f = X[:,self.colf]
            g = X[:,self.colg]
            self.medf = np.median(f)
            self.medg = np.median(g)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            if f.max() > self.medf and g.max() > self.medg:
                lst = [0.0,1.0,2.0,3.0]
            elif f.max() <= self.medf:
                if g.max() > self.medg:
                    lst = [0.0,1.0]
                else:
                    lst = [0.0]
            else: 
                lst = [0.0,2.0]
            self.Cats[X.shape[1]] = lst
            return self
        
        def transform(self, X):
            XCatOrig = deepcopy(self.XCatind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = deepcopy(f)
            for j in range(0,len(h)):
                if f[j] <= self.medf and g[j] <= self.medf:
                    h[j] = 0.0
                elif f[j] <= self.medf and g[j] > self.medf:
                    h[j] = 1.0
                elif f[j] > self.medf and g[j] <= self.medf:
                    h[j] = 2.0
                else:
                    h[j] = 3.0
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XCatind = np.append(self.XCatind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XCatind = XCatOrig
            return X2

class NumCat():
    class GroupMean(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.tab = pd.Series()
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.mean = 0
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.colf]
            g = X[:,self.colg]
            self.tab = pd.DataFrame([f,g]).T.groupby(1)[0].mean()
            self.mean = f.mean()
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = deepcopy(f)
            for j in range(len(h)):
                if g[j] in self.tab.keys():
                    h[j] = self.tab[g[j]]
                else:
                    h[j] = self.mean 
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class GroupMedian(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.tab = pd.Series()
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.median = 0
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.colf]
            g = X[:,self.colg]
            self.tab = pd.DataFrame([f,g]).T.groupby(1)[0].median()
            self.median = np.median(f)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = deepcopy(f)
            for j in range(len(h)):
                if g[j] in self.tab.keys():
                    h[j] = self.tab[g[j]]
                else:
                    h[j] = self.median 
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class GroupStd(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.tab = pd.Series()
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.std = 0
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.colf]
            g = X[:,self.colg]
            self.tab = pd.DataFrame([f,g]).T.groupby(1)[0].std()
            self.tab = self.tab.fillna(1)
            self.std = f.std()
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = deepcopy(f)
            for j in range(len(h)):
                if g[j] in self.tab.keys():
                    h[j] = self.tab[g[j]]
                else:
                    h[j] = self.std 
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class MinGroupMed(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.tab = pd.Series()
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.median = 0
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.colf]
            g = X[:,self.colg]
            self.tab = pd.DataFrame([f,g]).T.groupby(1)[0].median()
            self.median = np.median(f)
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = deepcopy(f)
            for j in range(len(h)):
                if g[j] in self.tab.keys():
                    h[j] = f[j] - self.tab[g[j]]
                else:
                    h[j] = f[j] - self.median 
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class MinGroupNorm(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.tab = pd.Series()
            self.tabstd = pd.Series()
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.mean = 0
            self.std = 0
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            f = X[:,self.colf]
            g = X[:,self.colg]
            self.tab = pd.DataFrame([f,g]).T.groupby(1)[0].mean()
            self.mean = f.mean()
            self.tabstd = pd.DataFrame([f,g]).T.groupby(1)[0].std()
            self.tabstd = self.tabstd.fillna(1)
            self.tabstd = self.tabstd.replace(0, 1)
            self.std = f.std()
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            h = deepcopy(f)
            for j in range(len(h)):
                if g[j] in self.tab.keys():
                    h[j] = (f[j]-self.tab[g[j]]) / self.tabstd[g[j]]
                else:
                    h[j] = (f[j]-self.mean) / self.std
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2
        
    class OneHotNum(BaseEstimator, TransformerMixin):
        def __init__(self, colf, colg, XCatind, XNumind):
            self.colf = colf
            self.colg = colg
            self.XCatind = XCatind
            self.XNumind = XNumind
            self.Cats = {key: [] for key in XCatind}
        
        def issuitable(self, X):
            Cont = True
            f = X[:,self.colf]
            g = X[:,self.colg]
            if len(set(f)) == 1:
                Cont = False
            if len(set(g)) == 1:
                Cont = False
            return Cont
        
        def fit(self, X, y=None):
            for i in self.XCatind:
                self.Cats[i] = list(set(X[:,i]))
            return self
        
        def transform(self, X):
            XNumOrig = deepcopy(self.XNumind)
            f = X[:,self.colf]
            g = X[:,self.colg]
            oh=pd.get_dummies(g,drop_first=False)
            if len(set(oh.columns.tolist())-set(self.Cats[self.colg])) > 0:
                oh=oh.drop(set(oh.columns.tolist())-set(self.Cats[self.colg]),axis =1)
            if len(set(self.Cats[self.colg])-set(oh.columns.tolist())) > 0:
                for j in range(0,len(self.Cats[self.colg])):
                    if self.Cats[self.colg][j] not in oh.columns.tolist():
                        oh.insert(loc = j, column = self.Cats[self.colg][j], value = 0)
            oh = oh.values
            h=f.reshape(-1,1)*oh
            X2 = deepcopy(X)
            X2 = np.column_stack([X2,h])
            self.XNumind = np.append(self.XNumind,[ind for ind in range(X.shape[1],X2.shape[1])])
            todel = []
            for i in self.XCatind:
                if len(self.Cats[i]) > 2:
                    todel.append(i)
                    a=pd.get_dummies(X2[:,i])
                    if len(set(a.columns.tolist())-set(self.Cats[i])) > 0:
                        a=a.drop(set(a.columns.tolist())-set(self.Cats[i]),axis =1)
                    if len(set(self.Cats[i])-set(a.columns.tolist())) > 0:
                        for j in range(0,len(self.Cats[i])):
                            if self.Cats[i][j] not in a.columns.tolist():
                                a.insert(loc = j, column = self.Cats[i][j], value = 0)
                    a=a.values[:,1:]
                    X2 = np.column_stack([X2,a])
            X2 = np.delete(X2, todel, axis=1)
            self.XNumind = XNumOrig
            return X2