# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:11:04 2019

@author: casper
"""

from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import scipy as sp
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

def MetaNum(f,MetaTarget,X,y,XCatind):
    meta_f = pd.DataFrame()
    meta_f['y'] = [MetaTarget]
    meta_f['mean'] = [f.mean()]
    meta_f['std'] = [f.std()]
    meta_f['skew'] = [sp.stats.skew(f)]
    meta_f['kurt'] = [sp.stats.kurtosis(f)]
    meta_f['median'] = [np.median(f)]
    meta_f['min'] = [min(f)]
    meta_f['max'] = [max(f)]
    meta_f['rangesize'] = [max(f)-min(f)]
    meta_f['posneg'] = [1.0 if max(f)>0 and min(f)<0 else 0.0]
    meta_f['mode'] = [sp.stats.mode(f)[0][0]]
    meta_f['modefreq'] = [sp.stats.mode(f)[1][0]/len(f)]
    
    kbd = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 10).fit(f.reshape(-1,1))
    q = kbd.transform(f.reshape(-1,1))[:,0]
    tab = pd.value_counts(q)
    bins = [tab[val]/len(f) if val in tab.keys() else 0.0 for val in range(0,10)]
    meta_f = meta_f.join(pd.DataFrame(
                 [bins], 
                 index=meta_f.index, 
                 columns=['bin1', 'bin2', 'bin3','bin4', 'bin5', 'bin6','bin7', 'bin8', 'bin9','bin10']
             ))
    meta_f['npeaks'] = len(find_peaks(bins, height = 0.10, distance=2)[0])
    #meta_f['nybinpeaks'] = len(find_peaks(pd.DataFrame([y,q]).T.groupby(1)[0].mean(), height = 0.10, distance=3)[0])
    
    one = [f[x] for x in range(0,len(f)) if y[x] == 1]
    zero = [f[x] for x in range(0,len(f)) if y[x] == 0]
    q0 = kbd.transform(np.asarray(zero).reshape(-1,1))[:,0]
    q1 = kbd.transform(np.asarray(one).reshape(-1,1))[:,0]
    bins0 = [pd.value_counts(q0)[val]/len(q0) if val in pd.value_counts(q0).keys() else 0.0 for val in range(0,10)]
    bins1 = [pd.value_counts(q1)[val]/len(q1) if val in pd.value_counts(q1).keys() else 0.0 for val in range(0,10)]
    meta_f['bincorr'] = [np.corrcoef(bins0, bins1)[0][1]]
    meta_f['binmaxdiff'] = [np.abs(pd.value_counts(q0).argmax() - pd.value_counts(q1).argmax())]
    meta_f['binchitest'] = [sum((np.asarray(bins0)-np.asarray(bins1))*(np.asarray(bins0)-np.asarray(bins1))/(np.asarray(bins0)+np.asarray(bins1)+0.00000000000001))]
    if np.mean(one) > np.mean(zero):
        meta_f['binmeanrate'] = [np.mean(one) / (np.mean(zero)+0.00000001)]
    else:
        meta_f['binmeanrate'] = [np.mean(zero) / (np.mean(one)+0.00000001)]
    meta_f['ycorr'] = [np.abs(np.corrcoef(f, y)[0][1])]
    meta_f['yfreq'] = [max(len(y)-sum(y),sum(y))/len(y)]
    meta_f['XCat'] = [len(XCatind)]
    meta_f['XNum'] = [np.shape(X)[1] - len(XCatind)]
    meta_f['Xtotal'] = [np.shape(X)[1]]
    meta_f=meta_f.fillna(0)
    meta_f=meta_f.replace(np.inf,1)
    return meta_f

def MetaCat(f,MetaTarget,X,y,XCatind):
    meta_f = pd.DataFrame()
    meta_f['y'] = [MetaTarget]
    meta_f['ncats'] = len(set(f))
    tab = pd.value_counts(f)
    bins = [sorted(tab,reverse = True)[val]/len(f) if val in tab.keys() else 0.0 for val in range(0,4)]
    meta_f = meta_f.join(pd.DataFrame(
                 [bins], 
                 index=meta_f.index, 
                 columns=['freqmax', 'freq2', 'freq3','freq4']))
    meta_f['remainingcats'] = [1 - sum(bins)]
    meta_f['freqmin'] = [sorted(tab)[0]/len(f)]
    meta_f['divmaxmin'] = [(meta_f['freqmax'] / meta_f['freqmin'])[0]]
        
    zero = [f[x] for x in range(0,len(f)) if y[x] == 0]
    one = [f[x] for x in range(0,len(f)) if y[x] == 1]
    bins0 = [pd.value_counts(zero)[tab.index.tolist()[ind]]/len(zero) if ind in tab.keys() and tab.index.tolist()[ind] in zero else 0.0 for ind in range(0,5)]
    bins1 = [pd.value_counts(one)[tab.index.tolist()[ind]]/len(one) if ind in tab.keys() and tab.index.tolist()[ind] in one else 0.0 for ind in range(0,5)]
    while bins0[-1]+bins1[-1] == 0.0:
        del bins0[-1] ; del bins1[-1]
    meta_f['bincorr'] = [np.corrcoef(bins0, bins1)[0][1]]
    meta_f['binsamemax'] = [1.0 if pd.value_counts(zero).argmax() == pd.value_counts(one).argmax() else 0.0]
    meta_f['binmaxfreqdiff'] = [max([np.abs(tab.sub(pd.value_counts(one),fill_value=0)[ind]/len(zero) - tab.sub(pd.value_counts(zero),fill_value=0)[ind]/len(one)) for ind in set(f)])]
    meta_f['ycorr'] = [np.abs(np.corrcoef(f, y)[0][1])]
    meta_f['yfreq'] = [max(len(y)-sum(y),sum(y))/len(y)]
    meta_f['XCat'] = [len(XCatind)]
    meta_f['XNum'] = [np.shape(X)[1] - len(XCatind)]
    meta_f['Xtotal'] = [np.shape(X)[1]]
    meta_f=meta_f.fillna(0)
    meta_f=meta_f.replace(np.inf,1)
    return meta_f

def MetaNumNum(f,g,MetaTarget,X,y,XCatind):
    meta_f = pd.DataFrame()
    meta_f['y'] = [MetaTarget]
    
    meta_f['fmean'] = [f.mean()]
    meta_f['fstd'] = [f.std()]
    meta_f['fskew'] = [sp.stats.skew(f)]
    meta_f['fmedian'] = [np.median(f)]
    q = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 5).fit_transform(f.reshape(-1,1))[:,0]
    ftab = pd.value_counts(q)
    fbins = [ftab[val]/len(f) if val in ftab.keys() else 0.0 for val in range(0,5)]
    meta_f = meta_f.join(pd.DataFrame(
                 [fbins], 
                 index=meta_f.index, 
                 columns=['fbin1', 'fbin2', 'fbin3','fbin4', 'fbin5']))    
    meta_f['fmin'] = [min(f)]
    meta_f['fmax'] = [max(f)]
    meta_f['frangesize'] = [max(f)-min(f)]
    meta_f['fmode'] = [sp.stats.mode(f)[0][0]]
    meta_f['fmodefreq'] = [sp.stats.mode(f)[1][0]/len(f)]
    kbd = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 10).fit(f.reshape(-1,1))
    one = [f[x] for x in range(0,len(f)) if y[x] == 1]
    zero = [f[x] for x in range(0,len(f)) if y[x] == 0]
    q0 = kbd.transform(np.asarray(zero).reshape(-1,1))[:,0]
    q1 = kbd.transform(np.asarray(one).reshape(-1,1))[:,0]
    bins0 = [pd.value_counts(q0)[val]/len(q0) if val in pd.value_counts(q0).keys() else 0.0 for val in range(0,10)]
    bins1 = [pd.value_counts(q1)[val]/len(q1) if val in pd.value_counts(q1).keys() else 0.0 for val in range(0,10)]
    meta_f['fbincorr'] = [np.corrcoef(bins0, bins1)[0][1]]
    meta_f['fbinchitest'] = [sum((np.asarray(bins0)-np.asarray(bins1))*(np.asarray(bins0)-np.asarray(bins1))/(np.asarray(bins0)+np.asarray(bins1)+0.00000000000001))]
    if np.mean(one) > np.mean(zero):
        meta_f['fbinmeanrate'] = [np.mean(one) / (np.mean(zero)+0.00000001)]
    else:
        meta_f['fbinmeanrate'] = [np.mean(zero) / (np.mean(one)+0.00000001)]
    
    meta_f['gmean'] = [g.mean()]
    meta_f['gstd'] = [g.std()]
    meta_f['gskew'] = [sp.stats.skew(g)]
    meta_f['gmedian'] = [np.median(g)]
    q = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 5).fit_transform(g.reshape(-1,1))[:,0]
    gtab = pd.value_counts(q)
    gbins = [gtab[val]/len(g) if val in gtab.keys() else 0.0 for val in range(0,5)]
    meta_f = meta_f.join(pd.DataFrame(
                 [gbins], 
                 index=meta_f.index, 
                 columns=['gbin1', 'gbin2', 'gbin3','gbin4', 'gbin5']))    
    meta_f['gmin'] = [min(g)]
    meta_f['gmax'] = [max(g)]
    meta_f['grangesize'] = [max(g)-min(g)]
    meta_f['gmode'] = [sp.stats.mode(g)[0][0]]
    meta_f['gmodefreq'] = [sp.stats.mode(g)[1][0]/len(g)]
    kbd = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 10).fit(g.reshape(-1,1))
    one = [g[x] for x in range(0,len(g)) if y[x] == 1]
    zero = [g[x] for x in range(0,len(g)) if y[x] == 0]
    q0 = kbd.transform(np.asarray(zero).reshape(-1,1))[:,0]
    q1 = kbd.transform(np.asarray(one).reshape(-1,1))[:,0]
    bins0 = [pd.value_counts(q0)[val]/len(q0) if val in pd.value_counts(q0).keys() else 0.0 for val in range(0,10)]
    bins1 = [pd.value_counts(q1)[val]/len(q1) if val in pd.value_counts(q1).keys() else 0.0 for val in range(0,10)]
    meta_f['gbincorr'] = [np.corrcoef(bins0, bins1)[0][1]]
    meta_f['gbinchitest'] = [sum((np.asarray(bins0)-np.asarray(bins1))*(np.asarray(bins0)-np.asarray(bins1))/(np.asarray(bins0)+np.asarray(bins1)+0.00000000000001))]
    if np.mean(one) > np.mean(zero):
        meta_f['gbinmeanrate'] = [np.mean(one) / (np.mean(zero)+0.00000001)]
    else:
        meta_f['gbinmeanrate'] = [np.mean(zero) / (np.mean(one)+0.00000001)]
        
    meta_f['fgcorr'] = [np.corrcoef(f, g)[0][1]]
    if f.mean() > g.mean():
        meta_f['fgmeanrate'] = [f.mean() / (g.mean()+0.00000001)]
        meta_f['fgstdrate'] = [f.std() / g.std()]
        meta_f['fgrangerate'] = [(meta_f['frangesize'] / meta_f['grangesize'])[0]]
    else:
        meta_f['fgmeanrate'] = [g.mean() / (f.mean()+0.00000001)]
        meta_f['fgstdrate'] = [g.std() / f.std()]
        meta_f['fgrangerate'] = [(meta_f['grangesize'] / meta_f['frangesize'])[0]]
    meta_f['fgmaxdiff'] = [np.abs(ftab.argmax() - gtab.argmax())]
    
    meta_f['yfcorr'] = [np.abs(np.corrcoef(f, y)[0][1])]
    meta_f['ygcorr'] = [np.abs(np.corrcoef(g, y)[0][1])]
    meta_f['yfreq'] = [max(len(y)-sum(y),sum(y))/len(y)]
    meta_f['XCat'] = [len(XCatind)]
    meta_f['XNum'] = [np.shape(X)[1] - len(XCatind)]
    meta_f['Xtotal'] = [np.shape(X)[1]]
    meta_f=meta_f.fillna(0)
    meta_f=meta_f.replace(np.inf,1)
    return meta_f

def MetaNumCat(f,g,MetaTarget,X,y,XCatind):
    meta_f = pd.DataFrame()
    meta_f['y'] = [MetaTarget]
    
    meta_f['fmean'] = [f.mean()]
    meta_f['fstd'] = [f.std()]
    meta_f['fskew'] = [sp.stats.skew(f)]
    meta_f['fmedian'] = [np.median(f)]    
    q = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 5).fit_transform(f.reshape(-1,1))[:,0]
    ftab = pd.value_counts(q)
    bins = [ftab[val]/len(f) if val in ftab.keys() else 0.0 for val in range(0,5)]
    meta_f = meta_f.join(pd.DataFrame(
                 [bins], 
                 index=meta_f.index, 
                 columns=['fbin1', 'fbin2', 'fbin3','fbin4', 'fbin5']))    
    meta_f['fmin'] = [min(f)]
    meta_f['fmax'] = [max(f)]
    meta_f['frangesize'] = [max(f)-min(f)]
    #meta_f['fposneg'] = [1.0 if max(f)>0 and min(f)<0 else 0.0]
    meta_f['fmode'] = [sp.stats.mode(f)[0][0]]
    meta_f['fmodefreq'] = [sp.stats.mode(f)[1][0]/len(f)]
    
    kbd = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform', n_bins = 10).fit(f.reshape(-1,1))
    one = [f[x] for x in range(0,len(f)) if y[x] == 1]
    zero = [f[x] for x in range(0,len(f)) if y[x] == 0]
    q0 = kbd.transform(np.asarray(zero).reshape(-1,1))[:,0]
    q1 = kbd.transform(np.asarray(one).reshape(-1,1))[:,0]
    bins0 = [pd.value_counts(q0)[val]/len(q0) if val in pd.value_counts(q0).keys() else 0.0 for val in range(0,10)]
    bins1 = [pd.value_counts(q1)[val]/len(q1) if val in pd.value_counts(q1).keys() else 0.0 for val in range(0,10)]
    meta_f['fbincorr'] = [np.corrcoef(bins0, bins1)[0][1]]
    meta_f['fbinchitest'] = [sum((np.asarray(bins0)-np.asarray(bins1))*(np.asarray(bins0)-np.asarray(bins1))/(np.asarray(bins0)+np.asarray(bins1)+0.00000000000001))]
    if np.mean(one) > np.mean(zero):
        meta_f['fbinmeanrate'] = [np.mean(one) / (np.mean(zero)+0.000000001)]
    else:
        meta_f['fbinmeanrate'] = [np.mean(zero) / (np.mean(one)+0.000000001)]
    
    meta_f['gncats'] = len(set(g))
    tab = pd.value_counts(g)
    tabsort = sorted(tab,reverse = True)
    bins = [tabsort[val]/len(g) if val < len(tabsort) else 0.0 for val in range(0,4)]
    meta_f = meta_f.join(pd.DataFrame(
                 [bins], 
                 index=meta_f.index, 
                 columns=['gfreqmax', 'gfreq2', 'gfreq3','gfreq4']))
    meta_f['gremainingcats'] = [1 - sum(bins)]
    meta_f['gfreqmin'] = [sorted(tab)[0]/len(g)]
    meta_f['gdivmaxmin'] = [(meta_f['gfreqmax'] / meta_f['gfreqmin'])[0]]
    
    one = [g[x] for x in range(0,len(g)) if y[x] == 1]
    zero = [g[x] for x in range(0,len(g)) if y[x] == 0]
    bins0 = [pd.value_counts(zero)[tab.index.tolist()[ind]]/len(zero) if ind in tab.keys() and tab.index.tolist()[ind] in zero else 0.0 for ind in range(0,5)]
    bins1 = [pd.value_counts(one)[tab.index.tolist()[ind]]/len(one) if ind in tab.keys() and tab.index.tolist()[ind] in one else 0.0 for ind in range(0,5)]
    while bins0[-1]+bins1[-1] == 0.0:
        del bins0[-1] ; del bins1[-1]
    meta_f['gbincorr'] = [np.corrcoef(bins0, bins1)[0][1]]
    meta_f['gbinsamemax'] = [1.0 if pd.value_counts(zero).argmax() == pd.value_counts(one).argmax() else 0.0]
    meta_f['gbinmaxfreqdiff'] = [max([np.abs(tab.sub(pd.value_counts(one),fill_value=0)[ind]/len(zero) - tab.sub(pd.value_counts(zero),fill_value=0)[ind]/len(one)) for ind in set(g)])]
    
    meta_f['fgcorr'] = [np.corrcoef(f, g)[0][1]]
    cat1 = [[f[x] for x in range(0,len(g)) if g[x] == ind] for ind in tab.index.tolist()[:min(3,len(set(g)))]]
    meta_f['fgfirst3catmeansrate'] = [100000.0 if min([np.mean(i) for i in cat1]) == 0.0 else max([np.mean(i) for i in cat1])/min([np.mean(i) for i in cat1])]
    
    meta_f['yfcorr'] = [np.abs(np.corrcoef(f, y)[0][1])]
    meta_f['ygcorr'] = [np.abs(np.corrcoef(g, y)[0][1])]
    meta_f['yfreq'] = [max(len(y)-sum(y),sum(y))/len(y)]
    meta_f['XCat'] = [len(XCatind)]
    meta_f['XNum'] = [np.shape(X)[1] - len(XCatind)]
    meta_f['Xtotal'] = [np.shape(X)[1]]
    meta_f=meta_f.fillna(0)
    meta_f=meta_f.replace(np.inf,1)
    return meta_f