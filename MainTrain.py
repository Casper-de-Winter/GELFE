# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:42:37 2019

@author: casper
"""

from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import warnings
import glob
import os
from time import time
warnings.filterwarnings("ignore")

from OperatorsV2 import Numerical
from OperatorsV2 import Categorical
from OperatorsV2 import NumNum
from OperatorsV2 import NumCat
import DataSetTrainV2

loc = r'C:\OneDrive - Building Blocks\Thesis\Data\OpenML\TrainingPhase'
loc = r'C:\Users\Casper\Dropbox\MSc\BA&QM\Thesis\Data\OpenML\TrainingPhase'
locsmall = r'C:\OneDrive - Building Blocks\Thesis\Data\OpenML\TooSmallLess500'

csvfiles = glob.glob(os.path.join(loc, '*.csv'))[150:]
5/0

Num_names = [method for method in dir(Numerical()) if callable(getattr(Numerical(), method)) if not method.startswith('_')]
Num_dict = {key: pd.DataFrame() for key in Num_names}
Cat_names = [method for method in dir(Categorical()) if callable(getattr(Categorical(), method)) if not method.startswith('_')]
Cat_dict = {key: pd.DataFrame() for key in Cat_names}
NumNum_names = [method for method in dir(NumNum()) if callable(getattr(NumNum(), method)) if not method.startswith('_')]
NumNum_dict = {key: pd.DataFrame() for key in NumNum_names}
NumCat_names = [method for method in dir(NumCat()) if callable(getattr(NumCat(), method)) if not method.startswith('_')]
NumCat_dict = {key: pd.DataFrame() for key in NumCat_names}

dicts = {'Num': Num_dict, 'Cat': Cat_dict, 'NumNum': NumNum_dict, 'NumCat': NumCat_dict}

#dicts = np.load('meta_data.npy').item()

#data = csvfiles[9]
#df = pd.read_csv(data,index_col='Unnamed: 0')
#start = time()
#dicts = DataSetTrain.Train(df,dicts)
#print(time() - start)

for data in csvfiles:
    file = open("results_lap4.txt","a")
    #file = open("results100.txt","a")
    print(data)
    start = time()
    df = pd.read_csv(data,index_col='Unnamed: 0')
    dicts = DataSetTrainV2.Train(df,dicts)
    #np.save('meta_data_100.npy', dicts)
    np.save('meta_data_lap4.npy', dicts)
    print(time() - start)
    file.write(data[70:]+" "+str(time() - start)+"\n")
    file.close()
