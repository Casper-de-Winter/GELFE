# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:24:32 2019

@author: casper
"""

#!pip install openml
import openml
import pandas as pd
import os
import kNN_Impute

openml_list = openml.datasets.list_datasets()
datalist = pd.DataFrame.from_dict(openml_list, orient='index')
a=datalist.query('1 < NumberOfClasses < 11')
a=a.query('-1 < NumberOfMissingValues < NumberOfInstances')
a=a.query('5 < NumberOfFeatures < 502') #plus one because it counts the target as feature.
a=a.query('299 < NumberOfInstances')
a=a.query('0.8749 < MajorityClassSize / NumberOfInstances < 0.9')
IDs = a['did'].tolist()

path = "C:\\OneDrive - Building Blocks\\Thesis\\Data\\OpenML\\Missing\\"

#40589 / 40592 / 40594 / 40595 / 40597 / 41228 wrong 1220 extra
for i in IDs:
    if a.loc[i]['format'] != 'Sparse_ARFF':#and i > 41228:
        data = openml.datasets.get_dataset(i)
        X, y, cat_indicator, attribute_names = data.get_data(target=data.default_target_attribute,return_attribute_names=True, return_categorical_indicator=True)
        df = pd.DataFrame(X, columns=[m+"_"+str(n) for m,n in zip(attribute_names,cat_indicator)])
        df.insert(loc=0, column='y', value=y)
        if a.loc[i]['NumberOfMissingValues'] > 0:
            df = kNN_Impute.kNNImpute(df,neighbors = 5) #fill in missing values
            #kNNImpute(df,a.loc[i],neighbors = 5)
        name = a.loc[i]['name']
        df.to_csv(os.path.join(path,name + r'.csv'))
        print(i)
