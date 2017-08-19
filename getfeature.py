import pandas as pd
import numpy as np

import math

import matplotlib.pyplot as plt


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

# get useful format of the data
def Data_Pre_Process(filename):

    data = pd.read_csv(filename, sep='\t', header=0)
    columns = data.columns
    p = list(filter(lambda x: x.find('NEG') != -1, columns))   # p = ['POS','POS1',...]
    n = list(filter(lambda x: x.find('POS') != -1, columns))   # n = ['NEG','NEG1',...]
    Feature = data['mdr']                                      # Feature = ['1990_at','10002_at',...]
    data = data.set_index(['mdr'])

    # POS & NEG
    data = data.ix[:, p].join(data.ix[:, n])                    # same lable has been merged together
    p_data = data.ix[:, p].T                                      # label is p
    n_data = data.ix[:, n].T                                      # label is n

    return p,n,Feature,p_data,n_data,data

# for each row calculate the pvalue
def get_Pvalue(p_data,n_data):
    Pvalues = []
    for row in range(len(p_data)):
        pos = p_data.ix[row]
        neg = n_data.ix[row]
        tvalue, pvalue = stats.ttest_ind(pos,neg)
        Pvalues.append(pvalue)
    return Pvalues

# after a t_test, get a sorted_feature
def get_sorted_feature(Pvalues,Feature):
    PvalueDict = dict(zip(Pvalues, Feature))
    Pvalues.sort()
    sortFeature = map(PvalueDict.get, Pvalues)
    return sortFeature,Pvalues
def return_data():
    p,n,Feature,p_data,n_data,data = Data_Pre_Process('ALL3.txt')

    Pvalues = get_Pvalue(p_data.T,n_data.T)

    sortFeature,Pvalues = get_sorted_feature(Pvalues,Feature)
    sortFeature = list(sortFeature)
    feature_data = []
    for i in range(10):
        feature_data.append(list(data.loc[sortFeature[i]]))
    feature_data = np.array(feature_data)
    feature_data = feature_data.T
    print(feature_data.shape)
    classes = []
    for i in range(len(p)):
        classes.append(1)
    for i in range(len(n)):
        classes.append(0)
    #print(classes)
    return feature_data,classes
if __name__=='__main__':
    return_data()


