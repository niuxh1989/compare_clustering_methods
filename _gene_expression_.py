# -*- coding: utf-8 -*-
"""Copy of article_1_gene_expression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MF1IjTB2Q3PdSL3RwyEguJcWcG1J_p-P
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn_som.som import SOM

def remove_nan_improved(arr_, axis = None):
    '''for solving error: ufunc 'isnan' not supported for the input types'''
    if len(arr_.shape) != 2:
        raise ValueError("arg shape must be (a, b)")
    h, w = arr_.shape
    if axis == 0: # along rows
        h_c = 0
        while h_c <= h-1:
            if np.any(np.isnan(arr_[h_c,:].astype(float))) ==True:
                arr_ = np.delete(arr_, h_c , axis = 0)
                h_c = h_c
                h = h-1
            else:
                h_c = h_c + 1
    elif axis == 1: # along columns
        w_c = 0
        while w_c <= w-1:
            if np.any(np.isnan(arr_[:,w_c].astype(float))) ==True:
                arr_ = np.delete(arr_, w_c  , axis = 1)
                w_c = w_c
                w = w -1
            else:
                w_c = w_c + 1
    return arr_

def record_index_nan(arr_, axis = None):
    ''' recoord index of row or column containing nan'''
    if len(arr_.shape) != 2:
        raise ValueError("arg shape must be (a, b)")
    h, w = arr_.shape
    if axis == 0: # along rows
        index =[]
        h_c = 0
        while h_c <= h-1:
            if np.any(np.isnan(arr_[h_c,:].astype(float))) ==True:
                index.append(h_c)
                h_c = h_c + 1
            else:
                h_c = h_c + 1
    elif axis == 1: # along columns
        index =[]
        w_c = 0
        while w_c <= w-1:
            if np.any(np.isnan(arr_[:,w_c].astype(float))) ==True:
                index.append(w_c)
                w_c = w_c + 1
            else:
                w_c = w_c + 1

    return np.array(index)

a = pd.read_csv(r'/content/Breast_GSE26304.csv').to_numpy()
a = a[0:50,:]
# # check type
# t_R = []
# for i in a[0]:
#     t_R.append(type(i))
# set(t_R)
ta_ = a[:,1]
a_1 = np.delete(a, [0,1], axis = 1)
inde_n = record_index_nan(a_1, axis =0)
a_1 = remove_nan_improved(a_1, axis =0 )
a_1 = a_1.astype(float)
da_ = a_1
ta_ = np.delete(ta_, list(inde_n), axis = None)
# change target
fir_ = sorted(list(set(ta_)), reverse = True)
sec_ = [i for i in range(0, len(fir_))]
ex_ = dict(zip(np.array(fir_), np.array(sec_ )))
for ind in range(0,len(ta_)):
    ta_[ind] = ex_[ta_[ind]]

from sklearn.cluster import KMeans
          1,2,3,4,|5|,6,7,8,9,10,11,|12|,|13|,14,15,16,17

          1,2,3,4,|5|,6,7,8,9                                                                    2,3,4,|5|,6,7,8,9,10,                                                    3,4,|5|,6,7,8,9,10,11                                                    4,|5|,6,7,8,9,10,11,|12|
1: can't choose
          1,2,3,4,  2,3,4,|5|   3,4,|5|,6   4,|5|,6,7   |5|,6,7,8    6,7,8,9


#         pieces_of_RNA_1   pieces_of_RNA_2...........
# k-means
# som

# k-means_objective record
# som_objective record

class identification_:
    def __init__(self, n, data, target, n_clusters):
        self.n = n
        self.data = data
        self.target = target
        self.n_clusters = n_clusters
        self.column_indexes = []
        self.RNA_pieces = []
        self.objective_k_means = []
        self.objective_som_ = []
        self.partition_ratio = [0.5, 0.4, 0.3]
    def check(self, arg_1, arg_2):
        '''check order of target, set self.order
          arg_1:target
          arg_2:labels
          order: 'dir_1' (no change)
            'dir_2' (des order)
            'dir_3' (inc order)'''
        if (np.mean(np.where(arg_1==np.max(arg_1))) - np.mean(np.where(arg_1==np.min(arg_1))))*(np.mean(np.where(arg_2==np.max(arg_2))) - np.mean(np.where(arg_2==np.min(arg_2)))) >= 0:
            self.order = 'dir_1'
        elif (np.mean(np.where(arg_1==np.max(arg_1))) - np.mean(np.where(arg_1==np.min(arg_1))))*(np.mean(np.where(arg_2==np.max(arg_2))) - np.mean(np.where(arg_2==np.min(arg_2)))) < 0:
            if np.mean(np.where(arg_1==np.max(arg_1))) - np.mean(np.where(arg_1==np.min(arg_1))) < 0:
                self.order = 'dir_2'
            elif np.mean(np.where(arg_1==np.max(arg_1))) - np.mean(np.where(arg_1==np.min(arg_1))) > 0:
                self.order = 'dir_3'

    def partition_data(self, data, n, min_num = 10):
        if n < 10:
          raise ValueError("n >= 10")
        index = 0
        while index+n <= len(data[0])-1:
              self.column_indexes.append([i for i in range(index,index+n)])
              self.RNA_pieces.append(data[:,index:index+n])
              index = index + 1

    @staticmethod
    def run_(ins):
        data = ins.data
        for i in ins.partition_ratio:
            n_ = i*len(data[0])
            ins.partition_data(data, n_)
            for da_ in ins.column_indexes:
                target_ = ins.target.copy()
                # k-means
                __labels_ = ins.kmeans_clustering(ins.n_clusters, da_ )
                ins.check( target_ ,__labels_  )
                __labels_= ins.change_labels(__labels_)
                ins.objective_k_means.append(ins.similarity_labels_target(__labels_, target_))
                # som

                # other
        ins.objective_k_means_compare_ = np.array(ins.objective_k_means)
        index_selected = np.where(ins.objective_k_means_compare_==np.min(ins.objective_k_means_compare_))
        ins.column_indexes[index_selected]
        ins.RNA_pieces[index_selected]


    def kmeans_clustering(self, num_of_clusters, data):
        ''' parameters:
        -------------------
        num_of_clusters: int
        data: array, shape (a, b)

        return
        -------------------
        labels for data: array'''
        kmeans = KMeans(n_clusters= self.n_clusters, random_state=0,init = 'k-means++').fit(data)
        labels = kmeans.labels_
        return labels

    # def other_clustering(num_of_clusters, data):
    #     return labels

    def similarity_labels_target(self, labels_, target):
        return np.linalg.norm(labels_ - target)
        # return dictionary_from_


    # def minimum_similarity_difference()
    #     return  pieces_of_RNA

    #########################################
    def change_labels(self, arg):  # for supervised training
        '''
        parameters:
        ------------------
        arg: array of target
        order: 'dir_1' (no change)
               'dir_2' (des order)
               'dir_3' (inc order)
        return:
        ------------------
        array with changed (or unchanged ) order of target
        '''
        order = self.order
        if order == 'dir_1': # no change
            return arg
        elif order == 'dir_2': # des
            fir_ = sorted(list(set(arg)), reverse=False)
            sec_ = sorted([i for i in range(0, len(fir_))] , reverse=True)
            ex_ = dict(zip(np.array(fir_), np.array(sec_ )))
            for ind in range(0,len(arg)):
                arg[ind] = ex_[arg[ind]]
            return arg
        elif order == 'dir_3': # inc
            fir_ = sorted(list(set(arg)), reverse=True)
            sec_ = sorted([i for i in range(0, len(fir_))] , reverse=False)
            ex_ = dict(zip(np.array(fir_), np.array(sec_ )))
            for ind in range(0,len(arg)):
                arg[ind] = ex_[arg[ind]]
            return arg

clustering = identification_(10, da_, ta_, 2)
identification_.run_(clustering)