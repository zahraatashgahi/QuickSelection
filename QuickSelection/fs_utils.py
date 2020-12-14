# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:31:37 2019

@author: 20194461
"""
#import math
import numpy as np

import random
from statistics import stdev
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy import sparse
import networkx as nx
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def feature_selection(W1, W2, k, method):
    if method == "Node Strength(in)": 
        indices =  fs_strength_w(sparse.csr_matrix.transpose(W1), k)
    elif method == "Node Strength(out)": 
        indices =  fs_strength_w(W2, k)
    elif method == "Node Degree(in)":
        indices =  fs_degree_w(sparse.csr_matrix.transpose(W1), k)
    elif method == "Node Degree(out)":
        indices =  fs_degree_w(W2, k)
    elif method == "Degree(in_out)":
        indices = fs_degree12(sparse.csr_matrix.transpose(W1), W2, k)
    elif method == "Degree(out_in)":
        indices = fs_degree12(W2, sparse.csr_matrix.transpose(W1), k)
    elif method == "Node Strength(in_out)":
        indices = fs_strength12(sparse.csr_matrix.transpose(W1), W2, k)
    elif method == "Node Strength(out_in)":
        indices = fs_strength12(W2, sparse.csr_matrix.transpose(W1), k)

    return indices 

def get_degree(W):
    weights = W.tolil()
    num_ws = np.zeros(W.shape[1]) 
    for i in range(W.shape[1]):    
        num_ws[i] = len(np.sum(weights[:,i].data))   
    return num_ws 
    
def fs_strength_w(W, k):
    weights = W.tolil()    
    abs_w = np.absolute(weights)
    sum_abs_w = np.sum(abs_w, axis = 0)
    new_sum_w = np.zeros(W.shape[1])
    #print(W.shape)
    new_sum_w = np.zeros(W.shape[1])
    for i in range(W.shape[1]):
        new_sum_w[i] = sum_abs_w[0,i]              
    indices = new_sum_w.argsort()[-k:][::-1]
    return indices

def fs_degree_w(W, k):
    num_ws = np.zeros(W.shape[1]) 
    for i in range(W.shape[1]):    
        num_ws[i] = len(W[:,i].data)
    fs_indices = num_ws.argsort()[-k:][::-1]  
    return fs_indices 
    
def fs_degree12(W1, W2, k):
    weights = W1.tolil()
    num_ws1 = np.zeros(W1.shape[1]) 
    for i in range(W1.shape[1]):    
        num_ws1[i] = len(np.sum(weights[:,i].data))
        
    weights = W2.tolil()
    num_ws2 = np.zeros(W2.shape[1]) 
    for i in range(W2.shape[1]):    
        num_ws2[i] = len(np.sum(weights[:,i].data))
        
    num_ws = num_ws1 + num_ws2
    c = int(num_ws.shape[0]*2/3)
    indices = num_ws.argsort()[-c:][::-1]
    
    weights = W1.tolil()
    num_ws = np.zeros(W1.shape[1]) 
    for i in range(W1.shape[1]):    
        num_ws[i] = len(np.sum(weights[indices,i].data))
    fs_indices = num_ws.argsort()[-k:][::-1]  
    return fs_indices
    
    
def fs_strength12(W1, W2, k):
    
    weights = W1.tolil()
    abs_w = np.absolute(weights)
    sum_abs_w = np.sum(abs_w, axis = 1)
    sum_abs_w = np.transpose(sum_abs_w)
    new_sum_w1 = np.zeros(W1.shape[0])
    for i in range(W1.shape[0]):
        new_sum_w1[i] = sum_abs_w[0,i]  

    weights = W2.tolil()    
    abs_w = np.absolute(weights)
    sum_abs_w = np.sum(abs_w, axis = 1)
    sum_abs_w = np.transpose(sum_abs_w)
    new_sum_w2 = np.zeros(W2.shape[0])   
    for i in range(W2.shape[0]):
        new_sum_w2[i] = sum_abs_w[0,i]  
        
    new_sum_w = new_sum_w1 + new_sum_w1
    c = int(new_sum_w.shape[0]*0.5)
    indices1 = new_sum_w.argsort()[-c:][::-1]
    
    weights = W1.tolil()
    abs_w = np.absolute(weights)
    sum_abs_w = np.sum(abs_w[indices1,:], axis = 0)
    sum_abs_w = np.transpose(sum_abs_w)
    new_sum_w = np.zeros(W1.shape[1])

    for i in range(W1.shape[1]):
        new_sum_w[i] = sum_abs_w[i,0] 
    
    indices = new_sum_w.argsort()[-k:][::-1]    
      
    return indices    




    