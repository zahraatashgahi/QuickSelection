# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:10:32 2019

@author: 20194461
"""
"""*************************************************************************"""
"""                           IMPORT LIBRARIES                              """
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from utils import check_path
import os
import errno
from fs_utils import get_degree
import seaborn as sns
from fs_utils import  fs_strength_w
from sklearn.utils import shuffle
from matplotlib import colors

 
def plt_acc(acc, path, type):
    plt.figure()
    dots = []
    colors = ['g', 'blue', 'salmon', 'k', 'r', 'purple', 'yellow', 'c', 'm', 'sienna']
    for i in range(1):
        dots.append([])
        dots[i], = plt.plot(acc , linestyle='--', marker='o', color=colors[i]) 
    plt.xlabel('# of epoch')
    plt.ylabel('Accuracy (%)')
    plt.title("Classification Accuracy") 
    plt.savefig(path+"accuracy.png")


def plt_loss(loss_train, loss_test, path):    
    plt.figure()   
    dot0, = plt.plot(loss_train , linestyle=':', marker='d', color='g')
    dot1, = plt.plot(loss_test , linestyle=':', marker='d', color='b')    
    plt.legend([dot0, dot1],  ["Train", "Test"])
    plt.xlabel('# of epoch')
    plt.ylabel('MSE loss')
    plt.title( "Reconstruction error") 
    plt.savefig(path + "/reconstruction_error.png")
    #plt.show()   
    

