### Reference
#If you use this code, please consider citing the following paper:
#
#@misc{atashgahi2020quick,
#      title={Quick and Robust Feature Selection: the Strength of Energy-efficient Sparse Training for Autoencoders}, 
#      author={Zahra Atashgahi and Ghada Sokar and Tim van der Lee and Elena Mocanu and Decebal Constantin Mocanu and Raymond Veldhuis and Mykola Pechenizkiy},
#      year={2020},
#      eprint={2012.00560},
#      archivePrefix={arXiv},
#      primaryClass={cs.LG}
#}

### Acknowledgements
#Starting of the code is "sparse-evolutionary-artificial-neural-networks" which is available at:
#https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks
#@article{Mocanu2018SET, 
#        author = {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio}, 
#        journal = {Nature Communications}, 
#        title = {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science}, 
#        year = {2018}, doi = {10.1038/s41467-018-04316-3}, 
#        url = {https://www.nature.com/articles/s41467-018-04316-3 }}
#

import os
import numpy as np
import sys; sys.path.append(os.getcwd())  
from Sparse_DAE import Sparse_DAE 
from other_classes import Sigmoid, MSE, tanh, Relu, LeakyRelu, Linear 
from utils import load_data, check_path
import datetime
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser(description='Train Sparse DAE')
parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
parser.add_argument("--epoch", help="epoch", type=int)
args = parser.parse_args()
dataset_name = args.dataset_name



strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
noise_factor = 0.2
weightDecay = 0.00001
noHiddenNeuronsLayer=1000
batchSize=100
dropoutRate=0.2
learningRate=0.01
momentum=0.9
epsilon = 13
zeta = 0.2

for i in range(5):
    filename = "./results/"+dataset_name+"/"+str(strtime)+"_epsilon_"+str(epsilon)+"_zeta_"+str(zeta)+"/run "+str(i)+"/"
    check_path(filename)
    print("*******************************************************************************")
    print("Dataset = ", dataset_name)
    print("epsilon = ", epsilon)
    print("zeta = ", zeta)
    X_train, Y_train, X_test, Y_test = load_data(dataset_name)
    noTrainingSamples = X_train.shape[0]

 
    start = time.time()
    if dataset_name == "madelon": 
        set_dae = Sparse_DAE((X_train.shape[1], noHiddenNeuronsLayer, X_train.shape[1]), (Sigmoid, tanh), epsilon=epsilon)
    else:
        set_dae = Sparse_DAE((X_train.shape[1], noHiddenNeuronsLayer, X_train.shape[1]), (Sigmoid, Linear), epsilon=epsilon)           
 
    set_dae.fit(X_train, Y_train.ravel(), X_test, Y_test.ravel(), loss=MSE, 
                epochs=args.epoch, batch_size=batchSize, learning_rate=learningRate,
                momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, 
                testing=False, save_filename=filename, noise_factor = noise_factor)
    end = time.time()
    print("running time QS_",args.epoch,"= ", end - start)





































