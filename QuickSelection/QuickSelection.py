import numpy as np
import os; import sys; sys.path.append(os.getcwd())
from utils import load_data, check_path
import datetime
import time
from fs_utils import feature_selection
import sklearn
import scipy
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score 




def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)

from sklearn.cluster import KMeans
def evaluation(X_selected, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results
    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels
    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy
    """
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)

    k_means.fit(X_selected)
    y_predict = k_means.labels_



    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return acc
    


from sklearn.ensemble import ExtraTreesClassifier
def eval_subset(train, test):
    n_clusters = len(np.unique(train[2]))
    
    clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1)
    clf.fit(train[0], train[2])
    DTacc = float(clf.score(test[0], test[2]))

    max_iters = 10
    cacc = 0.0
    for iter in range(max_iters):
        acc = evaluation(train[0], n_clusters = n_clusters, y = train[2])
        cacc += acc / max_iters
    return  DTacc,float(cacc)





import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='QuickSelection')

parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
args = parser.parse_args()


import time
dataset_name = args.dataset_name
methods = ['Node Strength(in)']
path = "./results/"+dataset_name+"/" 
#print("Data = ", dataset_name)
X_train, Y_train, X_test, Y_test = load_data(dataset_name)
print("Performing feature selection on ", dataset_name)
print("Dataset shape:")
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

k =50
if dataset_name == "madelon":
    k = 20

for file in os.listdir(path):
   
    cnt = 0 
    ls_acc = []
    ls_acc2 = []
    print("------------------------------------------------------------")
    print("Quick_Selection results:")
    for run in os.listdir(path+file+"/"):
        print("Run %d:" % cnt, end = "")
        cnt += 1
        w1_10  = scipy.sparse.load_npz(path+file+"/"+run+"/w1.npz")
        w2_10  = scipy.sparse.load_npz(path+file+"/"+run+"/w2.npz")
        for j in range(len(methods)):
            indices  = feature_selection(w1_10, w2_10, k, methods[j])
            DTacc, cacc = eval_subset([X_train[:,indices], X_train, Y_train.ravel()],
                            [X_test[:, indices], X_test,  Y_test.ravel()])
            print("ETacc=%.4f | cacc=%.4f |" % ( DTacc,cacc))
            ls_acc.append(cacc)
            ls_acc2.append(DTacc)
    ls_acc = np.array(ls_acc)
    mean = np.mean(ls_acc)
    std = np.std(ls_acc)
    print("cacc | mean=%.4f |  std=%.4f " % (mean, std))
    ls_acc2 = np.array(ls_acc2)
    mean = np.mean(ls_acc2)
    std = np.std(ls_acc2)
    print("Etacc | mean=%.4f |  std=%.4f"   % (mean, std))

           

