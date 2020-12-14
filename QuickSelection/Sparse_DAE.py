
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
#the "sparseoperations" Cython library was tested in Ubuntu 16.04. Please note that you may encounter some "solvable" issues if you compile it in Windows.
import sparseoperations
import datetime
from other_classes import Sigmoid, tanh
import scipy
from scipy import sparse  
import time


def backpropagation_updates_Numpy(a, delta, rows, cols, out):
    for i in range(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def createSparseWeights(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float64(np.random.randn() / 10)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (noRows * noCols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights

def array_intersect(A, B):
    # this are for array intersection
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype))  # boolean return

class Sparse_DAE:
    def __init__(self, dimensions, activations, epsilon=20):
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.droprate = 0  # dropout rate
        self.dimensions = dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            if (i<len(dimensions) - 2):
                self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i],
                                                dimensions[i + 1])  # create sparse weight matrices
            else:
                self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i],
                                                dimensions[i + 1])  # create sparse weight matrices

            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]
        
            
    def _feed_forward(self, x, drop=False):
        # w(x) + b
        z = {}
        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            if (drop == False):
                if (i > 1):
                    z[i + 1] = z[i + 1] * (1 - self.droprate)
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if (drop):
                if (i < self.n_layers - 1):
                    dropMask = np.random.rand(a[i + 1].shape[0], a[i + 1].shape[1])
                    dropMask[dropMask >= self.droprate] = 1
                    dropMask[dropMask < self.droprate] = 0
                    a[i + 1] = dropMask * a[i + 1]

        return z, a

    def _back_prop(self, z, a, X_train): 
        delta = self.loss.delta(X_train, a[self.n_layers])
        dw = coo_matrix(self.w[self.n_layers - 1])

        # compute backpropagation updates
        sparseoperations.backpropagation_updates_Cython(a[self.n_layers - 1],delta,dw.row,dw.col,dw.data)# If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        #backpropagation_updates_Numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)
        
        update_params = {
            self.n_layers - 1: (dw.tocsr(), delta)
        }
        for i in reversed(range(2, self.n_layers)):         
            delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
            dw = coo_matrix(self.w[i - 1])

            # compute backpropagation updates
            sparseoperations.backpropagation_updates_Cython(a[i - 1], delta, dw.row, dw.col, dw.data)# If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            #backpropagation_updates_Numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(), delta)
            for k, v in update_params.items():
                self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        # perform the update with momentum
        if (index not in self.pdw):
            self.pdw[index] = -self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * np.mean(delta, 0)
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * np.mean(delta, 0)

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.00001, zeta=0.2, dropoutrate=0, testing=True, save_filename="", noise_factor = 0.2):

        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
            
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.droprate = dropoutrate
        self.save_filename=save_filename        
        self.inputLayerConnections = []
        self.inputLayerConnections.append(self.getCoreInputConnections())
        np.savez_compressed(self.save_filename + "_input_connections.npz",
                        inputLayerConnections=self.inputLayerConnections)
        


        metrics = np.zeros((epochs, 2))
        mean = 0; stddev = 1
        noise = noise_factor * np.random.normal(mean, stddev, (x.shape[0], x.shape[1]))
        x_noisy = x + noise
        seed = np.arange(x.shape[0])
        
        for i in range(epochs):
            print("----------------------------------------------------")
            print("epoch ", i)

            
            # Shuffle the data
            np.random.shuffle(seed)
            x_noisy_ = x_noisy[seed]
            x_ = x[seed]
            y_ = y_true[seed]
            # training
            t1 = datetime.datetime.now()
            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_noisy_[k:l], True)
                self._back_prop(z, a, x_[k:l])
            t2 = datetime.datetime.now()       
            print("Training time: ", t2 - t1)
       
            if (testing):
                activations_hidden_test, activations_output_test  = self.predict(x_test)
                activations_hidden_train, activations_output_train = self.predict(x)
                loss_train = self.loss.loss(x, activations_output_train)
                metrics[i, 0] = loss_train
                loss_test = self.loss.loss(x_test, activations_output_test)
                metrics[i, 1] = loss_test
                print("Loss train: ", loss_train, "; Loss test: ", loss_test) 
                plt_loss(metrics[:i+1, 0], metrics[:i+1, 1], self.save_filename)                
          
            if (i < epochs - 1):  
                # do not change connectivity pattern after the last epoch
                # self.weightsEvolution_I() #this implementation is more didactic, but slow.
                w1 , w2= self.weightsEvolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
    


            if (self.save_filename != ""):
                np.savetxt(self.save_filename+"/metrics.txt", metrics)
        scipy.sparse.save_npz(self.save_filename+"w1", w1)
        scipy.sparse.save_npz(self.save_filename+"w2", w2)
        return metrics
    def getCoreInputConnections(self):
        values = np.sort(self.w[1].data)
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)

        largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
        smallestPositive = values[
            int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        wlil = self.w[1].tolil()
        wdok = dok_matrix((self.dimensions[0], self.dimensions[1]), dtype="float64")

        # remove the weights closest to zero
        keepConnections = 0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if ((val < largestNegative) or (val > smallestPositive)):
                    wdok[ik, jk] = val
                    keepConnections += 1
        return wdok.tocsr().getnnz(axis=1)

    def weightsEvolution_I(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers):

            values = np.sort(self.w[i].data)
            firstZeroPos = find_first_pos(values, 0)
            lastZeroPos = find_last_pos(values, 0)

            largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
            smallestPositive = values[
                int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float64")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float64")

            # remove the weights closest to zero
            keepConnections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if ((val < largestNegative) or (val > smallestPositive)):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keepConnections += 1

            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keepConnections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.randn() / 10
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weightsEvolution_II(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        #evolve all layers, except the one from the last hidden layer to the output layer
        for i in range(1, self.n_layers):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            #if(self.w[i].count_nonzero()/(self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8):
                t_ev_1 = datetime.datetime.now()
                # converting to COO form
                wcoo = self.w[i].tocoo()
                valsW = wcoo.data
                rowsW = wcoo.row
                colsW = wcoo.col

                pdcoo = self.pdw[i].tocoo()
                valsPD = pdcoo.data
                rowsPD = pdcoo.row
                colsPD = pdcoo.col
                # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
                values = np.sort(self.w[i].data)
                firstZeroPos = find_first_pos(values, 0)
                lastZeroPos = find_last_pos(values, 0)

                largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
                smallestPositive = values[
                    int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

                # remove the weights (W) closest to zero and modify PD as well
                valsWNew = valsW[(valsW > smallestPositive) | (valsW < largestNegative)]
                rowsWNew = rowsW[(valsW > smallestPositive) | (valsW < largestNegative)]
                colsWNew = colsW[(valsW > smallestPositive) | (valsW < largestNegative)]

                newWRowColIndex = np.stack((rowsWNew, colsWNew), axis=-1)
                oldPDRowColIndex = np.stack((rowsPD, colsPD), axis=-1)

                newPDRowColIndexFlag = array_intersect(oldPDRowColIndex, newWRowColIndex)  # careful about order

                valsPDNew = valsPD[newPDRowColIndexFlag]
                rowsPDNew = rowsPD[newPDRowColIndexFlag]
                colsPDNew = colsPD[newPDRowColIndexFlag]

                self.pdw[i] = coo_matrix((valsPDNew, (rowsPDNew, colsPDNew)),
                                         (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                if(i==1):
                    w1 = coo_matrix((valsWNew, (rowsWNew, colsWNew)),
                                       (self.dimensions[i - 1], self.dimensions[i])).tocsr()
                    self.inputLayerConnections.append(coo_matrix((valsWNew, (rowsWNew, colsWNew)),
                                       (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))
                    np.savez_compressed(self.save_filename + "_input_connections.npz",
                                        inputLayerConnections=self.inputLayerConnections)
                else:
                    w2 = coo_matrix((valsWNew, (rowsWNew, colsWNew)),
                                       (self.dimensions[i - 1], self.dimensions[i])).tocsr()


                # add new random connections
                keepConnections = np.size(rowsWNew)
                lengthRandom = valsW.shape[0] - keepConnections
                randomVals = np.random.randn(lengthRandom) / 10
                zeroVals = 0 * randomVals  # explicit zeros
                
                # adding  (wdok[ik,jk]!=0): condition
                while (lengthRandom > 0):
                    ik = np.random.randint(0, self.dimensions[i - 1], size=lengthRandom, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=lengthRandom, dtype='int32')

                    randomWRowColIndex = np.stack((ik, jk), axis=-1)
                    randomWRowColIndex = np.unique(randomWRowColIndex, axis=0)  # removing duplicates in new rows&cols
                    oldWRowColIndex = np.stack((rowsWNew, colsWNew), axis=-1)

                    uniqueFlag = ~array_intersect(randomWRowColIndex, oldWRowColIndex)  # careful about order & tilda

                    ikNew = randomWRowColIndex[uniqueFlag][:, 0]
                    jkNew = randomWRowColIndex[uniqueFlag][:, 1]
                    # be careful - row size and col size needs to be verified
                    rowsWNew = np.append(rowsWNew, ikNew)
                    colsWNew = np.append(colsWNew, jkNew)

                    lengthRandom = valsW.shape[0] - np.size(rowsWNew)  # this will constantly reduce lengthRandom

                # adding all the values along with corresponding row and column indices
                valsWNew = np.append(valsWNew, randomVals)
                # valsPDNew=np.append(valsPDNew, zeroVals)
                if (valsWNew.shape[0] != rowsWNew.shape[0]):
                    print("not good")
                self.w[i] = coo_matrix((valsWNew, (rowsWNew, colsWNew)),
                                       (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

                t_ev_2 = datetime.datetime.now()
                print("Weights evolution time for layer",i,"is", t_ev_2 - t_ev_1)
        return w1 , w2
    
    def predict(self, x):
        _, a = self._feed_forward(x)
        a_hidden = a[self.n_layers-1]
        a_output = a[self.n_layers]
        del a  
        return a_hidden, a_output
    
   

		
		
