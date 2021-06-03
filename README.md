## Quick and Robust Feature Selection: the Strength of Energy-efficient Sparse Training for Autoencoders
This repository contains code for the paper, Quick and Robust Feature Selection: the Strength of Energy-efficient Sparse Training for Autoencoders by Zahra Atashgahi, Ghada Sokar, Tim van der Lee, Elena Mocanu, Decebal Constantin Mocanu, Raymond Veldhuis, and Mykola Pechenizkiy. 
For more information please read the paper at https://arxiv.org/abs/2012.00560. 

### Prerequisites
We run this code on Python 3. Following Python packages have to be installed before executing the project code:
* numpy
* scipy
* sklearn
* Cython (optional - To use the fast implementation)


### Usage
To run the code you can use the following lines:
1. select dataset and the number of training epochs: 
    ```sh
    dataset="madelon"
    epoch=100
    ```
2. Train sparse-DAE:
    ```sh
    python3 ./QuickSelection/train_sparse_DAE.py --dataset_name $dataset --epoch $epoch
    ```
3. Use the trained model weights to select features:
    ```sh
    python3 ./QuickSelection/QuickSelection.py --dataset_name $dataset
    ```
There are two implementations for back-propagation in ```Sparse_DAE.py```. 
If you are running this code on Linux and you want to exploit fast implementation, you can use Cython to run it. You need to first install ```sparseoperation```. Use the following line to install it on your environment:```cythonize -a -i ./QuickSelection/sparseoperations.pyx```.
But if you are on Windows, please change the back-propagation method in the ```Sparse_DAE.py``` file. Please note that the running time will be much higher. More details can be found there.

### Results on MNIST
On the MNIST dataset, first, we train the sparse-denoising-autoencoder (sparse-DAE). Then, we select the 50 most important features using the strength of the input neurons of the trained sparse-DAE. We visualize the features selected for each class separately. In Figure below, each picture at different epochs is the average of the 50 selected features of all the samples of each class along with the average of the actual samples of the corresponding class. As we can see, during training, these features become more similar to the pattern of digits of each class. Thus, QuickSelection is able to find the most relevant features for all classes.

![mnist](https://github.com/zahraatashgahi/QuickSelection/blob/main/mnist.JPG)

### Reference
If you use this code, please consider citing the following paper:
```
@misc{atashgahi2020quick,
      title={Quick and Robust Feature Selection: the Strength of Energy-efficient Sparse Training for Autoencoders}, 
      author={Zahra Atashgahi and Ghada Sokar and Tim van der Lee and Elena Mocanu and Decebal Constantin Mocanu and Raymond Veldhuis and Mykola Pechenizkiy},
      year={2020},
      eprint={2012.00560},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
### Acknowledgements
Starting of the code is "sparse-evolutionary-artificial-neural-networks" which is available at:
https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks
```
@article{Mocanu2018SET, 
        author = {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio}, 
        journal = {Nature Communications}, 
        title = {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science}, 
        year = {2018}, doi = {10.1038/s41467-018-04316-3}, 
        url = {https://www.nature.com/articles/s41467-018-04316-3 }}
```

### Contact
email: z.atashgahi@utwente.nl
