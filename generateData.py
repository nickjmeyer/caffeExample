import h5py
import numpy as np
import numpy.linalg

np.random.seed(0)


## define dimensions
n = 10000 ## observations
p = 20 ## variables

m1 = 20 ## neurons in first layer


## param for first fully-connected layer
w1 = np.random.normal(scale=0.1,size=(m1,p))
b1 = np.random.normal(scale=0.1,size=(m1,))

## param for second fully-connected layer
w2 = np.random.normal(scale=0.1,size=(1,m1))
b2 = np.random.normal(scale=0.1)


## generate data
X= np.random.normal(size=(n,p))
eps = np.random.normal(size=(n,))
Y = np.zeros((n,))

## define sigmoid
def sigmoid(x):
    return 1.0 - 1.0/(1.0 + np.exp(x))

## fill Y values
for i in range(n):
    Y[i] = w2.dot(sigmoid(w1.dot(X[i,:].flatten())+b1))+b2 + eps[i]


## add data to hdf5 file
with h5py.File("train.h5","w") as f:
    ## data
    f.create_dataset("X",(n,p), dtype=np.float)
    f.create_dataset("Y",(n,),dtype=np.float)

    ## parameters
    f.create_dataset("w1",(m1,p),dtype=np.float)
    f.create_dataset("b1",(m1,),dtype=np.float)
    f.create_dataset("w2",(1,m1),dtype=np.float)
    f.create_dataset("b2",(1,),dtype=np.float)

    for i in range(n):
        f["X"][i,:] = X[i,:]
        f["Y"][i] = Y[i]


## create text file of database names
## needed for prototxt data layer
with open("data_list.txt","w") as f:
    f.write("train.h5\n")
