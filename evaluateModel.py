import sys
sys.path.append("/home/meyer_nicholas/caffe/python")

import caffe
import numpy as np
import h5py

caffe.set_mode_cpu()

## load trained network
## use deploy so batch_size is the entire data set
net = caffe.Net("./deploy.prototxt",
                "./sed_iter_10000.caffemodel",
                caffe.TEST)

## pull out data
with h5py.File("train.h5","r") as f:
    X = np.asarray(f["X"])
    Y = np.asarray(f["Y"]).flatten()


## calculate Yhat from caffe
YhatCaffe = net.forward(end="ip2")["ip2"].flatten()


## access estimated parameters
w1hat = net.params["ip1"][0].data
b1hat = net.params["ip1"][1].data
w2hat = net.params["ip2"][0].data
b2hat = net.params["ip2"][1].data


## define sigmoid
def sigmoid(x):
    return 1.0 - 1.0/(1.0 + np.exp(x))


## calculate Yhat by hand from estimated parameters
n = Y.shape[0]
YhatByHand = np.zeros((n,))
for i in range(n):
    YhatByHand[i] = w2hat.dot(sigmoid(w1hat.dot(X[i,:].flatten())+b1hat))+b2hat


## euclidean loss for Yhat calculated by hand and yHat calculatd
## forward in caffe
print sum((Y-YhatByHand)**2)/(2.0*float(n))
print sum((Y-YhatCaffe)**2)/(2.0*float(n))


## to make sure by hand calculation matches caffe
## squared l2 distance between two solutions
print sum((YhatByHand -YhatCaffe)**2)
