# Compare Caffe to hand calculations

The purpose of these scripts are to understand the mechanics of Caffe.
First, generate data so the true model is known.  Then, using Caffe,
estimate weights for a correctly specified model.  To make sure the
calculations of Caffe are understood, extract the estimated weights
and calculated the estimate by hand.

The three scripts are [`generateData.py`](./generateData),
[`train.sh`](./train.sh), and [`evaluateModel.py`](./evaluateModel.py).


1. Generate data from a simple 3 layer neural network
   - fully connected
   - sigmoid
   - fully connected
2. Save data to an HDF5 database
3. Train model using Caffe
4. Use Caffe to calculate estimated response
5. Extract estimated weights from Caffe model file
6. Using estimated weights, calcuate estimated response by hand
7. Compare estimate from Caffe and from manual calculation
