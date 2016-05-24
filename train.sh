#!/bin/bash

CAFFEDIR=~/caffe
MODEL=train.prototxt
SOLVER=solver.prototxt

$CAFFEDIR/build/tools/caffe train -model $MODEL -solver $SOLVER
