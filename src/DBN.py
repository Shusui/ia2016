# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from RBM import RBM
from utils import *


class DBN(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 rng=None):
        
        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if rng is None:
            rng = numpy.random.RandomState(1234)

        
        assert self.n_layers > 0


        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        rng=rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)


            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()



    def pretrain(self, lr=0.1, k=1, epochs=100, verbose=False):
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            
            for epoch in xrange(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)

                if verbose:
                    cost = rbm.get_reconstruction_cross_entropy()
                    print >> sys.stderr, 'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost


    def finetune(self, lr=0.1, epochs=100, verbose=False):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)

            if verbose:
                self.finetune_cost = self.log_layer.negative_log_likelihood()
                print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost
            
            lr *= 0.95
            epoch += 1


    def predict(self, x):
        layer_input = x
        
        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out
