# -*- coding: utf-8 -*-
# Copyright: (c) 2018, https://github.com/olivierRoche/neural_network_polynomial_layers-
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Created on Fri Jul 13 13:35:26 2018

@author: Olivier Roche
"""

import numpy as np

import AlgPoly as ap

import random

def norme(x):
    return np.sqrt(sum(x**2))

def Id(n):
      #the Identity matrix of R^n
      return np.array([[1*(i==j) for j in range(n)]for i in range (n)])

class ComponentwiseC1Function:
    def __init__(self,main,derivative):
        self.main=main
        self.derivative = derivative
        self.differential = lambda array : Id(len(array))*derivative(array)
        
    
#*********Activation functions*****************
    
sigm=lambda x:1/(1+np.exp(-x))
sigmP=lambda x:sigm(x)*(1-sigm(x))
sigmoid=ComponentwiseC1Function(sigm,sigmP)

relu=ComponentwiseC1Function(lambda x:x*(x>0),lambda x:1*(x>0))

identity=ComponentwiseC1Function(lambda x:x,lambda x:1)

step=ComponentwiseC1Function(lambda x:(-1.0*(x<-1)+x*(np.abs(x)<=1)+(x>1)), lambda x:(0.0*(x<1)+1*(np.abs(x)<=1)+0.0*(x>1)))

#*******************************************
""" We use the square of norm cost function : """
def cost(a,y):
    return 0.5 * sum((a-y)*(a-y))

def nabla_cost(a,y):
    return a-y

class PolyNN:
    def __init__(self,layersSizes,thresholdFunction,degrees,damping = None):
        self.numberLayers = len(layersSizes)
        self.layersSizes = layersSizes
        self.degrees = degrees
        self.functions = [ap.initialize_polyfunc_rand(inp,outp,deg)
            for inp,outp,deg in zip(layersSizes[:-1],layersSizes[1:],degrees)]
        self.thresh=thresholdFunction
        if damping is not None:
            self.damping = 10**damping
        else:
            self.damping = None
    def feedForward(self,x):
        activation = x
        for t in self.functions:
            z = ap.evaluate_poly_func(t,activation)
            activation=self.thresh.main(z)
        return activation

    def backPropagation(self,x,y):
        # ******forward pass*****
        activations=[]
        zValues=[]
        activation=x
        activations.append(activation)
        for t in self.functions:
            z = ap.evaluate_poly_func(t,activation)
            zValues.append(z)
            activation = self.thresh.main(z)
            activations.append(activation)
        #*********initialisation for backward pass***********************
        #delta_P will be our return value. It is a convenient way
        #to store the partial derivatives of cost with respect to the
        #different parameters of self.functions .
        delta_P = [ap.initialize_polyfunc_zero(inp,outp,deg) for inp,outp,deg  
                   in zip(self.layersSizes[:-1],self.layersSizes[1:],self.degrees)]
        #damping is a trick to avoid explosion of our return value
        if self.damping is not None:
            nabla_Cost_wrt_a = nabla_cost(activation,y)/(self.damping*norme(x))
        else:
            nabla_Cost_wrt_a = nabla_cost(activation,y)
        
        #*****backward passes*****
        for l in range(self.numberLayers-2,-1,-1):
            for i in range(self.layersSizes[l+1]):
                delta_P[l][i].coeff = {tuple : self.thresh.derivative(zValues[l][i]) 
                       * ap.monom(tuple,activations[l]) 
                       * nabla_Cost_wrt_a[i] 
                       for tuple in ap.iterTuple(nb_var=self.layersSizes[l],
                                             degree=self.degrees[l])
                                        }
            nabla_Cost_wrt_a = np.dot(np.dot(self.thresh.differential(zValues[l]),\
                ap.differential(self.functions[l],activations[l])).transpose(),
                nabla_Cost_wrt_a)
        return delta_P
    
    def steplearning(self,x,y,eta,normalize = False):
        """ modifies self.functions using self.backPropagation(x,y)
        eta is the change rate 
        if normalize is enabled, the norm of the change is bounded by eta,
        this is achieved by normalizing the vector delta_P if its norm is
        greater than 1"""     
        delta_P = self.backPropagation(x,y)
        for l in range(self.numberLayers-1):
            if normalize:
                w = norm_polyfunc(delta_P[l])
                if w>1:
                    learning_rate = eta / w
                else:
                    learning_rate = eta
            else:
                learning_rate = eta
            self.functions[l] -= learning_rate* delta_P[l]           
            
    def learn_minibatch(self, minibatch, eta, normalize = False):
        delta_P = [ap.initialize_polyfunc_zero(inp,outp,deg) for inp,outp,deg  
                   in zip(self.layersSizes[:-1],self.layersSizes[1:],self.degrees)]      
        for (x,y) in minibatch:
            change = self.backPropagation(x,y)
            for l in range(len(delta_P)):
                if normalize:
                    w = norm_polyfunc(change[l])
                    if w>1:
                        learning_rate = eta / w
                    else:
                        learning_rate = eta
                else:
                    learning_rate = eta
                delta_P[l] += learning_rate / len(minibatch) * change[l]
        for l in range(len(delta_P)):
            self.functions[l] -= delta_P[l]
            
    def batchlearning(self, training_data, batch_size, nb_epochs, eta, normalize = False):
        for epoch in range(nb_epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in 
                       range(0,len(training_data),batch_size)]
            for batch in batches:
                self.learn_minibatch(batch, eta, normalize)
            print("epoch {0}/{1} complete\n".format(epoch + 1,nb_epochs))

def norm_polynomial(polynomial):
    vec_coeff = np.array([c for c in polynomial.coeff.values()])
    return norme(vec_coeff)

def norm_polyfunc(P):
    return max([norm_polynomial(p) for p in P])