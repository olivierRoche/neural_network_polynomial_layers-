# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:30:45 2018

@author: Augustine
"""

import numpy as np

def norme(x):
    return np.sqrt(sum(x**2))

class C1Function:
    def __init__(self,main,derivative):
        self.main=main
        self.derivative=derivative
        
sigm=lambda x:1/(1+np.exp(-x))
sigmP=lambda x:sigm(x)*(1-sigm(x))
sigmoid=C1Function(sigm,sigmP)

relu=C1Function(lambda x:x*(x>0),lambda x:1*(x>0))

identity=C1Function(lambda x:x,lambda x:1)

step=C1Function(lambda x:(-1.0*(x<-1)+x*(np.abs(x)<=1)+(x>1)), lambda x:(0.0*(x<1)+1*(np.abs(x)<=1)+0.0*(x>1)))

def costDerivative(a,y):
    return a-y

class network:
    def __init__(self,layersSizes,thresholdFunction):
        self.numberLayers=len(layersSizes)
        self.layersSizes=layersSizes
        self.weights=[np.random.randn(y,x) for x,y in zip(layersSizes[:-1],layersSizes[1:])]
        self.biases=[np.random.randn(y,1) for y in layersSizes[1:]]
        self.thresh=thresholdFunction.main
        self.threshDerivative=thresholdFunction.derivative
    def feedForward(self,x):
        a=x.reshape(self.layersSizes[0],1)
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,a)+b
            a=self.thresh(z)
        return a
    def backPropagation(self,x,y):
        ###forward pass###
        activations=[]
        a=x.reshape(self.layersSizes[0],1)
        activations.append(a)
        zValues=[]
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,a)+b
            zValues.append(z)
            a=self.thresh(z)
            activations.append(a)
        ###initialisation for backward pass###
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta=costDerivative(a,y.reshape(self.layersSizes[-1],1))/(10*norme(x))
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        ###backward passes###
        for l in range(self.numberLayers-3,-1,-1):
            delta=np.dot(self.weights[l+1].transpose(),delta)*self.threshDerivative(zValues[l])
            nabla_b[l]=delta
            nabla_w[l]=np.dot(delta,activations[l].transpose())
        return nabla_b,nabla_w
    def stepLearning(self,x,y,eta):
        nb,nw=self.backPropagation(x,y)
        for l in range(self.numberLayers-1):
            self.biases[l]-=eta*nb[l]
            self.weights[l]-=eta*nw[l]
    
        
