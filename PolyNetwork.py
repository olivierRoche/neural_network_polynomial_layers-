# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 08:43:20 2018

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

def enumSquare(n):
    for i in xrange(n):
        for j in xrange(i,n):
            yield i,j

def pRead(ind,n):
    compt=0
    for i in xrange(n):
        for j in xrange(i,n):
            if compt==ind:
                return [i,j]
            compt+=1
    raise Exception("index out of bound")
    
def productsVector(a):
    n=a.shape[0]
    return np.array([a[pRead(ind,n)[0]]*a[pRead(ind,n)[1]] for ind in xrange(n*(n+1)/2)])

def factorial(n):
    if n<=1:
        return 1
    else:
        return n*factorial(n-1)
   
class Polynetwork:
    def __init__(self,layersSizes,thresholdFunction,damping=1):
        self.numberLayers=len(layersSizes)
        self.layersSizes=layersSizes
        self.XY=[np.random.randn(y,x*(x+1)/2) for x,y in zip(layersSizes[:-1],layersSizes[1:])]
        self.weights=[np.random.randn(y,x) for x,y in zip(layersSizes[:-1],layersSizes[1:])]
        self.biases=[np.random.randn(y,1) for y in layersSizes[1:]]
        self.thresh=thresholdFunction.main
        self.threshDerivative=thresholdFunction.derivative
        self.damping=10**damping
    def feedForward(self,x):
        a=x.reshape(self.layersSizes[0],1)
        for w,b,c in zip(self.weights,self.biases,self.XY):
            z=np.dot(w,a)+b+np.dot(c,productsVector(a))
            a=self.thresh(z)
        return a
    def backPropagation(self,x,y):
        ###forward pass###
        activations=[]
        a=x.reshape(self.layersSizes[0],1)
        activations.append(a)
        zValues=[]
        for w,b,c in zip(self.weights,self.biases,self.XY):
            z=np.dot(w,a)+b+np.dot(c,productsVector(a))
            zValues.append(z)
            a=self.thresh(z)
            activations.append(a)
        ###initialisation for backward pass###
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_c=[np.zeros(c.shape) for c in self.XY]
        nabla_Cost=costDerivative(a,y.reshape(self.layersSizes[-1],1))/(self.damping*norme(x))
        delta_b[-1]=nabla_Cost
        delta_w[-1]=np.dot(nabla_Cost,activations[-2].transpose())
        nbXY=self.XY[-1].shape[1]
        deriv_XY=np.array([[activations[-2][pRead(ind,self.layersSizes[-2])[0]]*activations[-2][pRead(ind,self.layersSizes[-2])[1]] for ind in xrange(nbXY)]for j in xrange(self.layersSizes[-1])])
        delta_c[-1]=nabla_Cost*deriv_XY[:,:,0]
        ###backward passes###
        for l in xrange(self.numberLayers-3,-1,-1):
            nabla_Cost=np.dot((self.differentialTrans(activations[l+1],l+1)).transpose(),nabla_Cost)*self.threshDerivative(zValues[l])
#            nabla_Cost=np.dot(self.weights[l+1].transpose(),nabla_Cost)*self.threshDerivative(zValues[l])
            delta_b[l]=nabla_Cost
            delta_w[l]=np.dot(nabla_Cost,activations[l].transpose())
            delta_c[l]=self.differentialCross(activations[l],l)*nabla_Cost
        return delta_b,delta_w,delta_c
    def stepLearning(self,x,y,eta):
        nb,nw,nc=self.backPropagation(x,y)        
        for l in xrange(self.numberLayers-1):
            self.biases[l]-=eta*nb[l]
            self.weights[l]-=eta*nw[l]
            self.XY[l]-=eta*nc[l]
    def differentialCross(self,a,l):
        sizeXY=self.layersSizes[l]*(self.layersSizes[l]+1)/2
        size=self.layersSizes[l+1]
        derivprod=[[0.0 for ind in range(sizeXY)] for i in xrange(size)]
        for i in xrange(size):
            for ind in xrange(sizeXY):
                j,k=pRead(ind,self.layersSizes[l])
                if j==k:
                    derivprod[i][ind]=2*a[j]
                else:
                    derivprod[i][ind]=a[j]*a[k]
        return np.array(derivprod)[:,:,0]
    def differentialTrans(self,a,l):
        n=self.layersSizes[l]
        diff=self.weights[l]
        for j in xrange(self.layersSizes[l+1]):
            for ind in xrange(n*(n+1)/2):
                o,p=pRead(ind,n)
                if o==j:
                    diff[j,o]+=self.XY[l][j,ind]*a[p]
                if p==j:
                    diff[j,p]+=self.XY[l][j,ind]*a[o]                    
        return diff
    def batchLearning(self,batch,eta):
        db=[np.zeros(b.shape) for b in self.biases]
        dw=[np.zeros(w.shape) for w in self.weights]
        dc=[np.zeros(c.shape) for c in self.XY]
        for x,y in batch:
            nb,nw,nc=self.backPropagation(x,y)
            for l in xrange(self.numberLayers -1):
                db[l]+=nb[l]
                dw[l]+=nw[l]
                dc[l]+=nc[l]
        for l in xrange(self.numberLayers-1):
            self.biases[l]-=eta*db[l]
            self.weights[l]-=eta*dw[l]
            self.XY[l]-=eta*dc[l]
            

