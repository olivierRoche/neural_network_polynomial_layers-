# -*- coding: utf-8 -*-
# Copyright: (c) 2018, https://github.com/olivierRoche/neural_network_polynomial_layers-
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Created on Fri Jul 13 13:35:26 2018

@author: Augustine
"""

import numpy as np

import AlgPoly as ap

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
    return sum((a-y)*(a-y))

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
        self.damping = 10**damping
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
    
    def steplearning(self,x,y,eta):
        """ modifies self.functions using self.backPropagation(x,y)
        eta is the change rate """     
        delta_P = self.backPropagation(x,y)
        for l in range(self.numberLayers-1):
            self.functions[l] -= eta* delta_P[l]           
#    def backPropagation(self,x,y):
#        ###forward pass###
#        activations=[]
#        a=x.reshape(self.layersSizes[0],1)
#        activations.append(a)
#        zValues=[]
#        for w,b,c in zip(self.weights,self.biases,self.XY):
#            z=np.dot(w,a)+b+np.dot(c,productsVector(a))
#            zValues.append(z)
#            a=self.thresh(z)
#            activations.append(a)
#        ###initialisation for backward pass###
#        delta_b = [np.zeros(b.shape) for b in self.biases]
#        delta_w = [np.zeros(w.shape) for w in self.weights]
#        delta_c=[np.zeros(c.shape) for c in self.XY]
#        nabla_Cost=costDerivative(a,y.reshape(self.layersSizes[-1],1))/(self.damping*norme(x))
#        delta_b[-1]=nabla_Cost
#        delta_w[-1]=np.dot(nabla_Cost,activations[-2].transpose())
#        nbXY=self.XY[-1].shape[1]
#        deriv_XY=np.array([[activations[-2][pRead(ind,self.layersSizes[-2])[0]]*activations[-2][pRead(ind,self.layersSizes[-2])[1]] for ind in range(nbXY)]for j in range(self.layersSizes[-1])])
#        delta_c[-1]=nabla_Cost*deriv_XY[:,:,0]
#        ###backward passes###
#        for l in range(self.numberLayers-3,-1,-1):
#            nabla_Cost=np.dot((self.differentialTrans(activations[l+1],l+1)).transpose(),nabla_Cost)*self.threshDerivative(zValues[l])
##            nabla_Cost=np.dot(self.weights[l+1].transpose(),nabla_Cost)*self.threshDerivative(zValues[l])
#            delta_b[l]=nabla_Cost
#            delta_w[l]=np.dot(nabla_Cost,activations[l].transpose())
#            delta_c[l]=self.differentialCross(activations[l],l)*nabla_Cost
#        return delta_b,delta_w,delta_c
#    def stepLearning(self,x,y,eta):
#        nb,nw,nc=self.backPropagation(x,y)        
#        for l in range(self.numberLayers-1):
#            self.biases[l]-=eta*nb[l]
#            self.weights[l]-=eta*nw[l]
#            self.XY[l]-=eta*nc[l]
#    def differentialCross(self,a,l):
#        sizeXY=self.layersSizes[l]*(self.layersSizes[l]+1)/2
#        size=self.layersSizes[l+1]
#        derivprod=[[0.0 for ind in range(sizeXY)] for i in range(size)]
#        for i in range(size):
#            for ind in range(sizeXY):
#                j,k=pRead(ind,self.layersSizes[l])
#                if j==k:
#                    derivprod[i][ind]=2*a[j]
#                else:
#                    derivprod[i][ind]=a[j]*a[k]
#        return np.array(derivprod)[:,:,0]
#    def differentialTrans(self,a,l):
#        n=self.layersSizes[l]
#        diff=self.weights[l]
#        for j in range(self.layersSizes[l+1]):
#            for ind in range(n*(n+1)/2):
#                o,p=pRead(ind,n)
#                if o==j:
#                    diff[j,o]+=self.XY[l][j,ind]*a[p]
#                if p==j:
#                    diff[j,p]+=self.XY[l][j,ind]*a[o]                    
#        return diff
#    def batchLearning(self,batch,eta):
#        db=[np.zeros(b.shape) for b in self.biases]
#        dw=[np.zeros(w.shape) for w in self.weights]
#        dc=[np.zeros(c.shape) for c in self.XY]
#        for x,y in batch:
#            nb,nw,nc=self.backPropagation(x,y)
#            for l in range(self.numberLayers -1):
#                db[l]+=nb[l]
#                dw[l]+=nw[l]
#                dc[l]+=nc[l]
#        for l in range(self.numberLayers-1):
#            self.biases[l]-=eta*db[l]
#            self.weights[l]-=eta*dw[l]
#            self.XY[l]-=eta*dc[l]
