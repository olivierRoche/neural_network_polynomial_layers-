# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:34:29 2018

@author: Augustine
"""
import network as nn
import numpy as np

def norme(x):
    return np.sqrt(sum(x**2))

dim=2
n=nn.network([dim,dim],nn.identity)
eta=1.0
worstMax=0.0
for i in xrange(10000):
    x=np.random.randn(dim,1)
    #x=x/(10*norme(x))
    y=x
    nb,nw = n.backPropagation(x,y)
    for j in xrange(n.numberLayers-1):
        n.biases[j]-=eta*nb[j]
        n.weights[j]-=eta*nw[j]
    if i%100==0:
        print "epoch{0} : {1}".format(i/100,worstMax)
        worstMax=0.0
    else:
        m=norme(y-n.feedForward(x)).max()
        worstMax=max(m,worstMax)