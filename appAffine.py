# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:45:12 2018

@author: Augustine
"""

import network as nn
import numpy as np

dimInput=2    
dimOutput=2
eta=1.0
M=np.random.randn(dimOutput, dimInput)
B=np.random.randn(dimOutput,1).reshape(dimOutput,1)
n=nn.network([dimInput,dimOutput],nn.identity)
worstMax=0.0
for i in xrange(10000):
    ###creation training data
    x=np.random.randn(dimInput).reshape(dimInput,1)
    #x=x/(10*nn.norme(x))
    y=np.dot(M,x)+B
    ###training
    n.stepLearning(x,y,1.0)
    if i%100==0 and i>0:
        print "epoch{0} : {1}".format(i/100,worstMax)
        worstMax=0.0
    else:
        m=nn.norme(y-n.feedForward(x)).max()
        worstMax=max(m,worstMax)