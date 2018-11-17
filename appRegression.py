# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:57:44 2018

@author: Augustine
"""

import network as nn
import numpy as np
import random as rd

dimInput=2
dimOutput=2

eta=1.0
M=np.random.randn(dimOutput, dimInput)
B=np.random.randn(dimOutput,1).reshape(dimOutput,1)
n=nn.network([dimInput,dimOutput],nn.identity)
worstMax=0.0
trainingData=[]
for i in range(1000):
    x=np.random.randn(dimInput).reshape(dimInput,1)
    #x=x/(10*nn.norme(x))
    y=np.dot(M,x)+B
    if rd.random()>0.97:
        y+=np.array([0.0,rd.random()]).reshape(dimOutput,1)
    trainingData.append((x,y))

for j in range(100):
    i=0
    rd.shuffle(trainingData)
    for x,y in trainingData:
    ###training
        i+=1
        n.stepLearning(x,y,1.0)
        if i%100==0 and i>0:
            print("epoch{0} : {1}".format(j*10+i/100,worstMax))
            worstMax=0.0
        else:
            m=nn.norme(y-n.feedForward(x)).max()
            worstMax=max(m,worstMax)