# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:07:01 2018

@author: Augustine
"""

import numpy as np
import PolyNetwork
from network import norme

n=PolyNetwork.Polynetwork([3,1],PolyNetwork.identity,2.5)


def P(a):
    return a[0]+3*a[1]+2*a[2]+4*a[0]*a[1]+ 5*a[0]*a[2]+a[1]*a[2]+100*a[0]**2+2*a[1]**2

def xPx(a):
    return (a,P(a))

eta=1.0
worstMax=0.0
for i in xrange(100):
    batch=[xPx(np.random.randn(3,1)) for j in xrange(100)]
    n.batchLearning(batch,eta)
    x,y=xPx(np.array([1,1,0.5]))
    print(y-n.feedForward(x))