#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:38:39 2018

@author: suppurax
"""

import PolyNN
import AlgPoly as ap
import numpy as np

net = PolyNN.PolyNN([784,10,10], thresholdFunction=PolyNN.sigmoid, degrees=[2,2])
x=np.random.randn(784,)
y=np.random.randn(10,)
dP=net.backPropagation(x,y)