#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:00:48 2018

@author: suppurax
"""

import numpy as np
import AlgPoly as ap
import timeit

P = ap.initialize_polyfunc_rand(50,10,2)
x = np.random.randn(50,)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped
