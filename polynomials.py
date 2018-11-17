# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:49:24 2018

@author: Augustine
"""

import numpy as np
class polynomial:
    def __init__(self,coef,shape=None):
        if shape:
            self.coef=coef.reshape(shape)
        else:
            self.coef=coef
        self.len=len(self.coef)
    def eval(self,x):
        px=1
        res=0
        for d in xrange(self.len-1,-1,-1):
            res+=self.coef[d]*px
            px*=x
        return res
    def __add__(self,other):
        coefSelf=expandArray(self.coef,other.len)
        coefOther=expandArray(other.coef,self.len)
        return polynomial(coefSelf+coefOther)
    def __sub__(self,other):
        coefSelf=expandArray(self.coef,other.len)
        coefOther=expandArray(other.coef,self.len)
        return polynomial(coefSelf-coefOther)
    def __mul__(self,other):
        n=(self.len-1)*(other.len-1)+1
        eSelf=expandArray(self.coef,n)
        mat=np.array([[eSelf[(n-1-j+i)*(j>=i)]*(j>=i) for j in range(n)]for i in range(n)])    
        return polynomial(np.dot(mat,expandArray(other.coef,n)))
    def derivative(self):
        mat=np.array([[(self.len-i)*(j+1==i) for j in range(self.len)] for i in range(1,self.len)])
        return polynomial(np.dot(mat, self.coef))
    def __str__(self):
        return str(self.coef)


        
def expandArray(a,n):
    #output=a completed with zeroes, of size n
    if n<=len(a):
        return a
    else:
        mat=np.array([[(j-i==n-len(a)) for i in range(len(a))] for j in range(n)])
        return np.dot(np.array(mat),a)    
        

a=np.array([2,1,2,3])
p=polynomial(a)
