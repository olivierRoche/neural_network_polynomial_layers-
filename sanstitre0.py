# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 23:59:23 2018

@author: Augustine
"""

class Polynomial:
    def __init__(self,coeff):
        self.coeff=coeff
        if '' not in coeff.keys():
            self.coeff['']=0
        self.vars=set([])
        for c in self.coeff.keys():
            if c!='':
                for i in c:
                    self.vars.add(i)
    def __add__(self,other):
        resCoeff=self.coeff.copy()
        if isinstance(other,Polynomial):
            for k in other.coeff.keys():
                if k in resCoeff.keys():
                    resCoeff[k]+=other.coeff[k]
                else:
                    resCoeff[k]=other.coeff[k]
        else:
            resCoeff['']+=other
        return Polynomial(resCoeff)
    def eval(self,values):
        #renvoie un nombre si la valuation est complète, un Polynomial sinon
        if len(set(values.keys())&self.vars)==len(self.vars):
            res=0
            for ck in self.coeff.keys():
                cur=self.coeff[ck]
                for i in xrange(len(ck)):
                    cur*=values[ck[i]]
                res+=cur
            return res
        else:
            

p=Polynomial({'' : 1,(0,) : 2, (0,0) : 1})
q=Polynomial({(0,) :1, (0,1): 2})
    