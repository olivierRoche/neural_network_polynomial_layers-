# -*- coding: utf-8 -*-
# Copyright: (c) 2018, https://github.com/olivierRoche/neural_network_polynomial_layers-
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)
"""
Created on Sat Jun 30 23:59:23 2018

@author: Augustine
"""
import numpy as np
import random as rd

def sort_tuple(t):
    t_as_list = list(t)
    t_as_list.sort()
    return tuple(t_as_list)

class Polynomial:
#vars is the set of the index of all variables.
#Say the variables are X_0,X_1,X_2. Then vars is the set {0,1,2}.

#The coeff is a dictionary of coefficients, which keys are tuples. 
#eg the key (0,1) in coeff corresponds to the monomial X_0X_1
# the key (0,0,1) corresponds to the monomial X_0X_0X_1, ie X_0^2X_1 .
# () corresponds to 1 (the constant term).
# (0,) (resp. (1,)) corresponds to X_0 (resp. X_1).
#To ensure that a given monomial only comes once in the keys of coeff,
#the corresponding tuple is required to be ordered. eg, X_0^2X_1 might
#be described either by (0,0,1), (1,0,0), (0,1,0) but only (0,0,1) is valid
#since it is ordered. We use the function sort_tuple to achieve that.
    def __init__(self,coeff = {}):  
        self.coeff={}
        if len(coeff.keys())==0:
            self.coeff[()]= 0 
        for k in coeff.keys():
            self.coeff[sort_tuple(k)] = coeff[k]
        self.vars=set([])
        for c in self.coeff.keys():
            if c!=():
                for i in c:
                    self.vars.add(i)
                    
    def evaluate(self,valuation):
        #Evaluates the value of the plynomial for a given valuation.
        #The valuation is given as a dictionary. 
        #eg, say the polynomial is X_0^2+X_1, hence variables are X_0 and X_1,
        #then evaluate({0: 4, 1: 13}) will return 4^2 + 13, ie 29.
        
        #First, check if the input is compatible :
        if len(self.vars.symmetric_difference(valuation.keys())) != 0:
            missing_vars = self.vars.difference(valuation.keys())
            extra_vars = set(valuation.keys()).difference(self.vars)
            message_fields = []
            if len(missing_vars)!=0:
                message_fields.append("variables {0} are missing \n".format(
                                                            missing_vars))
            if len(extra_vars)!=0:
                message_fields.append("variables {0} were not expected".format(
                                                                extra_vars))
            message = ''.join(message_fields)
            raise ValueError(message)
        ret = 0
        for monomial in self.coeff.keys():
            term = self.coeff[monomial]
            for var in monomial :
                term *= valuation[var]
            ret += term
        return ret
        
    def raw_evaluate(self,array):
        #This requires the variables of self to be range(len(array)).
        return self.evaluate(raw_valuation(array))
    
    def __str__(self):
        return str({k:self.coeff[k] for k in self.coeff.keys() if 
                    self.coeff[k] !=0})
    
    def __add__(self,other):
        sumcoeff=self.coeff.copy()
        if isinstance(other,Polynomial):
            for k in other.coeff.keys():
                if k in sumcoeff.keys():
                    sumcoeff[k]+=other.coeff[k]
                else:
                    sumcoeff[k]=other.coeff[k]
        else:
            #here, we assume that other is of numeric type
            sumcoeff[()]+=other
        return Polynomial(sumcoeff)
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __mul__(self,other):
        if not isinstance(other,Polynomial):
            #again, we assume that other is then of numeric type
            prodcoeff=self.coeff.copy()
            for k in prodcoeff.keys():
                prodcoeff[k]*=other
        else:
            prodcoeff={}
            for sck in self.coeff.keys():
                for ock in other.coeff.keys():
                    r=self.coeff[sck]*other.coeff[ock]
                    keyCtor=[]
                    for v in self.vars | other.vars:
                        for i in sck:
                            if i==v:
                                keyCtor.append(i)
                        for i in ock:
                            if i==v:
                                keyCtor.append(i)
                    newkey=tuple(keyCtor)
                    if newkey in prodcoeff.keys():
                        prodcoeff[newkey]+=r
                    else:
                        prodcoeff[newkey]=r
        return Polynomial(prodcoeff)
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __sub__(self,other):
        return self+(-1)*other
    
    def __rsub__(self,other):
        return other+(-1)*self
    
    def partial_derivative(self,var):
        coeff_deriv={tuple([var]) : 0} #forces var to appear as a variable in 
                                 # the returned polynomial 
        for ck in self.coeff.keys():
            t_deriv=self.coeff[ck]
            deg_v=sum(i==var for i in ck)
            t_deriv*=deg_v
            keyCtor=[]
            vNotMet=True
            for i in ck:
                if var==i and vNotMet:
                    vNotMet=False
                else:
                    keyCtor.append(i)
            newkey=tuple(keyCtor)
            if newkey in coeff_deriv.keys():
                coeff_deriv[newkey]+=t_deriv
            else:
                coeff_deriv[newkey]=t_deriv
        return Polynomial(coeff_deriv)        
    
    def differential(self,valuation):
       if len(set(valuation.keys()) ^ self.vars)!=0:
           raise Exception("Incompatible Variables {0} expected, {1} given"
                           .format(self.vars,valuation.keys()))
       else:
           return np.array([(self.partial_derivative(i)).evaluate(valuation) 
                            for i in self.vars])

#-----------------------------------------------------------------------------
    
"""
--------help functions for arrays of Polynomials----------------------------

in this section, poly_function refers to an array of instances of Polynomial
whose variables indices range from 0 to n-1.
Say poly_function is of shape (m,), poly_function must be thought as 
 a function from R^n to R^m, which is polynomial componentwise.
point is assumed to be an array of numbers of shape (n,), representing
 a point in R^n."""
    
def evaluate_poly_func(poly_function,point):
    return np.array([p.raw_evaluate(point) for p in poly_function])    
    
def differential(poly_function,point):
    return np.array([p.differential(raw_valuation(point)) for p in poly_function])

def initialize_polyfunc_zero(dim_input,dim_output,degree):
    return np.array([fullZeroPolynomial(dim_input,degree) for i in range(dim_output)])


def initialize_polyfunc_rand(dim_input,dim_output,degree):
    return np.array([fullRandomPolynomial(dim_input,degree) for i in range(dim_output)])


    #**********initializing functions***********************************
    
def iterTuple(nb_var,degree,to_yield = [],start = 0,empty_tuple = True):
    #iters through all (ordered) tuples with values in range(nb_var-1) of
    #degree less than degree in lexycographical order. eg iterTuple(2,3)
    #iters trough [(), (0,), (0, 0), (0, 1), (0, 2), (1,), (1, 1), (1, 2), (2,), (2, 2)]
    if empty_tuple:
        yield ()
    if len(to_yield)>=degree:
        return
    else:
        for x in range(start,nb_var):
            to_yield.append(x)
            yield tuple(to_yield)
            for res in iterTuple(nb_var,degree,to_yield,start=x,empty_tuple=False):
                yield res
            to_yield.pop()
   
def fullZeroPolynomial(nbVar,degree):
    coeff = {}
    for k in iterTuple(nbVar,degree):
        coeff[k]=0.0
    return Polynomial(coeff)

def fullRandomPolynomial(nbVar,degree):
    coeff = {}
    for k in iterTuple(nbVar,degree):
        coeff[k]=rd.random()
    return Polynomial(coeff)

    #***********************utilitaries**************************
def raw_valuation(array):
    #uses the values of array to build a valuation.
    #eg raw_valuation(np.array([4,9,13])) gives {0: 4, 1: 9, 2: 13}
    #that is, X_0 is evaluated as 4, X_1 as 9 and X_2 as 13
    return {var : value for var,value in zip(range(len(array)),array)}

def monom(tuple,array):
    ret = 1
    for i in tuple:
        ret *= array[i]
    return ret
