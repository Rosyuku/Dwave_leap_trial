#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 09:34:09 2018

@author: kazuyuki
"""

import dwavebinarycsp

csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.SPIN)
#csp.add_constraint(lambda a, b, c: a * b * c == 1, ['a', 'b', 'c'])
#csp.add_constraint(lambda a, b, c: a + b + c > 1,  ['a', 'b', 'c'])
csp.add_constraint(lambda a, b, c: a + b + c < 3,  ['a', 'b', 'c'])
bqm = dwavebinarycsp.stitch(csp)

print(bqm)

import sympy

expr = bqm.offset
for symb, coef in bqm.linear.items():
    expr += sympy.symbols(symb) * coef
for (symb1, symb2), coef in bqm.quadratic.items():
    expr += sympy.symbols(symb1) * sympy.symbols(symb2) * coef

sympy.init_printing()
print(expr)

def test(a, b, c, aux0, aux1):
    return -1.0*a*aux0 - 1.0*a*b - 2.0*a + 1.0*aux0*b - 1.0*aux0*c + 1.0*aux0 - 1.0*b*c - 1.0*b - 2.0*c + 11.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

n = 5

case = np.array(np.meshgrid(*list(itertools.repeat([0, 1], n)))).T.reshape(-1, n)
    
df = pd.DataFrame(case)

df['act'] = df.iloc[:, :3].sum(axis=1)
df['pred'] = test(case[:, 0], case[:, 1], case[:, 2], case[:, 3], case[:, 4])
