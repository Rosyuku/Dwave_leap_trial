#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 20:42:00 2018

@author: kazuyuki


base:https://myenigma.hatenablog.com/entry/2017/04/10/074451
"""

import cvxpy
import numpy as np
import time

start = time.time()

np.random.seed(2)

n = 200
weight_limit = n * 2

value = np.random.randint(1, n, n)
weight = np.random.randint(1, n, n)

#size = np.array([21, 11, 15, 9, 34, 25, 41, 52])
#weight = np.array([22, 12, 16, 10, 35, 26, 42, 53])
#capacity = 100

x = cvxpy.Variable(shape=n, boolean=True)
objective = cvxpy.Maximize(value * x)
constraints = [weight_limit >= weight * x]

prob = cvxpy.Problem(objective, constraints)
prob.solve(solver=cvxpy.ECOS_BB)
result = np.round(x.value)

print("status:", prob.status)
print("optimal value", prob.value)
print("result x:", result)

print(time.time() - start)