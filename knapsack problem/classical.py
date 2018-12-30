#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 01:28:28 2018

@author: kazuyuki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

start = time.time()

np.random.seed(2)

n = 5
weight_limit = n * 2

value = np.random.randint(1, n, n)
weight = np.random.randint(1, n, n)

case = np.array(np.meshgrid(*list(itertools.repeat([0, 1], n)))).T.reshape(-1, n)

case_value = (case * value).sum(axis=1).reshape(-1, 1)
case_weight = (case * weight).sum(axis=1).reshape(-1, 1)

df = pd.DataFrame(columns=list(range(n))+['value', 'weight'], data=np.concatenate([case, case_value, case_weight], axis=1))

tdf = df.loc[df['weight'] < weight_limit].sort_values('value', ascending=False)

print(tdf.head(2).T)

print(time.time() - start)