#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 09:34:09 2018

@author: kazuyuki
"""

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import setting

import pandas as pd
import numpy as np
import time

start = time.time()
np.random.seed(2)

detail=False

alpha = 1

n = 5
weight_limit = n * 2

value = np.random.randint(1, n, n)
weight = np.random.randint(1, n, n)
c = np.array([1]*weight_limit)

Q = dict()
for i in range(n+weight_limit):
    for j in range(n+weight_limit):
        if i==j:
            if i < n:
                Q.update({("x"+str(i), "x"+str(i)):-value[i]+alpha*weight[i]**2})
            else:
                Q.update({("x"+str(i), "x"+str(i)):alpha*c[i-n]**2})
        elif i > j:
            if (i < n) & (j < n):
                Q.update({("x"+str(i), "x"+str(j)):2*alpha*weight[i]*weight[j]})
            elif (i >= n) & (j < n):
                Q.update({("x"+str(i), "x"+str(j)):-2*alpha*c[i-n]*weight[j]})
            else:
                Q.update({("x"+str(i), "x"+str(j)):2*alpha*c[i-n]*c[j-n]})
        else:
            continue

print("Start anealing", time.time() - start)

response = EmbeddingComposite(DWaveSampler(token=setting.tokencode)).sample_qubo(Q, num_reads=1000)

print("End anealing", time.time() - start)

if detail == True:

    df_result = pd.DataFrame()
    k = 0
    for sample, energy, num_occurrences, chain_break_fraction in list(response.data()):
        #print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
        df_tmp = pd.DataFrame(dict(sample), index=[k])
        df_tmp['Energy'] = -energy
        df_tmp['Occurrences'] = num_occurrences
        df_result = df_result.append(df_tmp)
        k+=1
    
    result = df_result.pivot_table(index=df_result.columns[:n].tolist()+['Energy'], values=['Occurrences'], aggfunc='sum').sort_values('Energy', ascending=False)
        
    print(result)
    
else:
    print(list(response.data())[0])

print("Total_real_time ", response.info["timing"]["total_real_time"], "us")
print(time.time() - start)
