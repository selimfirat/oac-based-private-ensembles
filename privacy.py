# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:08:09 2021

@author: bh4418
"""

import numpy as np
from scipy.stats import norm


# Functions for calculating privacy loss

def binary_search_sigma(init_min, init_max, eps, delta, m, p):
    eps_p = np.log(1+(1/p)*(np.exp(eps)-1))
    
    def get_delta(sigma):
        return p*(norm.cdf(1/sigma-(eps_p*sigma)/2) - np.exp(eps_p)*norm.cdf(-1/sigma-(eps_p*sigma)/2))
        
    cur_min = init_min
    cur_max = init_max

    avg = 0.5*(cur_min + cur_max)
    while np.abs(cur_max-cur_min)>1e-10:
        avg = 0.5*(cur_min + cur_max)
        current_delta = get_delta(avg)
        # print(f"comp_delta_res: {comp_delta_res}, delta: {delt}")
        if current_delta == delta:
            return avg
        elif current_delta > delta:
            cur_min = avg
        else:
            cur_max = avg
    total_sigma = 0.5*(cur_min + cur_max)
    return total_sigma/np.sqrt(m)

# # Test:
# import matplotlib.pyplot as plt
# p_vect = np.arange(0.1,1.1,0.1)
# resVect = []
# for p in p_vect:
#     resVect.append(binary_search_sigma(0,20,1.0,1e-6,p*10,p))
# plt.plot(p_vect,resVect)
        
    

