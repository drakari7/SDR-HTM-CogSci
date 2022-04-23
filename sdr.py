from cmath import exp
import numpy as np
from math import comb

from torch import exp_

class SDR:
    def __init__(self, sdr:np.array):
        self.vec = sdr
        self.n = len(sdr)
        self.w = np.sum(sdr)
        self.s = self.w/self.n
    
    def __mul__(self, other):
        return np.sum(np.logical_and(self.vec,other.vec))
    
def match(sdr1, sdr2, theta):
    return sdr1 * sdr2 >= theta

def uniq_sdr(n, w):
    return comb(n, w)

def prob_identical(n, w):
    return 1/comb(n, w)

def num_overlap(sdr, w, b):
    return comb(sdr.w, b) * comb(sdr.n - sdr.w, w - b)

def prob_fp(sdr, w, theta):
    return np.sum([num_overlap(sdr, w, i) for i in range(theta, w+1, 1)]) / uniq_sdr(sdr.n, w)

def approx_prob_fp(sdr, w, theta):
    return num_overlap(sdr, w, theta) / uniq_sdr(sdr.n, w)

#For sub sampling pass :: w_y as w

def check_match(X, theta):
    for x in X:
        for y in X:
            if x!=y and match(x, y, theta):
                return False
    return True

def p_not(w, n, M):
    return (w/n)**M

def prob_fp_union(w, n, M):
    return (1-p_not(w, n, M))**w

def expected_size(sdr, w, b, M):
    exp_w_x = sdr.n * (1 - p_not(sdr.w, sdr.n, M))
    return comb(exp_w_x, b) * comb(sdr.n - exp_w_x, w - b)

a = np.array([1,1,0])
b = np.array([1,0,1])
print(SDR(a)*SDR(b)) 


