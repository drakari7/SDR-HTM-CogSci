from math import comb
import numpy as np
import random
import matplotlib.pyplot as plt

from sdr import SDR
import sdr as sdr_tools

def generate_random_sdr(n, w):
    rand_positions = random.sample(range(n), w)
    print(type(rand_positions))
    res = SDR.from_index_list(n, rand_positions)
    return res

def generate_random_NxC_matrix(n, c, P):
    return np.random.choice([0, 1], size=(n,c), p=[P, 1-P])
    
def main():
    n, w = 1024, 500
    a = generate_random_sdr(n, w)
    b = generate_random_sdr(n, w)
    P = 0.1
    c = 1024
    mat = generate_random_NxC_matrix(n, c, P)
    res = np.zeros(n)
	
    print(mat)
    print(mat[:,1])
    print(a)

    for i in range(n):
        # res[i] = np.dot(np.asarray(a), mat[i:])
        res[i] = SDR.overlap(a, SDR.from_index_list(n,mat[:,i].tolist()))
        # res[i] = SDR.overlap(a, b)
        # print(i, res[i])
    print(res)

if __name__ == "__main__":
    main()
