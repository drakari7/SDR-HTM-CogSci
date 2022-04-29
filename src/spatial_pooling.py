from math import comb
import numpy as np
import random
import matplotlib.pyplot as plt

from sdr import SDR
import sdr_utils


def generate_random_matrix(n, c, P):
    return np.random.choice([0, 1], size=(n,c), p=[1-P, P])

def main():
    n, w = 1024, 40
    k = w
    a = sdr_utils.generate_random_sdr(n, w)
    b = sdr_utils.generate_random_sdr(n, w)
    matrix_sparsity = 0.1
    c = 1024
    sp = generate_random_matrix(n, c, matrix_sparsity)

    res = np.matmul(a.vec, sp)

    # top_k_indices = np.argsort(res)[::-1][:k]

    # final_ans = np.zeros(c)
    # final_ans[top_k_indices] = 1
    # ans_sdr = SDR(final_ans)

    cutoff = 400
    top_k = np.sort(res)[::-1][:cutoff]
    plt.plot(range(400), top_k)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # main()
    pass
