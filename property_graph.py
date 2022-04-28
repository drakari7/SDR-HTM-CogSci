from math import comb
import numpy as np
import random
import matplotlib.pyplot as plt

from sdr import SDR
import sdr as sdr_tools

def generate_random_sdr(n, w):
    rand_positions = random.sample(range(n), w)
    res = SDR.from_index_list(n, rand_positions)
    return res


def main():
    n, w = 1024, 40
    a = generate_random_sdr(n, w)

    thetas = list(range(0, w+1, 1))
    ratios = []

    for theta in thetas:
        v1 = sdr_tools.exact_prob_fp(a, w, theta)
        v2 = sdr_tools.approx_prob_fp(a, w, theta)

        ratio = v2/v1
        ratios.append(ratio)
        
    noise = [1 - theta/w for theta in thetas[::-1]]
    plt.plot(noise, ratios[::-1])
    plt.xlabel(f"Noise % for n = {n}, w = {w}")
    plt.ylabel(f"approximate probability / exact probability")
    plt.title(f"Variation of false probability approximation")
    plt.grid()
    plt.savefig("./graphs/fp_approximation.jpg", dpi=250)

if __name__ == "__main__":
    main()
