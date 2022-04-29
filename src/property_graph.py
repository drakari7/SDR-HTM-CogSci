import numpy as np
import random
import matplotlib.pyplot as plt

from sdr import SDR
import sdr_utils

graph_dir = sdr_utils.get_graph_dir()

def main():
    n, w = 1024, 40
    a = sdr_utils.generate_random_sdr(n, w)

    thetas = list(range(0, w+1, 1))
    ratios = []

    for theta in thetas:
        v1 = sdr_utils.exact_prob_fp(a, w, theta)
        v2 = sdr_utils.approx_prob_fp(a, w, theta)

        ratio = v2/v1
        ratios.append(ratio)
        
    noise = [1 - theta/w for theta in thetas[::-1]]
    plt.plot(noise, ratios[::-1])
    plt.xlabel(f"Noise % for n = {n}, w = {w}")
    plt.ylabel(f"approximate probability / exact probability")
    plt.title(f"Variation of false probability approximation")
    plt.grid()

    file_name = graph_dir + 'fp_approximation.jpg'
    plt.savefig(file_name, dpi=250)

if __name__ == "__main__":
    main()
