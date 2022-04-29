import matplotlib.pyplot as plt

from sdr import SDR
import sdr_utils

# Returns union SDR of k random SDRs
def get_union_k(n, w, k) -> SDR:
    sdrs = [sdr_utils.generate_random_sdr(n, w) for _ in range(k)]
    return sdr_utils.sdr_union(sdrs)

if __name__ == "__main__":
    n, w = 1024, 20

    graph_dir = sdr_utils.get_graph_dir()

    M = 200
    num_unions = range(1, M)
    sparsities = [get_union_k(n, w, k).s*100 for k in num_unions]

    plt.plot(num_unions, sparsities)
    plt.grid()
    plt.xlabel(f"Number of SDRs in union")
    plt.ylabel(f"Percentage of ON bits")
    plt.title(f"Saturation of ON bits with size of Union Vector (n={n}, w={w})")

    file_name = graph_dir + 'union_saturation.jpg'
    plt.savefig(file_name, dpi=250)


