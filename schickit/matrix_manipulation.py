import numpy
from scipy import sparse


def weighted_hic_to_unweighted_graph(w_hic):
    unweighted = w_hic.copy()
    unweighted.data[unweighted.data > 0] = 1
    return unweighted
