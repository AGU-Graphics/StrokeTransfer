import numpy as np


def normalize_vectors(u):
    epsilon = 1e-10
    u_norm = np.sqrt(np.sum(u * u, axis=1))
    return np.einsum("ij,i->ij", u, 1.0 / (epsilon + u_norm))


def normalize_vector_image(V):
    epsilon = 1e-16
    V_norm = epsilon + np.sqrt(np.einsum("ijk,ijk->ij", V, V))
    V_normalized = np.einsum("ijk, ij->ijk", V, 1.0 / V_norm)
    return V_normalized
