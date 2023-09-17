import cupy as cp
import numpy as np


class Algorythms:

    def matricies_calc_np(self, bodies, G, DIMENSIONS):
        M = np.expand_dims(bodies[:, :2], axis=0).astype("float")
        direction = np.transpose(M, (1, 0, 2)) - M
        E = np.power(np.sum(direction**2, axis=2), (3/2))
        E[E<8] = 0
        bodies[:, 2:] -= np.sum(np.expand_dims(np.divide(np.expand_dims([G], axis=(0)), E, out=np.zeros((N, N)), where=E!=0), axis=2)*direction, axis=1)
        bodies[:, :2] += bodies[:, 2:]
        return bodies

    def matricies_calc_cp(self, bodies, G, DIMENSIONS):
        bodies_GPU = cp.asarray(bodies)
        M = cp.expand_dims(bodies_GPU[:, :2], axis=0)
        direction = cp.transpose(M, (1, 0, 2)) - M
        E = cp.power(cp.sum(direction**2, axis=2), (3/2))
        E[E<8] = 0
        G_MATRX = cp.expand_dims(cp.array([G]), axis=(0))
        attraction = cp.zeros_like(E)
        attraction = cp.divide(G_MATRX, E)
        attraction[cp.isinf(attraction)] = 0
        bodies_GPU[:, 2:] -= cp.sum(cp.expand_dims(attraction, axis=2)*direction, axis=1)
        bodies_GPU[:, :2] += bodies_GPU[:, 2:]
        return bodies_GPU.get() 