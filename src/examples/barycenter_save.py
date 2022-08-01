import sys, os

import torch

from barycenters.simplex import CliqSampler, AllCliq

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/'))

from dataset.lazy_loader import LazyLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset.toheatmap import ToGaussHeatMap
import ot
from barycenters.sampler import Uniform2DBarycenterSampler
from parameters.path import Paths
from joblib import Parallel, delayed


N = 300
print(N)
dataset = LazyLoader.w300().dataset_train

padding = 68
prob = np.ones(padding) / padding
NS = 7000

data_ids = np.random.permutation(np.arange(0, 3148))[0: N]

def LS(k):
    return dataset[k]["meta"]['keypts_normalized'].numpy()

ls = np.asarray([LS(k) for k in data_ids])

# distance matrix
D = np.zeros((N, N))

# uniform measures as points
a = np.ones(padding) / padding
b = np.ones(padding) / padding

for i in range(N):
    for j in range(i + 1, N):
        # squared Euclidean distance as the ground metric
        M_ij = ot.dist(ls[i], ls[j], metric="sqeuclidean")

        # 2-Wasserstein distance, take square root
        D[i, j] = np.sqrt(ot.emd2(a, b, M_ij))

# symmetrize distance matrix
D = D + D.T
print("D matrix")

def viz_mes(ms):
    heatmaper = ToGaussHeatMap(128, 1)

    kapusta = torch.zeros(128, 128)
    for m in ms:
        keyptsiki = torch.from_numpy(m)[None,].clamp(0, 1)
        tmp = heatmaper.forward(keyptsiki)
        kapusta += tmp.sum(axis=(0, 1))

    plt.imshow(kapusta)
    plt.show()

    return kapusta / kapusta.sum()

ls_mes = viz_mes(ls)

bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=1.0)

def juja(knn):

    def juja_inside(sample):
        landmarks = [ls[i] for i in sample]
        B, Bws = bc_sampler.sample(landmarks)
        print(sample)
        return B

    cliques, K = AllCliq(knn).forward(D)
    cl_sampler = CliqSampler(cliques)
    cl_samples = cl_sampler.sample(NS)

    bc_samples = list(Parallel(n_jobs=30)(delayed(juja_inside)(sample) for sample in cl_samples))

    bc_mes = viz_mes(bc_samples)
    ent = kl(ls_mes, bc_mes) + kl(bc_mes, ls_mes)

    return ent, bc_samples


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


ent, bcs = juja(knn=11)
print("KL distance between generated data and initial", ent)

#
os.mkdir(f"{Paths.default.data()}/w300_bc_{N}")
os.mkdir(f"{Paths.default.data()}/w300_bc_{N}/lmbc")
os.mkdir(f"{Paths.default.data()}/w300_bc_{N}/lm")

for i,b in enumerate(bcs):
    np.save(f"{Paths.default.data()}/w300_bc_{N}/lmbc/{i}.npy", b)

for i,b in enumerate(ls):
    np.save(f"{Paths.default.data()}/w300_bc_{N}/lm/{i}.npy", b)
