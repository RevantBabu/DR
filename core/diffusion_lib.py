import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

X = np.genfromtxt("../data/processed/hc_13/T22_4.csv", delimiter=',')[:, (1,2)]
dM1 = np.load("distance_matrix_T26_0_1s_20ms.npy")
dM2 = np.load("distance_matrix_T22_4_1s_20ms.npy")

dM = np.sqrt(dM1 **2 + dM2 ** 2)

#dM = np.load("distance_matrix_spatial.npy")

embedding = SpectralEmbedding(n_components=2, affinity="precomputed")
#coords = embedding.fit( 1/(dM + 0.01)).embedding_
#coords = embedding.fit( 1/(dM + 0.01)**2).embedding_
coords = embedding.fit( np.exp(-dM) ).embedding_
#coords = embedding.fit( np.exp(-dM/4) ).embedding_
#coords = embedding.fit( np.exp(-dM/64) ).embedding_
print(coords.shape)

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0], coords[:, 1], 'o-', label="Target neurons")
plt.title('DM_lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig('../results/test_dmlib.svg', format="svg")

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0], 'o-', label="Target neurons")
plt.title('DM_lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig('../results/test_dmlib_ev1.svg', format="svg")

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1], 'o-', label="Target neurons")
plt.title('DM_lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig('../results/test_dmlib_ev2.svg', format="svg")
