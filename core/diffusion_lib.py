import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

X = np.genfromtxt("../data/processed/hc_13/T22_4.csv", delimiter=',')[:, (1,2)]
d1 = np.load("distance_matrix_T22_0_1s_20ms.npy")
d2 = np.load("distance_matrix_T22_2_1s_20ms.npy")
d3 = np.load("distance_matrix_T22_3_1s_20ms.npy")
d4 = np.load("distance_matrix_T22_4_1s_20ms.npy")
d5 = np.load("distance_matrix_T26_0_1s_20ms.npy")
d6 = np.load("distance_matrix_T27_1_1s_20ms.npy")
d7 = np.load("distance_matrix_T27_2_1s_20ms.npy")
d8 = np.load("distance_matrix_T27_3_1s_20ms.npy")
d9 = np.load("distance_matrix_T27_4_1s_20ms.npy")
d10 = np.load("distance_matrix_T27_5_1s_20ms.npy")
d11 = np.load("distance_matrix_T27_6_1s_20ms.npy")
dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2)

#dM = np.load("distance_matrix_spatial.npy")


def thresholdMatrix(sM, topN):
  n = sM.shape[0]
  m = sM.shape[1]
  result = np.zeros(shape=(n,m))

  columnSorted = np.sort(sM,axis=0)
  rowSorted = np.sort(sM,axis=1)

  rowTopN = np.zeros(n)
  columnTopN = np.zeros(n)
  for i in range(0, n):
    rowTopN[i] = rowSorted[i, :][n-topN]
    columnTopN[i] = columnSorted[:, i][n-topN]


  for i in range(0, n):
    for j in range(0, m):
      if (sM[i][j]>=rowTopN[i] or sM[i][j]>=columnTopN[j]):
        result[i][j] = sM[i][j]

  return result

embedding = SpectralEmbedding(n_components=2, affinity="precomputed", n_neighbors=0)
coords = embedding.fit( thresholdMatrix(np.exp(-dM), 10) ).embedding_
#coords = embedding.fit( 1/(dM + 0.01)**2).embedding_
#coords = embedding.fit( np.exp(-dM) ).embedding_
#coords = embedding.fit( np.exp(-dM/4) ).embedding_
#coords = embedding.fit( np.exp(-dM/64) ).embedding_
print(coords.shape)

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0], coords[:, 1], 'o', label="Target neurons")
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
