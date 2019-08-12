import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def thresholdMatrix(sM, topN):
  n = sM.shape[0]
  m = sM.shape[1]
  result = np.zeros(shape=(n,m))

  columnSorted = np.sort(sM,axis=0)
  rowSorted = np.sort(sM,axis=1)

  rowTopN = np.zeros(n)
  columnTopN = np.zeros(n)

  for i in range(0, n):
    rowTopN[i] = rowSorted[i, :][topN-1]
    columnTopN[i] = columnSorted[:, i][topN-1]

  for i in range(0, n):
    for j in range(0, m):
      if (sM[i][j]<=rowTopN[i] or sM[i][j]<=columnTopN[j]): #How to Handle 0s ???
        result[i][j] = sM[i][j]

  return result

#dM = np.load("distance_matrix_" + sys.argv[1] + ".npy")
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
tM = thresholdMatrix(dM, 10)
graph = csr_matrix(tM)

# graph = [
# [1, 1 , 3, 8],
# [1, 2, 6, 1],
# [3, 6, 1, 7],
# [8, 1, 7, 4]
# ]
# graph = thresholdMatrix(np.asarray(graph), 2)
# print(graph)
# graph = csr_matrix(graph)

adist = dijkstra(csgraph=graph, directed=False, indices=range(0,805))
print(np.count_nonzero(adist==0))

amax = np.amax(adist)
adist /= amax

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(adist)

coords = results.embedding_

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0], coords[:, 1], 'o')
#plt.savefig('../results/' + sys.argv[1] + "_isomap.png")
plt.savefig('../results/all_isomap.png')
