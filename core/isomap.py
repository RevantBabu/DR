import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

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

if sys.argv[1]=="all":
  d1 = np.load("../distances/21/1/distance_matrix_T22_0_1s_20ms.npy")
  d2 = np.load("../distances/21/1/distance_matrix_T22_2_1s_20ms.npy")
  d3 = np.load("../distances/21/1/distance_matrix_T22_3_1s_20ms.npy")
  d4 = np.load("../distances/21/1/distance_matrix_T22_4_1s_20ms.npy")
  d5 = np.load("../distances/21/1/distance_matrix_T26_0_1s_20ms.npy")
  d6 = np.load("../distances/21/1/distance_matrix_T27_0_1s_20ms.npy")
  d7 = np.load("../distances/21/1/distance_matrix_T27_1_1s_20ms.npy")
  d8 = np.load("../distances/21/1/distance_matrix_T27_2_1s_20ms.npy")
  d9 = np.load("../distances/21/1/distance_matrix_T27_3_1s_20ms.npy")
  d10 = np.load("../distances/21/1/distance_matrix_T27_4_1s_20ms.npy")
  d11 = np.load("../distances/21/1/distance_matrix_T27_5_1s_20ms.npy")
  d12 = np.load("../distances/21/1/distance_matrix_T27_6_1s_20ms.npy")
  d1 = d1/np.amax(d1)
  d2 = d2/np.amax(d2)
  d3 = d3/np.amax(d3)
  d4 = d4/np.amax(d4)
  d5 = d5/np.amax(d5)
  d6 = d6/np.amax(d6)
  d7 = d7/np.amax(d7)
  d8 = d8/np.amax(d8)
  d9 = d9/np.amax(d9)
  d10 = d10/np.amax(d10)
  d11 = d11/np.amax(d11)
  d12 = d12/np.amax(d12)
  #dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d10**2 + d11**2)
  dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2 + d12**2)
elif sys.argv[1]=="all_sparse":
  d1 = np.load("distance_matrix_T22_0_1s_20ms.npy")
  d2 = np.load("distance_matrix_T22_2_1s_20ms.npy")
  d3 = np.load("distance_matrix_T22_3_1s_20ms.npy")
  d6 = np.load("distance_matrix_T27_1_1s_20ms.npy")
  d7 = np.load("distance_matrix_T27_2_1s_20ms.npy")
  d8 = np.load("distance_matrix_T27_3_1s_20ms.npy")
  d9 = np.load("distance_matrix_T27_4_1s_20ms.npy")
  d10 = np.load("distance_matrix_T27_5_1s_20ms.npy")
  dM = np.sqrt(d1**2 + d2**2 + d3**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2)
else:  
  dM = np.load("../distances/21/1/distance_matrix_" + sys.argv[1] + ".npy")

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

print(dM.shape[0])
adist = dijkstra(csgraph=graph, directed=False, indices=range(0,dM.shape[0]))
print(np.count_nonzero(adist==0))

amax = np.amax(adist)
print(amax)
adist /= amax

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(adist)

coords = results.embedding_

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(coords[:, 0]*10, coords[:, 1]*10, 'o', label="Target neurons")
plt.title('Isomap Dimensions')
plt.xlabel('dimension1 ($10^{-1}$)')
plt.ylabel('dimension2 ($10^{-1}$)')
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/' + sys.argv[1] + "_isomap.svg", format="svg")
plt.savefig('../results/21/1/isomap2d.png')
plt.savefig('../results/21/1/isomap2d.pdf')
