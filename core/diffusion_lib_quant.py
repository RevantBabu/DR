import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

from sklearn import manifold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)


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
dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2 + d12**2)

#dM = np.load("../distances/21/1/distance_matrix_spatial.npy")

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
coords = embedding.fit( thresholdMatrix(1/(dM+0.1), 10) ).embedding_
print(coords.shape)


total_count = 0;
similar_count1 = 0;
similar_count2 = 0;
sM = thresholdMatrix(1/(dM+0.1), 10)
for i in range(0, sM.shape[0]):
  r = sM[i, :]
  r = np.nonzero(r)
  r = r[0]
  l = r.size
  for a in range(0, l):
    for b in range(a+1, l):
        total_count += 1;
        if (sM[i,r[a]]>sM[i,r[b]] and (abs(coords[i,0]-coords[r[a],0]) < abs(coords[i,0]-coords[r[b],0]))):
          similar_count1 += 1
        elif (sM[i,r[a]]<sM[i,r[b]] and (abs(coords[i,0]-coords[r[a],0]) > abs(coords[i,0]-coords[r[b],0]))):
          similar_count1 += 1

        if (sM[i,r[a]]>sM[i,r[b]] and (abs(coords[i,1]-coords[r[a],1]) < abs(coords[i,1]-coords[r[b],1]))):
          similar_count2 += 1
        elif (sM[i,r[a]]<sM[i,r[b]] and (abs(coords[i,1]-coords[r[a],1]) > abs(coords[i,1]-coords[r[b],1]))):
          similar_count2 += 1


print(similar_count1, similar_count2, total_count, similar_count1/total_count, similar_count2/total_count)



tM = thresholdMatrix(dM, 10)
sM = thresholdMatrix(1/(dM+0.1), 10)
graph = csr_matrix(tM)

print(dM.shape[0])
adist = dijkstra(csgraph=graph, directed=False, indices=range(0,dM.shape[0]))
print(np.count_nonzero(adist==0))

amax = np.amax(adist)
print(amax)
adist /= amax

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(adist)
coords = results.embedding_

total_count = 0;
similar_count1 = 0;
similar_count2 = 0;
sM = thresholdMatrix(1/(dM+0.1), 10)
for i in range(0, sM.shape[0]):
  r = sM[i, :]
  r = np.nonzero(r)
  r = r[0]
  l = r.size
  for a in range(0, l):
    for b in range(a+1, l):
        total_count += 1;
        if (sM[i,r[a]]>sM[i,r[b]] and (abs(coords[i,0]-coords[r[a],0]) < abs(coords[i,0]-coords[r[b],0]))):
          similar_count1 += 1
        elif (sM[i,r[a]]<sM[i,r[b]] and (abs(coords[i,0]-coords[r[a],0]) > abs(coords[i,0]-coords[r[b],0]))):
          similar_count1 += 1

        if (sM[i,r[a]]>sM[i,r[b]] and (abs(coords[i,1]-coords[r[a],1]) < abs(coords[i,1]-coords[r[b],1]))):
          similar_count2 += 1
        elif (sM[i,r[a]]<sM[i,r[b]] and (abs(coords[i,1]-coords[r[a],1]) > abs(coords[i,1]-coords[r[b],1]))):
          similar_count2 += 1


print(similar_count1, similar_count2, total_count, similar_count1/total_count, similar_count2/total_count)