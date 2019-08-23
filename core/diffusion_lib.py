import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

#X = np.genfromtxt("../data/processed/hc_13/T22_4.csv", delimiter=',')[:, (1,2)]
d1 = np.load("../distances/23/5/distance_matrix_T22_0_1s_20ms.npy")
d2 = np.load("../distances/23/5/distance_matrix_T22_1_1s_20ms.npy")
d4 = np.load("../distances/23/5/distance_matrix_T22_2_1s_20ms.npy")
d3 = np.load("../distances/23/5/distance_matrix_T22_3_1s_20ms.npy")
d5 = np.load("../distances/23/5/distance_matrix_T26_0_1s_20ms.npy")
d6 = np.load("../distances/23/5/distance_matrix_T27_0_1s_20ms.npy")
d7 = np.load("../distances/23/5/distance_matrix_T27_1_1s_20ms.npy")
d8 = np.load("../distances/23/5/distance_matrix_T27_2_1s_20ms.npy")
#d9 = np.load("../distances/23/5/distance_matrix_T27_3_1s_20ms.npy")
d10 = np.load("../distances/23/5/distance_matrix_T27_4_1s_20ms.npy")
d11 = np.load("../distances/23/5/distance_matrix_T27_5_1s_20ms.npy")
d12 = np.load("../distances/23/5/distance_matrix_T27_6_1s_20ms.npy")
d13 = np.load("../distances/23/5/distance_matrix_T27_7_1s_20ms.npy")
# d1 = d1/np.amax(d1)
# d2 = d2/np.amax(d2)
# d3 = d3/np.amax(d3)
# d4 = d4/np.amax(d4)
# d5 = d5/np.amax(d5)
# d6 = d6/np.amax(d6)
# d7 = d7/np.amax(d7)
# d8 = d8/np.amax(d8)
# d9 = d9/np.amax(d9)
# d10 = d10/np.amax(d10)
# d11 = d11/np.amax(d11)
# d12 = d12/np.amax(d12)
dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d10**2 + d11**2 + d12**2 +  d13**2)
#dM = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2 + d12**2)

#dM = np.load("../distances/23/5/distance_matrix_spatial.npy")

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
#coords = embedding.fit( 1/(dM+0.0001) ).embedding_
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
#plt.savefig('../results/dm/23/5/test_dmlib.svg', format="svg")
plt.savefig('../results/23/5/dm2d.png')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0][200:245], 'o-', label="Target neurons")
plt.title('DM_lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/23/5/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/23/5/dmev1.png')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1][200:245], 'o-', label="Target neurons")
plt.title('DM_lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/23/5/test_dmlib_ev2.svg', format="svg")
plt.savefig('../results/23/5/dmev2.png')


rng = dM.shape[0]
fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, rng, rng)

ax.plot(coords[:, 0], coords[:, 1], z, 'o-', label='parametric curve', linewidth=0.6, markersize=1)# c = plt.cm.jet(z/max(z)))
ax.legend()

#plt.show()
plt.savefig('../results/23/5/dm3d.png')