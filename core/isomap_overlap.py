import sys
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
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

#tM = thresholdMatrix(dM, 10)
tM = dM
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
ax.plot(coords[:, 0]*10, coords[:, 1]*10, 'o-', label="Target neurons")
plt.title('Isomap Dimensions')
plt.xlabel('dimension1 ($10^{-1}$)')
plt.ylabel('dimension2 ($10^{-1}$)')
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/' + sys.argv[1] + "_isomap.svg", format="svg")
plt.savefig('../results/21/1/isomap2d.png')
plt.savefig('../results/21/1/isomap2d.pdf')


############################ CORNER OVERLAP ###########################



mat = scipy.io.loadmat('../data/raw/hc_13/T1rawpos.mat')
days = mat['rawpos']

day = int(sys.argv[1])
epoch = int(sys.argv[2])
cellNo = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])

d = days[0][day][0][epoch][0][0]['data']
spikes = d[:, 0]
xs = d[:, 1]
ys = d[:, 2]

fct = 1
rng = int((end-start)/fct)
resX = np.zeros(rng)
resY = np.zeros(rng)

for i in range(0, spikes.size):
  resX[int((int(spikes[i])-start)/fct)] = xs[i]
  resY[int((int(spikes[i])-start)/fct)] = ys[i]


cornerX = []
cornerY = []
cornerIdx1= np.zeros(rng)
cornerIdx2 = np.zeros(rng)
cornerIdx3 = np.zeros(rng)
cornerIdx4 = np.zeros(rng)
ncX = []
ncY = []


for i in range(0,rng):
  if (resX[i]>240 or resX[i]<115 or resY[i]>170 or resY[i]<35):
    cornerX.append(resX[i])
    cornerY.append(resY[i])
    if (resX[i]<115 and resY[i]>100):
      cornerIdx1[i] = 1
    elif (resX[i]<115 and resY[i]<100):
      cornerIdx2[i] = 1
    elif (resX[i]>240 and resY[i]<100):
      cornerIdx3[i] = 1
    elif (resX[i]>240 and resY[i]>100):
      cornerIdx4[i] = 1
  else:
    ncX.append(resX[i])
    ncY.append(resY[i])


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(coords[:, 0][200:300]*100, 'o-')
ax.plot(np.nonzero(cornerIdx1[200:300])[0], np.full(np.nonzero(cornerIdx1[200:300])[0].shape, -5), 'o', label="Corner1")
ax.plot(np.nonzero(cornerIdx2[200:300])[0], np.full(np.nonzero(cornerIdx2[200:300])[0].shape, -5), 'o', label="Corner2")
ax.plot(np.nonzero(cornerIdx3[200:300])[0], np.full(np.nonzero(cornerIdx3[200:300])[0].shape, -5), 'o', label="Corner3")
ax.plot(np.nonzero(cornerIdx4[200:300])[0], np.full(np.nonzero(cornerIdx4[200:300])[0].shape, -5), 'o', label="Corner4")
plt.title('Isomap Dimension1')
plt.xlabel('time (s)')
plt.ylabel('dimension1 ($10^{-2}$)')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/imev1_overlap_200.png')
plt.savefig('../results/21/1/imev1_overlap_200.pdf')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1][200:300]*100, 'o-', label="Target neurons")
ax.plot(np.nonzero(cornerIdx1[200:300])[0], np.full(np.nonzero(cornerIdx1[200:300])[0].shape, -5), 'o', label="Corner1")
ax.plot(np.nonzero(cornerIdx2[200:300])[0], np.full(np.nonzero(cornerIdx2[200:300])[0].shape, -5), 'o', label="Corner2")
ax.plot(np.nonzero(cornerIdx3[200:300])[0], np.full(np.nonzero(cornerIdx3[200:300])[0].shape, -5), 'o', label="Corner3")
ax.plot(np.nonzero(cornerIdx4[200:300])[0], np.full(np.nonzero(cornerIdx4[200:300])[0].shape, -5), 'o', label="Corner4")
plt.title('Isomap Dimension2')
plt.xlabel('time (s)')
plt.ylabel('dimension2 ($10^{-2}$)')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/imev2_overlap_200.png')
plt.savefig('../results/21/1/imev2_overlap_200.pdf')