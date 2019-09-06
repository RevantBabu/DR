import sys
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

#X = np.genfromtxt("../data/processed/hc_13/T22_4.csv", delimiter=',')[:, (1,2)]
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
#d13 = np.load("../distances/21/1/distance_matrix_T27_7_1s_20ms.npy")
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
#dM = np.sqrt(d1**2 + d3**2 + d4**2 + d5**2 + d6**2 + d7**2 + d8**2 + d9**2 + d10**2 + d11**2 + d12**2)
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

embedding = SpectralEmbedding(n_components=3, affinity="precomputed", n_neighbors=0)
coords = embedding.fit( thresholdMatrix(1/(dM+0.1), 10) ).embedding_
#coords = embedding.fit( thresholdMatrix(np.exp(-dM), 10) ).embedding_
#coords = embedding.fit( 1/(dM+0.0001) ).embedding_
#coords = embedding.fit( 1/(dM + 0.01)**2).embedding_
#coords = embedding.fit( np.exp(-dM) ).embedding_
#coords = embedding.fit( np.exp(-dM/4) ).embedding_
#coords = embedding.fit( np.exp(-dM/64) ).embedding_
print(coords.shape)






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


df1 = pd.read_csv("../data/processed/hc_13/" + sys.argv[1] + "/" + sys.argv[2] + "/" + cellNo + ".csv", header=None)
spikes = df1[0].values

n = end-start
fr = np.zeros(n)
for spike in spikes:
  fr[int(spike) - start] += 1

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(coords[:, 0][200:300]*100, 'o-')
ax.plot(np.nonzero(cornerIdx1[200:300])[0], np.full(np.nonzero(cornerIdx1[200:300])[0].shape, -5), 'o', label="Corner1")
ax.plot(np.nonzero(cornerIdx2[200:300])[0], np.full(np.nonzero(cornerIdx2[200:300])[0].shape, -5), 'o', label="Corner2")
ax.plot(np.nonzero(cornerIdx3[200:300])[0], np.full(np.nonzero(cornerIdx3[200:300])[0].shape, -5), 'o', label="Corner3")
ax.plot(np.nonzero(cornerIdx4[200:300])[0], np.full(np.nonzero(cornerIdx4[200:300])[0].shape, -5), 'o', label="Corner4")
plt.title('Diffusion Map Dimension1')
plt.xlabel('time (s)')
plt.ylabel('dimension1 ($10^{-2}$)')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/dmev1_overlap_200.png')
plt.savefig('../results/21/1/dmev1_overlap_200.pdf')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1][200:300]*100, 'o-', label="Target neurons")
ax.plot(np.nonzero(cornerIdx1[200:300])[0], np.full(np.nonzero(cornerIdx1[200:300])[0].shape, -5), 'o', label="Corner1")
ax.plot(np.nonzero(cornerIdx2[200:300])[0], np.full(np.nonzero(cornerIdx2[200:300])[0].shape, -5), 'o', label="Corner2")
ax.plot(np.nonzero(cornerIdx3[200:300])[0], np.full(np.nonzero(cornerIdx3[200:300])[0].shape, -5), 'o', label="Corner3")
ax.plot(np.nonzero(cornerIdx4[200:300])[0], np.full(np.nonzero(cornerIdx4[200:300])[0].shape, -5), 'o', label="Corner4")
plt.title('Diffusion Map Dimension2')
plt.xlabel('time (s)')
plt.ylabel('dimension2 ($10^{-2}$)')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/dmev2_overlap_200.png')
plt.savefig('../results/21/1/dmev2_overlap_200.pdf')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 2][200:300]*100, 'o-', label="Target neurons")
ax.plot(np.nonzero(cornerIdx1[200:300])[0], np.full(np.nonzero(cornerIdx1[200:300])[0].shape, -5), 'o', label="Corner1")
ax.plot(np.nonzero(cornerIdx2[200:300])[0], np.full(np.nonzero(cornerIdx2[200:300])[0].shape, -5), 'o', label="Corner2")
ax.plot(np.nonzero(cornerIdx3[200:300])[0], np.full(np.nonzero(cornerIdx3[200:300])[0].shape, -5), 'o', label="Corner3")
ax.plot(np.nonzero(cornerIdx4[200:300])[0], np.full(np.nonzero(cornerIdx4[200:300])[0].shape, -5), 'o', label="Corner4")
plt.title('Diffusion Map Dimension3')
plt.xlabel('time (s)')
plt.ylabel('dimension2 ($10^{-2}$)')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/dmev3_overlap_200.png')
plt.savefig('../results/21/1/dmev3_overlap_200.pdf')

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(10*coords[:, 0][200:300]/np.max(coords[:, 0]), 'o-')
ax.plot(10*fr[200:300]/np.max(fr), 'o-')
plt.title('DM Dimension1 and Firing Rate')
plt.xlabel('time (s)')
plt.ylabel('magnitude')
ax.yaxis.set_label_coords(-0.08,0.5)
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/dmev1_fr_200.png')
plt.savefig('../results/21/1/dmev1_fr_200.pdf')

print(np.max(fr), np.min(fr))
print(np.max(coords[:, 0]), np.min(coords[:, 0]))