import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)


mat = scipy.io.loadmat('../data/raw/hc_13/T1rawpos.mat')
days = mat['rawpos']

day = 21
epoch = 1
cellNo = "T26_0"
start = 790
end = 1595

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

cMap = []

for i in range(0,rng):
  if not (resX[i]>240 or resX[i]<115 or resY[i]>170 or resY[i]<35):
    if (resX[i]<177 and resY[i]<102.5):
      cMap.append('r')
    elif (resX[i]<177 and resY[i]>102.5):
      cMap.append('b')
    elif (resX[i]>177 and resY[i]<102.5):
      cMap.append('g')
    elif (resX[i]>177 and resY[i]>102.5):
      cMap.append('k')
    else:
      cMap.append('k')

print(len(cMap))

#X = np.genfromtxt("../data/processed/hc_13/T22_4.csv", delimiter=',')[:, (1,2)]
d1 = np.load("../distances/21/1/filtered_1s__distance_matrix_T22_4_1s_20ms.npy")
d2 = np.load("../distances/21/1/filtered_1s__distance_matrix_T26_0_1s_20ms.npy")
d3 = np.load("../distances/21/1/filtered_1s__distance_matrix_T27_6_1s_20ms.npy")
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
dM = np.sqrt(d1**2 + d2**2 + d3**2)

print(dM.shape)

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

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
#ax.plot(coords[:, 0], coords[:, 1], 'o', label="Target neurons", c=cMap)
for i in range(len(cMap)):
    ax.scatter(coords[i, 0]*100, coords[i, 1]*100, color=cMap[i])
plt.title('Diffusion Map Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/21/1/test_dmlib.svg', format="svg")
line1 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10, markerfacecolor="red")
line2 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10,markerfacecolor="green")
line3 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10, markerfacecolor="blue")
line4 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10,markerfacecolor="gray")
plt.legend((line1,line2,line3,line4),('SW','NW', 'NE', 'SE'),numpoints=1,
 bbox_to_anchor=(0.8, 0.22), borderaxespad=0., prop={'size': 17})

plt.savefig('../results/21/1/filtered_dm2d.png')
plt.savefig('../results/21/1/filtered_dm2d.pdf')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0][100:145], 'o-', label="Target neurons")
plt.title('DM lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/filtered_dmev1.png')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1][100:145], 'o-', label="Target neurons")
plt.title('DM lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/21/1/test_dmlib_ev2.svg', format="svg")
plt.savefig('../results/21/1/filtered_dmev2.png')


rng = dM.shape[0]
fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, rng, rng)

ax.plot(coords[:, 0], coords[:, 1], z, 'o-', label='parametric curve', linewidth=0.6, markersize=1)# c = plt.cm.jet(z/max(z)))
ax.legend()

#plt.show()
plt.savefig('../results/21/1/filtered_dm3d.png')