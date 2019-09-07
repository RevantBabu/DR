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


#X = np.genfromtxt("../data/processed/hc_13/T22_4.csv", delimiter=',')[:, (1,2)]
d1 = np.load("../distances/21/1/distance_matrix_T12_0_1s_20ms.npy")
d2 = np.load("../distances/21/1/distance_matrix_T15_0_1s_20ms.npy")
d3 = np.load("../distances/21/1/distance_matrix_T19_0_1s_20ms.npy")
# d1 = d1/np.amax(d1)
# d2 = d2/np.amax(d2)
# d3 = d3/np.amax(d3)
dM = np.sqrt(d1**2 + d2**2 + d3**2)

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
#coords = embedding.fit( thresholdMatrix(np.exp(-dM), 10) ).embedding_
#coords = embedding.fit( 1/(dM+0.0001) ).embedding_
#coords = embedding.fit( 1/(dM + 0.01)**2).embedding_
#coords = embedding.fit( np.exp(-dM) ).embedding_
#coords = embedding.fit( np.exp(-dM/4) ).embedding_
#coords = embedding.fit( np.exp(-dM/64) ).embedding_
print(coords.shape)

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0]*100, coords[:, 1]*100, 'o', label="Target neurons")
plt.title('Diffusion Map Dimensions on PFC')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/21/1/test_dmlib.svg', format="svg")
plt.savefig('../results/21/1/pfc/dm2d.png')
plt.savefig('../results/21/1/pfc/dm2d.pdf')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0], 'o-', label="Target neurons")
plt.title('DM lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/pfc/dmev1.png')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1]*100, 'o-', label="Target neurons")
plt.title('Diffusion Map Dimension2 (PFC)')
plt.xlabel('time (s)')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
#plt.savefig('../results/dm/21/1/test_dmlib_ev2.svg', format="svg")
plt.savefig('../results/21/1/pfc/dmev2.png')
plt.savefig('../results/21/1/pfc/dmev2.pdf')


rng = dM.shape[0]
fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, rng, rng)

ax.plot(coords[:, 0], coords[:, 1], z, 'o-', label='parametric curve', linewidth=0.6, markersize=1)# c = plt.cm.jet(z/max(z)))
ax.legend()

#plt.show()
plt.savefig('../results/21/1/pfc/dm3d.png')




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
      cornerIdx1[i] = -2.5
    elif (resX[i]<115 and resY[i]<100):
      cornerIdx2[i] = -2.5
    elif (resX[i]>240 and resY[i]<100):
      cornerIdx3[i] = -2.5
    elif (resX[i]>240 and resY[i]>100):
      cornerIdx4[i] = -2.5
  else:
    ncX.append(resX[i])
    ncY.append(resY[i])


fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 0][400:500], 'o-', label="Target neurons")
ax.plot(np.nonzero(cornerIdx1[400:500])[0], np.full(np.nonzero(cornerIdx1[400:500])[0].shape, -5), 'o', color="green")
ax.plot(np.nonzero(cornerIdx2[400:500])[0], np.full(np.nonzero(cornerIdx2[400:500])[0].shape, -5), 'o', color="red")
ax.plot(np.nonzero(cornerIdx3[400:500])[0], np.full(np.nonzero(cornerIdx3[400:500])[0].shape, -5), 'o', color="black")
ax.plot(np.nonzero(cornerIdx4[400:500])[0], np.full(np.nonzero(cornerIdx4[400:500])[0].shape, -5), 'o', color="blue")
plt.title('DM lib Dimensions')
plt.xlabel('dimension1')
plt.ylabel('dimension2')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/pfc/dmev1_overlap_200.png')

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(coords[:, 1][400:500]*100, 'o-', label="Target neurons")
ax.plot(np.nonzero(cornerIdx1[400:500])[0], np.full(np.nonzero(cornerIdx1[400:500])[0].shape, -2.5), 'o', color="green")
ax.plot(np.nonzero(cornerIdx2[400:500])[0], np.full(np.nonzero(cornerIdx2[400:500])[0].shape, -2.5), 'o', color="red")
ax.plot(np.nonzero(cornerIdx3[400:500])[0], np.full(np.nonzero(cornerIdx3[400:500])[0].shape, -2.5), 'o', color="black")
ax.plot(np.nonzero(cornerIdx4[400:500])[0], np.full(np.nonzero(cornerIdx4[400:500])[0].shape, -2.5), 'o', color="blue")
plt.title('Diffusion Map Dimension2')
plt.xlabel('time (s)')
plt.ylabel('dimension2')
ax.yaxis.set_label_coords(-0.08,0.5)
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
line1 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10, markerfacecolor="red")
line2 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10,markerfacecolor="green")
line3 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10, markerfacecolor="blue")
line4 = mlines.Line2D(range(1), range(1), color="white", marker='o',markersize=10,markerfacecolor="gray")
plt.legend((line1,line2,line3,line4),('SW','NW', 'NE', 'SE'),numpoints=1,
 bbox_to_anchor=(0.8, 0.9), borderaxespad=0., prop={'size': 17})
plt.savefig('../results/21/1/pfc/dmev2_overlap_200.png')
plt.savefig('../results/21/1/pfc/dmev2_overlap_200.pdf')