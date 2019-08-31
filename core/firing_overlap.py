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


day = int(sys.argv[1])
epoch = int(sys.argv[2])
cellNo = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])


df1 = pd.read_csv("../data/processed/hc_13/" + sys.argv[1] + "/" + sys.argv[2] + "/" + cellNo + ".csv", header=None)
spikes = df1[0].values

n = end-start
fr = np.zeros(n)
for spike in spikes:
  fr[int(spike) - start] += 1

############################ CORNER OVERLAP ###########################



mat = scipy.io.loadmat('../data/raw/hc_13/T1rawpos.mat')
days = mat['rawpos']

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
ax.plot(fr[200:300]*100, 'o-')
ax.plot(np.nonzero(cornerIdx1[200:300])[0], np.full(np.nonzero(cornerIdx1[200:300])[0].shape, -5), 'o', label="Corner1")
ax.plot(np.nonzero(cornerIdx2[200:300])[0], np.full(np.nonzero(cornerIdx2[200:300])[0].shape, -5), 'o', label="Corner2")
ax.plot(np.nonzero(cornerIdx3[200:300])[0], np.full(np.nonzero(cornerIdx3[200:300])[0].shape, -5), 'o', label="Corner3")
ax.plot(np.nonzero(cornerIdx4[200:300])[0], np.full(np.nonzero(cornerIdx4[200:300])[0].shape, -5), 'o', label="Corner4")
plt.title('Isomap Dimension1')
plt.xlabel('time (s)')
plt.ylabel('dimension1 ($10^{-2}$)')
#ax.legend(loc='upper left',  shadow=True, ncol=1)#bbox_to_anchor=(0.75, 1.075),
#plt.savefig('../results/dm/21/1/test_dmlib_ev1.svg', format="svg")
plt.savefig('../results/21/1/fr_overlap_200.png')
plt.savefig('../results/21/1/fr_overlap_200.pdf')