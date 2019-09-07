import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sys
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.manifold import SpectralEmbedding
from scipy import interpolate

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)


d1 = np.load("../distances/21/1/filtered_distance_matrix_T22_4_1s_20ms.npy")
d2 = np.load("../distances/21/1/filtered_distance_matrix_T26_0_1s_20ms.npy")
d3 = np.load("../distances/21/1/filtered_distance_matrix_T27_6_1s_20ms.npy")
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

# Number of samplepoints
N = 805
# sample spacing
T = 1.0
x = np.linspace(0.0, N*T, N)
y = coords[:, 1]*1000

f1 = interpolate.interp1d(x, y, kind='linear')
xnew = np.linspace(0.0, N*T, N*16)
ynew = f1(xnew)

# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

N = N*16

yf = scipy.fftpack.fft(ynew)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
#plt.show()
plt.savefig('../results/21/1/fft_ev2_250ms_filt.pdf')
plt.savefig('../results/21/1/fft_ev2_250ms_filt.png')