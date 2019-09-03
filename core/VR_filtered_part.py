import sys
import scipy.io
import numpy as np

day = int(sys.argv[1])
epoch = int(sys.argv[2])
cellNo = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])


n1 = np.genfromtxt("../data/processed/hc_13/" + sys.argv[1] + "/" + sys.argv[2] + "/" + sys.argv[3] + ".csv", delimiter=',')
if (n1.size!=n1.shape[0]):
	n1 = n1[:, 0]
slots = end-start
n1counts = {}

for spike_time in n1:
  key = int(spike_time-start)
  if key>=600:
    if ((key-600) in n1counts):
      n1counts[key-600].append(int((spike_time-int(spike_time))*1000))
    else:
      n1counts[key-600] = [int((spike_time-int(spike_time))*1000)]

print(len(n1counts.keys()))

nfcounts4 = {}
for i in range(0, slots-600):
  if i in n1counts:
  	s = np.asarray(n1counts[i])
  	s1 = s[s<250]
  	s2 = s[s>=250]
  	s2 = s2[s2<500]
  	s3 = s[s>=500]
  	s3 = s3[s3<750]
  	s4 = s[s>=750]
  	nfcounts4[i*4] = s1
  	nfcounts4[i*4+1] = s2-250
  	nfcounts4[i*4+2] = s3-500
  	nfcounts4[i*4+3] = s4-750
  else:
    nfcounts4[i*4] = np.asarray([])
    nfcounts4[i*4+1] = np.asarray([])
    nfcounts4[i*4+2] = np.asarray([])
    nfcounts4[i*4+3] = np.asarray([])

print(len(nfcounts4.keys()))
print(nfcounts4[0])
print(nfcounts4[1])
print(nfcounts4[2])
print(nfcounts4[3])
print(nfcounts4[4])


def vanRossumDistance(u, v, tau):
  componentU = 0;
  for i in range(0, u.shape[0]):
    for j in range(0, u.shape[0]):
      componentU += np.exp(-abs(u[i]-u[j])/tau)

  componentV = 0;
  for i in range(0, v.shape[0]):
    for j in range(0, v.shape[0]):
      componentV += np.exp(-abs(v[i]-v[j])/tau)

  componentUV = 0;
  for i in range(0, u.shape[0]):
    for j in range(0, v.shape[0]):
      componentUV += np.exp(-abs(u[i]-v[j])/tau)

  return (componentU + componentV - 2*componentUV)


def generateDistanceMatrix(n, tau, n1c):
  result = np.zeros(shape=(n,n))
  ncounts = {}

  for i in range(0, n):
    ncounts[i] = np.asarray(n1c[i] if (i in n1c) else [])

  for i in range(0, n):
    print(i)
    for j in range(i+1, n):
      result[i][j] = vanRossumDistance(ncounts[i], ncounts[j], tau)
  
  for i in range(0, n):
    for j in range(0, i):
      result[i][j] = result[j][i]

  return np.sqrt(result)

#change slots to nfcounts4 len
d = generateDistanceMatrix(len(nfcounts4), 20, nfcounts4)
np.save("../distances/" + sys.argv[1] + "/" + sys.argv[2] + "/" + "part_250ms_distance_matrix_" + sys.argv[3] + "_1s_20ms.npy", d)
