import os
import sys
import numpy as np

#path = '..\\data\\processed\\hc_13\\' + sys.argv[1] + '\\' + sys.argv[2] + '\\'
path = '../data/processed/hc_13/' + sys.argv[1] + '/' + sys.argv[2] + '/'
#dpath = '..\\distances\\' + sys.argv[1] + '\\' + sys.argv[2] + '\\'
dpath = '../distances/' + sys.argv[1] + '/' + sys.argv[2] + '/'
start = int(sys.argv[3])
end = int(sys.argv[4])


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


def generateDistanceMatrix(n, tau, n1counts):
  result = np.zeros(shape=(n,n))
  ncounts = {}

  for i in range(0, n):
    ncounts[i] = np.asarray(n1counts[i] if (i in n1counts) else [])

  for i in range(0, n):
    #print(i)
    for j in range(i+1, n):
      result[i][j] = vanRossumDistance(ncounts[i], ncounts[j], tau)
  
  for i in range(0, n):
    for j in range(0, i):
      result[i][j] = result[j][i]

  return np.sqrt(result)


def store_distance_matrix(f):
	n1counts = {}
	print(f)
	print(path + f)
	n1 = np.genfromtxt(path + f, delimiter=',')
	if (n1.size!=n1.shape[0]):
		n1 = np.genfromtxt(path + f, delimiter=',')[:, 0]
	slots = end-start

	for spike_time in n1:
		key = int(spike_time-start)
		if (key in n1counts):
			n1counts[key].append(int((spike_time-int(spike_time))*1000))
		else:
			n1counts[key] = [int((spike_time-int(spike_time))*1000)]

	d = generateDistanceMatrix(slots, 20, n1counts) #20ms fixed for now
	np.save(dpath + f[:-4] + "_1s_20ms.npy", d)


files = []
for r, d, f in os.walk(path):
    for fname in f:
    	print("------starting-------: ", fname[:-4])
    	store_distance_matrix(fname)
    	print("------finished-------: ", fname[:-4])
