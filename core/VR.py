import sys
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt


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


def generateDistanceMatrix(n, tau):
  result = np.zeros(shape=(n,n))
  ncounts = {}

  for i in range(0, n):
    ncounts[i] = np.asarray(n1counts[i] if (i in n1counts) else [])

  for i in range(0, n):
    print(i)
    for j in range(i+1, n):
      result[i][j] = vanRossumDistance(ncounts[i], ncounts[j], tau)
  
  for i in range(0, n):
    for j in range(0, i):
      result[i][j] = result[j][i]

  return np.sqrt(result)

#args : file_name start_time end_time window
#df1 = pd.read_csv("../data/processed/hc_13/" + sys.argv[1] + ".csv", header=None)
#n1 = df1[0].values
n1 = np.genfromtxt("../data/processed/hc_13/" + sys.argv[1] + ".csv", delimiter=',')[:, 0]
start = int(sys.argv[2])
end = int(sys.argv[3])
slots = end-start
n1counts = {}

for spike_time in n1:
	key = int(spike_time-start)
	if (key in n1counts):
		n1counts[key].append(int((spike_time-int(spike_time))*1000))
	else:
		n1counts[key] = [int((spike_time-int(spike_time))*1000)]

d = generateDistanceMatrix(slots, int(sys.argv[4]))

np.save("distance_matrix_" + sys.argv[1] + "_1s_" + sys.argv[4] + "ms.npy", d)
#print(d.shape)
