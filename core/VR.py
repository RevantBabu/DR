import sys
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt


def vanRossumDistance(u, v, tau1, tau2, tau3):
  componentU1 = 0;
  componentU2 = 0;
  componentU3 = 0;
  for i in range(0, u.shape[0]):
    for j in range(0, u.shape[0]):
      componentU1 += np.exp(-abs(u[i]-u[j])/tau1)
      componentU2 += np.exp(-abs(u[i]-u[j])/tau2)
      componentU3 += np.exp(-abs(u[i]-u[j])/tau3)

  componentV1 = 0;
  componentV2 = 0;
  componentV3 = 0;
  for i in range(0, v.shape[0]):
    for j in range(0, v.shape[0]):
      componentV1 += np.exp(-abs(v[i]-v[j])/tau1)
      componentV2 += np.exp(-abs(v[i]-v[j])/tau2)
      componentV3 += np.exp(-abs(v[i]-v[j])/tau3)

  componentUV1 = 0;
  componentUV2 = 0;
  componentUV3 = 0;
  for i in range(0, u.shape[0]):
    for j in range(0, v.shape[0]):
      componentUV1 += np.exp(-abs(u[i]-v[j])/tau1)
      componentUV2 += np.exp(-abs(u[i]-v[j])/tau2)
      componentUV3 += np.exp(-abs(u[i]-v[j])/tau3)

  return (componentU1 + componentV1 - 2*componentUV1), (componentU2 + componentV2 - 2*componentUV2), (componentU3 + componentV3 - 2*componentUV3)


def generateDistanceMatrix(n, tau1, tau2, tau3):
  result1 = np.zeros(shape=(n,n))
  result2 = np.zeros(shape=(n,n))
  result3 = np.zeros(shape=(n,n))
  for i in range(0, n):
    print(i)
    for j in range(i+1, n):
      result1[i][j], result2[i][j], result3[i][j] = vanRossumDistance(np.asarray(n1counts[i] if (i in n1counts) else []), np.asarray(n1counts[j] if (j in n1counts) else []), tau1, tau2, tau3)
  
  for i in range(0, n):
    for j in range(0, i):
      result1[i][j] = result1[j][i]
      result2[i][j] = result2[j][i]
      result3[i][j] = result3[j][i]

  return np.sqrt(result1), np.sqrt(result2), np.sqrt(result3)

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

d1, d2, d3 = generateDistanceMatrix(slots, int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

np.save("distance_matrix_" + sys.argv[1] + "_1s_" + sys.argv[4] + "ms.npy", d1)
np.save("distance_matrix_" + sys.argv[1] + "_1s_" + sys.argv[5] + "ms.npy", d2)
np.save("distance_matrix_" + sys.argv[1] + "_1s_" + sys.argv[6] + "ms.npy", d3)
#print(d.shape)
