import sys
import numpy as np
import csv
#import pandas as pd
#import matplotlib.pyplot as plt


def lastCrossSpike(u, v):
  n1 = u.shape[0]
  n2 = v.shape[0]
  result = np.zeros(n1)-1
  
  uIter = 0
  vIter = 0
  while True:
    if u[uIter]>v[vIter]:
      result[uIter] = vIter
      vIter = vIter + 1
    elif v[vIter]>u[uIter]:
      uIter = uIter + 1
    elif v[vIter]==u[uIter]:
      result[uIter] = vIter - 1 #verify the boundary case
      uIter = uIter + 1

    if (uIter==n1 or vIter==n2):
      break
  
  while uIter!=n1:
    result[uIter] = n2-1
    uIter = uIter + 1

  return result

def vanRossumDistanceOpt(u, v, tau):
  n1 = u.shape[0]
  n2 = v.shape[0]

  mU = np.zeros(n1)
  mV = np.zeros(n2)

  for i in range(1, n1):
    mU[i] = (mU[i-1]+1)*np.exp(-(u[i]-u[i-i])/tau)

  for i in range(1, n2):
    mV[i] = (mV[i-1]+1)*np.exp(-(v[i]-v[i-i])/tau)
  
  lU = lastCrossSpike(u, v)
  lV = lastCrossSpike(v, u)

  term1 = np.sum(mU)
  term2 = np.sum(mV)

  term3 = 0
  for i in range(0, n1):
    if lU[i]>=0:
      term3 += (1+mV[lU[i]])*np.exp(-(u[i]-v[lU[i]])/tau)

  term4 = 0
  for i in range(0, n2):
    if lV[i]>=0:
      term4 += (1+mU[lV[i]])*np.exp(-(v[i]-u[lV[i]])/tau)

  return (n1+n2)/2 + term1 + term2 - term3 - term4

def generateDistanceMatrix(n, tau):
  result = np.zeros(shape=(n,n))
  for i in range(0, n):
    print(i)
    for j in range(i+1, n):
      result[i][j] = vanRossumDistanceOpt(np.asarray(n1counts[i] if (i in n1counts) else []), np.asarray(n1counts[j] if (j in n1counts) else []), tau)

  for i in range(0, n):
    for j in range(0, i):
      result[i][j] = result[j][i]

  return result

#args : file_name start_time end_time window
#df1 = pd.read_csv("../data/processed/mj/" + sys.argv[1] + "_filtered.csv", header=None)
#n1 = df1[0].values
n1 = np.genfromtxt("../data/processed/mj/" + sys.argv[1] + "_filtered.csv", delimiter=',')
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
print(d)
