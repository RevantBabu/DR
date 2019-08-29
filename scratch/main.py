import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

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

  return componentU + componentV - 2*componentUV

def lastCrossSpike(u, v):
  n1 = u.shape[0]
  n2 = v.shape[0]
  result = np.full_like(u, -1)
  
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

v = np.array([2,3,5])
u = np.array([1,4,6,7,9])
print("v: ", v)
print("u: ", u)
print("lastCrossSpike J(i) = max(j|u_i > v_j")
print(lastCrossSpike(u,v))
print(lastCrossSpike(v,u))

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

print(vanRossumDistance(u, v, 15))
print(vanRossumDistanceOpt(u, v, 15))

def isiToSpikeTrain(u):
  result = np.zeros(np.sum(u))
  index = 0
  for i in range(0,u.shape[0]):
    index = index+u[i]
    result[index-1] = 1
  return result

def plotVRD(title, st1, st2, fname):
  fig = plt.figure(figsize=(8,8))
  ax = plt.subplot(111)
  ax.plot(st1, np.full_like(st1, 1), 'bo-', markersize=4, label="spike train1")
  ax.plot(st2, np.full_like(st2, 2), 'ko-', markersize=4, label="spike train2")
  plt.title(title +' d = {:.2f}'.format(vanRossumDistance(st1, st2, 15)))
  plt.ylim(0, 3)
  plt.gca().axes.get_yaxis().set_ticklabels([])
  plt.xlabel('time (ms)')
  ax.legend(loc='upper left', shadow=True, ncol=1)
  plt.savefig(fname + ".png")
  plt.savefig(fname + ".pdf")

np.random.seed(42)

isi1 = np.random.poisson(15, 20)
isi2 = np.random.poisson(15, 20)
st1 = np.cumsum(isi1)
st2 = np.cumsum(isi2)

#random spike trains
#plotVRD("Random spike-trains.(N:20) ", st1, st2)

#Same spike trains
#st2 = st1
#plotVRD("Same spike-trains.", st1, st2, "P0")

#Offset by +1
# st2 = st1 + 1
# plotVRD("spike-trains offset by +1.", st1, st2, "P1")

#Offset by +2
# st2 = st1 + 2
# plotVRD("spike-trains offset by +2.", st1, st2, "P2")

#Offset by +4
# st2 = st1 + 4
# plotVRD("spike-trains offset by +4.", st1, st2, "P3")

#Offset by +4
# st2 = st1 - 4
# plotVRD("spike-trains offset by -4.", st1, st2, "P4")

y = np.zeros(61)
for i in range(0,61):
  st2 = st1 + i
  y[i] = vanRossumDistance(st1, st2, 15)

# y = np.zeros(100)
# for i in range(0,100):
#   isi1 = np.random.poisson(15, 20)
#   isi2 = np.random.poisson(15, 20)
#   st1 = np.cumsum(isi1)
#   st2 = np.cumsum(isi2)
#   y[i] = vanRossumDistance(st1, st2, 15)


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(y, 'ko-', markersize=4)
plt.title('Change of VR with offset')
plt.xlabel('Offset between spike trains(ms)')
plt.ylabel('VR distance')
plt.savefig("offset.png")
plt.savefig("offset.pdf")

