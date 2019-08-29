import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

e = np.load("sleep_ephys.npy").item()
a =  e['TT10']['spike_times']


n = 1290
d = np.zeros(n)
for spike in a:
	d[int(spike) - 3333] += 1

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.bar(range(n), d)
plt.savefig("dist_TT10.png")