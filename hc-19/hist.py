import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

e = np.load("ephys.npy").item()
b = np.load("behaviour.npy").item()
a =  e['TT11']['spike_times']


n = a.size
d = np.zeros(n)
for i in range(n):
	d[i] = b['linear_position'][int((a[i]-2321.901286)/0.04)]

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(d, 'o', markersize=0.4)
plt.savefig("position_TT11.png")