import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=26)

day = sys.argv[1]
epoch = sys.argv[2]
cellNo = sys.argv[3]

df = pd.read_csv("../data/processed/hc_13/" + day + "/" + epoch + "/" + cellNo + ".csv", header=None)
df.columns = ['timestamp', 'xpos', 'ypos']

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.plot(df['xpos'], df['ypos'], 'o', label="Target neuron")
plt.title('spiking-position plot : T27-3')
plt.xlabel('x position (cm)')
plt.ylabel('y position (cm)')
#ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
plt.savefig("../plots/position_plot/" + day + "/" + epoch + "/" + cellNo + ".png")
plt.savefig("../plots/position_plot/" + day + "/" + epoch + "/" + cellNo + ".pdf")
#plt.savefig("position_plot_" + cellNo + ".svg", format="svg")
