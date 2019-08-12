import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cellNo = sys.argv[1]

df = pd.read_csv("../data/processed/hc_13/" + cellNo + ".csv", header=None)
df.columns = ['timestamp', 'xpos', 'ypos']

fig = plt.figure(figsize=(9,9))
ax = plt.subplot(111)
ax.plot(df['xpos'], df['ypos'], 'o', label="Position")
plt.title('Position plot')
plt.xlabel('x position')
plt.ylabel('y position')
ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1.075), shadow=True, ncol=1)
# plt.savefig("position_plot_" + cellNo + ".png")
plt.savefig("position_plot_" + cellNo + ".svg", format="svg")
