import scipy.io
import sys
import numpy as np

mat = scipy.io.loadmat('../data/raw/hc_13/T1spikes.mat')

day = int(sys.argv[1])
epoch = int(sys.argv[2])
tetrode = int(sys.argv[3])
cell = int(sys.argv[4])

pos = mat['spikes'][0][day][0][epoch][0][tetrode][0][cell][0][0]['data'][:, (0,1,2)]

pos[:,0] = np.around(pos[:,0], decimals=4)
pos[:,1] = np.around(pos[:,1], decimals=2)
pos[:,2] = np.around(pos[:,2], decimals=2)

np.savetxt('../data/processed/hc_13/' + str(day) + '/' + str(epoch) + '/T' + str(tetrode) + '_' + str(cell) +  '.csv', pos, delimiter=",", fmt="%.4f")
