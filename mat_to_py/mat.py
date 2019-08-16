import scipy.io

mat = scipy.io.loadmat('../data/raw/hc_13/T1spikes.mat')
tetInfo = scipy.io.loadmat('../data/raw/hc_13/T1tetinfo.mat')

days = mat['spikes']

for day in range(days.size):
	if days[0][day].size!=0:
		epochs = days[0][day]
		for epoch in range(epochs.size):
			if epochs[0][epoch].size!=0:
				print("------------------------------------------")
				tetrodes = epochs[0][epoch]
				for tetrode in range(tetrodes.size):
					if tetrodes[0][tetrode].size!=0:
						cells = tetrodes[0][tetrode]
						for cell in range(cells.size):
							if cells[0][cell].size!=0:
								area = 'NA'
								if tetInfo['tetinfo'][0][day][0][epoch][0][tetrode].size!=0:
									area = tetInfo['tetinfo'][0][day][0][epoch][0][tetrode][0][0]['area'][0]
								if area!='PFC':
									start = cells[0][cell][0][0]['timerange'][0][0]/10000
									end = cells[0][cell][0][0]['timerange'][0][1]/10000
									timerange = (cells[0][cell][0][0]['timerange'][0][1]-cells[0][cell][0][0]['timerange'][0][0])/10000
									numspikes = cells[0][cell][0][0]['data'].shape[0]
									print(day, "-", epoch, "-", tetrode, "-", cell, "-", area," : ", start, "-" , end, "-", timerange, "-", numspikes, "-", int(numspikes/timerange))


