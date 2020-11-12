import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pylab as plt


def read_data(name):
	file = open(name)
	data = eval(file.read())
	file.close()
	acc = None

	
	for i in data:
		if acc is None:
			acc = np.asarray(i)
		else:
			acc = acc + np.asarray(i)
	return acc/len(data)

def mean_c(data):
	col = []
	for i in data:
		row = []
		for j in i:
			row.append(np.mean(j))
		col.append(row)

	return col

def split_data(name):
	file = open(name)
	data = eval(file.read())
	file.close()


	acc1 = None
	acc2 = None

	for i in data:
		if acc1 is None:
			acc1 = np.asarray(get_col(i,0))
			acc2 = np.asarray(get_col(i,1))
		else:
			acc1 = acc1 + np.asarray(get_col(i,0))
			acc2 = acc2 + np.asarray(get_col(i,1))
	return acc1/len(data), acc2/len(data)

def get_col(table, n):
	n_col = []
	for row in table:
		n_col.append(row[n])
	return n_col

def read_group():
	file = open("group")
	acc1 = None
	acc2 = None

	for i in range(30):
		actions = file.readline()
		times = file.readline()
		if acc1 is None:
			acc1 = np.asarray(eval(actions))
			acc2 = np.asarray(eval(times))
		else:
			acc1 = acc1 + np.asarray(eval(actions))
			acc2 = acc2 + np.asarray(eval(times))

	file.close()
	return [np.mean(i) for i in acc1/30], list(acc2/30)

def stat_t_group():
	file = open("group")

	actions = []
	times = []
	for i in range(30):
		a = eval(file.readline())
		t = eval(file.readline())

		actions.append([np.mean(j) for j in a])
		times.append(t)

	file.close()

	
	temp = [get_col(actions,i) for i in range(5)]
	print("Mean C%")
	print(stats.kruskal(temp[0],temp[1],temp[2],temp[3],temp[4]))

	temp = [get_col(times,i) for i in range(5)]
	print("Time")
	print(stats.kruskal(temp[0],temp[1],temp[2],temp[3],temp[4]))
	
def heat_map_fm():
	file = open("fm")

	acc=None

	for i in range(30):
		if acc is None:
			acc = np.asarray(eval(file.readline()))
		else:
			acc = acc + np.asarray(eval(file.readline()))
		
	file.close()

	data = mean_c(acc/30)
	fig,ax = plt.subplots(figsize = (8,8))
	sns.set(font_scale=1.1)
	heatmap = sns.heatmap(data, linewidth=0.5, square=True,annot=True, fmt='.4g', cbar_kws={'label': 'Mean C%'})
	heatmap.set_xticklabels([1,2,3,4,5], fontsize=15) 
	heatmap.set_yticklabels([1, 3, 5, 7, 9], fontsize=15) 
	


	plt.xlabel("M", fontsize=20)
	plt.ylabel("F", fontsize=20) 
	plt.show()

def heat_map_lp():
	file = open("lambda psi")

	acc=None

	for i in range(30):
		if acc is None:
			acc = np.asarray(eval(file.readline()))
		else:
			acc = acc + np.asarray(eval(file.readline()))
		
	file.close()

	data = mean_c(acc/30)
	fig,ax = plt.subplots(figsize = (8,8))
	sns.set(font_scale=1.1)
	heatmap = sns.heatmap(data, linewidth=0.5, square=True,annot=True, fmt='.4g', cbar_kws={'label': 'Mean C%'}, vmin=0.715,vmax=0.75)
	heatmap.set_xticklabels([20, 40, 80, 160, 320], fontsize=15) 
	heatmap.set_yticklabels([25, 50, 100, 200, 400], fontsize=15) 
	

	plt.xlabel("Psi", fontsize=20)
	plt.ylabel("Lambda", fontsize=20) 
	plt.show()


	# return 

def stat_t_fm():
	file = open("fm")

	data = []

	for i in range(30):
		data.append(np.asarray(mean_c(eval(file.readline()))))
	file.close()

	F = [1, 3, 5, 7, 9]
	M = [1,2,3,4,5]

	for col in range(5):
		temp = [[],[],[],[],[]]
		for row in range(5):
			for d in data:
				temp[row].append(d[col][row])
		print("For F = ", F[col])
		print(stats.kruskal(temp[0],temp[1],temp[2],temp[3],temp[4]))

	for row in range(5):
		temp = [[],[],[],[],[]]
		for col in range(5):
			for d in data:
				temp[col].append(d[col][row])
		print("For M = ", M[row])
		print(stats.kruskal(temp[0],temp[1],temp[2],temp[3],temp[4]))

	# return np.column_stack(data), np.row_stack(data)

def stat_t_lp():
	file = open("lambda psi")

	data = []

	for i in range(30):
		data.append(np.asarray(mean_c(eval(file.readline()))))
	file.close()

	LAMBDA = [25, 50, 100, 200, 400]
	PSI = [20, 40, 80, 160, 320]

	for col in range(5):
		temp = [[],[],[],[],[]]
		for row in range(5):
			for d in data:
				temp[row].append(d[col][row])
		print("For Lambda = ", LAMBDA[col])
		print(stats.kruskal(temp[0],temp[1],temp[2],temp[3], temp[4]))

	for row in range(5):
		temp = [[],[],[],[],[]]
		for col in range(5):
			for d in data:
				temp[col].append(d[col][row])
		print("For Psi = ", PSI[row])
		print(stats.kruskal(temp[0],temp[1],temp[2],temp[3],temp[4]))

	# return np.column_stack(data), np.row_stack(data)