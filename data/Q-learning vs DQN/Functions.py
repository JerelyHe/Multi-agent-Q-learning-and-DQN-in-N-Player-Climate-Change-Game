import numpy as np
import matplotlib.pylab as plt


def mean_c(data):
	col = []
	for i in data:
		row = []
		for j in i:
			row.append(np.mean(j))
		col.append(row)

	return col

def list_average(l):
	acc = None

	for i in l:
		if acc is None:
			acc = np.asarray(i)
		else:
			acc = acc + np.asarray(i)
	return acc/len(l)

def get_col(table, n):
	n_col = []
	for row in table:
		n_col.append(row[n])
	return n_col

def read_q_dqn_F7M4():
	file = open("q dqn F7M4")
	q_action = []
	dqn_action = []
	q_acc = []
	dqn_acc = []

	for i in range(30):
		actions = eval(file.readline())
		accs = eval(file.readline())

		q_action.append(actions[:10])
		dqn_action.append(actions[10:])

		q_acc.append(accs[:10])
		dqn_acc.append(accs[10:])


	file.close()
	return q_action, dqn_action, q_acc, dqn_acc

def read_q_dqn_F3M2():
	file = open("q dqn F3M2")
	q_action = []
	dqn_action = []
	q_acc = []
	dqn_acc = []

	for i in range(30):
		actions = eval(file.readline())
		accs = eval(file.readline())

		q_action.append(actions[:10])
		dqn_action.append(actions[10:])

		q_acc.append(accs[:10])
		dqn_acc.append(accs[10:])


	file.close()
	return q_action, dqn_action, q_acc, dqn_acc


def c_episode(q_action, dqn_action):
	plt.plot(range(1,101),to_c_percent(list_average(q_action)), label='Q-learning') 
	plt.plot(range(1,101),to_c_percent(list_average(dqn_action)), label = "DQN")
	plt.xticks([1,25,50,75, 100], [1,25,50,75, 100]) 
	plt.legend(loc="upper left")
	plt.xlabel("Episode", fontsize=15)
	plt.ylabel("Cooperation%", fontsize=15) 

	plt.show()

def reward_episode(q_acc, dqn_acc):
	plt.plot(range(1,101),total_reward(list_average(q_acc)), label = "Q-learning") 
	plt.plot(range(1,101),total_reward(list_average(dqn_acc)), label = "DQN")
	plt.xticks([1,25,50,75, 100], [1,25,50,75, 100]) 
	plt.legend(loc="upper left")
	plt.xlabel("Episode", fontsize=15)
	plt.ylabel("Acc Reward", fontsize=15) 
	plt.show()

def to_c_percent(agent_actions):
	p = [ [0] for i in range(100)]
	for i in range(100):
		for a in agent_actions:
			p[i]+=(1-a[i])
		p[i] = p[i]/len(agent_actions)
	return p

def total_reward(agent_accs):
	total_r = []

	for i in range(100):
		temp = 0
		for a in agent_accs:
			temp += a[i]
		total_r.append(temp)

	return total_r

