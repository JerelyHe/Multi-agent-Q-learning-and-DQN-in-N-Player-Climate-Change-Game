import numpy as np
import Agent
import random
import time
import csv

class Game(object):
	def __init__(self, group_size, episodes, F = 5, M = 3, ratio=(0,0),
		lamda = 10, loss = 3, risk_function = "linear",
		num_hidden_layer = 1, memory_size = 50, batch_size = 25):

		(num_Q, num_DQN) = ratio
		self.population_size = num_Q + num_DQN
		self.group_size = group_size
		self.episodes = episodes
		self.F = F
		self.M = M

		# threat factor used in calculating the probability of a collective loss
		self.lamda = lamda
		# collective loss after a climate change
		self.loss = loss
		self.risk_function = risk_function


		self.agents = []
		i = 0
		for n in range(num_Q):
			self.agents.append(Agent.QAgent(i))
			i+=1

		for n in range(num_DQN):
			self.agents.append(Agent.DQNAgent(i,group_size, num_hidden_layer,
				memory_size, batch_size))
			i+=1
				

		self.climate_loss = 0
		self.public_good = 0

	def play_game(self):


		polulation_actions = [[]for i in range(self.population_size)]
		polulation_fitness = [[]for i in range(self.population_size)]
		disaster_prob = []


		for e in range(self.episodes):

			random.shuffle(self.agents)
			groups = [ self.agents[i:i+self.group_size] for i in range(0, 
				len(self.agents), self.group_size)]

			group_num_c = get_num_c(self.agents)
			disaster_prob.append(self.calculate_prob(get_num_c(self.agents), len(self.agents)))

			for group in groups:
				# let each agent chooses its action, based their opponents history
				actions = []

				for agent in group:
					agent_action = agent.choose_action(e/self.episodes, group)
					# record the action for this episode
					actions.append(agent_action)

				group_num_c = get_num_c(group)
				rewards = self.calculate_reward(actions, group_num_c)

				# if 1st episode, s0 is not defined
				# update starts from the 2nd 
				for (agent, reward) in zip(group, rewards):
					agent.update_reward(reward)

				for (agent, reward) in zip(group, rewards):
					agent.update_model(reward, group, self.public_good, 
						self.climate_loss, e)
		

			for agent in self.agents:

				polulation_actions[agent.id].append(agent.action)
				polulation_fitness[agent.id].append( agent.acc)


		return (polulation_actions, polulation_fitness, disaster_prob)


	""" reward formula """
	def calculate_reward(self, actions, group_num_c):

		num_c = 0
		for a in actions:
			# agent chooses "D"
			if a == 1:
				continue
			else:
				num_c += 1

		cost = 3

		if num_c < self.M:  
			c_reward = -cost
			d_reward = 0
			self.public_good = 0
		else:
			d_reward = (num_c * self.F * cost / self.group_size )
			c_reward = d_reward - cost
			self.public_good = 1

		if np.random.uniform(0, 1) < self.calculate_prob(group_num_c, len(actions)):
			self.climate_loss = 1
			d_reward -= self.loss
			c_reward -= self.loss
			# print("disaster happens")
		else:
			self.climate_loss = 0

		return [(c_reward, d_reward)[a] for a in actions]

	def calculate_prob(self, group_num_c, N):
		if self.risk_function == "linear":
			return 1 - (group_num_c/N/self.lamda)
		elif self.risk_function == "concave":
			if group_num_c == 0:
				return 1
			return 0.1/(group_num_c/N/self.lamda)
		else:
			if group_num_c == 0:
				return 0
			return -0.1/(group_num_c/N/self.lamda)+1




def get_num_c(group):
	acc = 0
	for agent in group:
		acc += agent.num_c
	return acc


	