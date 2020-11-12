
import numpy as np

from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf


# state is modeled as the opponents played against


class QAgent(object):
	"""docstring for ClassName"""
	def __init__(self, id):
		# print("agent init")
		self.id = id
		self.q_table = QTable()

		self.alpha = 0.1

		self.action = 0
		self.acc = 0
		self.num_c = 0
		self.gemma = 0.9

		# from_state
		self.s0 = []
		# to_state
		self.s1 = []
	

	def choose_action(self, epsilon, group):

		if np.random.uniform(0, 1) < (epsilon):
			[q_c, q_d] = self.q_table.get_q_list(self.s0)
			self.action = int(q_d > q_c)
		else:
			self.action = np.random.randint(2)

		return self.action

	def update_reward(self, reward):	
		if self.action == 0:
			self.num_c += 1
		self.acc += reward

	def update_model(self, reward, group, public_good, climate_loss, episode):
		self.s1 = self.make_state(group, public_good, climate_loss)

		# skip if 1st episode, as s0 == null
		if episode != 0:
			q_s0 = self.q_table.get_q_list(self.s0)
			q_s1 = self.q_table.get_q_list(self.s1)
			q_new = q_s0[self.action] * (1 - self.alpha)
			q_new += self.alpha * (reward + self.gemma*max(q_s1))
			self.q_table.set_q_value(q_new, self.s0, self.action)
		
		self.s0 = self.s1

	def make_state(self, group, public_good, climate_loss):
		state = [self.action, public_good, climate_loss]
		for agent in group:
			if agent != self:
				state.append(agent.action)

		return state

	def getId(self):
		return self.id

	def print(self):
		print("Agent: ", self.id)
		print("Fitness: ", self.acc)
		print("", self.qTable)


class QTable(object):
	"""docstring for Q"""
	def __init__(self):
		# size of action space
		self.a_size = 2
		# state space
		self.q_states = []
		# q_value list for each state
		self.q_value_lists = []

	def get_q_list(self, state):
		if state not in self.q_states:
			self.q_states.append(state)
			self.q_value_lists.append(np.random.rand(self.a_size) * 0.01)

		i = self.q_states.index(state)
		return self.q_value_lists[i]

	def set_q_value(self, q_new, state, action):
		i = self.q_states.index(state)
		self.q_value_lists[i][action] = q_new


class DQNAgent(object):
	"""docstring for ClassName"""
	def __init__(self, id, group_size, num_hidden_layers, memory_size, batch_size):
		# print("agent init")
		self.id = id

		input_size = group_size*3+2
		output_size = 2
		hidden_size = int((group_size*3+4)/2)

		self.policy_net = Net(input_size, output_size, hidden_size)
		self.target_net = Net(input_size, output_size, hidden_size)
		self.target_net.set_params(self.policy_net.get_params())
		
		self.memory = ReplayMemory(memory_size)
		self.batch_size = batch_size

		self.alpha = 0.1
		self.gemma = 0.9
		self.action = 0
		self.acc = 0
		self.num_c = 0

		# from_state
		self.s0 = []
		# to_state
		self.s1 = []

		self.target_update_counter = 0
		self.UPDATE_TARGET_EVERY = batch_size/5


	def choose_action(self, epsilon, group):
		if np.random.uniform(0, 1) < epsilon:
			[q_c, q_d] = self.policy_net.forward(self.s0)

			self.action = int(q_d > q_c)

		else:
			self.action = np.random.randint(2)

		# print(self.action)
		return self.action

	def update_reward(self, reward):
		if self.action == 0:
			self.num_c += 1
		self.acc += reward

	def update_model(self, reward, group, public_good, climate_loss, episode):
		self.s1 = self.make_state(group, public_good, climate_loss)
		
		if episode != 0:
			self.memory.push([self.s0, self.s1, self.action, reward])

		if len(self.memory.memory) >= self.batch_size:
			if self.target_update_counter == self.UPDATE_TARGET_EVERY:
				self.target_net.set_params(self.policy_net.get_params())
				self.target_update_counter = 0

			batch = self.memory.sample(self.batch_size)

			X = []
			y = []

			for [s0, s1, action, reward] in batch:
				X.append(s0)

				q_s0 = self.policy_net.forward(s0)
				q_s1 = self.target_net.forward(s1)
				q_s0[action] = q_s0[action]* (1 - self.alpha)
				q_s0[action] += self.alpha * (reward + self.gemma*max(q_s1))

				y.append(q_s0)

			self.policy_net.train(X, y)
			self.target_update_counter += 1

		self.s0 = self.s1

	def getId(self):
		return self.id

	def make_state(self, group, public_good, climate_loss):
		# training data format: [self_action, agent1_numc%, agent1_fitness, ...]
		state = [self.action, self.num_c, self.acc, public_good, climate_loss]
		for agent in group:
			if agent != self:
				state.append(agent.action)
				state.append(agent.num_c)
				state.append(agent.acc)

		return state


class ReplayMemory(object):
	"""docstring for ReplayMemory"""
	def __init__(self, size):
		self.size = size
		self.memory = []
		self.position = 0

	def push(self, trainsition):
		if len(self.memory) < self.size:
			self.memory.append(None)

		self.memory[self.position] = trainsition
		self.position = (self.position + 1) % self.size

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)
		

class Net(object):
	"""docstring for Net"""
	def __init__(self, input_size, output_size, hidden_size):
		self.model = Sequential()
		self.model.add(keras.Input(shape= input_size))

		for i in range(hidden_size):
			self.model.add(Dense(hidden_size, activation='relu'))

		self.model.add(Dense(output_size, activation='relu'))
		self.model.compile(loss="mse", optimizer=Adam(lr=0.1), metrics=['accuracy'])

	def forward(self, state):
		return self.model.predict(tf.convert_to_tensor([state]))[0]

	def train(self, X, y):
		self.model.fit(tf.convert_to_tensor(X), tf.convert_to_tensor(y))
	
	def get_params(self):
		return self.model.get_weights()

	def set_params(self, params):
		self.model.set_weights(params)
