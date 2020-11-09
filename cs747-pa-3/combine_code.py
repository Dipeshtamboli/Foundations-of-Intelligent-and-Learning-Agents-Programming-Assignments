import argparse
from time import time
import matplotlib.pyplot as plt
import pdb
import numpy as np
#row-column format is used
# top-left is (r=0,c=0)
start_time = time()
def plot(x,y, title,x_lab, y_lab,savename):
	plt.clf()
	plt.plot(x, y)
	plt.title(title, fontsize=10)
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	plt.grid()
	plt.savefig(savename)
	# plt.show()	
class GridWorld(object):
	"""docstring for GridWorld"""
	def __init__(self, height, width, start, end, wind_vector,num_actions,alpha,epsilon, stochastic):
		super(GridWorld, self).__init__()
		self.stochastic = stochastic
		self.steps_for_to_end = 0
		self.reached_end = 0
		self.epsilon =epsilon
		self.gamma = 1
		self.alpha = alpha
		self.num_actions = num_actions
		self.wind_vector = wind_vector
		self.width = width
		self.height = height
		self.start = start
		self.end = end
		self.r = start[0]
		self.c = start[1]
		# self.Q_tab = np.random.randint(5, size=(self.height, self.width, self.num_actions))
		self.Q_tab = np.zeros((self.height, self.width, self.num_actions))
		self.action_to_id = {"up":0,
						 "right":1,
						 "down":2,
						 "left":3}
		self.id_to_action = {0:"up",
						 1:"right",
						 2:"down",
						 3:"left"}
		if self.num_actions==8:						 
			self.id_to_action = {0:"N",
							  1:"E",
							  2:"S",
							  3:"W",
							  4:"NE",
							  5:"SE",
							  6:"SW",
							  7:"NW"}
		
	def get_next_s_r_8_actions(self, action,i):
		wind_action = self.wind_vector[self.c]
		self.r = self.r + wind_action
		if (self.stochastic) and (wind_action != 0) :
			self.r += np.random.choice([-1,1,0],1)		
		if "N" in action:
			self.r -= 1
		if "S" in action:
			self.r +=1
		if "W" in action:
			self.c -= 1
		if "E" in action:
			self.c += 1
		if self.r < 0:
			self.r=0
		if self.r > (self.height-1): 
			self.r= (self.height-1)
		if self.c < 0: 
			self.c=0
		if self.c > (self.width-1): 
			self.c = (self.width-1)
		reward = -1
		if (self.r == self.end[0] and self.c == self.end[1]):
			self.r = start[0]
			self.c = start[1]
			reward = 10
			self.reached_end += 1

			self.steps_for_eps[i] = self.steps_for_to_end
			self.steps_for_to_end = 0
		next_s = {"r":self.r, "c":self.c}
		return next_s, reward

	def get_next_s_r(self, action,i):
		wind_action = self.wind_vector[self.c]
		self.r = self.r + wind_action
		
		if action == "up":
			self.r -= 1
		elif action == "down":
			self.r +=1
		elif action == "left":
			self.c -= 1
		elif action == "right":
			self.c += 1
		if self.r < 0:
			self.r=0
		if self.r > (self.height-1): 
			self.r= (self.height-1)
		if self.c < 0: 
			self.c=0
		if self.c > (self.width-1): 
			self.c = (self.width-1)
		reward = -1
		if (self.r == self.end[0] and self.c == self.end[1]):
			self.r = start[0]
			self.c = start[1]
			reward = 10
			self.reached_end += 1

			self.steps_for_eps[i] = self.steps_for_to_end
			self.steps_for_to_end = 0
		next_s = {"r":self.r, "c":self.c}
		return next_s, reward


	# def update_Q(self,state,action,reward,next_s):
	# 	next_a = np.argmax(self.Q_tab[next_s["r"],next_s["c"],:])
	# 	Target = reward + self.gamma*self.Q_tab[next_s["r"],next_s["c"],next_a]
	# 	self.Q_tab[state["r"],state["c"],action] = self.Q_tab[state["r"],state["c"],action]*(1-self.alpha) + self.alpha*(Target)

	def sarsa_update(self,state,action,reward,next_s, next_action):
		Target = reward + self.gamma*self.Q_tab[next_s["r"],next_s["c"],next_action]
		self.Q_tab[state["r"],state["c"],action] = self.Q_tab[state["r"],state["c"],action]*(1-self.alpha) + self.alpha*(Target)

	def Q_learning_update(self,state,action,reward,next_s,next_action):
		next_greedy_a = np.argmax(self.Q_tab[next_s["r"],next_s["c"],:])
		Target = reward + self.gamma*self.Q_tab[next_s["r"],next_s["c"],next_greedy_a]
		self.Q_tab[state["r"],state["c"],action] = self.Q_tab[state["r"],state["c"],action]*(1-self.alpha) + self.alpha*(Target)

	def expected_Q_learning_update(self,state,action,reward,next_s,next_action):
		next_greedy_a = np.argmax(self.Q_tab[next_s["r"],next_s["c"],:])
		Target = reward 
		for act in range(self.num_actions):
			Target += self.epsilon/self.num_actions * self.gamma*self.Q_tab[next_s["r"],next_s["c"],act]
		Target += (1-self.epsilon) * self.gamma*self.Q_tab[next_s["r"],next_s["c"],next_greedy_a]
		self.Q_tab[state["r"],state["c"],action] = self.Q_tab[state["r"],state["c"],action]*(1-self.alpha) + self.alpha*(Target)
	def find_path(self, steps, algo):
		self.episodes, self.steps_for_eps = np.zeros((steps)),np.zeros((steps))
		state = {"r":self.r,"c":self.c}
		do_explore = np.random.binomial(1, self.epsilon)
		if do_explore:
			current_action = np.random.randint(self.num_actions)
		else:
			current_action = np.argmax(self.Q_tab[self.r,self.c,:])
		for i in range(steps):
			if self.num_actions == 4:
				next_s, reward = self.get_next_s_r(self.id_to_action[current_action],i)
			elif self.num_actions == 8:
				next_s, reward = self.get_next_s_r_8_actions(self.id_to_action[current_action],i)			
			# next_s, reward = self.get_next_s_r(self.id_to_action[current_action],i)
			do_explore = np.random.binomial(1, self.epsilon)
			if do_explore:
				next_action = np.random.randint(self.num_actions)
			else:
				next_action = np.argmax(self.Q_tab[next_s["r"],next_s["c"],:])
			if algo=="sarsa":
				self.sarsa_update(state,current_action,reward,next_s,next_action)
			elif algo=="q_learning":
				self.Q_learning_update(state,current_action,reward,next_s,next_action)
			elif algo=="exp_q_learn":
				self.expected_Q_learning_update(state,current_action,reward,next_s,next_action)
			self.episodes[i] = self.reached_end
			self.steps_for_to_end += 1
			state = next_s
			current_action = next_action
		return self.episodes, self.steps_for_eps

		# for i in range(steps):
		# 	state = {"r":self.r,"c":self.c}
		# 	do_explore = np.random.binomial(1, self.epsilon)
		# 	if do_explore:
		# 		current_action = np.random.randint(self.num_actions)
		# 	else:
		# 		current_action = np.argmax(self.Q_tab[self.r,self.c,:])
			# print(self.id_to_action[current_action])
			# if self.num_actions == 4:
			# 	next_s, reward = self.get_next_s_r(self.id_to_action[current_action],i)
			# elif self.num_actions == 8:
			# 	next_s, reward = self.get_next_s_r_8_actions(self.id_to_action[current_action],i)
			# print(state, current_action)
			# self.update_Q(state,current_action,reward,next_s)
			# self.episodes[i] = self.reached_end
			# self.steps_for_to_end += 1
			# # steps_for_eps[i] = 
			# if ((i+1) % 1000 == 0):
			# 	print(i+1, self.reached_end)

		# return self.episodes, self.steps_for_eps




	def state_r_c_to_num(self, r, c):
		return r*self.width + c 


	def get_current_status(self):
		print(f"current (row, column): ({self.r},{self.c})")
		# print(f"current column {self.c}")
		# print(f"current matrix {self.width}x{self.height}")



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Windy Gridworld, Parse #Actions, algorithm, alpha, epsilon, horizon_steps, stochasticity.')
	parser.add_argument('--num_actions', default=4, help='Number of available actions(up, down, east, west, etc.).')
	parser.add_argument('--steps', default=8000, help='Number of steps(horizon).')
	parser.add_argument('--alpha', default=0.5, help='Learning rate')
	parser.add_argument('--epsilon', default=0.1, help='Exploit vs explore')
	parser.add_argument('--algorithm', default="sarsa", help='q_learning,sarsa,exp_q_learn')
	parser.add_argument('--stochastic', default=False, help='q_learning,sarsa,exp_q_learn')
	args = parser.parse_args()
	start = (3,0)
	end = (3,7)
	height = 7
	width = 10
	wind_vector = -np.array([0,0,0,1,1,1,2,2,1,0])
	stochastic = args.stochastic
	num_actions = int(args.num_actions)
	steps = int(args.steps)
	alpha = float(args.alpha)
	epsilon = float(args.epsilon)
	algo = args.algorithm
	total_eps = 0
	for randomseed in range(1,11):
		np.random.seed(randomseed)	
		world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon,stochastic)
		episodes, steps_for_eps = world.find_path(steps,algo)
		total_eps += episodes
	# plot(x,y, title,x_lab, y_lab,savename)
	episodes = total_eps/10
	plot(range(steps), episodes, f"Episodes against time steps\nactions:{num_actions},alpha:{alpha},eps:{epsilon},max_episodes:{int(episodes[-1])}\nstochastic:{stochastic},algorithm:{algo}","Time steps", "Episodes", f"episodes_vs_time_actions:{num_actions},stochastic:{stochastic},algorithm:{algo}.jpg")
	# plot(range(steps), steps_for_eps, f"Steps taken for completing a episode against time steps;actions{num_actions}","Time steps", "Steps taken for completing a episode", f"episodes_vs_time_alp:{alpha:.2f}_eps:{epsilon:.2f}_score:{np.min(steps_for_eps[np.nonzero(steps_for_eps)])}.jpg")
	print(f'total_time taken: {(time()- start_time)//3600} hrs {(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')
	