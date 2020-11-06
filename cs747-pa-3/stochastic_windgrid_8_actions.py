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
	plt.title(title)
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	plt.grid()
	plt.savefig(savename)
	# plt.show()	
class GridWorld(object):
	"""docstring for GridWorld"""
	def __init__(self, height, width, start, end, wind_vector,num_actions,alpha,epsilon):
		super(GridWorld, self).__init__()
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
		# pdb.set_trace()
		# print(self.Q_tab.shape)
		# self.arg = arg
		
	def get_next_s_r_8_actions(self, action,i):
		wind_action = self.wind_vector[self.c]
		self.r = self.r + wind_action
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


	def update_Q(self,state,action,reward,next_s):
		# pdb.set_trace()
		# do_explore = np.random.binomial(1, self.epsilon)
		next_a = np.argmax(self.Q_tab[next_s["r"],next_s["c"],:])
		Target = reward + self.gamma*self.Q_tab[next_s["r"],next_s["c"],next_a]
		self.Q_tab[state["r"],state["c"],action] = self.Q_tab[state["r"],state["c"],action]*(1-self.alpha) + self.alpha*(Target)

	def find_path(self, steps):

		self.episodes, self.steps_for_eps = np.zeros((steps)),np.zeros((steps))
		for i in range(steps):
			state = {"r":self.r,"c":self.c}
			do_explore = np.random.binomial(1, self.epsilon)
			if do_explore:
				current_action = np.random.randint(self.num_actions)
			else:
				current_action = np.argmax(self.Q_tab[self.r,self.c,:])
			# print(self.id_to_action[current_action])
			if self.num_actions == 4:
				next_s, reward = self.get_next_s_r(self.id_to_action[current_action],i)
			elif self.num_actions == 8:
				next_s, reward = self.get_next_s_r_8_actions(self.id_to_action[current_action],i)
			# print(state, current_action)
			self.update_Q(state,current_action,reward,next_s)
			self.episodes[i] = self.reached_end
			self.steps_for_to_end += 1
			# # steps_for_eps[i] = 
			# if ((i+1) % 1000 == 0):
			# 	print(i+1, self.reached_end)

		return self.episodes, self.steps_for_eps




	def state_r_c_to_num(self, r, c):
		return r*self.width + c 


	def get_current_status(self):
		print(f"current (row, column): ({self.r},{self.c})")
		# print(f"current column {self.c}")
		# print(f"current matrix {self.width}x{self.height}")



if __name__ == '__main__':
	start = (3,0)
	end = (3,7)
	height = 7
	width = 10
	wind_vector = -np.array([0,0,0,1,1,1,2,2,1,0])
	num_actions = 4
	steps = 10000
	alpha = 0.5
	epsilon = 0.05
	world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon)
	# pdb.set_trace()
	episodes, steps_for_eps = world.find_path(steps)
	# plot(x,y, title,x_lab, y_lab,savename)
	plot(range(steps), episodes, f"Episodes against time steps;actions{num_actions}","Time steps", "Episodes", f"episodes_vs_time_alp:{alpha:.2f}_eps:{epsilon:.2f}_score:{episodes[-1]}.jpg")
	plot(range(steps), steps_for_eps, f"Steps taken for completing a episode against time steps;actions{num_actions}","Time steps", "Steps taken for completing a episode", f"episodes_vs_time_alp:{alpha:.2f}_eps:{epsilon:.2f}_score:{np.min(steps_for_eps[np.nonzero(steps_for_eps)])}.jpg")

	# for alpha in np.arange(0,1,0.1):
	# 	for epsilon in np.arange(0,0.1,0.01):
	# 		world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon)
	# 		episodes, steps_for_eps = world.find_path(steps)
	# 		# plot(x,y, title,x_lab, y_lab,savename)
	# 		plot(range(steps), episodes, "Episodes against time steps","Time steps", "Episodes", f"eps_plots/episodes_vs_time_alp:{alpha:.2f}_eps:{epsilon:.2f}_score:{episodes[-1]}.jpg")
	# 		# plot(range(steps), steps_for_eps, "Steps taken for completing a episode against time steps","Time steps", "Steps taken for completing a episode", f"steps_taken_vs_time_alp:{alpha}_eps:{epsilon}.jpg")
	print(f'total_time taken: {(time()- start_time)//3600} hrs {(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')
	