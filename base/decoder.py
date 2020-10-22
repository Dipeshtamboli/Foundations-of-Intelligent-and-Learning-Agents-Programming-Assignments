import numpy as np
import pdb
from time import time
import argparse
# import pulp as p 
start_time = time()

class Decoder(object):
	"""docstring for Decoder"""
	def __init__(self, grid, policy):
		super(Decoder, self).__init__()
		self.grid = grid
		self.policy_file = policy
		self.arr_grid = self.read_grid(grid)
		self.numStates = (self.arr_grid.shape[0]-2)*(self.arr_grid.shape[1]-2)
		self.v_star = np.zeros((self.numStates))
		self.pi_star = np.zeros((self.numStates), dtype = int)
		self.dir_dict = {'N':[-1,0], 'E':[0,1],'W':[0,-1],'S':[1,0]}
		self.act_dict = {0:'N', 2:'E',3:'W',1:'S'}
		self.read_policy()
		start_st = np.where(self.arr_grid == 2)
		self.start_st = (start_st[0][0]-1)*(self.arr_grid.shape[0]-2)+start_st[1][0]-1		
		end_st = np.where(self.arr_grid == 3)
		self.end_st = (end_st[0][0]-1)*(self.arr_grid.shape[0]-2)+end_st[1][0]-1		
		self.plan_path()
	def plan_path(self):
		# print(self.start_st, self.end_st)
		# print(self.st_to_id(self.start_st),self.st_to_id(self.end_st))
		cur_st = self.start_st
		while(cur_st != self.end_st):
			action = (self.pi_star[cur_st])
			print(self.act_dict[action], end =" ")
			x,y = self.st_to_id(cur_st)
			next_st = self.dir_dict[self.act_dict[action]]
			next_st = self.id_to_st(x+next_st[0], y+next_st[1])
			cur_st = next_st
			# print(x,y)
			# pdb.set_trace()
	def id_to_st(self, x, y):
		return (x)*(self.arr_grid.shape[0]-2) + y
	def st_to_id(self,state):
		column = state % (self.arr_grid.shape[0]-2)
		row = state / (self.arr_grid.shape[0]-2)
		return int(row),int(column)		
	def read_policy(self):
		policy_file = open(self.policy_file, "r")
		for it_id,line in enumerate(policy_file):
			v_and_pi = line.strip().split(" ")
			self.v_star[it_id] = v_and_pi[0]
			self.pi_star[it_id] = v_and_pi[1]
		# print(self.v_star)
		# print(self.pi_star)
	def read_grid(self,grid):
		grid_file = open(grid, "r")
		# print(grid_file)
		first_time = True
		for line in grid_file:
			nums = line.strip().split(" ")
			arr = np.array(nums)
			arr = arr.astype(np.int)
			if first_time:
				final_arr = arr
				final_arr = np.expand_dims(final_arr,0)
				first_time = False
			else:
				final_arr = np.concatenate((final_arr,np.expand_dims(arr,0)), axis=0)
		
		# print(final_arr)
		return final_arr
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Grid Decoding')
	parser.add_argument('--grid', default="data/maze/grid10.txt", help='path to grid file.')
	parser.add_argument('--value_policy', default="policy", help='path to the policy file.')
	args = parser.parse_args()

	dec = Decoder(args.grid, args.value_policy)
	# print(f'total_time taken: {(time()- start_time)//3600} hrs \
	# 	{(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')	