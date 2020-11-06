import numpy as np
import pdb
from time import time
import argparse
start_time = time()

class Encoder:
	def __init__(self, grid):
		self.grid = grid
		self.arr_grid = self.read_grid(grid)
		self.numStates = (self.arr_grid.shape[0]-2)*(self.arr_grid.shape[1]-2)
		# self.mdp_file = open(f"mdp_{grid.split('/')[-1]}", "w")
		self.discount = 0.9
		self.dim0 = self.arr_grid.shape[0]-2
		self.dim1 = self.arr_grid.shape[1]-2
		self.action_dict = {"0": -self.dim1,
							"1": self.dim0,
							"2": 1,
							"3": -1
		}
		self.write_mdp(self.arr_grid)
	def id_to_st(self, x, y):
		return (x)*(self.arr_grid.shape[0]-2) + y
	def st_to_id(self,state):
		column = state % (self.arr_grid.shape[0]-2)
		row = state / (self.arr_grid.shape[0]-2)
		return int(row),int(column)
	def write_mdp(self, arr_grid):
		line_to_write = "numStates "+str(self.dim0*self.dim1)
		self.write(line_to_write)
		self.write("numActions 4")
		start_st = np.where(arr_grid == 2)
		start_st = (start_st[0][0]-1)*(self.dim0)+start_st[1][0]-1
		self.write(f"start {start_st}")
		end_st = np.where(arr_grid == 3)
		end_st = (end_st[0][0]-1)*(self.dim0)+end_st[1][0]-1
		self.write(f"end {end_st}")
		for x in range(0,self.dim0): #rows
			for y in range(0,self.dim1): #columns
				c_st = self.id_to_st(x,y)
				for act in range(4):
					next_st = c_st + self.action_dict[str(act)]
					x_next, y_next = self.st_to_id(next_st)
					if next_st<0 or next_st>=self.numStates or np.abs(x- x_next)>1 or np.abs(y- y_next)>1:
						reward = -10
						prob = 1
						next_st = c_st						
					elif next_st>=0:
						if arr_grid[x_next+1][y_next+1] == 1:
							reward = -10
							prob = 1
							next_st = c_st
						elif arr_grid[x_next+1][y_next+1] == 0:
							reward = -1
							prob = 1
						elif arr_grid[x_next+1][y_next+1] == 3:
							reward = 10
							prob = 1
					print(line_to_write)
					line_to_write = f"transition {c_st} {act} {next_st} {reward} {prob}"
					self.write(line_to_write)

		self.write(f"mdptype episodic")
		self.write(f"discount  {self.discount}")
		# self.mdp_file.close()		

	def write(self, line):
		print(line)
		# self.mdp_file.write(line+'\n')
	def read_grid(self,grid):
		grid_file = open(grid, "r")
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
		
		print(final_arr)
		return final_arr
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Grid Encoding')
	parser.add_argument('--grid', default="data/maze/grid10.txt", help='path to mdp file.')
	args = parser.parse_args()

	enc = Encoder(args.grid)
	print(f'total_time taken: {(time()- start_time)//3600} hrs \
		{(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')	