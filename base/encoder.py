import numpy as np
import pdb
from time import time
import argparse
# import pulp as p 
start_time = time()

class Encoder:
	def __init__(self, grid):
		self.grid = grid
		# self.age = age
		self.arr_grid = self.read_grid(grid)
		pdb.set_trace()
		self.mdp_file = open(f"mdp_{grid.split('/')[-1]}", "w")
	def write_mdp(self, arr_grid):

		self.mdp_file.write("Now the file has more content!")
		self.mdp_file.close()		

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
		
		print(final_arr)
		return final_arr
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Grid Encoding')
	parser.add_argument('--grid', default="data/maze/grid10.txt", help='path to mdp file.')
	args = parser.parse_args()

	enc = Encoder(args.grid)
	print(f'total_time taken: {(time()- start_time)//3600} hrs \
		{(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')	