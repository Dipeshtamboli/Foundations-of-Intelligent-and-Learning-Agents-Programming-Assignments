import pdb
import numpy as np
#row-column format is used
# top-left is (r=0,c=0)
class GridWorld(object):
	"""docstring for GridWorld"""
	def __init__(self, height, width, start, end, wind_vector):
		super(GridWorld, self).__init__()
		self.wind_vector = wind_vector
		self.width = width
		self.height = height
		self.start = start
		self.end = end
		self.r = start[0]
		self.c = start[1]
		# self.arg = arg
		
	def get_next_s_r(self, action):
		env = np.zeros((self.width, self.height))
		print(env.shape)
		wind_action = self.wind_vector[self.c]
		self.r = self.r + wind_action
		# if next_r < 0:

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
		self.get_current_status()

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
	print(wind_vector.shape)

	world = GridWorld(height,width,start,end, wind_vector)
	print(world.state_r_c_to_num(1,2))
	world.get_current_status()
	pdb.set_trace()