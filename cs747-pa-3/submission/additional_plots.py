from all_functions import *
import argparse
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

parser = argparse.ArgumentParser(description='Windy Gridworld, Parse #Actions, algorithm, alpha, epsilon, horizon_steps, stochasticity.')
parser.add_argument('--num_actions', default=4, help='Number of available actions(up, down, east, west, etc.).')
parser.add_argument('--steps', default=10000, help='Number of steps(horizon).')
parser.add_argument('--alpha', default=0.6, help='Learning rate')
parser.add_argument('--epsilon', default=0.05, help='Exploit vs explore')
parser.add_argument('--algorithm', default="sarsa", help='q_learning,sarsa,exp_sarsa')
parser.add_argument('--stochastic', default=False, help='stochasticity: True or False')
parser.add_argument('--total_seeds', default=10, help='Number of seeds')
parser.add_argument('--plot_save_name', default="", help='plot save directory/name')
parser.add_argument('--what_to_plot', default="episodes_vs_al_epsi", help='episodes_vs_al_epsi, sarsa_task_2-3-4_comparison')


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
total_seeds = args.total_seeds
save_name = args.plot_save_name
what_to_plot = args.what_to_plot


total_eps = 0

if what_to_plot == "sarsa_task_2-3-4_comparison":
	plot_dicts = {}
	algo = "sarsa"
	for task in ["task_2","task_3","task_4"]:
		for randomseed in range(1,total_seeds+1):
			np.random.seed(randomseed)
			if task == "task_2":
				num_actions,stochastic = 4, False
			if task == "task_3":
				num_actions,stochastic = 8, False
			if task == "task_4":
				num_actions,stochastic = 8, True								
			world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon,stochastic)
			episodes, steps_for_eps = world.find_path(steps, algo)		
			if randomseed == 1:
				plot_dicts[task] = episodes
			plot_dicts[task] += episodes
			if randomseed == total_seeds:
				plot_dicts[task] /= total_seeds	
	plt.clf()
	x = range(steps)
	plt.plot(x, plot_dicts["task_2"], label='task_2: actions=4')
	plt.plot(x, plot_dicts["task_3"], label='task_3: actions=8')
	plt.plot(x, plot_dicts["task_4"], label='task_4: actions=8,stochastic')
	plt.title(f"Comparison of 3 tasks\nalpha:{alpha},eps:{epsilon}", fontsize=10)
	plt.xlabel("Time steps", fontsize=10)
	plt.ylabel("Episodes against time steps", fontsize=10)
	plt.legend(loc=2, prop={'size': 10})
	plt.grid()
	save_name = save_name if save_name != '' else f"Comparison_of_3_tasks.jpg"
	plt.savefig(save_name)				

if what_to_plot == "episodes_vs_al_epsi":
	alphas = np.arange(0,1,0.1)
	epsis = np.arange(0,1,0.1)
	al_epsi = np.zeros((len(alphas), len(epsis)))
	for i, alpha in enumerate(alphas):
		for j, epsilon in enumerate(epsis):
			episodes_reached = 0
			for randomseed in range(1,total_seeds+1):
				np.random.seed(randomseed)	
				world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon,stochastic)
				episodes, steps_for_eps = world.find_path(steps,algo)
				total_eps += episodes
				episodes_reached += (episodes[-1])
			episodes = total_eps/total_seeds
			al_epsi[i][j] = episodes_reached/total_seeds

	# np.save(f"al_epsi_{len(alphas)*len(alphas)}points.npy",al_epsi)
	# zs = np.load("al_epsi.npy")
	zs = al_epsi
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	xs, ys = np.meshgrid(alphas, epsis)
	ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
	ax.set_xlabel('Epsilon')
	ax.set_ylabel('Alpha')
	ax.set_zlabel('Episodes reached')
	ax.set_title("Episodes reached in 1000 steos for varying alpha and epsilon")
	save_name = save_name if save_name != '' else f"3d_{len(alphas)*len(epsis)}points.jpg"
	plt.savefig(save_name)
	# pdb.set_trace()