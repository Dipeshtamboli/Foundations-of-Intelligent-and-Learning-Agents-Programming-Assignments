import argparse
from all_functions import *

parser = argparse.ArgumentParser(description='Windy Gridworld, Parse #Actions, algorithm, alpha, epsilon, horizon_steps, stochasticity.')
parser.add_argument('--num_actions', default=4, help='Number of available actions(up, down, east, west, etc.).')
parser.add_argument('--steps', default=10000, help='Number of steps(horizon).')
parser.add_argument('--alpha', default=0.6, help='Learning rate')
parser.add_argument('--epsilon', default=0.05, help='Exploit vs explore')
parser.add_argument('--algorithm', default="sarsa", help='q_learning,sarsa,exp_sarsa')
parser.add_argument('--stochastic', default=False, help='stochasticity: True or False')
parser.add_argument('--total_seeds', default=10, help='Number of seeds')
parser.add_argument('--plot_save_name', default="", help='plot save directory/name')
parser.add_argument('--task_5', default=False, help='Special treatment to task 5')
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
task_5 = args.task_5

save_name = save_name if save_name != '' else f"plots/alpha:{alpha},eps:{epsilon}_task5_seed:{total_seeds}.jpg"

total_eps = 0
for randomseed in range(1,total_seeds+1):
	np.random.seed(randomseed)	
	world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon,stochastic)
	episodes, steps_for_eps = world.find_path(steps,algo)
	total_eps += episodes
# plot(x,y, title,x_lab, y_lab,savename)
episodes = total_eps/total_seeds
plot(range(steps), episodes, f"Episodes against time steps\nactions:{num_actions},alpha:{alpha},eps:{epsilon},max_episodes:{int(episodes[-1])}\nstochastic:{stochastic},algorithm:{algo}","Time steps", "Episodes", save_name)

if algo=="all":
	plot_dicts = {}
	for algo in ["q_learning","sarsa","exp_sarsa"]:
		for randomseed in range(1,total_seeds+1):
			np.random.seed(randomseed)
			world = GridWorld(height,width,start,end, wind_vector,num_actions,alpha,epsilon,stochastic)
			episodes, steps_for_eps = world.find_path(steps, algo)		
			if randomseed == 1:
				plot_dicts[algo] = episodes
			plot_dicts[algo] += episodes
			if randomseed == total_seeds:
				plot_dicts[algo] /= total_seeds

	plt.clf()
	x = range(steps)
	plt.plot(x, plot_dicts["q_learning"], label='Q_Learning')
	plt.plot(x, plot_dicts["sarsa"], label='Sarsa')
	plt.plot(x, plot_dicts["exp_sarsa"], label='Expected_Sarsa')
	plt.title(f"GridWorld with three agents\nalpha:{alpha},eps:{epsilon}", fontsize=15)
	plt.xlabel("Time steps", fontsize=15)
	plt.ylabel("Episodes against time steps", fontsize=15)
	plt.legend(loc=2, prop={'size': 10})
	plt.grid()
	plt.savefig(save_name)
	# print(f'total_time taken: {(time()- start_time)//3600} hrs {(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')
