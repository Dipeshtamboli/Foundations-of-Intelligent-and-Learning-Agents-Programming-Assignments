import pdb
import numpy as np
import matplotlib.pyplot as plt
'''

	epsilon greedy =>16 mins 
	ucb =>13 mins 
	thompson-sampling =>25 mins 

'''

# regret-vector loading for epsilon-greedy algorithm
# algorithm = "epsilon-greedy"
# /home/dipesh/Desktop/acads/cs747-fila/asmts/cs747-pa1/submission/epsilon-greedy/i:i-1_s:0.npy
regret_dict = {}
for algo in ["epsilon-greedy","ucb","kl-ucb","thompson-sampling"]:
	inst_dict = {}
	for instance in [1,2,3]:
		regret_step_all = np.zeros((102400))
		for seed in range(50):
			if np.load(f'{algo}/i:i-{instance}_s:{seed}.npy').shape[0] != 102400:
				pdb.set_trace()
			regret_step_all += np.load(f'{algo}/i:i-{instance}_s:{seed}.npy')
		regret_step_all /= 50
		inst_dict[instance] = regret_step_all
	regret_dict[algo] = inst_dict

scatter_dict = {}
for algo in regret_dict:
	inst_scat = {}

	for inst in regret_dict[algo]:
		# print(inst)
		scatter_reg = np.zeros((6))
		for id, iter_i in enumerate([100, 400, 1600, 6400, 25600, 102400]):
			# print(regret_dict[algo][instance][iter_i-1], algo, instance, iter_i)
			scatter_reg[id] = regret_dict[algo][inst][iter_i-1]
		# print(scatter_reg)
		inst_scat[inst] = scatter_reg
	scatter_dict[algo] = inst_scat

pdb.set_trace()
x_axis = [100, 400, 1600, 6400, 25600, 102400]
# pdb.set_trace()
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 10)) #, dpi=100
ax1.plot(x_axis,scatter_dict['epsilon-greedy'][1], label='epsilon-greedy')
ax1.plot(x_axis,scatter_dict['ucb'][1], label='ucb')
ax1.plot(x_axis,scatter_dict['kl-ucb'][1], label='kl-ucb')
ax1.plot(x_axis,scatter_dict['thompson-sampling'][1], label='thompson-sampling')

ax2.plot(x_axis,scatter_dict['epsilon-greedy'][2], label='epsilon-greedy')
ax2.plot(x_axis,scatter_dict['ucb'][2], label='ucb')
ax2.plot(x_axis,scatter_dict['kl-ucb'][2], label='kl-ucb')
ax2.plot(x_axis,scatter_dict['thompson-sampling'][2], label='thompson-sampling')

ax3.plot(x_axis,scatter_dict['epsilon-greedy'][3], label='epsilon-greedy')
ax3.plot(x_axis,scatter_dict['ucb'][3], label='ucb')
ax3.plot(x_axis,scatter_dict['kl-ucb'][3], label='kl-ucb')
ax3.plot(x_axis,scatter_dict['thompson-sampling'][3], label='thompson-sampling')

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

ax1.set_title("Instance i-1", fontsize=20)
ax2.set_title("Instance i-2", fontsize=20)
ax3.set_title("Instance i-3", fontsize=20)

# ax1.rcParams["font.size"] = "150"

ax1.set_xlabel("Horizon", fontsize=20)
ax2.set_xlabel("Horizon", fontsize=20)
ax3.set_xlabel("Horizon", fontsize=20)
ax1.set_ylabel("Average regret over 50 seeds", fontsize=20)
# ax2.set_ylabel("Average regret over 50 seeds")
# ax3.set_ylabel("Average regret over 50 seeds")
ax1.legend(loc=2, prop={'size': 20})
ax2.legend(loc=2, prop={'size': 20})
ax3.legend(loc=2, prop={'size': 20})

ax1.set_xscale('log', basex = 10)
ax2.set_xscale('log', basex = 10)
ax3.set_xscale('log', basex = 10)

fig.savefig(f'test2.jpg')

# plt.figure(2,1)

# plt.plot(range(102400),regret_dict[1])
# plt.plot(range(102400),regret_dict[2])
# plt.plot(range(102400),regret_dict[3])
