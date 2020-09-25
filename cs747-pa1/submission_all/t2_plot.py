import pdb
import numpy as np
import matplotlib.pyplot as plt



regret_dict = {}
for algo in ["thompson-sampling", "thompson-sampling-with-hint"]:
	inst_dict = {}
	for instance in [1,2,3]:
		regret_step_all = np.zeros((102400))
		for seed in range(50):
			# print(f'{algo}/i:i-{instance}_s:{seed}.npy')
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
		print(scatter_reg)
	scatter_dict[algo] = inst_scat

# for algo in ["thompson-sampling-with-hint"]:
# 	inst_dict = {}
# 	for instance in [1,2,3]:
# 		regret_step_all = np.zeros((6))
# 		for seed in range(50):
# 			if np.load(f'{algo}/i:i-{instance}_s:{seed}.npy').shape[0] != 6:
# 				pdb.set_trace()
# 			regret_step_all += np.load(f'{algo}/i:i-{instance}_s:{seed}.npy')
# 		regret_step_all /= 50
# 		inst_dict[instance] = regret_step_all
# 		print(regret_step_all)
# 	scatter_dict[algo] = inst_dict


x_axis = [100, 400, 1600, 6400, 25600, 102400]
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 10)) #, dpi=100
ax1.plot(x_axis,scatter_dict['thompson-sampling'][1], label='thompson-sampling')
ax1.plot(x_axis,scatter_dict['thompson-sampling-with-hint'][1], label='thompson-sampling-with-hint')

ax2.plot(x_axis,scatter_dict['thompson-sampling'][2], label='thompson-sampling')
ax2.plot(x_axis,scatter_dict['thompson-sampling-with-hint'][2], label='thompson-sampling-with-hint')

ax3.plot(x_axis,scatter_dict['thompson-sampling'][3], label='thompson-sampling')
ax3.plot(x_axis,scatter_dict['thompson-sampling-with-hint'][3], label='thompson-sampling-with-hint')

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

ax1.set_title("Instance i-1", fontsize=20)
ax2.set_title("Instance i-2", fontsize=20)
ax3.set_title("Instance i-3", fontsize=20)

ax1.set_xlabel("Horizon", fontsize=20)
ax2.set_xlabel("Horizon", fontsize=20)
ax3.set_xlabel("Horizon", fontsize=20)
ax1.set_ylabel("Average regret over 50 seeds", fontsize=20)
ax1.legend(loc=2, prop={'size': 20})
ax2.legend(loc=2, prop={'size': 20})
ax3.legend(loc=2, prop={'size': 20})

ax1.set_xscale('log', basex = 10)
ax2.set_xscale('log', basex = 10)
ax3.set_xscale('log', basex = 10)

fig.savefig(f'test__T2_new_2_2.jpg')