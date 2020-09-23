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
for algo in ["epsilon-greedy","ucb","thompson-sampling"]:
	inst_dict = {}
	for instance in [1,2,3]:
		regret_step_all = np.zeros((102400))
		for seed in range(50):
			regret_step_all += np.load(f'{algo}/i:i-{instance}_s:{seed}.npy')
		regret_step_all /= 50
		inst_dict[instance] = regret_step_all
	regret_dict[algo] = inst_dict

# pdb.set_trace()
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 10), dpi=1000)
ax1.plot(range(102400),regret_dict['epsilon-greedy'][1])
ax1.plot(range(102400),regret_dict['ucb'][1])
ax1.plot(range(102400),regret_dict['thompson-sampling'][1])

ax2.plot(range(102400),regret_dict['epsilon-greedy'][2])
ax2.plot(range(102400),regret_dict['ucb'][2])
ax2.plot(range(102400),regret_dict['thompson-sampling'][2])

ax3.plot(range(102400),regret_dict['epsilon-greedy'][3])
ax3.plot(range(102400),regret_dict['ucb'][3])
ax3.plot(range(102400),regret_dict['thompson-sampling'][3])

ax1.set_xscale('log', basex = 10)
ax2.set_xscale('log', basex = 10)
ax3.set_xscale('log', basex = 10)

fig.savefig(f'test2.jpg')

# plt.figure(2,1)

# plt.plot(range(102400),regret_dict[1])
# plt.plot(range(102400),regret_dict[2])
# plt.plot(range(102400),regret_dict[3])
