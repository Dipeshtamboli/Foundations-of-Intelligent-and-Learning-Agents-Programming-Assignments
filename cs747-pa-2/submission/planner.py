import numpy as np
import pdb
from time import time
import argparse
import pulp as p 
start_time = time()
def cal_v_from_pi(pi,v,numStates,numActions,transition_probs, transition_reward, gamma):
	while True:
		v_new = np.zeros((numStates))
		for state in range(numStates):
			for next_state in range(numStates):
				v_new[state] += transition_probs[state,int(pi[state]),next_state]*(transition_reward[state,int(pi[state]),next_state] + gamma*v[next_state])
		if np.sum(np.abs(v-v_new)) < 1e-10:
			return v_new
		v = v_new
def calculate_Q(v,numStates,numActions,transition_probs, transition_reward, gamma):
	q_s = transition_probs*(transition_reward + gamma*v)
	q_s = np.sum(q_s, 2)
	return q_s
def hpi_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	pi = np.zeros((numStates), dtype = int)
	v = np.zeros((numStates))
	while(1):
		v = cal_v_from_pi(pi,v,numStates,numActions,transition_probs, transition_reward, gamma)
		q_s = calculate_Q(v,numStates,numActions,transition_probs, transition_reward, gamma)
		next_pi = np.argmax(q_s, 1)
		if np.all(next_pi == pi):
			return pi, v
		pi = next_pi

def lp_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	Lp_prob = p.LpProblem('finding_V', p.LpMinimize)  
	state_var =[]
	for i in range(numStates):
		state_var.append(p.LpVariable(f"state_{i}"))
	Lp_prob += np.sum(state_var)
	for state in range(numStates):
		for action in range(numActions):
			sum_over_next_state = 0
			for next_state in range(numStates):
				sum_over_next_state += transition_probs[state,action,next_state]*(transition_reward[state,action,next_state] + gamma*state_var[next_state])
			Lp_prob += sum_over_next_state <= state_var[state]
	status = p.PULP_CBC_CMD(msg=0).solve(Lp_prob)
	v_star = [p.value(state_var[i]) for i in range(numStates)]
	return np.array(v_star)
def vi_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	V = np.zeros((numStates))
	while(1):
		next_V = np.zeros((numStates))
		action_reward = np.zeros((numActions, numStates))
		for a in range(numActions):
			red_sum = 0
			red_sum += transition_probs[:,a,:]*(transition_reward[:,a,:]+ gamma*V[:])
			red_sum = np.sum(red_sum, 1)
			action_reward[a] =red_sum
		next_V = np.max(action_reward, 0)
		if np.sum(np.abs(V-next_V)) < 1e-10:
			return V
		V = next_V

def vi_pi_star(v_star,numStates,numActions,transition_probs, transition_reward, gamma):
	V = np.zeros((numStates))
	action_reward = np.zeros((numActions, numStates))
	for a in range(numActions):
		red_sum = 0
		red_sum += transition_probs[:,a,:]*(transition_reward[:,a,:]+ gamma*v_star[:])
		red_sum = np.sum(red_sum, 1)
		action_reward[a] =red_sum	
	return np.argmax(action_reward, 0)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='MDP Planning')
	parser.add_argument('--mdp', default="data/mdp/episodic-mdp-10-5.txt", help='path to mdp file.')
	parser.add_argument('--algorithm', default="vi", help='one of vi, hpi, and lp')

	args = parser.parse_args()
	mdp = args.mdp    
	algorithm = args.algorithm    
	mdp_file = open(mdp, "r")
	end_st =[]
	for line in mdp_file:
		line_words = line.strip().split(" ")
		if line_words[0] == "numStates":
			numStates = int(line_words[1])
		if line_words[0] == "numActions":
			numActions = int(line_words[1])
			transition_probs = np.zeros((numStates,numActions,numStates))
			transition_reward = np.zeros((numStates,numActions,numStates))
		if line_words[0] == "start":
			start_st = int(line_words[1])
		if line_words[0] == "end":
			for st in range(1, len(line_words)):
				end_st.append(int(line_words[st]))

		if line_words[0] == "transition":
			transition_reward[int(line_words[1]),int(line_words[2]),int(line_words[3])] = float(line_words[4])
			transition_probs[int(line_words[1]),int(line_words[2]),int(line_words[3])] = float(line_words[5])

		if line_words[0] == "mdptype":
			mdptype = (line_words[1])
		if line_words[0] == "discount":
			# Discount has 2 spaces
			gamma = float(line_words[-1])

	if algorithm=='vi':
		v_star = vi_vstar(numStates,numActions,transition_probs, transition_reward, gamma)
	if algorithm=='lp':
		v_star = lp_vstar(numStates,numActions,transition_probs, transition_reward, gamma)
	if algorithm=='hpi':
		pi, v_star = hpi_vstar(numStates,numActions,transition_probs, transition_reward, gamma)
	pi_star = vi_pi_star(v_star,numStates,numActions,transition_probs, transition_reward, gamma)
	for v, a in zip(v_star, pi_star):
		if v==0.0:
			v = 0
		print(f'{round(v, 6):.6f} {a}')

	# print(f'total_time taken: {(time()- start_time)//3600} hrs \
	# 	{(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')