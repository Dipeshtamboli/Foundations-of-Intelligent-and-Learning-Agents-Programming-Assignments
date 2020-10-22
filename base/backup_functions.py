def calculate_Q(v,numStates,numActions,transition_probs, transition_reward, gamma):
	q_s = transition_probs*(transition_reward + gamma*v)
	q_s = np.sum(q_s, 2)
	return q_s
def calculate_Q(v,numStates,numActions,transition_probs, transition_reward, gamma):
	q_s = np.zeros((numStates,numActions))
	for next_state in range(numStates):
		q_s += transition_probs[:,:,next_state]*(transition_reward[:,:,next_state] + gamma*v[next_state])
	return q_s
def calculate_Q(v,numStates,numActions,transition_probs, transition_reward, gamma):
	q_s = np.zeros((numStates,numActions))
	for state in range(numStates):
		for action in range(numActions):
			for next_state in range(numStates):
				q_s[state][action] += transition_probs[state,action,next_state]*(transition_reward[state,action,next_state] + gamma*v[next_state])
	return q_s
def hpi_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	pi = np.zeros((numStates), dtype = int)
	v = np.zeros((numStates))
	while(1):
		v = cal_v_from_pi(pi,v,numStates,numActions,transition_probs, transition_reward, gamma)
		q_s = calculate_Q(v,numStates,numActions,transition_probs, transition_reward, gamma)
		# print(v)
		# pdb.set_trace()
		next_pi = np.argmax(q_s, 1)
		if np.all(next_pi == pi):
			return pi, v
		pi = next_pi
def cal_v_from_pi(pi,v,numStates,numActions,transition_probs, transition_reward, gamma):
	while True:
		v_new = np.zeros((numStates))
		for state in range(numStates):
			for next_state in range(numStates):
				v_new[state] += transition_probs[state,int(pi[state]),next_state]*(transition_reward[state,int(pi[state]),next_state] + gamma*v[next_state])
		if np.sum(np.abs(v-v_new)) < 1e-10:
			return v_new
		v = v_new
# def cal_v_from_pi(pi,numStates,numActions,transition_probs, transition_reward, gamma):
# 	v = np.zeros((numStates))
# 	for next_state in range(numStates):
# 		pdb.set_trace()
# 		v += transition_probs[:,pi[:],next_state]*(transition_reward[:,pi[:],next_state] + gamma*v[next_state])
# 	return v
def cal_v_from_pi(pi,numStates,numActions,transition_probs, transition_reward, gamma):
	v = np.zeros((numStates))
	for state in range(numStates):
		for next_state in range(numStates):
			v[state] += transition_probs[state,int(pi[state]),next_state]*(transition_reward[state,int(pi[state]),next_state] + gamma*v[next_state])
	return v
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
	print(Lp_prob) 
	status = Lp_prob.solve()   # Solver 
	for i in range(numStates):
		print(p.value(state_var[i]))
def lp_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	  
	# Create a LP Minimization problem 
	Lp_prob = p.LpProblem('Problem', p.LpMinimize)  
	  
	# Create problem Variables  
	x = p.LpVariable("x", lowBound = 0)   # Create a variable x >= 0 
	y = p.LpVariable("y", lowBound = 0)   # Create a variable y >= 0 
	  
	# Objective Function 
	Lp_prob += 3 * x + 5 * y    
	  
	# Constraints: 
	Lp_prob += 2 * x + 3 * y >= 12
	Lp_prob += -x + y <= 3
	Lp_prob += x >= 4
	Lp_prob += y <= 3
	  
	# Display the problem 
	print(Lp_prob) 
	  
	status = Lp_prob.solve()   # Solver 
	print(p.LpStatus[status])   # The solution status 
	  
	# Printing the final solution 
	print(p.value(x), p.value(y), p.value(Lp_prob.objective))   
def vec_vec_calculate_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	# Takes 14 seconds to execute on "data/mdp/episodic-mdp-10-5.txt"
	# vstar: [  0.         530.21982208 530.51367352 504.79864674 472.94806492
	#    0.         526.95299216 518.4643137  354.45767988 529.29214498]	
	V = np.zeros((numStates))
	while(1):
		next_V = np.zeros((numStates))
		action_reward = np.zeros((numActions, numStates))
		for a in range(numActions):
			red_sum = 0
			# for states in range(numStates):
			red_sum += transition_probs[:,a,:]*(transition_reward[:,a,:]+ gamma*V[:])
			red_sum = np.sum(red_sum, 1)
			# pdb.set_trace()
			action_reward[a] =red_sum
		next_V = np.max(action_reward, 0)
		if np.sum(np.abs(V-next_V)) < 1e-10:
			return V
		V = next_V

def vect_calculate_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	# Takes 44 seconds to execute on "data/mdp/episodic-mdp-10-5.txt"
	# vstar: [  0.         530.21982208 530.51367352 504.79864674 472.94806492
	#    0.         526.95299216 518.4643137  354.45767988 529.29214498]	
	V = np.zeros((numStates))
	while(1):
		next_V = np.zeros((numStates))
		action_reward = np.zeros((numActions, numStates))
		for a in range(numActions):
			red_sum = 0
			for states in range(numStates):
				red_sum += transition_probs[:,a,states]*(transition_reward[:,a,states]+ gamma*V[states])
			action_reward[a] =red_sum
		next_V = np.max(action_reward, 0)
		if np.sum(np.abs(V-next_V)) < 1e-10:
			return V
		V = next_V

def calculate_vstar(numStates,numActions,transition_probs, transition_reward, gamma):
	# Takes 1 min 57 seconds to execute on "data/mdp/episodic-mdp-10-5.txt"
	# vstar: [  0.         530.21982208 530.51367352 504.79864674 472.94806492
	#    0.         526.95299216 518.4643137  354.45767988 529.29214498]
	V = np.zeros((numStates))
	while(1):
		next_V = np.zeros((numStates))
		# action_reward = np.zeros((numActions, numStates))
		action_reward = np.zeros((numActions))
		for init_state in range(numStates):
			for a in range(numActions):
				red_sum = 0
				for states in range(numStates):
					# pdb.set_trace()
					# print(init_state,a,states)
					red_sum += transition_probs[init_state,a,states]*(transition_reward[init_state,a,states]+ gamma*V[states])
				action_reward[a] =red_sum
			next_V[init_state] = np.max(action_reward)
		# print(f"diff: {np.sum((V-np.max(action_reward, 0)))}")
		# if np.sum((V-np.max(action_reward, 0))) < 1e-10:
		# print(np.sum(np.abs(V-next_V)))
		if np.sum(np.abs(V-next_V)) < 1e-10:
			print((np.abs(V-next_V)))
			print(((next_V)))
			print(((V)))
			print(V)
			return V
		# V = np.max(action_reward, 0)
		V = next_V
