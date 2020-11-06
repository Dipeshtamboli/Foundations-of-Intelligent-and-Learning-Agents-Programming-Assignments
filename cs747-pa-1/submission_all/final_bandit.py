from time import time
import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse

'''
--instance in, where in is a path to the instance file.
--algorithm al, where al is one of epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint.
--randomSeed rs, where rs is a non-negative integer.
--epsilon ep, where ep is a number in [0, 1].
--horizon hz, where hz is a non-negative integer.
'''
# def kl(p,q):
#     # if q!=0 and p!=0:
#     # first_term = np.where(p!= 0 and q!=0,p * np.log(p / q),0 ) 
#     first_term = np.where(p!= 0,np.where(q!=0,p * np.log(p / q),0 ) ,0 ) 
#     # print(first_term)
#     # else:
#     #     first_term = 0
#     # if q!=1 and p!=1:
#     # second_term = np.where(q!=1 and p!=1,(1-p) * np.log((1-p) / (1-q)),0) 
#     second_term = np.where(q!=1,np.where(p!=1,(1-p) * np.log((1-p) / (1-q)),0) ,0) 
#     # else:
#         # second_term = 0

#     return first_term + second_term
def thompson_sampling_with_hint(instance, algorithm,arm_prob,randomSeed,epsilon,horizon):
    pass
def kl(p,q):
    first_term = p * np.log((p + 1e-10) / (q+ 1e-10))
    second_term = (1-p) * np.log((1-p+ 1e-10) / (1-q+ 1e-10))
    return first_term + second_term
def kl_ucb(instance, algorithm,arm_prob,randomSeed,epsilon,horizon):
    sum_reward = 0
    empirical_probs = np.zeros((len(arm_prob)))
    q_values = np.zeros((len(arm_prob)))
    regret_step = np.zeros((horizon))
    pull_count = np.zeros((len(arm_prob)))
    c = 0
    for t in range(1,horizon+1):
        for arm_id_iter in range(len(arm_prob)):
            steps = np.arange(empirical_probs[arm_id_iter],1,0.001)
            value = np.ones(steps.shape)*empirical_probs[arm_id_iter]
            kl_d = kl(value,steps)
            smaller_vals = (pull_count[arm_id_iter]*kl_d <= np.log(t) + c*np.log(np.log(t)))
            q_max_arg = [iter_i for iter_i, val in enumerate(smaller_vals) if val]
            # pdb.set_trace()
            if len(q_max_arg) == 0:
                q_values[arm_id_iter] = empirical_probs[arm_id_iter]
            else:
                q_values[arm_id_iter] = steps[q_max_arg[-1]]
        arm_id = np.argmax(q_values)
        reward = np.random.binomial(1, arm_prob[arm_id])
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward        
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
        if t in [100, 400, 1600, 6400, 25600, 102400]:
            print(instance, algorithm, randomSeed, epsilon, t, regret_step[t-1])        
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step
# def kl_ucb(arm_prob,randomSeed,epsilon,horizon):
#     sum_reward = 0
#     empirical_probs = np.zeros((len(arm_prob)))
#     q_values = np.zeros((len(arm_prob)))
#     pull_count = np.zeros((len(arm_prob)))
#     # print("kl",kl(0.5,0.7))
#     # out = np.zeros((1000))
#     # for i in range(1,1001):
#     #     out[i-1] = kl(i/1000*1.0, 0.7)
#     # steps = np.arange(0,1,0.001)
#     # value = np.ones((1000))*0.0
#     # out = kl(steps, value) 
#     # # out = kl(steps, value) < 0.2
#     # max_q = [iter for iter, val in enumerate(out) if val][-1]
#     # print(max_q, out)
#     # plt.figure()
#     # plt.plot(range(1000), out)
#     # plt.savefig("kl2.jpg")
#     # plt.show()
#     for t in range(1,horizon+1):
#         for arm_id_iter in range(len(arm_prob)):
#             steps = np.arange(empirical_probs[arm_id_iter],1,0.001)
#             value = np.ones(steps.shape)*empirical_probs[arm_id_iter]
#             kl_d = kl(value,steps)
#             smaller_vals = kl_d <= np.log(t)
#             # pdb.set_trace()
#             q_max_arg = [iter_i for iter_i, val in enumerate(smaller_vals) if val]
#             q_values[arm_id_iter] = steps[q_max_arg[-1]]
#             # if t == 2:
#             #     break
#         arm_id = np.argmax(q_values)
#         reward = np.random.binomial(1, arm_prob[arm_id])
#         empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
#         pull_count[arm_id] += 1
#         sum_reward += reward        
#     regret = horizon*np.max(arm_prob) - sum_reward
#     return empirical_probs, pull_count, regret

def thompson_sampling(instance, algorithm,arm_prob,randomSeed,epsilon,horizon):
    empirical_probs = np.zeros((len(arm_prob)))
    successes = np.zeros((len(arm_prob)))
    failures = np.zeros((len(arm_prob)))
    pull_count = np.zeros((len(arm_prob)))
    beta_value = np.zeros((len(arm_prob)))
    regret_step = np.zeros((horizon))
    sum_reward = 0
    for t in range(1, horizon+1):
        beta_value = np.random.beta(successes+1,failures+1)
        # for arm_iter in range(len(arm_prob)):
        #   beta_value[arm_iter] = np.random.beta(successes[arm_iter]+1,failures[arm_iter]+1)

        arm_id = np.argmax(beta_value)
        reward = np.random.binomial(1, arm_prob[arm_id])
        successes[arm_id] += reward
        failures[arm_id] += (1-reward)
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward        
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
        if t in [100, 400, 1600, 6400, 25600, 102400]:
            print(instance, algorithm, randomSeed, epsilon, t, regret_step[t-1])        
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step
def ucb(instance, algorithm,arm_prob,randomSeed,epsilon,horizon):
    empirical_probs = np.zeros((len(arm_prob)))
    pull_count = np.ones((len(arm_prob))) * 1e-10
    ucb_value = np.zeros((len(arm_prob)))
    # empirical_probs = [0] * len(arm_prob)
    # pull_count = [0] * len(arm_prob)
    # ucb_value = [0] * len(arm_prob) 
    sum_reward = 0
    regret_step = np.zeros((horizon))

    for t in range(1,horizon+1):
        arm_id = np.argmax(ucb_value)
        reward = np.random.binomial(1, arm_prob[arm_id])
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward

        # print("t,arm_id, reward",t,arm_id, reward)
        # print("pull_count",pull_count)
        # print("empirical_probs",empirical_probs)
        # print("ucb_value",ucb_value)
        # print("---------------------------")
        ucb_value = empirical_probs + np.sqrt((2*np.log(t))/pull_count)
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
        if t in [100, 400, 1600, 6400, 25600, 102400]:
            print(instance, algorithm, randomSeed, epsilon, t, regret_step[t-1])
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step

def epsilon_g(instance, algorithm,arm_prob,randomSeed,epsilon,horizon):
    explore_steps = int(epsilon*horizon)
    empirical_probs = [0] * len(arm_prob)
    pull_count = [0] * len(arm_prob)
    sum_reward = 0
    regret_step = np.zeros((horizon))
    # print(empirical_probs)
    for t in range(1, horizon+1):
        do_explore = np.random.binomial(1, epsilon)
        if do_explore:
            arm_id = np.random.randint(len(arm_prob))
        else:
            # reward = np.random.binomial(1, arm_prob[arm_id])
            # empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
            # pull_count[arm_id] += 1
            arm_id = np.argmax(empirical_probs)
        reward = np.random.binomial(1, arm_prob[arm_id])
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
        if t in [100, 400, 1600, 6400, 25600, 102400]:
            print(instance, algorithm, randomSeed, epsilon, t, regret_step[t-1])
    regret = horizon*np.max(arm_prob) - sum_reward
    # print(sum_reward, np.max(arm_prob))

    return empirical_probs, pull_count, regret, regret_step




    # print(explore_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse instance, algorithm, randomSeed, epsilon, horizon.')
    parser.add_argument('--instance', default="../instances/i-3.txt", help='instance in, where in is a path to the instance file.')
    parser.add_argument('--algorithm', default="kl-ucb", help='algorithm al, where al is one of epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint.')
    parser.add_argument('--randomSeed', default=0, help='randomSeed rs, where rs is a non-negative integer.')
    parser.add_argument('--epsilon', default=0.02, help='epsilon ep, where ep is a number in [0, 1].')
    parser.add_argument('--horizon', default=102400, help='horizon hz, where hz is a non-negative integer.')

    args = parser.parse_args()
    # print("----------------------------------------")
    instance = args.instance
    algorithm = args.algorithm
    randomSeed = int(args.randomSeed)
    epsilon = float(args.epsilon)
    horizon = int(args.horizon)
    # print(f"instance: {args.instance}")
    # print(f"algorithm: {args.algorithm}")
    # print(f"randomSeed: {args.randomSeed}")
    # print(f"epsilon: {args.epsilon}")
    # print(f"horizon: {args.horizon}")
    algo = {}
    algo["ucb"] = ucb
    algo["epsilon-greedy"] = epsilon_g
    algo["kl-ucb"] = kl_ucb
    algo["thompson-sampling"] = thompson_sampling
    algo["thompson_sampling_with_hint"] = thompson_sampling_with_hint


    start_time = time()

    # print(arms_prob)
    # epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint    

    # empirical_probs, pull_count, regret, regret_step = algo[algorithm](arms_prob,randomSeed,epsilon,horizon)
    regret_step_all = np.zeros((horizon))
    for instance in ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]:
        bandit_instance_file = open(instance, "r")
        arm_file = instance.split('/')[-1].split('.')[0]
        arms_prob =[]
        for i_arm_prob in bandit_instance_file:
            arms_prob.append(float(i_arm_prob.strip()))

        for randomSeed in range(50):
            # print(randomSeed)
            np.random.seed(randomSeed)
            empirical_probs, pull_count, regret, regret_step = algo[algorithm](instance, algorithm,arms_prob,randomSeed,epsilon,horizon)
            np.save(f'{algorithm}/i:{arm_file}_s:{randomSeed}.npy',regret_step)
            regret_step_all += regret_step
    regret_step_all /= 50
    # plt.figure()
    # plt.plot(range(horizon),regret_step_all)
    # plt.xscale('log', basex = 10)
    # plt.savefig(f'allseeds_{arm_file}_{algorithm}_h:{horizon}_s:{randomSeed}_eps:{epsilon}.jpg')

    # print(arms_prob ,empirical_probs, pull_count, regret, regret/horizon)
    # print(instance, algorithm, randomSeed, epsilon, horizon, regret)
    print(f'total_time taken: {(time()- start_time)//3600} hrs {(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')