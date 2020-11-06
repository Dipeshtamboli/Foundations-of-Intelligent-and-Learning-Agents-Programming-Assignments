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
def thompson_sampling_with_hint(instance, algorithm,arm_prob,randomSeed,epsilon,horizon,sorted_true_means):
    empirical_probs = np.zeros((len(arm_prob)))
    successes = np.zeros((len(arm_prob)))
    failures = np.zeros((len(arm_prob)))
    pull_count = np.zeros((len(arm_prob)))
    beta_value = np.zeros((len(arm_prob)))
    regret_step = np.zeros((horizon))
    sum_reward = 0
    posterior = np.ones((len(arm_prob),len(arm_prob))) / len(arm_prob)
    sorted_p = sorted_true_means
    sorted_1_p = 1- sorted_true_means
    p_true_max = max(sorted_true_means)
    for t in range(1, horizon+1):
        arm_id = np.argmax(posterior[:,-1])
        reward = np.random.binomial(1, arm_prob[arm_id])
        if reward>0:
            posterior[arm_id] *= sorted_p
        else:
            posterior[arm_id] *= sorted_1_p
        posterior[arm_id] /= np.sum(posterior[arm_id])
        successes[arm_id] += reward
        failures[arm_id] += (1-reward)
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward        
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step    

def kl(p,q):
    first_term = p * np.log((p + 1e-10) / (q+ 1e-10))
    second_term = (1-p) * np.log((1-p+ 1e-10) / (1-q+ 1e-10))
    return first_term + second_term
def kl_ucb(instance, algorithm,arm_prob,randomSeed,epsilon,horizon,sorted_true_means):
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
            # smaller_vals = (pull_count[arm_id_iter]*kl_d <= np.log(t) + c*np.log(np.log(t)))
            smaller_vals = (pull_count[arm_id_iter]*kl_d <= np.log(t) )
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
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step
def thompson_sampling(instance, algorithm,arm_prob,randomSeed,epsilon,horizon,sorted_true_means):
    empirical_probs = np.zeros((len(arm_prob)))
    successes = np.zeros((len(arm_prob)))
    failures = np.zeros((len(arm_prob)))
    pull_count = np.zeros((len(arm_prob)))
    beta_value = np.zeros((len(arm_prob)))
    regret_step = np.zeros((horizon))
    sum_reward = 0
    for t in range(1, horizon+1):
        beta_value = np.random.beta(successes+1,failures+1)
        arm_id = np.argmax(beta_value)
        reward = np.random.binomial(1, arm_prob[arm_id])
        successes[arm_id] += reward
        failures[arm_id] += (1-reward)
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward        
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
        # if t in [100, 400, 1600, 6400, 25600, 102400]:
        #     print(instance, algorithm, randomSeed, epsilon, t, regret_step[t-1])        
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step
def ucb(instance, algorithm,arm_prob,randomSeed,epsilon,horizon,sorted_true_means):
    empirical_probs = np.zeros((len(arm_prob)))
    pull_count = np.ones((len(arm_prob))) * 1e-10
    ucb_value = np.zeros((len(arm_prob)))
    sum_reward = 0
    regret_step = np.zeros((horizon))

    for t in range(1,horizon+1):
        arm_id = np.argmax(ucb_value)
        reward = np.random.binomial(1, arm_prob[arm_id])
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward
        ucb_value = empirical_probs + np.sqrt((2*np.log(t))/pull_count)
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
        # if t in [100, 400, 1600, 6400, 25600, 102400]:
        #     print(instance, algorithm, randomSeed, epsilon, t, regret_step[t-1])
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step

def epsilon_g(instance, algorithm,arm_prob,randomSeed,epsilon,horizon,sorted_true_means):
    explore_steps = int(epsilon*horizon)
    empirical_probs = [0] * len(arm_prob)
    pull_count = [0] * len(arm_prob)
    sum_reward = 0
    regret_step = np.zeros((horizon))
    for t in range(1, horizon+1):
        do_explore = np.random.binomial(1, epsilon)
        if do_explore:
            arm_id = np.random.randint(len(arm_prob))
        else:
            arm_id = np.argmax(empirical_probs)
        reward = np.random.binomial(1, arm_prob[arm_id])
        empirical_probs[arm_id] = (empirical_probs[arm_id] * pull_count[arm_id] + reward)/(pull_count[arm_id]+1)*1.0
        pull_count[arm_id] += 1
        sum_reward += reward
        regret_step[t-1] = (t*np.max(arm_prob) - sum_reward)
    regret = horizon*np.max(arm_prob) - sum_reward
    return empirical_probs, pull_count, regret, regret_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse instance, algorithm, randomSeed, epsilon, horizon.')
    parser.add_argument('--instance', default="../instances/i-3.txt", help='instance in, where in is a path to the instance file.')
    parser.add_argument('--algorithm', default="thompson-sampling", help='algorithm al, where al is one of epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint.')
    parser.add_argument('--randomSeed', default=0, help='randomSeed rs, where rs is a non-negative integer.')
    parser.add_argument('--epsilon', default=0.02, help='epsilon ep, where ep is a number in [0, 1].')
    parser.add_argument('--horizon', default=102400, help='horizon hz, where hz is a non-negative integer.')

    args = parser.parse_args()
    instance = args.instance
    algorithm = args.algorithm
    randomSeed = int(args.randomSeed)
    epsilon = float(args.epsilon)
    horizon = int(args.horizon)
    algo = {}
    algo["ucb"] = ucb
    algo["epsilon-greedy"] = epsilon_g
    algo["kl-ucb"] = kl_ucb
    algo["thompson-sampling"] = thompson_sampling
    algo["thompson-sampling-with-hint"] = thompson_sampling_with_hint

    bandit_instance_file = open(instance, "r")
    arm_file = instance.split('/')[-1].split('.')[0]
    arms_prob =[]

    for i_arm_prob in bandit_instance_file:
        arms_prob.append(float(i_arm_prob.strip()))

    sorted_true_means = np.sort(arms_prob)
    np.random.seed(randomSeed)
    empirical_probs, pull_count, regret, regret_step = algo[algorithm](instance, algorithm,arms_prob,randomSeed,epsilon,horizon,sorted_true_means)
    print(f'{instance}, {algorithm}, {randomSeed}, {epsilon}, {horizon}, {regret}')
