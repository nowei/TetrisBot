import envs
import gym
import math
import numpy as np  
np.set_printoptions(linewidth=200)

from scipy.linalg import sqrtm
import os 
import configparser

config = configparser.ConfigParser()
config.read('settings.config')
config = config['DEFAULT']

env = gym.make('Tetris-v0')




filename = config['TRAIN_SAVE_PATH']

# train on a 2/9 prob for S and Z
hard = config.getboolean('hard')

# train on a 3/11 prob for S and Z
harder = config.getboolean('harder')

# algorithm as in physics paper
physics = config.getboolean('physics')

multiprocess = config.getboolean('multiprocessing')
if multiprocess: 
    from multiprocessing import Pool, cpu_count

num_episodes = int(config['num_episodes'])

mean = np.array(eval(config['mean']))
p_c = np.array(eval(config['p_c']))
p_sigma = np.array(eval(config['p_sigma']))
sigma = float(config['sigma'])


import json
import io

def save_params(filename, m_t, sigma_t, C_t, t, p_c_t, p_sigma_t):
    print("saving params to {}".format(filename))
    m = {}

    def write_np(a):
        memfile = io.BytesIO()
        np.save(memfile, a)
        memfile.seek(0)
        return json.dumps(memfile.read().decode('latin-1'))
        
    with open(filename, 'w') as f:
        # print(m_t)
        # print(sigma_t)
        # print(C_t)
        # print(t)
        # print(p_c_t)
        # print(p_sigma_t)
        m['m_t'] = write_np(m_t)
        m['sigma_t'] = sigma_t 
        m['C_t'] = write_np(C_t)
        m['t'] = t 
        m['p_c_t'] = write_np(p_c_t)
        m['p_sigma_t'] = write_np(p_sigma_t)
        # print(m)
        f.write(json.dumps(m))

num_features = 6
# TODO: Variables to define, look at physics paper
# Parameters described in https://hal.inria.fr/inria-00276216/document

def load_params(filename):
    print("loading params from {}".format(filename))
    m = None
    m_t = None 
    sigma_t = None 
    C_t = None 
    t = None 
    p_c_t = None 
    p_sigma_t = None

    def load_np(serialized):
        memfile = io.BytesIO()
        memfile.write(json.loads(serialized).encode('latin-1'))
        memfile.seek(0)
        return np.load(memfile)

    with open(filename) as f:
        m = json.loads(f.read())
        m_t = load_np(m['m_t'])
        sigma_t = float(m['sigma_t'])
        C_t = load_np(m['C_t'])
        t = int(m['t'])
        p_c_t = load_np(m['p_c_t'])
        p_sigma_t = load_np(m['p_sigma_t'])
        # print(m_t)
        # print(sigma_t)
        # print(C_t)
        # print(t)
        # print(p_c_t)
        # print(p_sigma_t)
    return m_t, sigma_t, C_t, t, p_c_t, p_sigma_t 



# dimension of mean
n = num_features # Not sure if right? (????)

if os.path.exists(filename):
    print('picking up from previous generation')
    mean, sigma, C, t, p_c, p_sigma = load_params(filename)
    print('--------- generation {} ---------'.format(t))
    print('mean')
    print(mean)
    print('step size')
    print(sigma)
    print('Covariance matrix')
    print(C)
    print('covariance evolution path')
    print(p_c)
    print('conjugate evolution path')
    print(p_sigma)

    # TODO: Tune this when we have a good mean to choose from
    lamb = 4 + math.floor(3 * math.log(n))
else:
    print('starting new spawns')
    # Number of children, as suggested in https://hal.inria.fr/inria-00276216/document
    lamb = 4 + math.floor(3 * math.log(n))

    # Covariance matrix
    C = np.eye(num_features)

    # generation number
    t = 1
    
    print('--------- generation {} ---------'.format(t))
    print('mean')
    print(mean)
    print('step size')
    print(sigma)
    print('Covariance matrix')
    print(C)
    print('covariance evolution path')
    print(p_c)
    print('conjugate evolution path')
    print(p_sigma)

def CMA_ES(lamb, m, sigma, C, t, p_c, p_sigma):
    mu = math.floor(lamb / 2)
    m_t = m 
    sigma_t = sigma             # step size
    C_t = C                     # Covariance
    t = t                       # generation counter
    p_c_t = p_c
    p_sigma_t = p_sigma

    print("number of children = {}".format(lamb))
    
    # Parameters described in https://hal.inria.fr/inria-00276216/document
    # w on page 6 (paper page, not pdf page)
    # c_sigma on page 6
    # d_sigma on page 6
    # c_c on page 6
    # mu_cov on page 6
    # c_cov on page 6

    w = np.array([(math.log(mu + 1) - math.log(i)) / (mu * math.log(mu + 1) - sum([math.log(j) for j in range(1, mu + 1)])) for i in range(1, mu + 1)])
    # print(w)
    mu_eff = 1 / (np.sum(w * w))
    c_sigma = (mu_eff + 2) / (n + mu_eff + 3)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma
    c_c = 4 / (n + 4)
    mu_cov = mu_eff 
    c_cov = (1 / mu_cov) * (2 / (n + math.sqrt(2)) ** 2) + (1 - 1 / mu_cov) * min(1, (2 * mu_eff - 1)/(((n + 2) ** 2) + mu_eff))

    # create environment
    print('creating environment')
    if not multiprocess:
        env = gym.make('Tetris-v0')
        env.reset()
    else:
        print('attempting multiprocessing, cpu_count = {}'.format(os.cpu_count()))
        envs = [[None, gym.make('Tetris-v0'), num_episodes] for i in range(lamb)]
        for env in envs:
            env[1].reset()

    # Calculate expected N(0, I) vector norm
    expected_norm_normal = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    while (True):
        Z = np.random.multivariate_normal(np.zeros(n), C_t, (lamb))
        X = m_t + sigma * Z
        performance = []
        if not multiprocess:
            for i in range(lamb):
                child = X[i]
                # Eval(child)
                # env.reset() Done in tetris.py
                performance.append(eval_child(child, env, num_episodes))
        else:
            for i in range(lamb):
                envs[i][0] = X[i]
            with Pool(processes=cpu_count()) as pool:
                performance = pool.starmap(eval_child, envs)
            print(performance)
        # argsort Eval(X), reverse
        ordering = np.argsort(performance)[::-1]
        
        # Take top \mu from argsort
        X_sorted = X[ordering]
        Z_sorted = Z[ordering] 

        # Linear combine top \mu X with w
        m_new = np.sum(X_sorted[:mu] * w[:, np.newaxis], axis=0)

        # dim = mu x n
        Y_selected = Z_sorted[:mu]
        Y_ranked = np.sum(Y_selected * w[:, np.newaxis], axis=0) # \langle ⟨y⟩ \rangle
        
        # https://math.stackexchange.com/questions/827826/average-norm-of-a-n-dimensional-vector-given-by-a-normal-distribution
        # https://math.stackexchange.com/questions/2255173/mean-of-squared-l-2-norm-of-gaussian-random-vector/2255666
        # E[||N(0,I)||^2] = trace(covariance)
        # According to: https://en.wikipedia.org/wiki/CMA-ES#Algorithm and https://hal.inria.fr/inria-00276216/document (page 6)
        # E[||N(0, I)||] \approx sqrt(n) * (1 - 1 / (4n) + 1 / (21 * n^2))
        if physics:
            covar_evol_path_calc = (1.5 + 1 / (n - 0.5)) * expected_norm_normal * math.sqrt(1 - (1 - c_sigma) ** (2 * (t + 1)))
            h_sigma = 1 if np.linalg.norm(p_sigma_t) > covar_evol_path_calc else 0
            print("covar path update", np.linalg.norm(p_sigma_t), covar_evol_path_calc, h_sigma)

        if physics:
            # Physics paper
            p_sigma_new = (1 - c_sigma) * p_sigma_t + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrtm(np.linalg.inv(C_t))).dot(Y_ranked)
            sigma_new = sigma_t * math.exp((c_sigma / d_sigma) * ((np.linalg.norm(p_sigma_new)/ expected_norm_normal) - 1))
            p_c_new = (1 - c_c) * p_c_t + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (Y_ranked)
            C_new = (1 - c_cov) * C_t + c_cov/mu_cov * np.outer(p_c_new, p_c_new) + c_cov * (1 - 1 / mu_cov) * (Y_selected * w[:, np.newaxis]).T.dot(Y_selected)
        else:
            # Tetris paper
            p_sigma_new = (1 - c_sigma) * p_sigma_t + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrtm(C_t)).dot((m_new - m_t)/sigma_t)
            sigma_new = sigma_t * math.exp((c_sigma / d_sigma) * ((np.linalg.norm(p_sigma_new)/ expected_norm_normal) - 1))
            p_c_new = (1 - c_c) * p_c_t + math.sqrt(c_c * (2 - c_c) * mu_cov) * ((m_new - m_t)/sigma_t)
            C_new = (1 - c_cov) * C_t + c_cov/mu_cov * np.outer(p_c_new, p_c_new) + c_cov * (1 - 1 / mu_cov) * (Y_selected * w[:, np.newaxis]).T.dot(Y_selected)
        
        # Update variables
        t = t + 1
        m_t = m_new 
        p_sigma_t = p_sigma_new 
        sigma_t = sigma_new
        p_c_t = p_c_new 
        C_t = C_new

        print('--------- generation {} ---------'.format(t))
        print('mean')
        print(m_t)
        print('step size')
        print(sigma_t)
        print('Covariance matrix')
        print(C_t)
        print('covariance evolution path')
        print(p_c_t)
        print('conjugate evolution path')
        print(p_sigma_t)

        save_params(filename, m_t, sigma_t, C_t, t, p_c_t, p_sigma_t)
        # m_t, sigma_t, C_t, t, p_c_t, p_sigma_t = load_params('curr_iter.txt')
    return m_t 

def eval_child(child, env, episodes=5):
    total = 0
    for i in range(episodes):
        total += env.get_reward_child(child, hard, harder)
        print(i, end='\r')
    average = total / episodes
    print('cleared an average of {} lines'.format(average))
    return average

if __name__ == "__main__":
    CMA_ES(lamb, mean, sigma, C, t, p_c, p_sigma)
