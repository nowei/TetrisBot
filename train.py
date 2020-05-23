import envs
import gym
import math
import numpy as np  
from scipy.linalg import sqrtm
import os 
env = gym.make('Tetris-v0')

filename = "curr_iter.txt"

import json
import io

def save_params(filename, m_t, sigma_t, C_t, t, p_c_t, p_sigma_t):
    print("saving params to ./weights/{}".format(filename))
    m = {}

    def write_np(a):
        memfile = io.BytesIO()
        np.save(memfile, a)
        memfile.seek(0)
        return json.dumps(memfile.read().decode('latin-1'))
        
    with open('./weights/{}'.format(filename), 'w') as f:
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
    print("loading params from ./weights/{}".format(filename))
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

    with open('./weights/{}'.format(filename)) as f:
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

if os.path.exists("./weights/" + filename):
    print('picking up from previous iteration')
    mean, sigma, C, t, p_c, p_sigma = load_params(filename)

    # TODO: Tune this when we have a good mean to choose from
    lamb = 4 + math.floor(3 * math.log(n))
else:
    # Number of children
    lamb = 4 + math.floor(3 * math.log(n))

    # initial mean value
    # TODO: actually have values
    mean = np.array([0, 0, 0, 0, 0, 0]) 

    # Covariance matrix
    C = np.eye(num_features)

    # Generation number
    t = 1

    # evolution path
    p_c = np.ones((n,))

    # conjugate evolution path (to control step size)
    p_sigma = np.ones((n,))

    # Step size 
    sigma = 0.1

def CMA_ES(lamb, m, sigma, C, t, p_c, p_sigma):
    mu = math.floor(lamb / 2)
    m_t = m 
    sigma_t = sigma             # 
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
    mu_eff = 1 / (np.linalg.norm(w) ** 2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 3)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma
    c_c = 4 / (n + 4)
    mu_cov = mu_eff 
    c_cov = (1 / mu_cov) * (2 / (n + math.sqrt(2)) ** 2) + (1 - 1 / mu_cov) * min(1, (2 * mu_eff - 1)/((n + 2) ** 2 + mu_eff))
    env = gym.make('Tetris-v0')
    env.reset()

    while (True):
        print("t = {}".format(t))
        Z = np.random.multivariate_normal(np.zeros(n), C_t, (lamb))
        X = m_t + sigma * Z
        performance = []
        # create environment
        for i in range(lamb):
            child = X[i]
            # Eval(child)
            # env.reset() # Done in 
            performance.append(eval_child(child, env))

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
        expected_norm_normal = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        h_sigma = 0 if np.linalg.norm(p_sigma_t) > (1.5 + 1 / (n - 0.5)) * expected_norm_normal * math.sqrt(1 - (1 - c_sigma) ** (2 * (t + 1))) else 1

        p_sigma_new = (1 - c_sigma) * p_sigma_t + h_sigma * math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrtm(np.linalg.inv(C_t))).dot(Y_ranked)
        sigma_new = sigma_t * math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma_new)/ (expected_norm_normal - 1)))
        p_c_new = (1 - c_c) * p_c_t + math.sqrt(c_c * (2 - c_c) * mu_eff) * (Y_ranked)
        C_new = (1 - c_cov) * C_t + c_cov/mu_cov * np.outer(p_c_new, p_c_new) + c_cov * (1 - 1 / mu_cov) * (Y_selected * w[:, np.newaxis]).T.dot(Y_selected)
        
        # Update variables
        t = t + 1
        m_t = m_new 
        p_sigma_t = p_sigma_new 
        sigma_t = sigma_new
        p_c_t = p_c_new 
        C_t = C_new

        save_params('curr_iter.txt', m_t, sigma_t, C_t, t, p_c_t, p_sigma_t)
        # m_t, sigma_t, C_t, t, p_c_t, p_sigma_t = load_params('curr_iter.txt')
    return m_t 

def eval_child(child, env, episodes=5):
    total = 0
    for i in range(episodes):
        total += env.get_reward_child(child)
    average = total / episodes
    print('cleared an average of {} lines'.format(average))
    return average

if __name__ == "__main__":
    CMA_ES(lamb, mean, sigma, C, t, p_c, p_sigma)