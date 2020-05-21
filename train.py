import envs
import gym
import math
import numpy as np  
env = gym.make('Tetris-v0')

num_features = 6
# TODO: Variables to define, look at physics paper
# c_c = 
# c_sigma = 
# mu_co = 
# c_co = 
# w = 


def CMA_ES(lamb, m, sigma, f, n):
    mu = math.floor(lamb / 2)
    m_t = m 
    sigma_t = sigma 
    C_t = np.eye(num_features)
    t = 1
    p_c_t = 0
    p_sigma_t = 0
    

    Z = np.random.multivariate_normal(0, C_t, (lamb))
    X = m_t + sigma * Z
    # Eval(X)
    # argsort Eval(X), reverse
    # Take top \mu from argsort
    # Linear combine top \mu X with w, matmul = 

    

    m_new = w * X
    p_sigma_new = (1 - c_sigma) * p_sigma_t + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.matrix_power(C_t, 0.5) * ((m_new - m_t)/sigma_t)
    # https://math.stackexchange.com/questions/827826/average-norm-of-a-n-dimensional-vector-given-by-a-normal-distribution
    # https://math.stackexchange.com/questions/2255173/mean-of-squared-l-2-norm-of-gaussian-random-vector/2255666
    # E[||N(0,I)||^2] = trace(covariance)
    # https://en.wikipedia.org/wiki/CMA-ES#Algorithm
    sigma_new = sigma_t * math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma_new)/ (math.sqrt(num_features) * (1 - 1 / (4 * num_features) + 1 / (21 * num_features * num_features))) - 1))
    p_c_new = (1 - c_c) * p_c_t + math.sqrt(c_c * (2 - c_c) * mu_co) * (m_new - m_t) / sigma_t
    C_new = (1 - c_co) * C_t + c_co/mu_co * np.outer(p_c_new, p_c_new) + c_co * (1 - 1 / mu_co) * (w * Z).T.dot(Z)
    
    # Update variables
    t = t + 1
    m_t = m_new 
    p_sigma_t = p_sigma_new 
    sigma_t = sigma_new
    p_c_t = p_c_new 
    C_t = C_new

import pickle
import json

def save_params(filename, m_t, sigma_t, C_t, t, p_c_t, p_sigma_t):
    m = {}
    with open('./weights/{}'.format(filename), 'w') as f:
        m['m_t'] = pickle.dumps(m_t, protocol=0)
        m['sigma'] = sigma_t
        m['C_t'] = pickle.dumps(C_t, protocol=0)
        m['t'] = t 
        m['p_c_t'] = pickle.dumps(p_c_t, protocol=0)
        m['p_sigma_t'] = pickle.dumps(p_sigma_t, protocol=0)
        f.write(json.dumps(m))

def load_params(filename):
    m = None
    m_t = None 
    sigma_t = None 
    C_t = None 
    t = None 
    p_c_t = None 
    p_sigma_t = None
    with open('./weights/{}'.format(filename)) as f:
        m = json.loads(f.read())
        m_t = pickle.loads(m['m_t'])
        sigma_t = float(m['sigma_t'])
        C_t = pickle.loads(m['m'])
        t = int(m['t'])
        p_c_t = pickle.loads(m['p_c_t'])
        p_sigma_t = pickle.loads(m['p_sigma_t'])
    return m, m_t, sigma_t, C_t, t, p_c_t, p_sigma_t 