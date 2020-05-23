import envs
import gym
import io
import json
import numpy as np
import configparser
np.set_printoptions(linewidth=200)

config = configparser.ConfigParser()
config.read('settings.config')
config = config['DEFAULT']

env = gym.make('Tetris-v0')
env.reset()
PATH = config['TEST_PATH']

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
        print('--------- generation {} ---------'.format(t))
        print('mean')
        print(m_t)
        print('step size')
        print(sigma_t)
        print('Covariance matrix')
        print(C_t)
        print('evolution path')
        print(p_c_t)
        print('conjugate evolution path')
        print(p_sigma_t)
    return m_t, sigma_t, C_t, t, p_c_t, p_sigma_t 

candidate, _, _, _, _, _, = load_params(PATH)

while (not env.state.lost and env.state.cleared < 12500):
    prev_state = env.state.copy()
    best_val = -float("inf")
    # best_action = None
    # best_val_reward = None
    best_state = None
    for action in env.get_actions():
        env.step(action)
        val = np.dot(candidate, env.features)
        if val > best_val:
            best_val = val
            # best_action = action
            best_state = env.state.copy()
        env.set_state(prev_state)
    env.set_state(best_state)
print('cleared {} lines'.format(env.state.cleared))
