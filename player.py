import envs
import gym
import train 
env = gym.make('Tetris-v0')
PATH = './weights/curr_iter.txt'

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

m_t, _, _, _, _, _, = load_params(PATH)