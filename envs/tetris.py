"""
Tetris Simulator

Author - Anqi Li (anqil4@cs.washington.edu)
Adapted from the java simulator from Drew Bagnell's
course at Carnegie Mellon University

"""


import gym
from gym.utils import seeding
import numpy as np


class TetrisState:
    """
    the tetris state
    """
    def __init__(self, field, top, next_piece, lost, turn, cleared, features=None):
        # the board configuration
        self.field = field
        # the top position
        self.top = top
        # the piece ID of the next piece
        self.next_piece = next_piece
        # whether the game has lost
        self.lost = lost
        # the current turn
        self.turn = turn
        # the number of rows cleared so far
        self.cleared = cleared
        # last feature object
        self.features = features

    def copy(self):
        return TetrisState(
            self.field.copy(),
            self.top.copy(),
            self.next_piece,
            self.lost,
            self.turn,
            self.cleared,
            self.features
        )

class TetrisEnv(gym.Env):
    metadata = {
        'render.modes': ['ascii']
    }

    def __init__(self):
        self.n_cols = 10
        self.n_rows = 21
        self.n_pieces = 7

        # the next several lists define the piece vocabulary in detail
        # width of the pieces [piece ID][orientation]
        # pieces: O, I, L, J, T, S, Z
        self.piece_orients = [1, 2, 4, 4, 4, 2, 2]
        self.piece_width = [
            [2],
            [1, 4],
            [2, 3, 2, 3],
            [2, 3, 2, 3],
            [2, 3, 2, 3],
            [3, 2],
            [3, 2]
        ]
        # height of pieces [piece ID][orientation]
        self.piece_height = [
            [2],
            [4, 1],
            [3, 2, 3, 2],
            [3, 2, 3, 2],
            [3, 2, 3, 2],
            [2, 3],
            [2, 3]
        ]
        self.piece_bottom = [
            [[0, 0]],
            [[0], [0, 0, 0, 0]],
            [[0, 0], [0, 1, 1], [2, 0], [0, 0, 0]],
            [[0, 0], [0, 0, 0], [0, 2], [1, 1, 0]],
            [[0, 1], [1, 0, 1], [1, 0], [0, 0, 0]],
            [[0, 0, 1], [1, 0]],
            [[1, 0, 0], [0, 1]]
        ]
        self.piece_top = [
            [[2, 2]],
            [[4], [1, 1, 1, 1]],
            [[3, 1], [2, 2, 2], [3, 3], [1, 1, 2]],
            [[1, 3], [2, 1, 1], [3, 3], [2, 2, 2]],
            [[3, 2], [2, 2, 2], [2, 3], [1, 2, 1]],
            [[1, 2, 2], [3, 2]],
            [[2, 2, 1], [2, 3]]
        ]

        # initialize legal moves for all pieces
        self.legal_moves = []
        for i in range(self.n_pieces):
            piece_legal_moves = []
            for j in range(self.piece_orients[i]):
                for k in range(self.n_cols + 1 - self.piece_width[i][j]):
                    piece_legal_moves.append([j, k])
            self.legal_moves.append(piece_legal_moves)

        self.state = None
        self.cleared_current_turn = 0

    def seed(self, seed=None):
        """
        set the random seed for the environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        make a move based on the orientation and slot
        """
        self.features = np.zeros((6,))

        orient, slot = action
        self.state.turn += 1

        # height of the field
        height = max(
            self.state.top[slot+c] - self.piece_bottom[self.state.next_piece][orient][c]
            for c in range(self.piece_width[self.state.next_piece][orient])
        )

        self.features[0] = height

        # check if game ended
        if height + self.piece_height[self.state.next_piece][orient] >= self.n_rows:
            self.state.lost = True
            return self.state, self._get_reward(), True, {}

        # for each column in the piece - fill in the appropriate blocks
        for i in range(self.piece_width[self.state.next_piece][orient]):
            # from bottom to top of brick
            for h in range(height + self.piece_bottom[self.state.next_piece][orient][i], height + self.piece_top[self.state.next_piece][orient][i]):
                self.state.field[h, i+slot] = self.state.turn

        # adjust top
        for c in range(self.piece_width[self.state.next_piece][orient]):
            self.state.top[slot+c] = height + self.piece_top[self.state.next_piece][orient][c]

        # check for full rows - starting at the top
        self.cleared_current_turn = 0
        for r in range(height + self.piece_height[self.state.next_piece][orient] - 1, height - 1, -1):
            # if the row was full - remove it and slide above stuff down
            if np.all(self.state.field[r] > 0):
                self.cleared_current_turn += 1
                self.state.cleared += 1
                # for each column
                for c in range(self.n_cols):
                    self.features[1] += (1 if self.state.field[r, c] == self.state.turn else 0)
                    # slide down all bricks
                    self.state.field[r:self.state.top[c], c] = self.state.field[(r+1):(self.state.top[c]+1), c]
                    # lower the top
                    self.state.top[c] -= 1
                    while self.state.top[c] >= 1 and self.state.field[self.state.top[c]-1, c] == 0:
                        self.state.top[c] -= 1

        self.features[1] *= self.cleared_current_turn

        # Feature 3
        for r_top in self.state.top:
            for r in range(r_top):
                filled = True
                for c in range(self.n_cols):
                    if (filled and self.state.field[r, c] == 0) or (not filled and self.state.field[r, c] > 0):
                        filled = not filled
                        self.features[2] += 1
                if not filled: 
                    self.features[2] += 1
    
        # Feature 4,5,6
        for c in range(self.n_cols):
            filled = True
            well = 0
            f_3_tmp = 0
            for r in range(self.state.top[c]):
                if (filled and self.state.field[r, c] == 0) or (not filled and self.state.field[r, c] > 0):
                   filled = not filled 
                   f_3_tmp += 1
                   well = 0
                if self.state.field[r, c] == 0 and (c - 1 < 0 or self.state.field[r, c] > 0) and (c + 1 == self.n_cols or self.state.field[r, c + 1] > 0):
                    well += 1
            self.features[3] += f_3_tmp
            self.features[4] += f_3_tmp // 2
            self.features[5] += (well * (well + 1)) // 2
        
        # pick a new piece
        self.state.next_piece = self._get_random_piece()
        return self.state.copy(), self._get_reward(), False, {}

    def reset(self):
        lost = False
        turn = 0
        cleared = 0

        field = np.zeros((self.n_rows, self.n_cols), dtype=np.int)
        top = np.zeros(self.n_cols, dtype=np.int)
        next_piece = self._get_random_piece()

        self.state = TetrisState(field, top, next_piece, lost, turn, cleared)
        return self.state.copy()

    def render(self, mode='ascii'):
        print('\nThe wall:')
        print('-' * (2 * self.n_cols + 1))
        for r in range(self.n_rows - 1, -1, -1):
            render_string = '|'
            for c in range(self.n_cols):
                if self.state.field[r, c] > 0:
                    render_string += '*|'
                else:
                    render_string += ' |'
            render_string += ''
            print(render_string)
        print('-' * (2 * self.n_cols + 1))

        print('\nThe next piece:')
        if self.state.next_piece == 0:
            print('**\n**')
        elif self.state.next_piece == 1:
            print('****')
        elif self.state.next_piece == 2:
            print('*\n*\n**')
        elif self.state.next_piece == 3:
            print(' *\n *\n**')
        elif self.state.next_piece == 4:
            print(' * \n***')
        elif self.state.next_piece == 5:
            print(' **\n**')
        elif self.state.next_piece == 6:
            print('**\n **')

    def close(self):
        pass

    def _get_random_piece(self):
        """
        return an random integer 0-6
        """
        return np.random.randint(self.n_pieces)

    def _get_reward(self):
        """
        reward function
        """
        # TODO: change it to your own choice of rewards
        return self.cleared_current_turn
        # return 0.0

    def get_actions(self):
        """
        gives the legal moves for the next piece
        :return:
        """
        return self.legal_moves[self.state.next_piece]

    def set_state(self, state):
        """
        set the field and the next piece
        """
        self.state = state.copy()

    def get_reward_child(self, child, hard=True, harder=False):
        # Create x = (# of valid Orientation, # of valid placements, 6)
        while (not self.state.lost):
            prev_state = self.state.copy()
            best_val = -float("inf")
            # best_action = None
            # best_val_reward = None
            best_state = None
            for action in self.get_actions():
                self.step(action)
                val = np.dot(child, self.features)
                if val > best_val:
                    best_val = val
                    # best_action = action
                    best_state = self.state.copy()
                self.set_state(prev_state)
            self.set_state(best_state)
            if hard:
                self._set_hard_random_piece(harder)
        cummu_reward = self.state.cleared
        self.reset()
        return cummu_reward
    
    def _set_hard_random_piece(self, harder=False):
        # pieces: O, I, L, J, T, S, Z
        
        if harder: # suggested in the paper
            self.state.next_piece = np.random.choice(range(7), p=[1/11, 1/11, 1/11, 1/11, 1/11, 3/11, 3/11])
            
        else:      # slightly easier game
            self.state.next_piece = np.random.choice(range(7), p=[1/9, 1/9, 1/9, 1/9, 1/9, 2/9, 2/9])


if __name__ == "__main__":

    # run a random policy on the tetris simulator

    # np.random.seed(1)
    env = TetrisEnv()
    env.reset()
    env.render()

    for _ in range(50):
        actions = env.get_actions()
        action = actions[np.random.randint(len(actions))]
        state, reward, done, _ = env.step(action)
        if done:
            break
        env.render()

