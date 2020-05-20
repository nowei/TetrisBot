import envs
import gym
env = gym.make('Tetris-v0')

print(env.lost, env.cleared, env.turn)