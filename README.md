# TetrisBot instructions

## How to run
To run, you need a `settings.config` file like in the main directory.

`settings.config`:
```
[DEFAULT]
TRAIN_SAVE_PATH = ./weights/curr_iter.txt
TEST_PATH = ./test_params/test_iter.txt
hard = true
harder = false
multiprocessing = false
num_episodes = 5
mean = [0, 0, 0, 0, 0, 0]
p_c = [0, 0, 0, 0, 0, 0]
p_sigma = [0, 0, 0, 0, 0, 0]
sigma = 0.1
```

Then you can simply run `train.py` and it will save weights.

## Params for running the scripts
### Save locations
* `TRAIN_SAVE_PATH` - location where the results of your training will go, used for running `train.py`
* `TEST_PATH` - File of weights you want to test, used for running `player.py`

### Training options
* `hard` - enable training with hard mode (2/9 probability for S and Z)
* `harder` - enable training with harder mode (3/11 probability for S and Z) (requires `hard` to be `true`)
* `multiprocessing` - enable simple multithreading of child environments using multiprocessing.Pool, sends one child and environment to each pool worker
* `num_episodes` - number of episodes to average performance of children across
* `mean` - initial weighting of the 6 Dellacherie features
* `p_c` - evolution path
* `p_sigma` - conjugate evolution path
* `sigma` - step size (gets adaptively tuned by CMA-ES algorithm)


