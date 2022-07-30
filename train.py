from time import clock_getres
import torch
import os
import sys
os.chdir(sys.path[0])
from torch import nn
import numpy as np
# from ceva_env_large import CEVA_LARGE
from src.sac import sac
from src.armor_test import ARMOR_TEST
from src.armor_train import ARMOR_TRAIN
from src.common import OPT_DIR, check_exist_dir_path

# from test_policy import test_ppo
# from spinup import ddpg_pytorch, ppo_pytorch, td3_pytorch, sac_pytorch

exp_name = 'sac_train_test'
opt_directory = os.path.join(OPT_DIR, exp_name)
check_exist_dir_path(opt_directory)
tmp_predict_directory = os.path.join(opt_directory, 'tmp_predict')
check_exist_dir_path(tmp_predict_directory)

logger_kwargs = dict(output_dir=opt_directory,
                     exp_name=exp_name)

# sac(env_fn=MyEnv, ac_kwargs={}, seed=0, steps_per_epoch=100, epochs=40, replay_size=100000, gamma=0.99,
#             polyak=0.995, lr=0.001, alpha=0.2, batch_size=100, start_steps=50, update_after=10, update_every=50,
#             num_test_episodes=10, max_ep_len=100, logger_kwargs=logger_kwargs, save_freq=1)

# test_ppo(opt_dir)


sac(train_env_fn=ARMOR_TRAIN, test_env_fn=ARMOR_TEST, logger_2_file=os.path.join(opt_directory, 'log_2.txt'), 
    tmp_predict=tmp_predict_directory, ac_kwargs={}, seed=0, steps_per_epoch=2500, epochs=20, replay_size=1000000, gamma=0.9,
    polyak=0.995, lr=0.0005, alpha=0.2, batch_size=1024, start_steps=1500, update_after=2000, update_every=50,
    num_test_episodes=1, max_ep_len=4952, logger_kwargs=logger_kwargs, save_freq=1)

# sac(env_fn=CEVA, ac_kwargs={}, seed=0, steps_per_epoch=100, epochs=10, replay_size=1000000, gamma=0.99,
#             polyak=0.995, lr=0.001, alpha=0.2, batch_size=5, start_steps=40, update_after=40, update_every=10,
#             num_test_episodes=1, max_ep_len=4952, logger_kwargs=logger_kwargs, save_freq=1)


# 0.05 0.05 3
# 0.025 0.025 4
# 0.001 0.001 15 not suit

# 0.05 0.05 25 4

# 0.05 0.05 50 0.1
