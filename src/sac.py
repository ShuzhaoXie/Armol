from copy import deepcopy
import os
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import src.sac_core as core
from spinup.utils.logx import EpochLogger
import logging
from src.common import CUR_DEVICE, cur_device, cal_mAP, get_top_k, write_results_to_file, to_tensor, to_numpy, check_exist_dir_path

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=cur_device)
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=cur_device)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32, device=cur_device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=cur_device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=cur_device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: v for k,v in batch.items()}



def sac(train_env_fn, test_env_fn, logger_2_file, tmp_predict, wolp=1, cur_device = CUR_DEVICE,
        actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    logger_2 = logging.getLogger(__name__)
    logger_2.setLevel(level=logging.INFO)

    handler = logging.FileHandler(logger_2_file)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_2.addHandler(handler)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = train_env_fn(), test_env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    
    max_map = 0

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=to_numpy(q1),
                      Q2Vals=to_numpy(q2))

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=to_numpy(logp_pi))

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(o, deterministic)

    # def wolper(a):
    #     k = wolp
    #     a_np = to_numpy(a)
        
    #     out = np.zeros(10)
        
    #     nxt = []
    #     for i in range(len(a_np)):
    #         x = float(a_np[i])
    #         x_1 = abs(x-1)
    #         x_0 = abs(x-0)
    #         if x_1 < x_0:
    #             out[i] = 1
    #             nxt.append(x_0 - x_1)
    #         else:
    #             nxt.append(x_1 - x_0)
        
    #     if np.sum(out) == 0:
    #         ind = int(np.argmin(nxt))
    #         out[ind] = 1
        
        
    #     return to_tensor(out)
        
    def wolper(a):
        k = wolp
        # a_np = to_numpy(a)
        
        # out = np.zeros(10)
        
        # nxt = []
        # for i in range(len(a_np)):
        #     x = float(a_np[i])
        #     x_1 = abs(x-1)
        #     x_0 = abs(x-0)
        #     if x_1 < x_0:
        #         out[i] = 1
        #         nxt.append(0)
        #     else:
        #         nxt.append(1)
        
        # top_k_actions = []
        
        # top_k_actions.append(out)
        
        # for i in range(k):
        #     x = out
        #     x[i] = nxt[i]
        #     top_k_actions.append(x)
            
        # top_k_actions = to_tensor(np.array(top_k_actions))
            
        
        top_k_actions = get_top_k(a, k)
        
        return top_k_actions[0]
        
        # with torch.no_grad():
        #     os = torch.ones((k, 1000), device=cur_device).mul(o)
        #     # print(os.shape, topk_actions.shape)
        #     qv1 = ac.q1(os, top_k_actions)
        #     qv2 = ac.q2(os, top_k_actions)
            
        #     # sum
        #     # qv1 = qv1.add(qv2)
        #     # top1_qv, top1_id = torch.topk(qv1, 1)

        #     # min
        #     qv_min = torch.min(qv1, qv2)
        #     top1_qv, top1_id = torch.topk(qv_min, 1)
            
            # q1 only
            # top1_qv, top1_id = torch.topk(qv1, 1)

            
        
        # return to_tensor(out)
        
    
    def write_predict(file_path, action): 
        results = test_env.get_results(action)
        write_results_to_file(results, file_path)

    def test_agent():
        print('start test')
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            
            cost_all = 0
            actions = []
            sum_action = np.zeros(3)
            check_exist_dir_path(tmp_predict)
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                a = wolper(get_action(o, True))
                a_np = to_numpy(a, dtype=int)
                # print(test_env.t, a_np)
                actions.append(a_np)
                
                predict_file_path = os.path.join(tmp_predict, '{}.txt'.format(ep_len))
                write_predict(predict_file_path, a_np)

                o, r, d, _ = test_env.step(a)

                cost_all += np.sum(a_np)
                sum_action = sum_action + a_np
                ep_ret += r
                ep_len += 1

            # print(actions[:5])
            
            map = cal_mAP(tmp_predict)

            print('mAP: ', map)
            print('cost:', cost_all / ep_len)
            print('sum action:', sum_action)
            logger_2.info('mAP:{} - cost:{} - sum act: {}'.format(map, cost_all / ep_len, sum_action))

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            return map

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    print('interact with environment')
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        
        if t > start_steps:
            a = get_action(o)
        else:
            a = to_tensor(env.action_space.sample())
        # print(a)
        a_step = wolper(a)

        # Step the env
        o2, r, d, _ = env.step(a_step)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            # print('Step {}: update network'.format(t))
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            epoch_end_map = test_agent()
            
            # Save model
            if epoch_end_map > max_map:
                max_map = epoch_end_map
                logger.save_state({'env': env}, None)

            
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     logger.save_state({'env': env}, None)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
