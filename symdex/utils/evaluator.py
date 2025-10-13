import time
from copy import deepcopy
import torch

from symdex.utils.common import Tracker
#from symdex.utils.model_util import save_model
from symdex.utils.symmetry import SymmetryManager


class Evaluator:
    def __init__(self, cfg, env_cfg, env, wandb_run):
        cfg = deepcopy(cfg)
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.env = env
        self.wandb_run = wandb_run
        self.eval_policy = self.eval_policy_sp

        self.start_time = time.time()

    def eval_policy_sp(self, policy, value, algo_multi_cfg=None, normalizer=None, success_max=None):
        num_envs = self.cfg.num_envs
        max_step = self.env.max_episode_length
        if self.cfg.algo.multi_agent:
            symmetry_manager = SymmetryManager(cfg=algo_multi_cfg, symmetric_envs=self.cfg.task.symmetry.symmetric_envs)

        tracker_capacity = num_envs
        return_tracker = Tracker(tracker_capacity)
        step_tracker = Tracker(tracker_capacity)
        success_tracker = Tracker(tracker_capacity)
        current_returns = torch.zeros(num_envs, dtype=torch.float32, device=self.cfg.device)
        current_lengths = torch.zeros(num_envs, dtype=torch.float32, device=self.cfg.device)
        if_done = torch.ones(num_envs, dtype=torch.float32, device=self.cfg.device)
        obs, _ = self.env.reset()

        for i_step in range(max_step):  # run an episode
            if self.cfg.algo.obs_norm:
                obs = normalizer.normalize(obs)

            obs_list = symmetry_manager.get_multi_agent_obs(obs, self.env.unwrapped.symmetry_tracker)
            assert len(policy) == 2, "This version only support bimanual case"
            action1 = policy[0](obs_list[0], sample=False)
            action2 = policy[1](obs_list[1], sample=False)
            action = symmetry_manager.get_execute_action(action1, action2, self.env.unwrapped.symmetry_tracker)
            next_obs, reward, done, info = self.env.step(action)

            current_returns += reward
            current_lengths += 1
            env_done_indices = torch.where(done)[0]
            first_done = torch.logical_and(done, if_done)
            first_done = torch.where(first_done)[0]
            if_done[first_done] = 0.0

            return_tracker.update(current_returns[first_done])
            step_tracker.update(current_lengths[first_done])
            success_tracker.update(info['success'][first_done])
            current_returns[env_done_indices] = 0
            current_lengths[env_done_indices] = 0
            obs = next_obs
        
        self.env.obs = obs
        self.env.dones = done

        ret_mean = return_tracker.mean()
        step_mean = step_tracker.mean()
        success_mean = success_tracker.mean()

        return_dict = {'eval/return': ret_mean, 'eval/episode_length': step_mean, 'eval/success_rate': success_mean}
        if success_max is not None:
            if success_mean > success_max:
                success_max = success_mean
                if self.cfg.save_model:
                    save_model(path=f"{self.wandb_run.dir}/model.pth",
                                actor=policy if isinstance(policy, list) else policy.state_dict(),
                                critic=value if isinstance(value, list) or value is None else value.state_dict(),
                                rms=normalizer.get_states() if self.cfg.algo.obs_norm else None,
                                wandb_run=self.wandb_run,
                                description=f"success_rate={success_max:.2f} reward={ret_mean:.2f} step={step_mean:.2f}")
            return return_dict, success_max
        return return_dict

    def check_if_should_stop(self, step=None):
        if self.cfg.max_step is not None:
            return step > self.cfg.max_step
        else:
            return (time.time() - self.start_time) > self.cfg.max_time
