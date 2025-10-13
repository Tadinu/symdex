
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import torch
import hydra
from omegaconf import DictConfig
from loguru import logger
import gymnasium as gym

import symdex
from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, load_class_from_path, customize_cfg, Tracker
from symdex.algo.network import model_name_to_path
from symdex.algo.network.mlp import EquivariantMLPNet
from symdex.utils.model_util import load_model
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.symmetry import SymmetryManager
from symdex.utils.common import CONFIG_DIR, CONFIG_NAME

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base=None)
def main(hydra_cfg: DictConfig):
    set_random_seed(hydra_cfg.seed)
    capture_keyboard_interrupt()
    hydra_cfg, _ = customize_cfg(hydra_cfg)
    env_cfg = parse_env_cfg(hydra_cfg.env_name, device=hydra_cfg.device, num_envs=hydra_cfg.num_envs)
    env_cfg.seed = hydra_cfg.seed
    env = gym.make(hydra_cfg.env_name, cfg=env_cfg, hydra_cfg=hydra_cfg)
    env = VecEnvWrapper(env, rl_device=hydra_cfg.rl_device)
    device = torch.device(hydra_cfg.device)
    act_class: type[EquivariantMLPNet] = load_class_from_path(hydra_cfg.algo.act_class,
                                                              model_name_to_path[hydra_cfg.algo.act_class])
    
    multi_agent_cfg = hydra_cfg.task.multi.SYMDEX
    symmetry_cfg = hydra_cfg.task.symmetry.SYMDEX
    action_dim = [22, 22]
    actor = []
    for k in range(len(multi_agent_cfg.single_agent_obs_dim)):
        if "Equivariant" in hydra_cfg.algo.act_class:
            cur_actor = act_class(env.unwrapped.G, symmetry_cfg.actor_input_fields[k], symmetry_cfg.actor_output_fields[k], multi_agent_cfg.single_agent_obs_dim[k], action_dim[k]).to(device)
        else:
            cur_actor = act_class(hydra_cfg.task.multi.SYMDEX.single_agent_obs_dim[k], hydra_cfg.task.multi.SYMDEX.single_agent_action_dim).to(device)
        load_model(cur_actor, f"actor_{k}", hydra_cfg.artifact)
        actor.append(cur_actor)    
    symmetry_manager = SymmetryManager(cfg=multi_agent_cfg, symmetric_envs=hydra_cfg.task.symmetry.symmetric_envs)

    return_tracker = Tracker(hydra_cfg.num_envs)
    step_tracker = Tracker(hydra_cfg.num_envs)
    current_rewards = torch.zeros(hydra_cfg.num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(hydra_cfg.num_envs, dtype=torch.float32, device=device)

    if hydra_cfg.task.randomize.eval:
        env.unwrapped.update_randomization(1.0)
    else:
        env.unwrapped.update_randomization(0.0)
    obs, _ = env.reset()
    for _ in range(env.max_episode_length * 100):  # run 100 episodes
        with torch.no_grad():
            obs_list = symmetry_manager.get_multi_agent_obs(obs, env.unwrapped.symmetry_tracker)
            action1 = actor[0](obs_list[0], sample=False)
            action2 = actor[1](obs_list[1], sample=False)
            action = symmetry_manager.get_execute_action(action1, action2, env.unwrapped.symmetry_tracker)
        next_obs, reward, done, info = env.step(action)
        current_rewards += reward
        current_lengths += 1
        env_done_indices = torch.where(done)[0]
        return_tracker.update(current_rewards[env_done_indices])
        step_tracker.update(current_lengths[env_done_indices])
        current_rewards[env_done_indices] = 0
        current_lengths[env_done_indices] = 0
        obs = next_obs

    r_exp = return_tracker.mean()
    step_exp = step_tracker.mean()
    logger.warning(f"Cumulative return: {r_exp}, Episode length: {step_exp}")


if __name__ == '__main__':
    main()