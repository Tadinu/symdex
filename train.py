from itertools import count
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

# isaaclab
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# symdex
from symdex.algo import alg_name_to_path
from symdex.algo.ac_base import ActorCriticBase
from symdex.utils.common import init_wandb, load_class_from_path, set_random_seed, capture_keyboard_interrupt, customize_cfg
from symdex.utils.model_util import load_model
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.evaluator import Evaluator
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.common import CONFIG_DIR, CONFIG_NAME

# trackio
# NOTE: Must be after [symdex.utils.common], which defines os env TRACKIO_DIR
import trackio as wandb

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base=None)
def main(hydra_cfg: DictConfig):
    set_random_seed(hydra_cfg.seed)
    capture_keyboard_interrupt()
    hydra_cfg, _ = customize_cfg(hydra_cfg)
    env_cfg = parse_env_cfg(hydra_cfg.env_name, device=hydra_cfg.device, num_envs=hydra_cfg.num_envs)
    env_cfg.seed = hydra_cfg.seed
    env = gym.make(hydra_cfg.env_name, cfg=env_cfg, hydra_cfg=hydra_cfg)
    env = VecEnvWrapper(env, rl_device=hydra_cfg.rl_device, clip_obs=50.0)

    algo_name = hydra_cfg.algo.name
    if 'Agent' not in algo_name:
        algo_name = 'Agent' + algo_name
    agent_class: type[ActorCriticBase] = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    agent = agent_class(env=env,
                        cfg=hydra_cfg,
                        obs_dim=hydra_cfg.task.multi.single_agent_obs_dim if hydra_cfg.algo.multi_agent
                            else env.observation_space.shape[-1],
                        action_dim=hydra_cfg.task.multi.single_agent_action_dim if hydra_cfg.algo.multi_agent
                            else env.action_space.shape[-1])

    wandb_run = init_wandb(hydra_cfg)
    if hydra_cfg.artifact is not None:
        load_model(agent.actor, "actor_0", hydra_cfg.artifact)
        load_model(agent.actor_left, "actor_1", hydra_cfg.artifact)
        if hydra_cfg.algo.obs_norm:
            load_model(agent.obs_rms, "obs_rms", hydra_cfg.artifact)

    global_steps = 0
    success_max = float('-inf')
    evaluator = Evaluator(cfg=hydra_cfg, env_cfg=env_cfg, env=env, wandb_run=wandb_run)

    randomization_state, best_so_far = None, None
    if hydra_cfg.task.randomize.eval:
        env.unwrapped.update_randomization(1.0)
    else:
        env.unwrapped.update_randomization(0.0)
    agent.reset_agent()

    for iter_t in count():
        if iter_t % hydra_cfg.algo.eval_freq == 0:
            return_dict, success_max = (
                evaluator.eval_policy([agent.actor, agent.actor_left] if hydra_cfg.algo.multi_agent else agent.actor,
                                      [agent.critic, agent.critic_left] if hydra_cfg.algo.multi_agent else agent.critic,
                                      algo_multi_cfg=agent.algo_multi_cfg if hydra_cfg.algo.multi_agent else None,
                                      normalizer=agent.obs_rms, success_max=success_max))
            wandb.log(return_dict, step=global_steps)
            agent.reset_agent()
            
        trajectory, steps = agent.explore_env(env, hydra_cfg.algo.horizon_len, random=False)
        global_steps += steps
        log_info = agent.update_net(trajectory)

        if iter_t % hydra_cfg.algo.log_freq == 0:
            log_info['global_steps'] = global_steps
            for key in agent.detailed_tracker.keys():
                log_info[f'Rewards/{key}'] = agent.detailed_tracker[key].mean()
                
            if randomization_state is not None:
                for param in randomization_state.keys():
                    log_info[f'Randomization/{param}_sigma'] = randomization_state[param]['sigma']
                for parm in curriculum_state.keys():
                    val = curriculum_state[parm]
                    if isinstance(val, DictConfig):
                        continue
                    log_info[f'Curriculum/{parm}_value'] = val[min(best_so_far, len(val) - 1)]
                log_info['Randomization/best_so_far'] = best_so_far

            wandb.log(log_info, step=global_steps)

        if iter_t % hydra_cfg.task.randomize.update_freq == 0 and hydra_cfg.task.randomize.enable:
            # domain randomization
            randomization_state, curriculum_state, best_so_far = env.unwrapped.update_randomization(log_info['train/success_rate'])
            success_max = float('-inf')

        if evaluator.check_if_should_stop(global_steps):
            print(f"Training ended after {evaluator.cfg.max_step} steps or {evaluator.cfg.max_time} seconds")
            #print(OmegaConf.to_yaml(hydra_cfg))
            wandb.finish()
            break

if __name__ == '__main__':
    main()
