import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym

# isaaclab
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# symdex
import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, customize_cfg


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath('cfg').as_posix(), config_name="default", version_base=None)
def main(hydra_cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(hydra_cfg.seed)
    capture_keyboard_interrupt()
    hydra_cfg, _ = customize_cfg(hydra_cfg)
    env_cfg = parse_env_cfg(hydra_cfg.env_name, device=hydra_cfg.device, num_envs=hydra_cfg.num_envs)
    env = gym.make(hydra_cfg.env_name, cfg=env_cfg, hydra_cfg=hydra_cfg)
    env = VecEnvWrapper(env, rl_device=hydra_cfg.rl_device)
    count = 0
    env.reset()
    env.unwrapped.update_randomization(0.0)
    while simulation_app.is_running():
        actions = torch.rand(env.action_space.shape) * 2 - 1
        env.step(actions)
        count += 1
        if count % 10 == 0:
            count = 0
            env.reset()
            print("-" * 80)
            print("[INFO]: Resetting environment...")
        
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
