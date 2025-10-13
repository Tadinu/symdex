from __future__ import annotations

import gym.spaces
import torch
import numpy as np
import math
from isaaclab.envs import VecEnvObs

"""
Vectorized environment wrapper.
"""


class VecEnvWrapper:
    def __init__(self, env, rl_device: str, clip_obs: float = np.inf, clip_actions: float = 1):
        """Initializes the wrapper instance.

        Args:
            env: The environment to wrap around.
            rl_device: The device on which agent computations are performed.
            clip_obs: The clipping value for observations.
            clip_actions: The clipping value for actions.

        Raises:
            ValueError: The environment is not inherited from :class:`ManagerBasedRLEnv`.
            ValueError: If specified, the privileged observations (critic) are not of type :obj:`gym.spaces.Box`.
        """
        # initialize the wrapper
        self.env = env
        # store provided arguments
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.unwrapped.device
        self.max_episode_length_s = env.unwrapped.cfg.episode_length_s
        self.max_episode_length = math.ceil(env.unwrapped.cfg.episode_length_s / (env.unwrapped.cfg.decimation * env.unwrapped.physics_dt))
        
    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return (
            f"<{type(self).__name__}{self.env}>"
            f"\n\tObservations clipping: {self._clip_obs}"
            f"\n\tActions clipping     : {self._clip_actions}"
            f"\n\tAgent device         : {self._rl_device}"
        )

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.sim.render_mode

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        obs_space = self.unwrapped.observation_space["policy"]
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, obs_space.shape)

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        action_space = self.unwrapped.action_space
        action_space_shape = action_space.shape
        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space_shape)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        """Returns the base environment simulation device."""
        return self.unwrapped.device

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        """Returns the Gym spaces for the environment."""
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):  # noqa: D102
        obs_dict, extras = self.env.reset()
        # process observations and states
        return self._process_obs(obs_dict), extras

    def step(self, actions):  # noqa: D102
        actions = actions.detach().clone().to(device=self._sim_device)
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        # process observations and states
        obs = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }
        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs, rew, dones.float(), extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _process_obs(self, obs_dict: VecEnvObs) -> torch.Tensor | dict[str, torch.Tensor]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms.

        Args:
            obs_dict: The current observations from environment.

        Returns:
            If environment provides states, then a dictionary containing the observations and states is returned.
            Otherwise just the observations tensor is returned.
        """
        # process policy obs
        obs = obs_dict["policy"]
        # clip the observations
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        # move the buffer to rl-device
        obs = obs.to(device=self._rl_device).clone()
        
        return obs

