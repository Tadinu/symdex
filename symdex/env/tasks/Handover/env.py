from __future__ import annotations

import torch
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from symdex.env.tasks.manager_based_env import BaseEnv
from symdex.env.tasks.Handover.env_cfg import HandoverEnvCfg

class HandoverEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: HandoverEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        self.reach_middle += self.command_manager.get_term("target_pos").metrics["consecutive_success"] >= 5
        # Debug only
        if self.cfg.visualize_marker:
            right_palm_idx = self.scene["robot"].find_bodies("palm_link")[0]
            right_palm = self.scene["robot"].data.body_state_w[:, right_palm_idx, :7].reshape(-1, 7)
            left_palm_idx = self.scene["robot_left"].find_bodies("palm_link")[0]
            left_palm = self.scene["robot_left"].data.body_state_w[:, left_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_r']['ee_marker'].visualize(right_palm[:, :3], right_palm[:, 3:7])
            self.markers['arm_r']['goal_marker'].visualize(left_palm[:, :3], left_palm[:, 3:7])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _post_init(self):
        self.reach_middle = torch.zeros(self.hydra_cfg.num_envs, device=self.device)
        self.success_tracker_step = torch.zeros(self.hydra_cfg.num_envs, device=self.device, dtype=torch.long)

    def _post_reset(self, env_ids):
        super()._post_reset(env_ids)
        self.reach_middle[env_ids] = 0.0
        self.success_tracker_step[env_ids] = 0.0