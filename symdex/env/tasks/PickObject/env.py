from __future__ import annotations

import torch
from typing import Any, ClassVar
from functools import reduce
from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from symdex.env.tasks.manager_based_env import BaseEnv
from symdex.env.tasks.PickObject.env_cfg import PickObjectEnvCfg


class PickObjectEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: PickObjectEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # check if the object is on top of the tote
        object_ids = [1, 2]
        tote_top_pos = self.command_manager.get_command("target_pos") + self.scene.env_origins
        for object_id in object_ids:
            object = self.scene[f"object_{object_id}"]
            distance = torch.norm(tote_top_pos - object.data.root_pos_w, dim=1)
            idx = torch.where(distance < 0.08)[0]
            self.object_on_tote_tracker[object_id][idx] += 1

        # check if all objects are in the tote and track the success steps
        success = reduce(torch.logical_and, [self.object_in_tote_tracker[1] >= 3, self.object_in_tote_tracker[2] >= 3]).bool()
        success_idx = torch.where(success)[0]
        unsuccess_idx = torch.where(~success)[0]
        self.success_step_tracker[success_idx] += 1
        self.success_step_tracker[unsuccess_idx] = 0

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
        super()._post_init()
        self.success_step_tracker = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.object_on_tote_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)
        self.object_in_tote_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)

    def _post_reset(self, env_ids):
        super()._post_reset(env_ids)
        self.object_on_tote_tracker[:, env_ids] = 0.0
        self.object_in_tote_tracker[:, env_ids] = 0.0
        self.success_step_tracker[env_ids] = 0
