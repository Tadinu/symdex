import torch
from typing import Any, ClassVar

from symdex.env.tasks.manager_based_env import *
from symdex.env.tasks.InsertDrawer.env_cfg import InsertDrawerEnvCfg


class InsertDrawerEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: InsertDrawerEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            self.markers['arm_r']['ee_marker'].visualize(self.scene["drawer"].data.root_pos_w, self.scene["drawer"].data.root_quat_w)
            ee_idx = self.scene["drawer"].find_bodies("handle_grip")[0]
            self.markers['arm_r']['goal_marker'].visualize(self.scene["drawer"].data.body_state_w[:, ee_idx, :3].reshape(-1, 3), self.scene["drawer"].data.body_state_w[:, ee_idx, 3:7].reshape(-1, 4))
            # self.markers['arm_l']['ee_marker'].visualize(right_bottom, self.scene["robot"].data.root_quat_w)
            # self.markers['arm_l']['goal_marker'].visualize(right_top, self.scene["robot"].data.root_quat_w)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_init(self):
        super()._pre_init()
        self.success_tracker_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _post_reset(self, env_ids):
        super()._post_reset(env_ids)
        self.success_tracker_step[env_ids] = 0.0