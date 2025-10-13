from __future__ import annotations

import gymnasium as gym
from collections.abc import Sequence
from typing import Any, ClassVar, Union

import carb
from isaacsim.core.version import get_version
import omni.log
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.utils.timer import Timer
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.contact_sensor import ContactSensor

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from symdex.env.tasks.manager_based_env_cfg import BaseEnvCfg
from symdex.utils.domain_random import DomainRandomizer
from symdex.env.mdps.randomization_mdps import *
from symdex.utils.symmetry import load_symmetric_system


class BaseEnv(ManagerBasedRLEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: BaseEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        self._pre_init(cfg)
        super().__init__(cfg=cfg)
        self.hydra_cfg = kwargs['hydra_cfg']

        # allocate dictionary to store metrics
        self.extras = {}

        # environment specific initialization
        self._post_init()

    def _pre_init(self, cfg):
        pass

    def _post_init(self):
        # initialize IK visualizer
        if self.cfg.visualize_marker:
            self.markers = dict()
            # initialize marker for each arm
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = self.cfg.visualizer_scale
            self.markers['arm_l'] = dict()
            self.markers['arm_l']['ee_marker'] = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
            self.markers['arm_l']['goal_marker'] = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/left_ee_goal"))
            self.markers['arm_r'] = dict()
            self.markers['arm_r']['ee_marker'] = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
            self.markers['arm_r']['goal_marker'] = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/right_ee_goal"))

        # save the scale as tensors
        if self.cfg.action_scale is not None:
            self._scale = torch.tensor(self.cfg.action_scale, device=self.device)
        else:
            self._scale = torch.ones(self.action_space.shape, device=self.device)

        self.last_action = torch.zeros(self.action_space.shape, device=self.device)
        self.object_init_pos = torch.zeros((self.num_object, self.num_envs, 3), device=self.device)
        self.object_init_orient = torch.zeros((self.num_object, self.num_envs, 4), device=self.device)
        self.success_tracker = None
        self.symmetry_tracker = None
        self.domain_randomizer = DomainRandomizer(self.hydra_cfg.task.randomize, num_envs=self.num_envs)

        # symmetry
        self.G = load_symmetric_system(cfg=self.hydra_cfg.task.symmetry)
        assert self.hydra_cfg.task.symmetry.group_label == 'C2'
        symmetric_group = self.G.elements[-1]
        self.rep_Rd = self.G.representations['R3'](symmetric_group)
        self.rep_Rd = torch.tensor(self.rep_Rd, device=self.device, dtype=torch.float32)
        self.rep_SO3_flat = self.G.representations['SO3_flat'](symmetric_group)
        self.rep_SO3_flat = torch.tensor(self.rep_SO3_flat, device=self.device, dtype=torch.float32)
        self.rep_Q_js = self.G.representations['Q_js'](symmetric_group)
        self.rep_Q_js = torch.tensor(self.rep_Q_js, device=self.device, dtype=torch.float32)

        self.all_time_first = True

    @property
    def action_dim(self):
        return self.cfg.action_dim

    @property
    def num_object(self):
        return self.cfg.num_object

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # update last action inferred from the policy. this is different from the last action used in IsaacLab action manager
        self.last_action = action.clone()
        # scale the action
        action = action * self._scale
        super().step(action)

        self.extras['success'] = self.success_tracker
        self.extras['detailed_reward'] = self.reward_manager._episode_sums
        # post reset process
        reset_indices = torch.where(self.episode_length_buf == self.num_envs)[0]
        if len(reset_indices) > 0:
            self._post_reset(reset_indices)
        
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def update_randomization(self, success_rate):
        self.domain_randomizer.update(success_rate)
        randomized_values, randomization_state, curriculum_state = self.domain_randomizer.sample()
        if "object_mass" in randomized_values:
            randomize_mass(self, randomized_values["object_mass"])
        if "static_friction" in randomized_values:
            randomize_material(self, static_friction=randomized_values["static_friction"],
                               static_friction_range=list(self.hydra_cfg.task.randomize.randomization.static_friction),
                               dynamic_friction=randomized_values["dynamic_friction"], 
                               dynamic_friction_range=list(self.hydra_cfg.task.randomize.randomization.dynamic_friction),
                               restitution=randomized_values["restitution"], 
                               restitution_range=list(self.hydra_cfg.task.randomize.randomization.restitution), 
                               num_buckets=250)
        if "action_scale" in randomized_values:
            self._scale[:6] = randomized_values["action_scale"]
        if "energy_penalty" in randomized_values:
            for rew_name in curriculum_state["energy_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["energy_penalty"])
        if "collision_penalty" in randomized_values:
            for rew_name in curriculum_state["collision_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["collision_penalty"])
        if "external_force_torque" in randomized_values:
            randomize_external_force_torque(self, force=randomized_values["external_force_torque"], torque=randomized_values["external_force_torque"])
        # reset pose
        for rand_name in randomized_values.keys():
            if "reset_pose" in rand_name:
                randomize_reset_pose(self, curriculum_state[rand_name]["names"], randomized_values[rand_name])
        return randomization_state, curriculum_state, self.domain_randomizer.best_so_far

    """
    Helper functions
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            group_term_dim = self.observation_manager.group_obs_term_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_term_dim)
                })
        # action space (unbounded since we don't impose any limits)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
    
    def _reset_idx(self, env_ids: Union[Sequence[int], torch.Tensor]):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene["robot"]._ALL_INDICES
        self.scene["robot"].reset(env_ids)
        if 'robot_left' in self.scene.keys():
            self.scene["robot_left"].reset(env_ids)
        
        if self.symmetry_tracker is None:
            self.symmetry_tracker = torch.zeros(self.num_envs, device=self.device)

        if self.hydra_cfg.task.symmetry.symmetric_envs:
            if self.all_time_first:
                self.all_time_first = False
                symmetric_env_ids = None
            else:
                mask = torch.rand(env_ids.shape) < 0.5
                symmetric_env_ids = env_ids[mask]
                other_env_ids = env_ids[~mask]
                self.symmetry_tracker[symmetric_env_ids] = 1
                self.symmetry_tracker[other_env_ids] = 0

        super()._reset_idx(env_ids)
        if self.hydra_cfg.task.symmetry.symmetric_envs and symmetric_env_ids is not None:
            if self.hydra_cfg.task.symmetry.group_label == 'C4':
                self._reset_symmetry_c4(env_ids)
            else:
                self._reset_symmetry(symmetric_env_ids)
        self._post_reset(env_ids)

    def _post_reset(self, env_ids):
        """
        Post reset process for the environment. Call to reset the information at step 1 to ensure the correctness of the simulation.
        """
        for obj_id in range(self.num_object):
            self.object_init_pos[obj_id, env_ids, :3] = self.scene[f"object_{obj_id}"].data.root_pos_w[env_ids, :3]
            self.object_init_orient[obj_id, env_ids, :4] = self.scene[f"object_{obj_id}"].data.root_quat_w[env_ids, :4]
        if not self.hydra_cfg.task.symmetry.symmetric_envs:
            info = self.command_manager.reset(env_ids)
            self.extras["log"].update(info)
    
    def _reset_symmetry(self, symmetric_env_ids: Union[Sequence[int], torch.Tensor]):
        reset_symmetry = dict(self.hydra_cfg.task.symmetry.reset_symmetry)
        for key, value in reset_symmetry.items():
            if value["type"] == "articulation" or value["type"] == "rigid_object":
                asset = self.scene[key]
                if "position" in value["symmetry"]:
                    cur_position = asset.data.root_pos_w[symmetric_env_ids] - self.scene.env_origins[symmetric_env_ids]
                    symmetry_position = cur_position @ self.rep_Rd.T
                    symmetry_position = symmetry_position + self.scene.env_origins[symmetric_env_ids]
                else:
                    symmetry_position = asset.data.root_pos_w[symmetric_env_ids]
                if "quat" in value["symmetry"]:
                    cur_R = math_utils.matrix_from_quat(asset.data.root_quat_w[symmetric_env_ids])
                    cur_R_flat = cur_R.transpose(1, 2).reshape(-1, 9)
                    symmetry_R_flat = cur_R_flat @ self.rep_SO3_flat.T
                    symmetry_R = symmetry_R_flat.view(-1, 3, 3).transpose(1, 2)
                    symmetry_quat = math_utils.quat_from_matrix(symmetry_R)
                else:
                    symmetry_quat = asset.data.root_quat_w[symmetric_env_ids]
                asset.write_root_pose_to_sim(torch.cat([symmetry_position, symmetry_quat], dim=-1), env_ids=symmetric_env_ids)
            elif value["type"] == "command":
                command = self.command_manager.get_term(key)
                if "position" in value["symmetry"]:
                    cur_pos_command_e = command.pos_command_e[symmetric_env_ids]
                    command.pos_command_e[symmetric_env_ids] = cur_pos_command_e @ self.rep_Rd
                    command.pos_command_w[symmetric_env_ids] = command.pos_command_e[symmetric_env_ids] + self.scene.env_origins[symmetric_env_ids]
                if "quat" in value["symmetry"]:
                    cur_R = math_utils.matrix_from_quat(command.quat_command_w[symmetric_env_ids])
                    cur_R_flat = cur_R.transpose(1, 2).reshape(-1, 9)
                    symmetry_R_flat = cur_R_flat @ self.rep_SO3_flat.T
                    symmetry_R = symmetry_R_flat.view(-1, 3, 3).transpose(1, 2)
                    symmetry_quat = math_utils.quat_from_matrix(symmetry_R)
                    command.quat_command_w[symmetric_env_ids] = symmetry_quat
            else:
                raise ValueError(f"Unknown symmetry type: {value['type']}")