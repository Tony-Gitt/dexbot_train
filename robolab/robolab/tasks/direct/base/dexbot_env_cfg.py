from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.tasks.direct.base import mdp
from robolab.assets.robots import DEXBOT_CFG
from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg, 
    BaseEnvCfg, 
    RewardCfg, 
    HeightScannerCfg, 
    SceneContextCfg, 
    RobotCfg, 
    ObsScalesCfg, 
    NormalizationCfg, 
    CommandRangesCfg, 
    CommandsCfg, 
    NoiseScalesCfg, 
    NoiseCfg, 
    EventCfg,
    GRAVEL_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    ROUGH_HARD_TERRAINS_CFG,
    SceneCfg
)

@configclass
class DEXBOTRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.5, params={"std": 0.4})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.04)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-3)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!leg_.*_6.*).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso.*"])},)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    alive_reward = RewTerm(func=mdp.is_alive,weight=1)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_.*_6.*"), "threshold": 1.2},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_.*_6.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_.*_6.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-9e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_.*_6.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_6.*"]), "min": 0.24, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_4.*"]), "min": 0.24, "max": 0.35},
    )
    base_height = RewTerm(
        func=mdp.body_distance_z,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis_.*"]), "ref": 0.78},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_.*_6.*"])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_6.*"])},
    )
    feet_forward_orientation = RewTerm(
        func=mdp.feet_forward_orientation,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_6.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.01,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=["leg_.*_1", "leg_.*_2", "leg_.*_3"]
    #         )
    #     },
    # )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso.*",]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[ "arm_.*_1", "arm_.*_2", "arm_.*_3", "arm_.*_4", "arm_.*_5", "arm_.*_6", "arm_.*_7",],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_1", "leg_.*_2", "leg_.*_3", "leg_.*_4", "leg_.*_5", "leg_.*_6"])},  #"leg_.*_4"
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["leg_.*_6"])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.16, params={"pos_cfg": SceneEntityCfg("robot", joint_names=["arm.*","leg.*","torso.*"]),
                                                                     "vel_cfg": SceneEntityCfg("robot", joint_names=["arm.*","leg.*","torso.*"]), 
                                                                     "pos_weight": 1.0, "vel_weight": 0.04})
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="leg_.*_6"),
                "asset_cfg": SceneEntityCfg("robot", body_names="leg_.*_6"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.18,"threshold":0.02})



@configclass
class DEXBOTFlatEnvCfg(BaseEnvCfg):

    reward = DEXBOTRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 27
        self.observation_space = 90
        self.state_space = 159
        self.scene_context.robot = DEXBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene_context.height_scanner.enable_height_scan = False
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.feet_body_names = ["leg_.*_6"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link", "pelvis_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link", "pelvis_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["leg_.*_link_.*", "arm_.*_link_.*"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint.*"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint.*"]
        self.robot.action_scale = 0.25
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03


@configclass
class DEXBOTRoughEnvCfg(DEXBOTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.state_space = 346
        # flat state space+11*17
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.reward.ang_vel_xy_l2.weight = -0.05
        self.reward.lin_vel_z_l2.weight = -0.05


