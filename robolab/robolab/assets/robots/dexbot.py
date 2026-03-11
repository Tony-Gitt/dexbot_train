import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

DEXBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        force_usd_conversion=True,
        asset_path=f"{ISAAC_DATA_DIR}/robots/dexbot/urdf/dexbot.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "leg_r_joint_1": -0.35,
            "leg_r_joint_2": 0.06,
            "leg_r_joint_3": 0.18,
            "leg_r_joint_4": 0.72,
            "leg_r_joint_5": -0.4,
            "leg_r_joint_6": 0.00,
            "leg_l_joint_1": -0.35,
            "leg_l_joint_2": -0.06,
            "leg_l_joint_3": -0.18,
            "leg_l_joint_4": 0.72,
            "leg_l_joint_5": -0.4,
            "leg_l_joint_6": 0.00,
            "torso_joint": 0.0,
            "arm_r_joint_1": -0.04,
            "arm_r_joint_2": -0.40,
            "arm_r_joint_3": 0.13,
            "arm_r_joint_4": -1.38,
            "arm_r_joint_5": 0,
            "arm_r_joint_6": 0,
            "arm_r_joint_7": 0,
            "arm_l_joint_1": -0.04,
            "arm_l_joint_2": 0.40,
            "arm_l_joint_3": 0.13,
            "arm_l_joint_4": 1.38, 
            "arm_l_joint_5": 0,
            "arm_l_joint_6": 0,
            "arm_l_joint_7": 0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs1": DelayedPDActuatorCfg(
            joint_names_expr=[
                "leg_.*_joint_3",
                "leg_.*_joint_5",
                "leg_.*_joint_6",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=25.0,
            stiffness={
                "leg_.*_joint_3":100,
                "leg_.*_joint_5":40,
                "leg_.*_joint_6":40,
            },
            damping={
                "leg_.*_joint_3":5,
                "leg_.*_joint_5":2,
                "leg_.*_joint_6":2,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "legs2": DelayedPDActuatorCfg(
            joint_names_expr=[
                "leg_.*_joint_1",
                "leg_.*_joint_2",
                "leg_.*_joint_4",
                ".*torso.*",
            ],
            effort_limit_sim=260.0,
            velocity_limit_sim=25.0,
            stiffness={
                "leg_.*_joint_1":100,
                "leg_.*_joint_2":100,
                "leg_.*_joint_4":150,
                ".*torso.*":150,
            },
            damping={
                "leg_.*_joint_1":5,
                "leg_.*_joint_2":5,
                "leg_.*_joint_4":7.5,
                ".*torso.*":7.5,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
                        
        # "feet": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        #     effort_limit_sim=27.0,
        #     velocity_limit_sim=8.0,
        #     stiffness=40.0,
        #     damping=2.0,
        #     armature=0.01,
        #     min_delay=0,
        #     max_delay=2,
        # ),
        # "shoulders": DelayedPDActuatorCfg(
        #     joint_names_expr=[
        #         ".*_arm_pitch_joint",
        #         ".*_arm_roll_joint",
        #         ".*_arm_yaw_joint",
        #     ],
        #     effort_limit_sim=27.0,
        #     velocity_limit_sim=8.0,
        #     stiffness=40.0,
        #     damping=2.0,
        #     armature=0.01,
        #     min_delay=0,
        #     max_delay=2,
        # ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*arm_.*"
            ],
            stiffness={
                ".*arm_.*": 30.0,
            },
            damping={
                ".*arm_.*": 1.5,
            },
            effort_limit_sim=60.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)
