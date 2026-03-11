
import os

from robolab.assets.robots import DEXBOT_CFG
from robolab.tasks.manager_based.beyondmimic.beyondmimic_env_cfg import BeyondMimicEnvCfg

from isaaclab.utils import configclass
from robolab import ROBOLAB_ROOT_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
import robolab.tasks.manager_based.beyondmimic.mdp as mdp
from isaaclab.managers import SceneEntityCfg


@configclass
class DexbotBeyondMimicEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DEXBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "dexbot_bm", "dance.npz"
        )
        self.commands.motion.anchor_body_name = "pelvis_link"
        self.commands.motion.body_names = [
            "pelvis_link",
            "leg_r_link_1",
            # "leg_r_link_2",
            # "leg_r_link_3",
            "leg_r_link_4",
            # "leg_r_link_5",
            "leg_r_link_6",
            "leg_l_link_1",
            # "leg_l_link_2",
            # "leg_l_link_3",
            "leg_l_link_4",
            # "leg_l_link_5",
            "leg_l_link_6",
            "torso_link",
            "arm_r_link_1",
            # "arm_r_link_2",
            # "arm_r_link_3",
            "arm_r_link_4",
            # "arm_r_link_5",
            # "arm_r_link_6",
            "arm_r_link_7",
            "arm_l_link_1",
            # "arm_l_link_2",
            # "arm_l_link_3",
            "arm_l_link_4",
            # "arm_l_link_5",
            # "arm_l_link_6",
            "arm_l_link_7",
        ]

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names=[
            r"^(?!leg_l_link_6$)(?!leg_r_link_6$).+$"
        ]
        self.rewards.feet_slide.params["sensor_cfg"].body_names="leg_.*_link_6"
        self.rewards.feet_slide.params["asset_cfg"].body_names="leg_.*_link_6"
        self.rewards.feet_orientation_l2.params["sensor_cfg"].body_names="leg_.*_link_6"
        self.rewards.feet_orientation_l2.params["asset_cfg"].body_names="leg_.*_link_6"
        # self.terminations.ee_body_pos.params["body_names"]=None

        self.commands.motion.reset_on_motion_end = False
        self.rewards.stand_still_after_motion = RewTerm(
            func=mdp.stand_still_after_motion,
            weight=-0.2,
            params={
                "command_name": "motion",
                "pos_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "vel_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_weight": 1.0,
                "vel_weight": 0.04,
            },
        )

        self.events.randomize_push_robot.interval_range_s = (0.0, 5.0)

        self.episode_length_s = 20.0