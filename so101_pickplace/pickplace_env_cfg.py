"""SO-101 single-block pick-and-place environment for GR00T N1.7 post-training data.

Scene: SO-101 follower inside the white hackathon light box (dimensions of ``lightbox-simple.usd``
from isaac-sim/Sim-to-Real-SO-101-Workshop), one blue block, the A6 flyer as placement target and two
fixed cameras on the top edges of the box walls: ``front`` on the wall facing the arm, ``side`` on
the wall to the arm's right. This replaces the starter repo's SeattleLab table and wrist camera so
the scene matches the real station.

Three variants share the scene and differ only in the action space:

* :class:`SO101PickPlaceIKEnvCfg`     -- task-space pose IK (used by the scripted demo collector).
* :class:`SO101PickPlaceJointEnvCfg`  -- absolute joint targets [rad] (used for GR00T closed-loop
  evaluation and for converting leader-arm recordings).
* :class:`SO101PickPlaceTeleopEnvCfg` -- joint env + IsaacTeleop leader-arm pipeline for
  ``scripts/tools/record_demos.py`` (requires the optional ``isaacteleop`` package).
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.so101 import SO101_HIGH_PD_CFG
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import SO101PoseIKControllerCfg
from isaaclab_tasks.contrib.stack.mdp import stack_events
from isaaclab_tasks.contrib.stack.stack_env_cfg import PhysicsCfg

from . import mdp
from .so101_units import ARM_JOINTS, GRIPPER_CLOSE_RAD, GRIPPER_OPEN_RAD

##
# Geometry constants
##

# Camera resolution (matches the real C270 recordings in ~/Desktop/nvidia/datasets).
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Block: the printable cube set from ~/Downloads/cube_blocks (copied to assets/cube_blocks). The
# .usda files are generated from the same STL meshes (1 mm chamfer, mass, convex-hull collision,
# PLA friction), so the simulated block matches the printed one. 30 mm is the size recommended for
# SO-100/101 grippers; override with CUBE_SIZE_MM=25|30|40 and CUBE_COLOR=red|green|blue|yellow.
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
CUBE_SIZE_MM = int(os.environ.get("CUBE_SIZE_MM", "30"))
CUBE_COLOR = os.environ.get("CUBE_COLOR", "blue")
CUBE_USD = os.path.join(ASSETS_DIR, "cube_blocks", "usd", f"cube_{CUBE_SIZE_MM}mm_{CUBE_COLOR}.usda")
CUBE_SIZE = CUBE_SIZE_MM / 1000.0
# Target: the A6 flyer taped to the station floor (long side pointing into the box). Position read
# off a top-down phone photo and the two camera views; physics uses an invisible cuboid of the same
# footprint, the look comes from assets/flyer/flyer_a6.usda (photo texture).
PAD_SIZE = (0.148, 0.105, 0.002)
FLYER_POS = (float(os.environ.get("FLYER_X", "0.11")), float(os.environ.get("FLYER_Y", "-0.235")))
FLYER_USD = os.path.join(ASSETS_DIR, "flyer", "flyer_a6.usda")

# Workspace in the env frame [m]. The robot base is at the origin, rotated 90 deg about +Z so the
# arm faces +X (same seat as the contrib stack task). Block spawns on the +Y half, pad on the -Y half.
CUBE_POSE_RANGE = {"x": (0.16, 0.27), "y": (0.02, 0.13), "z": (CUBE_SIZE / 2, CUBE_SIZE / 2), "yaw": (-0.8, 0.8)}

# Seat pose (see contrib stack task for the derivation of the Z correction and 90 deg yaw).
_SO101_MOUNT_Z = -0.03008
_SO101_BASE_SEAT_POS = (0.0, 0.0, _SO101_MOUNT_Z)
_SO101_BASE_SEAT_ROT = (0.0, 0.0, 0.70710678, 0.70710678)  # xyzw

# Reset joint pose [rad]: gripper above the workspace, pointing down.
SO101_INIT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 1.4,  # jaws point straight down (measured: gripper-link +Z == base +Z)
    "wrist_roll": 0.0,
    "gripper": GRIPPER_OPEN_RAD * 0.5,
}

# Gripper link frame (measured with scripts/view_env.py from the USD mesh bounds):
#   -Z = jaw direction (fixed-jaw tip at z ~= -0.104), +X = closing direction of the moving jaw
#   (pivot at (0.020, 0.019, -0.024), open jaw sweeps x in [0.005, 0.089]), +Y = lateral (servo side).
# Calibrated empirically with scripts/calibrate_grasp.py (sweep of waypoint offsets, success =
# block lifted). Relative to the mesh-derived guess (0.007, 0, -0.082) the gripper had to sit
# 20 mm lower, 12.5 mm further along link -X and 5 mm along link -Y to close on the block, which
# means the grasp point moves the opposite way: +12.5 mm X, +5 mm Y, +20 mm Z (closer to the link
# origin). Success region ~+-1 cm in each axis for the 30 mm block.
GRASP_POINT_OFFSET = (-0.008 + CUBE_SIZE / 2 + 0.0125, 0.005, -0.062)

# Light box (inner size of lightbox-simple.usd: 0.506 deep, 0.762 wide, 0.508 high). The arm sits at
# the side that is open in the workshop model and closed with a white backdrop at the real station.
BOX_X_MIN, BOX_X_MAX = -0.10, 0.406
BOX_Y_HALF = 0.381
BOX_HEIGHT = 0.508
BOX_WALL = 0.006
BOX_WHITE = (0.92, 0.92, 0.92)

# Both cameras sit on the top edge of a wall, centred on it. Look-at points and focal length are
# matched by eye against Photo Booth frames of the real station (logs/station/real).
CAM_FOCAL_LENGTH = float(os.environ.get("CAM_FOCAL_LENGTH", "15.0"))
FRONT_CAM_POS = (BOX_X_MAX - 0.01, 0.0, BOX_HEIGHT + 0.02)
FRONT_CAM_LOOK_AT = (0.05, 0.0, 0.0)
SIDE_CAM_POS = ((BOX_X_MIN + BOX_X_MAX) / 2, -BOX_Y_HALF + 0.01, BOX_HEIGHT + 0.02)
SIDE_CAM_LOOK_AT = (0.32, -0.04, 0.0)


def quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for extrinsic XYZ Euler angles [rad] (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def look_at_quat_xyzw(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Orientation (xyzw, "world" camera convention: forward +X, up +Z) looking from ``eye`` at ``target``."""
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    return quat_xyzw_from_rpy(0.0, pitch, yaw)


def _box_panel(name: str, size: tuple[float, float, float], pos: tuple[float, float, float], collide: bool) -> AssetBaseCfg:
    """Static white panel of the light box (only the floor needs collision)."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/LightBox/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_WHITE, roughness=0.9),
            semantic_tags=[("class", "table" if name == "Floor" else "wall")],
        ),
    )


_BOX_DEPTH = BOX_X_MAX - BOX_X_MIN
_BOX_X_MID = (BOX_X_MIN + BOX_X_MAX) / 2


##
# Scene
##


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Light box, SO-101, one block, one target pad, two fixed cameras."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    box_floor = _box_panel("Floor", (_BOX_DEPTH, 2 * BOX_Y_HALF, 0.02), (_BOX_X_MID, 0.0, -0.01), collide=True)
    box_back = _box_panel("Back", (BOX_WALL, 2 * BOX_Y_HALF, BOX_HEIGHT), (BOX_X_MIN - BOX_WALL / 2, 0.0, BOX_HEIGHT / 2), False)
    box_far = _box_panel("Far", (BOX_WALL, 2 * BOX_Y_HALF, BOX_HEIGHT), (BOX_X_MAX + BOX_WALL / 2, 0.0, BOX_HEIGHT / 2), False)
    box_left = _box_panel("Left", (_BOX_DEPTH, BOX_WALL, BOX_HEIGHT), (_BOX_X_MID, BOX_Y_HALF + BOX_WALL / 2, BOX_HEIGHT / 2), False)
    box_right = _box_panel("Right", (_BOX_DEPTH, BOX_WALL, BOX_HEIGHT), (_BOX_X_MID, -BOX_Y_HALF - BOX_WALL / 2, BOX_HEIGHT / 2), False)
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(semantic_tags=[("class", "ground")]),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.21, 0.07, CUBE_SIZE / 2], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=CUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "cube")],
        ),
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[FLYER_POS[0], FLYER_POS[1], PAD_SIZE[2] / 2], rot=[0, 0, 0, 1]),
        spawn=sim_utils.CuboidCfg(
            size=PAD_SIZE,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2), roughness=0.8),
            semantic_tags=[("class", "target")],
        ),
    )
    flyer = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Flyer",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[FLYER_POS[0], FLYER_POS[1], PAD_SIZE[2] + 0.0005]),
        spawn=UsdFileCfg(usd_path=FLYER_USD),
    )

    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM_FOCAL_LENGTH, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=FRONT_CAM_POS, rot=look_at_quat_xyzw(FRONT_CAM_POS, FRONT_CAM_LOOK_AT), convention="world"),
    )
    side_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/side_cam",
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM_FOCAL_LENGTH, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=SIDE_CAM_POS, rot=look_at_quat_xyzw(SIDE_CAM_POS, SIDE_CAM_LOOK_AT), convention="world"),
    )


##
# MDP
##


@configclass
class EventCfg:
    """Reset-time randomisation."""

    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_cube_pose = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={"pose_range": CUBE_POSE_RANGE, "min_separation": 0.0, "asset_cfgs": [SceneEntityCfg("cube")]},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target)
        ee_pose_b = ObsTerm(func=mdp.ee_pose_in_base)
        cube_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("cube")})
        target_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("target")})
        front = ObsTerm(
            func=base_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False}
        )
        side = ObsTerm(
            func=base_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp = ObsTerm(func=mdp.object_grasped)
        place = ObsTerm(func=mdp.cube_placed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    success = DoneTerm(func=mdp.cube_placed)


@configclass
class IKActionsCfg:
    """``[pos_xyz, quat_xyzw, gripper_rad]`` -- 8-D, consumed by the scripted expert."""

    arm_action: SO101PoseIKActionCfg = MISSING
    gripper_action: JointPositionActionCfg = MISSING


@configclass
class JointActionsCfg:
    """``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`` absolute [rad]."""

    arm_action: JointPositionActionCfg = MISSING
    gripper_action: JointPositionActionCfg = MISSING


##
# Environments
##


@configclass
class SO101PickPlaceBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Shared scene / MDP; subclasses set the action terms."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(anchor_pos=(-0.1, -0.5, -1.05), anchor_rot=(0, 0, -0.5, 0.866))

    # Gripper bookkeeping used by the scripts.
    gripper_joint_names: list[str] = ["gripper"]
    gripper_open_val: float = GRIPPER_OPEN_RAD
    gripper_close_val: float = GRIPPER_CLOSE_RAD
    image_obs_list: list[str] = ["front", "side"]

    def __post_init__(self):
        # 30 Hz control (LeRobot datasets are 30 fps), 60 Hz physics, render every control step.
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 25.0
        self.num_rerenders_on_reset = 3

        # PhysX with the contact tuning of the contrib stack task (concrete cfg: these scripts do
        # not go through Hydra preset resolution). Cameras need Kit/RTX anyway.
        self.sim.physics = PhysicsCfg().isaacsim_physx
        self.scene.replicate_physics = True

        # Robot: stiff PD (tracks IK and absolute joint targets well), gentle gripper so it does
        # not punch the block away (same tuning as the contrib IK-Abs stack task).
        self.scene.robot = SO101_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=SO101_HIGH_PD_CFG.spawn.replace(
                variants={
                    "Robot": "robot",
                    "Sensor": "sensors",
                    "Physics": "physx",
                },
                semantic_tags=[("class", "robot")],
                # the station arm is printed in white PLA
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.88), roughness=0.6),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=_SO101_BASE_SEAT_POS, rot=_SO101_BASE_SEAT_ROT, joint_pos=SO101_INIT_JOINT_POS
            ),
            actuators={
                "arm": SO101_HIGH_PD_CFG.actuators["arm"],
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    joint_effort_limit=1.0,
                    joint_velocity_limit=2.0,
                    stiffness=17.8,
                    damping=0.60,
                ),
            },
        )

        # Grasp-point frame (base -> gripper link + offset) and the moving jaw for diagnostics.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper", name="end_effector", offset=OffsetCfg(pos=GRASP_POINT_OFFSET)
                ),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1", name="moving_jaw"),
            ],
        )


@configclass
class SO101PickPlaceIKEnvCfg(SO101PickPlaceBaseEnvCfg):
    """Task-space variant for the scripted expert: action = ``[x, y, z, qx, qy, qz, qw, gripper_rad]``."""

    actions: IKActionsCfg = IKActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = SO101PoseIKActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            body_name="gripper",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=GRASP_POINT_OFFSET),
            scale=1.0,
            controller=SO101PoseIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="adaptive_dls",
                ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                orientation_weight=0.5,
                orientation_joint_names=("wrist_flex", "wrist_roll"),
                joint_limit_avoidance_gain=0.5,
                joint_limit_avoidance_margin=0.3,
            ),
        )
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
        )


@configclass
class SO101PickPlaceJointEnvCfg(SO101PickPlaceBaseEnvCfg):
    """Joint-space variant: action = absolute joint targets [rad] in :data:`JOINTS` order."""

    actions: JointActionsCfg = JointActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False
        )
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
        )


@configclass
class SO101PickPlaceTeleopEnvCfg(SO101PickPlaceJointEnvCfg):
    """Joint env driven by the physical SO-101 leader arm through IsaacTeleop (``so101_leader`` device)."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_teleop import IsaacTeleopCfg

        from isaaclab_tasks.contrib.stack.config.so101.stack_joint_teleop_env_cfg import (
            _build_so101_joint_teleop_pipeline,
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_so101_joint_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
