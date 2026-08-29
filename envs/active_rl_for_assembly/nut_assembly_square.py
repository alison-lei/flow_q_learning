from __future__ import annotations

from typing import Any, Dict, Literal

from envs.active_rl_for_assembly.nut_assembly_square_env_cfg import NutAssemblySquareEnvConfig
from envs.active_rl_for_assembly.robosuite_base_env import RobosuiteBaseEnv
import numpy as np
from scipy.spatial.transform import Rotation as R

# Grasp point on the square-nut handle bar, in nut local frame.
# square-nut.xml: the handle bar geom sits at x=0.054 (= handle_site, the bar
# center) with x half-size 0.02525, so the bar spans x=0.0288..0.0793. The
# square nut body (outer frame) extends to x=0.0438. The handle bar geometry
# is identical to the round nut's; only its position differs (0.054 vs 0.06),
# so this offset is the round nut's 0.07 shifted in by the same 0.006.
# Grasping at x=0.064 sits 2.0cm past the nut body (gripper hand clears it in
# EE-y) and stays 1.5cm inside the bar tip with margin for jaw thickness.
SQUARE_NUT_HANDLE_OFFSET = np.array([0.064, 0.0, 0.0])

# ---- Unrecoverable-state thresholds (env-side episode termination) ----
# nut_height = nut_pos_world[2] - table_height. Flat-on-table reads ~+0.01;
# floor reads ~-0.7. -0.05 sits cleanly between.
NUT_OFF_TABLE_HEIGHT = -0.05
# Below this nut_height the nut is resting on the table (vs. lifted in
# transit, ~0.18, or sitting on the peg, ~0.13). Used to gate the
# handle-past-peg check so transit poses do not trigger it.
NUT_ON_TABLE_HEIGHT = 0.05
# Padding past ABS_POSE_LIMIT_LOW/HIGH[:2] before declaring out-of-reach.
WORKSPACE_MARGIN = 0.01
# |cos(angle between nut local +z and world +z)| below this => tilted >~31°.
# Square nut is top/bottom symmetric (all geoms centered at z=0), so use the
# absolute value.
NUT_UPRIGHT_THRESHOLD = 0.85
# Velocity thresholds for declaring the nut "settled". A nut mid-fall or
# tumbling reads transient tilt that resolves once it lands; gating the tilt
# (and handle-past-peg) checks on settled velocity prevents premature
# termination during the drop. Tuned for MuJoCo defaults; loosen if the nut
# never reads as settled in practice.
NUT_SETTLED_LIN_VEL = 0.002 #0.02   # m/s
NUT_SETTLED_ANG_VEL = 0.02 #0.5    # rad/s
# Match nut_assembly_expert.py — keeps "is grasping" semantics consistent.
GRIPPER_CLOSED_THRESH = 0.8
EMPTY_GRASP_THRESH = 0.4

# Gripper-broken thresholds. Healthy Panda fingers sit symmetrically around
# the hand frame; when MuJoCo lets a finger separate from its joint under
# heavy contact penetration the body drops several cm below the other while
# qpos still reads in-range, so we check body world poses, not joint qpos.
# Calibrate via sim_env/collect_data/calibrate_gripper_thresholds.py.
FINGER_Z_ASYMMETRY_LIMIT = 0.005   # m, in hand frame (~20x healthy max)
FINGER_MAX_DISTANCE_FROM_HAND = 0.15  # m (~3cm slack over healthy max ~0.119)

class PandaNutAssemblySquareGymEnv(RobosuiteBaseEnv):
    def __init__(
        self,
        config: NutAssemblySquareEnvConfig = NutAssemblySquareEnvConfig,
        fake_env: bool = False,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        save_video: bool = False,
        video_path: str = "./videos",
        privileged_obs: bool = False,
        **kwargs,
    ):
        super().__init__(config, fake_env, render_mode, save_video, video_path,
                         privileged_obs=privileged_obs,
                         single_object_mode=2,
                         nut_type="square", #"round" or "square"
        )

        for name, sampler in self._env.placement_initializer.samplers.items():
            if name in ["RoundNutSampler", "SquareNutSampler"]:
                sampler.rotation = np.array([np.pi/2, 3 * np.pi/2])
                print(f"Sampler: {name}")
                print(f"  rotation: {sampler.rotation}")

        if not fake_env:
            self.table_height = self._env.model.mujoco_arena.table_offset[2]
            self._resolve_gripper_body_ids()
            # peg2 (round peg) is not part of the square-nut task -- only
            # peg1 (square peg) is the target. But peg2 sits on the table
            # within the gripper's reachable workspace, and the square nut
            # can land at peg2's base after a regrasp drop, leaving the
            # gripper unable to reach the handle past the obstructing peg
            # (the round-nut env hits the same failure with the unused
            # square peg). Move peg2 far below the floor so it can't
            # interfere physically or visually. The body is welded to the
            # arena, so the offset persists across resets.
            # peg2_body_id = self._env.peg2_body_id
            # self._env.sim.model.body_pos[peg2_body_id] = np.array(
            #     [0.0, 0.0, -2.0]
            # )
            # self._env.sim.forward()

    def _resolve_gripper_body_ids(self):
        """Cache MuJoCo body ids for the gripper hand and two fingers. Body
        names depend on robosuite version (gripper0_leftfinger,
        gripper0_right_leftfinger, etc.); probe candidates and fall back to
        scanning body_names. Stores None if not found, in which case
        _check_gripper_broken is a no-op."""
        sim_model = self._env.sim.model
        self._hand_body_id = None
        self._finger1_body_id = None
        self._finger2_body_id = None

        def _try_id(name):
            try:
                return sim_model.body_name2id(name)
            except (KeyError, ValueError):
                return None

        for cand in ("gripper0_right_hand", "robot0_right_hand", "gripper0_hand"):
            self._hand_body_id = _try_id(cand)
            if self._hand_body_id is not None:
                break
        if self._hand_body_id is None:
            for n in sim_model.body_names:
                if "hand" in n and ("gripper0" in n or "robot0" in n):
                    self._hand_body_id = sim_model.body_name2id(n)
                    break

        for left, right in (
            # Panda parallel-jaw gripper.
            ("gripper0_leftfinger", "gripper0_rightfinger"),
            ("gripper0_right_leftfinger", "gripper0_right_rightfinger"),
            # Robotiq 85 (this build): inner_finger is the body holding the
            # fingertip pad; outer_knuckle is the driven 4-bar link, so
            # asymmetry there reflects normal closing motion, not breakage.
            ("gripper0_right_left_inner_finger",
             "gripper0_right_right_inner_finger"),
        ):
            l_id, r_id = _try_id(left), _try_id(right)
            if l_id is not None and r_id is not None:
                self._finger1_body_id, self._finger2_body_id = l_id, r_id
                break
        if self._finger1_body_id is None:
            for n in sim_model.body_names:
                if ("leftfinger" in n or "left_inner_finger" in n) \
                        and self._finger1_body_id is None:
                    self._finger1_body_id = sim_model.body_name2id(n)
                elif ("rightfinger" in n or "right_inner_finger" in n) \
                        and self._finger2_body_id is None:
                    self._finger2_body_id = sim_model.body_name2id(n)

        if (self._hand_body_id is None
                or self._finger1_body_id is None
                or self._finger2_body_id is None):
            candidates = [n for n in sim_model.body_names
                          if "finger" in n or "hand" in n or "gripper" in n]
            # print("[gripper_check] WARNING: could not resolve gripper bodies; "
            #       "_check_gripper_broken disabled. "
            #       f"hand={self._hand_body_id} "
            #       f"f1={self._finger1_body_id} "
            #       f"f2={self._finger2_body_id}. "
            #       f"Candidate body names: {candidates}")

    def _gripper_finger_offsets_in_hand_frame(self):
        """Return (left_finger, right_finger) positions in hand frame, or
        None if body ids could not be resolved."""
        if (self._hand_body_id is None
                or self._finger1_body_id is None
                or self._finger2_body_id is None):
            return None
        sim = self._env.sim
        hand_pos = sim.data.body_xpos[self._hand_body_id]
        hand_mat = sim.data.body_xmat[self._hand_body_id].reshape(3, 3)
        f1 = hand_mat.T @ (sim.data.body_xpos[self._finger1_body_id] - hand_pos)
        f2 = hand_mat.T @ (sim.data.body_xpos[self._finger2_body_id] - hand_pos)
        return f1, f2

    def _check_gripper_broken(self) -> bool:
        offsets = self._gripper_finger_offsets_in_hand_frame()
        if offsets is None:
            return False
        f1, f2 = offsets
        z_asym = abs(float(f1[2] - f2[2]))
        max_dist = float(max(np.linalg.norm(f1), np.linalg.norm(f2)))
        return (
            z_asym > FINGER_Z_ASYMMETRY_LIMIT
            or max_dist > FINGER_MAX_DISTANCE_FROM_HAND
        )

    def _change_camera_pose(self):
        # Mirror of the round-nut robot0_robotview camera across the y=0 plane
        # (the square task lives on +y). Computed as a look-at from the
        # y-mirrored eye toward the y-mirrored gaze with world-up, expressed
        # in the robot0_base parent frame (MuJoCo wxyz); reproduces the round
        # camera's framing on the +y side (verified by render). Round-nut
        # reference: cam_pos=[-0.12,-0.56,0.32],
        #   cam_quat=[0.69858682, 0.45040619, -0.26362669, -0.48950106].
        cam_name = "robot0_robotview"
        cam_id = self._env.sim.model.camera_name2id(cam_name)

        # cam_pos = [-0.12, 0.56, 0.32]
        # cam_quat = [-0.460581, -0.281790, 0.439271, 0.717983]

        cam_pos = [1.27, 0.43, 0.37]
        cam_quat = [0.43888116, 0.21882318, 0.51291811, 0.70456713]

        self._env.sim.model.cam_pos[cam_id] = cam_pos
        self._env.sim.model.cam_quat[cam_id] = cam_quat

    def get_privileged_shape(self):
        # delta_pos_ee_handle(3) + sym_yaw_handle(2) + nut_height(1)
        # + delta_pos_ee_target(3) + nut_world_rz(1) + wrist_qpos(1) = 11
        return (11,)

    def _process_privileged_obs(
        self,
        obs: Dict[str, Any],
        out: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:

        privileged_obs = self._extract_privileged_from_raw_obs(obs)
        # Stash the latest extracted nut pose so step() can reuse it for the
        # unrecoverable-state check without re-walking the raw obs.
        self._last_nut_pos_world = np.asarray(
            privileged_obs["SquareNut_pos"], dtype=np.float64
        ).copy()
        self._last_nut_quat_world = np.asarray(
            privileged_obs["SquareNut_quat"], dtype=np.float64
        ).copy()

        # EE pose in base frame
        ee_pos = out["state"]["tcp_pose"][:3]
        ee_rot = R.from_quat(out["state"]["tcp_pose"][3:])
        ee_mat = ee_rot.as_matrix()
        R_wb = self.world_to_base[:3, :3]
        t_wb = self.world_to_base[:3, 3]

        # ---- Square nut: center pose (world) and handle pose (world) ----
        nut_pos_world = privileged_obs["SquareNut_pos"]
        nut_rot_world = R.from_quat(privileged_obs["SquareNut_quat"])
        handle_pos_world = nut_pos_world + nut_rot_world.apply(SQUARE_NUT_HANDLE_OFFSET)

        # Diagnostic: nut orientation in world frame.
        # - Euler "xyz" gives RX/RY/RZ deviations from world axes (in rad).
        # - nut_z_dot_world_z = component of nut local +z along world +z;
        #   1.0 = perfectly upright, 0.0 = on its side, |.|<1 = tilted.
        # nut_euler_world = nut_rot_world.as_euler("xyz")
        # nut_z_dot_world_z = float(nut_rot_world.as_matrix()[2, 2])
        # print(
        #     f"[nut_orient_world] rx={nut_euler_world[0]:+.4f} "
        #     f"ry={nut_euler_world[1]:+.4f} rz={nut_euler_world[2]:+.4f} rad   "
        #     f"({np.degrees(nut_euler_world[0]):+.2f}, "
        #     f"{np.degrees(nut_euler_world[1]):+.2f}, "
        #     f"{np.degrees(nut_euler_world[2]):+.2f}) deg   "
        #     f"nut_z.world_z={nut_z_dot_world_z:+.4f}"
        # )

        # Transform to base frame
        nut_pos = R_wb @ nut_pos_world + t_wb
        nut_mat = R_wb @ nut_rot_world.as_matrix()
        handle_pos = R_wb @ handle_pos_world + t_wb

        # Delta from EE to handle (grasp target), in EE frame
        delta_pos_ee_handle = ee_rot.inv().apply(handle_pos - ee_pos)

        # Handle direction (nut local +x in base frame), projected to xy plane
        # for yaw alignment.
        handle_dir = nut_mat[:, 0]
        handle_dir_xy = handle_dir[:2].copy()

        # Wrist-camera-aware grasp side: empirically the wrist camera
        # (robot0_eye_in_hand) on this Panda is mounted looking ~straight
        # along EE-z, so its EE-frame gaze has no y-component to key off of.
        # That means there's no live geometric criterion for which 180-deg
        # grasp puts the nut hole on the camera-visible side - both options
        # are symmetric in EE frame. We just pick a fixed orientation so
        # that the camera view is at least *consistent* across episodes.
        # FLIP_HANDLE_DIR is the empirical knob: flip it if the camera ends up
        # looking at the handle tip instead of the hole.
        #
        # No nut-flip correction is applied. nut_mat[:, 0] is the nut's local +x,
        # where the handle is rigidly attached, so it points at the PHYSICAL
        # handle for ANY orientation; and because the square nut is top/bottom
        # symmetric, a z-flip leaves the handle (and the hole seen from above)
        # unchanged. So the grasp side is fully fixed by the handle direction
        # plus this one constant -- it does NOT depend on whether the nut reads
        # top-up or top-down. (An earlier nut_top_up-based flip double-counted
        # and picked the WRONG side for flipped nuts -- that was the "wrong
        # camera side" that appeared after a nut flip.)
        FLIP_HANDLE_DIR = False
        if np.linalg.norm(handle_dir_xy) > 1e-3 and FLIP_HANDLE_DIR:
            handle_dir_xy = -handle_dir_xy

        handle_angle = np.arctan2(handle_dir_xy[1], handle_dir_xy[0])
        gripper_angle = np.arctan2(ee_mat[1, 0], ee_mat[0, 0])
        # 1-fold encoding (camera-aware): unique target yaw, gripper grasps
        # from the side that keeps the nut visible.
        relative_angle = handle_angle - gripper_angle - np.pi / 2
        sym_yaw_handle = np.array([
            np.sin(relative_angle),
            np.cos(relative_angle),
        ])

        nut_height = np.array([nut_pos_world[2] - self.table_height])

        # ---- Target placement: align nut hole over peg1 (square, brass) ----
        # After grasp, ee ~= handle; the EE-frame delta to position the nut hole over
        # the peg simplifies to (peg_pos - nut_pos): the handle->nut offset cancels.
        peg1_pos_world = self._env.sim.data.body_xpos[self._env.peg1_body_id].copy()
        peg1_pos = R_wb @ peg1_pos_world + t_wb
        delta_pos_ee_target = ee_rot.inv().apply(peg1_pos - nut_pos)

        # Nut yaw in world frame, used by the expert to orient the nut to a
        # specific world-frame target during PLACE.
        nut_world_rz = np.array(
            [float(nut_rot_world.as_euler("xyz")[2])], dtype=np.float32
        )

        # Wrist (joint7) qpos. The expert uses it to detect when the wrist is
        # approaching its hard limit (+/-2.8973 rad) and reverse rotation
        # direction (long-way arc) instead of saturating the joint. Required
        # for intervention robustness: rz_relative alone can't disambiguate
        # joint configurations, so we expose the joint angle directly.
        arm_joint_ids = self._env.robots[0]._ref_joint_pos_indexes
        wrist_qpos = np.array(
            [float(self._env.sim.data.qpos[arm_joint_ids[-1]])], dtype=np.float32
        )
        # Diagnostic: dump full 7-DoF Panda arm qpos so we can spot which
        # joints (not just joint7) are saturating against their limits.
        # Panda hard limits: j1 +/-2.8973, j2 +/-1.7628, j3 +/-2.8973,
        #                    j4 [-3.0718, -0.0698], j5 +/-2.8973,
        #                    j6 [-0.0175, 3.7525], j7 +/-2.8973.
        arm_qpos_full = np.asarray(
            self._env.sim.data.qpos[arm_joint_ids], dtype=np.float64
        )
        # print(
        #     "[panda_qpos] j1={:+.3f} j2={:+.3f} j3={:+.3f} j4={:+.3f} "
        #     "j5={:+.3f} j6={:+.3f} j7={:+.3f}".format(*arm_qpos_full)
        # )

        out["state"]["privileged"] = np.concatenate([
            delta_pos_ee_handle,   # [0:3]
            sym_yaw_handle,        # [3:5]
            nut_height,            # [5]
            delta_pos_ee_target,   # [6:9]
            nut_world_rz,          # [9]
            wrist_qpos,            # [10]
        ]).astype(np.float32)
        return out

    def _nut_pos_base(self, body_name: str = "SquareNut_main") -> np.ndarray:
        body_id = self._env.sim.model.body_name2id(body_name)
        pos_world = np.asarray(self._env.sim.data.body_xpos[body_id])
        R_wb = self.world_to_base[:3, :3]
        return R_wb @ pos_world + self.world_to_base[:3, 3]

    def _nut_is_settled(self, body_name: str = "SquareNut_main") -> bool:
        body_id = self._env.sim.model.body_name2id(body_name)
        cvel = self._env.sim.data.cvel[body_id]   # [wx, wy, wz, vx, vy, vz]
        ang = float(np.linalg.norm(cvel[:3]))
        lin = float(np.linalg.norm(cvel[3:]))
        settled = lin <= NUT_SETTLED_LIN_VEL and ang <= NUT_SETTLED_ANG_VEL
        # print(f"[nut_velocity] lin={lin:.4f} m/s  ang={ang:.4f} rad/s  "
        #       f"settled={settled}  (thresholds lin<{NUT_SETTLED_LIN_VEL}, "
        #       f"ang<{NUT_SETTLED_ANG_VEL})")
        return settled

    _OUT_OF_BOUNDS_XY_MARGIN: float = 0.01

    def _nut_out_of_bounds(
        self,
        grasping: bool,
        *,
        nut_pos_base: np.ndarray,
        handle_pos_base: np.ndarray,
    ) -> bool:
        """OOB termination check.

        z is checked on the nut center (off-table is about the body falling
        off). xy is checked on the HANDLE: 'recoverable' means the gripper
        can re-grasp the nut, and the gripper grasps the handle, which sits
        ~6cm off the nut center along the nut's local +x. Checking nut
        center against the EE workspace bounds spuriously fires when the
        nut is correctly placed at the peg if the peg sits inside the
        margin band but the handle lies outside (or vice versa)."""
        low = self.config.ABS_POSE_LIMIT_LOW[:3]
        high = self.config.ABS_POSE_LIMIT_HIGH[:3]
        m = self._OUT_OF_BOUNDS_XY_MARGIN

        # print("NUT POS: ", nut_pos_base)
        # print("HANDLE POS: ", handle_pos_base)

        if nut_pos_base[2] < low[2] - 0.1:
            return True

        if not grasping:
            if handle_pos_base[0] < low[0] + m or handle_pos_base[0] > high[0] - m:
                return True
            if handle_pos_base[1] < low[1] + m or handle_pos_base[1] > high[1] - m:
                return True

        return False

    def _gripper_in_contact_with_nut(self) -> bool:
        """True if any gripper geom is currently touching the nut.

        Used alongside the gripper-width 'grasping' test to gate the
        unrecoverable-nut checks. The width test reads "not grasping" the
        instant the jaws open past EMPTY_GRASP_THRESH, but during a human
        intervention the open jaws often keep PUSHING the nut around -- it is
        under control, not released-and-lost -- so the OOB / workspace / tilt /
        handle-past-peg checks (which assume a FREE nut) must not fire. Checking
        contact catches that case the width test misses. Body-based (robust to
        unnamed collision geoms); returns False if the sim or gripper bodies
        couldn't be resolved, so it falls back to the width test alone.
        """
        env = getattr(self, "_env", None)
        sim = getattr(env, "sim", None)
        if sim is None:
            return False
        gripper_roots = {
            b for b in (self._hand_body_id, self._finger1_body_id,
                        self._finger2_body_id) if b is not None
        }
        if not gripper_roots:
            return False
        model, data = sim.model, sim.data
        try:
            nut_body = model.body_name2id("SquareNut_main")
        except (KeyError, ValueError):
            return False

        def _is_gripper_body(b):
            steps = 0
            while b != -1 and steps < 64:
                if b in gripper_roots:
                    return True
                b = model.body_parentid[b]
                steps += 1
            return False

        for i in range(data.ncon):
            c = data.contact[i]
            bodies = (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            if nut_body in bodies and any(_is_gripper_body(b) for b in bodies):
                return True
        return False

    def _check_unrecoverable_nut(self, obs):
        """Return a string failure reason if the nut is in an unrecoverable
        state, else None. Categories: off the table, past the EE workspace,
        out-of-bounds while not grasped, tilted steeply, or resting on the
        table with the handle slid to or past the peg's x (the workspace,
        out-of-bounds, tilt, and handle-past-peg checks only while not
        grasping)."""
        if getattr(self, "_last_nut_pos_world", None) is None:
            return None

        nut_pos_world = self._last_nut_pos_world
        nut_rot_world = R.from_quat(self._last_nut_quat_world)
        handle_pos_world = nut_pos_world + nut_rot_world.apply(SQUARE_NUT_HANDLE_OFFSET)

        R_wb = self.world_to_base[:3, :3]
        t_wb = self.world_to_base[:3, 3]
        nut_pos_base = R_wb @ nut_pos_world + t_wb
        handle_pos_base = R_wb @ handle_pos_world + t_wb

        nut_height = nut_pos_world[2] - self.table_height
        if nut_height < NUT_OFF_TABLE_HEIGHT:
            return "off_table"

        grip = float(np.ravel(obs["state"]["gripper_pose"])[0])
        grasping = (grip < GRIPPER_CLOSED_THRESH) and (grip >= EMPTY_GRASP_THRESH)
        # Treat the nut as held/controlled if the gripper is grasping it (width
        # test) OR merely in contact with it. The width test flips to "not
        # grasping" the moment the jaws open past EMPTY_GRASP_THRESH, but a human
        # intervention often keeps pushing the nut with open jaws -- it's under
        # control, not released-and-lost. Gating on contact too stops the OOB /
        # workspace / tilt / handle-past-peg checks from false-failing a human
        # recovery (e.g. dragging the nut near the workspace edge).
        held = grasping or self._gripper_in_contact_with_nut()

        if self._nut_out_of_bounds(
            held,
            nut_pos_base=nut_pos_base,
            handle_pos_base=handle_pos_base,
        ):
            return "out_of_bounds"

        # Workspace + tilt checks only apply when the nut is free. While the
        # gripper holds OR is touching the nut, the handle legitimately reaches
        # the workspace edge during transit/manipulation, and tilt is whatever
        # the carry/push pose imposes.
        if not held:
            handle_xy_base = handle_pos_base[:2]
            xy_low = np.asarray(self.config.ABS_POSE_LIMIT_LOW[:2]) - WORKSPACE_MARGIN
            xy_high = np.asarray(self.config.ABS_POSE_LIMIT_HIGH[:2]) + WORKSPACE_MARGIN
            if np.any(handle_xy_base < xy_low) or np.any(handle_xy_base > xy_high):
                return "out_of_workspace"

            nut_uprightness = float(nut_rot_world.as_matrix()[2, 2])
            # print("NUT UPRIGHTNESS: ", nut_uprightness)
            if abs(nut_uprightness) < NUT_UPRIGHT_THRESHOLD and self._nut_is_settled():
                return "tilted_unreachable"

            # Yaw out of spawn range: a free, settled nut whose world yaw has
            # settled outside the initial spawn range [pi/2, 3pi/2] (the
            # SquareNutSampler.rotation set in __init__) is unrecoverable -- that
            # band, centered on the place target pi, is the one the 1-fold expert
            # can rotate to grasp, so it can't re-align to a yaw outside it.
            # Because the band is centered on pi == the edge of the euler
            # [-pi, pi] range, distance-from-pi is (pi - |yaw|) and "in band"
            # reduces to |yaw| >= pi/2 -- so outside-band is just |yaw| < pi/2,
            # no normalization needed. Checked after tilt so a toppled nut
            # reports 'tilted_unreachable' first. NOTE: the gripper's 2-fold
            # symmetry makes some of these graspable from the opposite side --
            # this deliberately mirrors the expert's fixed grasp-side limitation.
            nut_yaw_world = float(nut_rot_world.as_euler("xyz")[2])
            if abs(nut_yaw_world) < np.pi / 2 and self._nut_is_settled():
                return "yaw_out_of_spawn_range"

            # Handle-past-peg: when the nut has fallen flat on the table and
            # the handle has slid to or past the peg's x position, the
            # gripper would have to reach across the peg to re-grasp the
            # handle - treat as unrecoverable so the policy doesn't get
            # rewarded for stalling here.
            if nut_height < NUT_ON_TABLE_HEIGHT and self._nut_is_settled():
                peg_pos_world = self._env.sim.data.body_xpos[self._env.peg1_body_id]
                # print("HANDLE POS WORLD: ", handle_pos_world[0])
                # print("PEG POS WORLD: ", peg_pos_world[0])
                if handle_pos_world[0] >= peg_pos_world[0] - 0.01: #0.016 is square peg half-width
                    return "handle_past_peg"

        return None

    # Panda joint4 (elbow) range: [-3.0718, -0.0698]. The asymmetric upper
    # bound is what saturates in practice -- when the arm extends fully
    # outward, j4 hits -0.0698 and the OSC loses authority near singularity.
    # Once j4 is within this margin of -0.0698, the controller cannot
    # reliably retract the arm with EE-frame commands (confirmed by
    # spacemouse teleop attempts producing no joint motion). Treat the
    # state as terminal failure rather than burning ~100 wasted steps.
    JOINT4_UPPER_LIMIT = -0.0698
    JOINT4_SATURATION_MARGIN = 0 #0.05  # j4 > -0.12 -> within margin

    def _check_joint4_saturated(self) -> bool:
        arm_joint_ids = self._env.robots[0]._ref_joint_pos_indexes
        j4 = float(self._env.sim.data.qpos[arm_joint_ids[3]])
        return j4 > self.JOINT4_UPPER_LIMIT - self.JOINT4_SATURATION_MARGIN

    # Panda wrist joints j5 (range +/-2.8973) and j6 (range [-0.0175, +3.7525])
    # are the pair that pin against their hard stops during the PLACE rotate-
    # to-pi: holding the gripper-down EE orientation while rotating the nut to
    # the world-yaw target winds j5 down to -2.8973 and j6 up to +3.7525. Once
    # pinned, OSC xy commands toward the peg silently fail and the arm burns the
    # rest of the step budget hovering ~15cm short of the peg (observed: 80+
    # wasted steps). Terminate as an unrecoverable failure instead.
    #
    # j5/j6 also sit *near* their limits in a healthy grasp here, so a bare
    # margin trip would false-positive -- require the joint to stay pinned
    # (margin <= 0, i.e. at/over the stop) for WRIST_SATURATION_PERSISTENCE
    # consecutive steps; the counter resets the instant it comes off the limit,
    # so transient excursions (including during a human/spacemouse intervention)
    # recover without ending the episode. A catastrophic overshoot (>=
    # WRIST_SATURATION_OVERSHOOT past the stop) trips immediately. Mirrors the
    # joint-saturation termination in door_env.py.
    WRIST_JOINT_INDICES = (4, 5)        # j5, j6 (0-indexed j1..j7 -> 0..6)
    WRIST_SATURATION_PERSISTENCE = 5    # consecutive at-limit steps -> done
    WRIST_SATURATION_OVERSHOOT = 0.05   # rad past a stop -> done immediately

    def _update_wrist_saturation(self) -> bool:
        """Tick the j5/j6 at-limit counters once per step and report whether
        either wrist joint is saturated (pinned for WRIST_SATURATION_PERSISTENCE
        steps, or overshooting its stop by WRIST_SATURATION_OVERSHOOT). Counters
        reset when the joint leaves its limit and on episode end (see step())."""
        joint_ids = self._env.robots[0]._ref_joint_indexes
        qpos_ids = self._env.robots[0]._ref_joint_pos_indexes
        if not hasattr(self, "_wrist_at_limit_steps"):
            self._wrist_at_limit_steps = {j: 0 for j in self.WRIST_JOINT_INDICES}

        saturated = False
        self._last_wrist_saturation = None
        for j in self.WRIST_JOINT_INDICES:
            q = float(self._env.sim.data.qpos[qpos_ids[j]])
            lo = float(self._env.sim.model.jnt_range[joint_ids[j], 0])
            hi = float(self._env.sim.model.jnt_range[joint_ids[j], 1])
            margin = min(q - lo, hi - q)   # >=0 inside range, <0 past a stop
            if margin <= 0.0:
                self._wrist_at_limit_steps[j] += 1
            else:
                self._wrist_at_limit_steps[j] = 0
            if (self._wrist_at_limit_steps[j] >= self.WRIST_SATURATION_PERSISTENCE
                    or margin <= -self.WRIST_SATURATION_OVERSHOOT):
                saturated = True
                self._last_wrist_saturation = (
                    j, q, lo, hi, self._wrist_at_limit_steps[j]
                )
        return saturated

    def _failure_summary(self, obs) -> str:
        """Compact one-line state summary for a non-success episode end
        (timeout / uncaught termination) so failures aren't silent in the
        logs. Best-effort: returns '' if the nut pose isn't cached yet."""
        if getattr(self, "_last_nut_pos_world", None) is None:
            return ""
        nut_pos_world = self._last_nut_pos_world
        nut_yaw = float(R.from_quat(self._last_nut_quat_world).as_euler("xyz")[2])
        nut_height = float(nut_pos_world[2] - self.table_height)
        try:
            grip = float(np.ravel(obs["state"]["gripper_pose"])[0])
            grasping = (grip < GRIPPER_CLOSED_THRESH) and (grip >= EMPTY_GRASP_THRESH)
        except (KeyError, IndexError, TypeError):
            grip, grasping = float("nan"), False
        peg_xy = np.asarray(
            self._env.sim.data.body_xpos[self._env.peg1_body_id]
        )[:2]
        nut_to_peg_xy = float(np.linalg.norm(nut_pos_world[:2] - peg_xy))
        arm_ids = self._env.robots[0]._ref_joint_pos_indexes
        jq = np.asarray(self._env.sim.data.qpos[arm_ids], dtype=np.float64)
        return (
            f"grasping={grasping} grip={grip:.2f} nut_h={nut_height:.3f} "
            f"nut_to_peg_xy={nut_to_peg_xy:.3f} nut_yaw={nut_yaw:+.3f} "
            f"j4={jq[3]:+.3f} j5={jq[4]:+.3f} j6={jq[5]:+.3f} j7={jq[6]:+.3f}"
        )

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        wrist_saturated = self._update_wrist_saturation()
        if not info.get("succeed", False):
            reason = self._check_unrecoverable_nut(obs)
            if reason is not None:
                done = True
                info["unrecoverable"] = reason
                print(
                    f"[EARLY_TERMINATION] step={self.curr_path_length} "
                    f"reason='unrecoverable_nut:{reason}'"
                )
            elif self._check_joint4_saturated():
                done = True
                info["unrecoverable"] = "joint4_saturated"
                arm_joint_ids = self._env.robots[0]._ref_joint_pos_indexes
                j4 = float(self._env.sim.data.qpos[arm_joint_ids[3]])
                print(
                    f"[EARLY_TERMINATION] step={self.curr_path_length} "
                    f"reason='joint4_saturated' j4={j4:+.4f} "
                    f"(limit={self.JOINT4_UPPER_LIMIT:+.4f}, "
                    f"margin={self.JOINT4_SATURATION_MARGIN:.4f})"
                )
            elif wrist_saturated:
                done = True
                info["unrecoverable"] = "wrist_saturated"
                if self._last_wrist_saturation is not None:
                    j, q, lo, hi, p = self._last_wrist_saturation
                    print(
                        f"[EARLY_TERMINATION] step={self.curr_path_length} "
                        f"reason='wrist_saturated' j{j + 1}={q:+.4f} "
                        f"(range=[{lo:+.4f},{hi:+.4f}], at_limit_steps={p}, "
                        f"persistence={self.WRIST_SATURATION_PERSISTENCE})"
                    )
            elif self._check_gripper_broken():
                done = True
                info["unrecoverable"] = "gripper_broken"
                offsets = self._gripper_finger_offsets_in_hand_frame()
                f1, f2 = offsets
                print(
                    f"[EARLY_TERMINATION] step={self.curr_path_length} "
                    f"reason='gripper_broken' "
                    f"z_asym={abs(f1[2] - f2[2]):.4f} "
                    f"f1_dist={np.linalg.norm(f1):.4f} "
                    f"f2_dist={np.linalg.norm(f2):.4f} "
                    f"(z_limit={FINGER_Z_ASYMMETRY_LIMIT:.3f}, "
                    f"dist_limit={FINGER_MAX_DISTANCE_FROM_HAND:.3f})"
                )
        elif done:
            print(
                f"[EARLY_TERMINATION] step={self.curr_path_length} reason='success'"
            )

        # Catch-all failure reason: the episode is ending without success and
        # without one of the explicit unrecoverable reasons above -- it ran out
        # of time (truncation / horizon) or the base env terminated it. Print a
        # categorized reason + state summary so failures aren't silent in the
        # logs (every non-success episode now ends with an explicit reason).
        if (done or truncated) and not info.get("succeed", False) \
                and "unrecoverable" not in info:
            reason = "timeout" if truncated else "terminated_no_success"
            info["failure"] = reason
            print(
                f"[EARLY_TERMINATION] step={self.curr_path_length} "
                f"reason='{reason}' {self._failure_summary(obs)}"
            )

        if done or truncated:
            # Clear the at-limit counters so the next episode starts clean
            # (mirrors door_env; avoids the counter leaking across episodes).
            self._wrist_at_limit_steps = {j: 0 for j in self.WRIST_JOINT_INDICES}
        return obs, reward, done, truncated, info


class PandaNutAssemblySquareGymEnvRG(PandaNutAssemblySquareGymEnv):
    def step(self, action_dict: dict):
        return super().step(action_dict["action"])
