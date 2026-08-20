#!/bin/bash
set -e

REPO_ROOT="$HOME/flow_q_learning/active-rl-for-assembly"
SERL_ROOT="$HOME/flow_q_learning/serl"

cd "$REPO_ROOT"

echo "=== [1] Init submodules ==="
git submodule update --init --recursive --force

echo "=== [2] Manual clone submodules at pinned commits if empty ==="
if [ -z "$(ls -A external/conrft 2>/dev/null)" ]; then
    git clone https://github.com/cccedric/conrft.git external/conrft
    cd external/conrft && git checkout 05b5a4e && cd "$REPO_ROOT"
fi
if [ -z "$(ls -A external/hil-serl 2>/dev/null)" ]; then
    git clone https://github.com/rail-berkeley/hil-serl.git external/hil-serl
    cd external/hil-serl && git checkout 8f1daa0 && cd "$REPO_ROOT"
fi
if [ -z "$(ls -A external/serl 2>/dev/null)" ]; then
    git clone https://github.com/rail-berkeley/serl.git external/serl
    cd external/serl && git checkout cd5f7e7 && cd "$REPO_ROOT"
fi

echo "=== [3] Install serl_launcher ==="
SERL_LAUNCHER=$(find external -name "setup.py" -o -name "pyproject.toml" 2>/dev/null | grep serl_launcher | head -1 | xargs dirname)
if [ -n "$SERL_LAUNCHER" ]; then
    pip install -e "$SERL_LAUNCHER"
else
    echo "WARNING: serl_launcher not found, skipping"
fi

echo "=== [4] Fix numpy version ==="
pip install "numpy>=1.24.1,<2.0"

echo "=== [5] Fix jax/jaxlib version mismatch ==="
pip install "jax==0.4.38" "jaxlib==0.4.38"

echo "=== [6] Fix protobuf conflicts ==="
pip install "tensorflow-metadata==1.14.0" "protobuf>=4.21.0,<5.0.0"

echo "=== [7] Fix opencv ==="
pip install "opencv-python<5.0"

echo "=== [8] Fix mujoco/robosuite compatibility ==="
pip install "mujoco==2.3.7"

echo "=== [9] Fix gym vs gymnasium in franka_env wrappers ==="
WRAPPERS="$SERL_ROOT/serl_robot_infra/franka_env/envs/wrappers.py"
if [ -f "$WRAPPERS" ]; then
    sed -i 's/^from gym import Env, spaces/from gymnasium import Env, spaces/' "$WRAPPERS"
    sed -i 's/^from gym.spaces import Box/from gymnasium.spaces import Box/' "$WRAPPERS"
    sed -i 's/^import gym$/import gymnasium as gym/' "$WRAPPERS"
    echo "Patched $WRAPPERS"
else
    echo "WARNING: wrappers.py not found at $WRAPPERS"
fi

echo "=== [10] Fix proprio_keys to include gripper_pose ==="
CFG="$REPO_ROOT/train/configs/default_robosuite_train_cfg.py"
# Comment out the wipe line, uncomment the most tasks line
sed -i 's/^    proprio_keys = \["tcp_pose", "tcp_vel", "wrist_force", "wrist_torque"\] #for wipe/    #proprio_keys = ["tcp_pose", "tcp_vel", "wrist_force", "wrist_torque"] #for wipe/' "$CFG"
sed -i 's/^    #proprio_keys = \["tcp_pose", "tcp_vel", "gripper_pose", "wrist_force", "wrist_torque"\] #for most tasks/    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "wrist_force", "wrist_torque"] #for most tasks/' "$CFG"
echo "Patched proprio_keys in $CFG"

echo ""
echo "=== All fixes applied. Run your script to verify: ==="
echo "python -m sim_env.collect_data.record_demos_scripted_expert --exp_name lift --successes_needed 20 --env_seed=500 --noise_seed=500 --no_render --save_video"