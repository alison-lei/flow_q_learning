for ((i = 1; i < 15; i++))
do
    epoch=$((i * 20000))
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python main.py --restore_path=exp/nut/Debug/sd000_20260829_130814 --restore_epoch=$epoch --eval_seed=345
done

