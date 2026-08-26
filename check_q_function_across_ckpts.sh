for ((i = 1; i < 10; i++))
do
    epoch=$((i * 20000))
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python main.py --restore_path=exp/image_proprio/Debug/sd000_20260821_131218 --restore_epoch=$epoch --eval_seed=345
done

