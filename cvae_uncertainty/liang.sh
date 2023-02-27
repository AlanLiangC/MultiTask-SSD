cd cvae_uncertainty;mkdir -p logs;
exp_id=exp20 # you can set other exp_id
for iter in `seq 0 1`;do
    sed "s@# FOLD_IDX: 0@FOLD_IDX: ${iter}@" cfgs/${exp_id}_gen_ori.yaml > cfgs/${exp_id}_gen.yaml
    grep FOLD cfgs/${exp_id}_gen.yaml
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 --cfg_file cfgs/${exp_id}_gen.yaml --tcp_port 18889  --max_ckpt_save_num 10  --workers 1 --extra_tag fold_${iter} &>> logs/${exp_id}_gen_fold_${iter}.log
done