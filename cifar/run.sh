CUDA_VISIBLE_DEVICES=2 nohup python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir DIR \
--data_path /home/xuzihan/data \
--dataset cifar10 \
--weight_hist 0 \
--epoch 400 \
--lr 0.1 \
-b 256 \
-bt 128 \
--rotation_update 2 \
--Tmin 1e-2 \
--Tmax 1e1 \
--lr_type cos \
> output2.log 2>&1 &
