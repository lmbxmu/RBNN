CUDA_VISIBLE_DEVICES=2 nohup python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir /home/sda1/xzh/L1/new/ \
--data_path /home/xuzihan/data \
--dataset cifar10 \
--weight_hist 0 \
--epoch 400 \
--lr 0.1 \
-b 256 \
-bt 128 \
--lr_type step \
--lr_decay_step 1 2 \
> output2.log 2>&1 &
