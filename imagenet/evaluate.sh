nohup python -u main.py \
--gpus 0 \
-e best_model_path \
--model resnet18_1w1a \
--results_dir ./ \
--save result \
--data_path /media/disk2/zyc/ImageNet2012 \
--dataset imagenet \
--weight_hist 0 \
--epoch 400 \
--lr 0.1 \
-b 256 \
-bt 128 \
> evaluate.log 2>&1 &
