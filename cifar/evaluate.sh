python -u main.py \
--gpus 0 \
-e [best_model_path] \
--model resnet20_1w1a \
--data_path [DATA_PATH] \
--dataset cifar10 \
-bt 128 \
