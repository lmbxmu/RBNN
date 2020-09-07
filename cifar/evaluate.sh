python -u main.py \
--gpus 0 \
-e [best_model_path] \
--model resnet20_bireal_1w1a \
--data_path [DATA_PATH] \
--dataset cifar10 \
-bt 128 \
