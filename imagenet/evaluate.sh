python -u main.py \
--gpus 0 \
-e best_model_path \
--model resnet18_1w1a \
--data_path DATA_PATH \
--dataset imagenet \
-b 256 \
-bt 128 \
