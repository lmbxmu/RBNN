torch.\_\_version\_\_=1.3.0  

# 1. Cifar
```bash
nohup python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir ./ \
--save result \
--data_path /data \
--dataset cifar10 \
--epoch 400 \
--lr 0.1 \
--rotation_update 2 \
--Tmin 1e-2 \
--Tmax 1e1 \
--lr_type cos \
-b 256 \
-bt 128 \
> output.log 2>&1 &
```
`--results_dir` &emsp;保存目录  
`--save` &emsp;保存文件夹名  
`--resume` &emsp;加载ckpt  
`--evaluate / -e`  &emsp;evaluate  
`--model / -a` &emsp;选择模型，  
&emsp;&emsp;默认resnet20_1w1a,可选择  
&emsp;&emsp; mobilenetv1_bireal_025_1w1a  
&emsp;&emsp; mobilenetv1_bireal_05_1w1a  
&emsp;&emsp; mobilenetv1_bireal_025_noGroup_1w1a  
&emsp;&emsp; mobilenetv1_bireal_05_noGroup_1w1a  
&emsp;&emsp; vgg_small_1w1a  
`--dataset` &emsp;选择数据集，默认cifar10，可选cifar100 / tinyimagenet  
`--data_path` &emsp;数据集路径  
`--gpus` &emsp;eg: 0,1  
`--lr` &emsp;初始学习率，默认0.1  
`--weight_decay` &emsp;默认1e-4  
`--momentum` &emsp;默认0.9  
`--workers` &emsp;data loading workers，默认8  
`--epochs` &emsp;epoch数，默认400  
`--batch_size / -b` &emsp;batch size，默认256   
`--batch_size_test / -bt` &emsp;evaluate batch size, 默认128  
`--print_freq` &emsp;打印频率，默认100  
`--time_estimate` &emsp;程序结束时间估计，设为0取消，默认1 开启  
`--mixup` &emsp;使用mixup  
`--labelsmooth` &emsp;使用label smooth  
`--weight_hist` &emsp;输出weight的直方图，默认0，设为n表示每隔n个epoch输出一次   
`--rotation_update` &emsp;每n个epoch更新一次旋转矩阵，默认2   
`--Tmin` &emsp;梯度近似函数参数T的最小值，默认1e-2  
`--Tmax` &emsp;梯度近似函数参数T的最大值，默认1e1  
`--lr_type` &emsp;lr_scheduler类型，默认cos，可选step  
`--lr_decay_step` &emsp;step lr的更新点，eg: 30 60 90   

# 2. ImageNet
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -u main.py \
--gpus 0,1 \
--model resnet18_1w1a \
--results_dir ./ \
--save result \
--data_path /data \
--dataset imagenet \
--epoch 120 \
--lr 0.1 \
--rotation_update 2 \
--Tmin 1e-2 \
--Tmax 1e1 \
--lr_type cos \
-b 256 \
-bt 128 \
> output.log 2>&1 &
```  
注：多卡时，pytorch1.3以下版本需在开头注明CUDA_VISIBLE_DEVICES=..., 1.3及以上不需要注明    
默认使用DALI  
其他参数和cifar10相同  
`--model / -a` &emsp;选择模型，  
&emsp;&emsp;默认resnet18_1w1a,可选择  
&emsp;&emsp; resnet34_1w1a  
&emsp;&emsp; mobilenetv1等尚未添加   
`--print_freq` &emsp;打印频率，默认500  


