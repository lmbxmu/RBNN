torch.\_\_version\_\_=1.3.0  

# 1. Cifar
```bash
nohup python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir ./ \
--save result \
--data_path /home/xuzihan/data \
--dataset cifar10 \
--weight_hist 0 \
--epoch 400 \
--lr 0.1 \
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

# 2. ImageNet
默认使用DALI  
其他参数和cifar10相同  
`--model / -a` &emsp;选择模型，  
&emsp;&emsp;默认resnet18_1w1a,可选择  
&emsp;&emsp; resnet34_1w1a  
&emsp;&emsp; mobilenetv1等尚未添加   


