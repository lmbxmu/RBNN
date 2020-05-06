torch.\_\_version\_\_=1.3.0  

# 1. Cifar
```bash
nohup python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir ./ \
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
`--a32` &emsp;使用w1a32    
`--use_gpu` &emsp;使用gpu来计算svd   

# 2. ImageNet
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -u main.py \
--gpus 0,1 \
--model resnet18_1w1a \
--results_dir ./ \
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

旋转操作嵌入到卷积核内，在文件`modules-binarized_modules`里，  
```python
V = self.R1.t()@X.detach()@self.R2
B = torch.sign(V)
#* update R1
D1=sum([Bi@(self.R2.t())@(Xi.t()) for (Bi,Xi) in zip(B,X.detach())]).cpu()
U1,S1,V1=torch.svd(D1)
self.R1=(V1@(U1.t())).to(X.device)
#* update R2
D2=sum([(Xi.t())@self.R1@Bi for (Xi,Bi) in zip(X.detach(),B)]).cpu()
U2,S2,V2=torch.svd(D2)
self.R2=(U2@(V2.t())).to(X.device)
```
这一部分将svd计算转移到cpu上，将计算后的R1，R2重新转移到GPU上，服务器上实测放在cpu上更快，但是GPU整体利用率很低    

可调参数：  
Bi-Realnet中参数设置： 
1. Resnet18
* SGD
* momentum 0.9
* weight_decay 0
* lr 0.01
* batch_size=128 (cifar上发现小batchsize效果更好)
* epochs 20 (没写错)
* lr_type step  
* lr_decay_step [10,15]
1. Resnet34 
* SGD
* momentum 0.9
* weight_decay 0
* lr 0.08
* batch_size=1024  
* epochs 40 (没写错)
* lr_type step  
* lr_decay_step [20,30]
论文中提到训练完后，固定weight到-1，1，单独对BatchNorm层再训一个epoch（这部分代码还没写）

ReActNet中参数设置：   
backbone是修改的mobilenetV1-0.5  
两阶段训练方法，用到蒸馏(Training binary neural networks with real-to- binary convolutions.)  
* Adam (代码里用的sgd，在cifar上发现adam效果不如sgd)  
* epoch 120 
* batch_size 256  
* lr 5e-4 (没写错)
* linear lr decay  
* weight_decay 第一阶段1e-5,第二阶段0
* 用到了Distributional Loss,替换原始CE loss（公式上看着像无教师蒸馏）

--weight_decay 默认1e-4，BI-Real中设为0  
--Tmin/Tmax 默认1e-2 / 1e1 , 可调1e-3/1e1 ; 1e-0.5,1e1等  
--batchsize 默认256 ,可调128  
--rotation_update 默认2，可调大到5/10等  

