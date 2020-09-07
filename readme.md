Pytorch implementation of **Rotated Binary Neural Network**  
torch.\_\_version\_\_=1.1.0 


# CIFAR-10
```bash
python -u main.py \
--gpus 0 \
--model resnet20_1w1a \
--results_dir ./result \
--data_path ./data \
--dataset cifar10 \
--epochs 1000 \
--lr 0.1 \
-b 256 \
-bt 128 \
--Tmin 1e-2 \
--Tmax 1e1 \
--lr_type cos \
--warm_up \
```
`--results_dir` &emsp;Path to save directory  
`--save` &emsp;Path to save folder    
`--resume` &emsp;Load checkpoint    
`--evaluate / -e`  &emsp;Evaluate  
`--model / -a` &emsp;Choose model   
&emsp;&emsp; default:&emsp;resnet20_1w1a,   
&emsp;&emsp; options:&emsp;resnet18_1w1a;&emsp;vgg_small_1w1a       
`--dataset` &emsp;Choose dataset，default: cifar10，options: cifar100 / tinyimagenet / imagenet  
`--data_path` &emsp;Path to dataset    
`--gpus` &emsp;Specify gpus, e.g. 0,1  
`--lr` &emsp;Learning rate，default: 0.1  
`--weight_decay` &emsp;Weight decay, default: 1e-4  
`--momentum` &emsp;Momentum, default: 0.9  
`--workers` &emsp;Data loading workers，default: 8  
`--epochs` &emsp;Number of training epochs，default:1000  
`--batch_size / -b` &emsp;Batch size，default: 256   
`--batch_size_test / -bt` &emsp;Evaluating batch size, default: 128  
`--print_freq` &emsp;Print frequency，default: 100  
`--time_estimate` &emsp;Estimate finish time of the program，set to 0 to disable，default: 1     
`--rotation_update` &emsp;Update rotaion matrix every n epoch，default: 1   
`--Tmin` &emsp;The minimum of param T in gradient approximation function，default: 1e-2  
`--Tmax` &emsp;The maximum of param T in gradient approximation function，default: 1e1  
`--lr_type` &emsp;Type of learning rate scheduler，default: cos (which means CosineAnnealingLR)，options: step (which means MultiStepLR)  
`--lr_decay_step` &emsp;If choose MultiStepLR, set milestones，eg: 30 60 90    
`--a32` &emsp;Don't binarize activation, namely w1a32    
`--warm_up` &emsp;Use warm up  

## Implementation details on Cifar10
args | resnet20_1w1a | resnet18_1w1a | vgg_small_1w1a
-|:-:|:-:|:-:
lr | 0.1 | 0.1 | 0.1
weight_decay | 1e-4 | 1e-4 | 1e-4 
momentum | 0.9 | 0.9 | 0.9
epochs | 1000 | 1000 | 1000
batch_size | 256 | 256 | 256
batch_size_test | 128 | 128 | 128
Tmin | 1e-2 | 1e-2 | 1e-2 
Tmax | 1e1 | 1e1 | 1e1
lr_type | cos | cos | cos
rotation_update | 1 | 1 | 1
warm_up | True | True | True

Note: Training for 400 epochs on cifar10 can also achieve impressive results, but training for 1000 epochs performs better, so we set `epochs` to 1000 as default.

# ImageNet
```bash
python -u main.py \
--gpus 0,1,2,3 \
--model resnet18_1w1a \
--results_dir ./result \
--data_path ./data \
--dataset imagenet \
--epochs 120 \
--lr 0.1 \
-b 256 \
-bt 128 \
--Tmin 1e-2 \
--Tmax 1e1 \
--lr_type cos \
--warm_up \
--use_dali \
```   
Other args are the same as those in CIFAR  
`--model / -a` &emsp;Choose model，  
&emsp;&emsp;default: resnet18_1w1a.   
&emsp;&emsp;options: resnet34_1w1a     
  
Imagenet DataLoaders are implemented with [nvidia-dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html), if you have never used dali before, you only need to install nvidia-dali package and the version of nvidia-dali should be >= 0.12
```
#for cuda9.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
#for cuda10.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```
More details and documents can be found [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html#)

## Implementation details on ImageNet

|<font size="6">lr</font>| weight_decay | momentum | epochs | batch_size | batch_size_test | Tmin | Tmax | lr_type |rotation_update | warm_up | use_dali| model | Top-1| Top-5 | Link | Paper data|
|:--:|:------------:|:--------:|:------:|:----------:|:---------------:|:----:|:----:|:-------:|:---------------:|:-------:|:-------:|:-----:|:----:|:-----:|:----:|:---------:|
| 0.1|    1e-4      |    0.9   |  120   |   512      |  256            | 1e-2 | 10.0 |  cos    |        1        |   False | Yes |resnet18_1w1a|59.550|81.581|      |  Yes | 
| 0.1|    1e-4      |    0.9   |  120   |   256      |  256            | 1e-2 | 10.0 |  cos    |        1        |   False | Yes |resnet18_1w1a |58.757|80.935|      |  No | 
| 0.1|    1e-4      |    0.9   |  150   |   512      |  256            | 1e-2 | 10.0 |  cos    |        1        |   False | Yes |resnet18_1w1a |59.941|81.892|      |  No | 
| 0.1|    1e-4      |    0.9   |  150   |   512      |  256            | 1e-2 | 10.0 |  cos    |        1        |   False | Yes |resnet34_1w1a |63.141| 84.379|      |  Yes |
