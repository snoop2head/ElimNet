# ELimNet

[![Wandb Log](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/elimnet/ElimNet)

**ELimNet: Eliminating Layers in a Neural Network Pretrained with Large Dataset for Downstream Task**

[📂 Please refer to README.pdf for further information.](https://github.com/snoop2head/ELimNet/blob/main/README.pdf)

- Removed top layers from pretrained EfficientNetB0 and ResNet18 to construct lightweight CNN model with less than 1M #params.
- Assessed on [Trash Annotations in Context(TACO) Dataset](http://tacodataset.org/) sampled for 6 classes with 20,851 images.
- Compared performance with lightweight models generated with Optuna's Neural Architecture Search(NAS) constituted with same convolutional blocks.

## Quickstart

### Installation

```shell
# clone the repository
git clone https://github.com/snoop2head/elimnet

# fetch image dataset and unzip
!wget -cq https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000081/data/data.zip
!unzip ./data.zip -d ./
```

### Train

```shell
# finetune on the dataset with pretrained model
python train.py --model ./model/efficientnet/efficientnet_b0.yaml

# finetune on the dataset with ElimNet
python train.py --model ./model/efficientnet/efficientnet_b0_elim_3.yaml
```

### Inference

```shell
# inference with the lastest ran model
python inference.py --model_dir ./exp/latest/
```

## Performance

Performance is compared with (1) original pretrained model and (2) Optuna NAS constructed models with no pretrained weights.

- Indicates that top convolutional layers eliminated pretrained CNN models outperforms empty Optuna NAS models generated with same convolutional blocks.
- Suggests that eliminating top convolutional layers creates lightweight model that shows similar(or better) classifcation performance with original pretrained model.
- **Reduces parameters to 7%(or less) of its original parameters while maintaining(or improving) its performance.** Saves inference time by 20% or more by eliminating top convolutional layters.

### ELimNet vs Pretrained Models (Train)

| [100 epochs]               | # of Parameters | # of Layers | Train                                             | Validation                                         | Test F1    |
| -------------------------- | --------------- | ----------- | ------------------------------------------------- | -------------------------------------------------- | ---------- |
| Pretrained EfficientNet B0 | 4.0M            | 352         | **Loss: 0.43**<br />Acc: 81.23%<br />F1: 0.84     | **Loss: 0.469**<br />Acc: 82.17%<br />F1: 0.76     | 0.7493     |
| EfficientNet B0 Elim 2     | 0.9M            | 245         | Loss:0.652<br />**Acc: 87.22%**<br />**F1: 0.84** | Loss: 0.622<br />**Acc: 87.22%**<br />**F1: 0.77** | **0.7603** |
| EfficientNet B0 Elim 3     | **0.30M**       | 181         | Loss: 0.602<br />Acc: 78.17%<br />F1: 0.74        | Loss: 0.661<br />Acc: 77.41% <br />F1: 0.74        | 0.7349     |
|                            |                 |             |                                                   |                                                    |            |
| Resnet18                   | 11.17M          | 69          | Loss: 0.578<br />Acc: 78.90%<br />F1: 0.76        | Loss: 0.700<br />Acc: 76.17%<br />F1: 0.719        | -          |
| Resnet18 Elim 2            | 0.68M           | **37**      | Loss: 0.447<br />Acc: 83.73%<br />F1: 0.71        | Loss: 0.712<br />Acc: 75.42%<br />F1: 0.71         | -          |

### ELimNet vs Pretrained Models (Inference)

|                            | # of Parameters | # of Layers | CPU times (sec) | CUDA time (sec) | Test Inference Time (sec) |
| -------------------------- | --------------- | ----------- | --------------- | --------------- | ------------------------- |
| Pretrained EfficientNet B0 | 4.0M            | 352         | 3.9s            | **4.0s**        | 105.7s                    |
| EfficientNet B0 Elim 2     | 0.9M            | 245         | 4.1s            | 13.0s           | 83.4s                     |
| EfficientNet B0 Elim 3     | **0.30M**       | 181         | **3.0s**        | 9.0s            | **73.5s**                 |
|                            |                 |             |                 |                 |                           |
| Resnet18                   | 11.17M          | 69          | -               | -               | -                         |
| Resnet18 Elim 2            | 0.68M           | **37**      | -               | -               | -                         |

### ELimNet vs Empty Optuna NAS Models (Train)

| [100 epochs]                          | # of Parameters | # of Layers | Train                                              | Valid                                                  | Test F1    |
| ------------------------------------- | --------------- | ----------- | -------------------------------------------------- | ------------------------------------------------------ | ---------- |
| Empty MobileNet V3                    | 4.2M            | 227         | Loss 0.925<br />Acc: 65.18%<br />F1: 0.58          | Loss 0.993<br />Acc: 62.83%<br />F1: 0.56              | -          |
| Empty EfficientNet B0                 | 1.3M            | 352         | Loss 0.867<br />Acc: 67.28%<br />F1: 0.61          | Loss 0.898<br />Acc: 66.80%<br />F1: 0.61              | 0.6337     |
|                                       |                 |             |                                                    |                                                        |            |
| Empty DWConv & InvertedResidualv3 NAS | **0.08M**       | 66          | -                                                  | Loss: 0.766<br />Acc: 71.71%<br />F1: 0.68             | 0.6740     |
| Empty MBConv NAS                      | 0.33M           | 141         | Loss: 0.786<br />Acc: 70.72%<br />F1: 0.66         | Loss: 0.866<br />Acc: 68.09%<br />F1: 0.62             | 0.6245     |
|                                       |                 |             |                                                    |                                                        |            |
| Resnet18 Elim 2                       | 0.68M           | **37**      | **Loss: 0.447**<br />**Acc: 83.73%**<br />F1: 0.71 | Loss: 0.712<br />Acc: 75.42%<br />F1: 0.71             | -          |
| EfficientNet B0 Elim 3                | 0.30M           | 181         | Loss: 0.602<br />Acc: 78.17%<br />F1: 0.74         | **Loss: 0.661**<br />**Acc: 77.41%**<br />**F1: 0.74** | **0.7603** |

### ELimNet vs Empty Optuna NAS Models (Inference)

|                                            | # of Parameters | # of Layers | CPU times (sec) | CUDA time (sec) | Test Inference Time (sec) |
| ------------------------------------------ | --------------- | ----------- | --------------- | --------------- | ------------------------- |
| Empty MobileNet V3                         | 4.2M            | 227         | 4               | 13              | -                         |
| Empty EfficientNet B0                      | 1.3M            | 352         | 3.780           | 3.782           | 68.4s                     |
|                                            |                 |             |                 |                 |                           |
| Empty DWConv &<br />InvertedResidualv3 NAS | **0.08M**       | 66          | 1               | **3.5**         | **61.1s**                 |
| Empty MBConv NAS                           | 0.33M           | 141         | 2.14            | 7.201           | 67.1s                     |
|                                            |                 |             |                 |                 |                           |
| Resnet18 Elim 2                            | 0.68M           | **37**      | -               | -               | -                         |
| EfficientNet B0 Elim 3                     | 0.30M           | 181         | 3.0s            | 9s              | 73.5s                     |

## Background & WiP

### Background

- NLP tasks are usually downstream tasks of finetuning large pretrained transformers models(i.e. BERT, RoBERTa, XLNet).
- [Removing top transformers layers may yield 40% reduction in size while preserving up to 98.2% of the performance.](https://arxiv.org/pdf/2004.03844.pdf)
- Likewise, for computer vision's classification task, removing convolutional top layers from pretrained models are tested.

### Work in Progress

- Will test the performance of replacing convolutional blocks with pretrained weights with a single convolutional layer without pretrained weights.
- Will add ResNet18's inference time data and compare with Optuna's NAS constructed lightweight model.
- Will test on pretrained MobileNetV3, MnasNet on torchvision with elimination based lightweight model architecture search.
- Will be applied on other small datasets such as Fashion MNIST dataset and Plant Village dataset.

### Others

- "Empty" stands for model with no pretrained weights.
- "EfficientNet B0 Elim 2" means 2 convolutional blocks have been eliminated from pretrained EfficientNet B0. Number next to "Elim" annotates how many convolutional blocks have been removed.
- Table's performance illustrates best performance out of 100 epochs of finetuning on TACO Dataset.

---

### Authors

- [@hihellohowareyou](https://github.com/hihellohowareyou)
- [@lkm2835](https://github.com/lkm2835)
- [@shawnhyeonsoo](https://github.com/shawnhyeonsoo)
- [@snoop2head](https://github.com/snoop2head)
- [@JoonHong-Kim](https://github.com/JoonHong-Kim)
- [@jjonhwa](https://github.com/jjonhwa)
- [@kimyeondu](https://github.com/kimyeondu)
