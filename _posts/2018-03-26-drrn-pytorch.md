---
title: DRRN-Pytorch
date: 2018-03-25 21:52:53
categories:
- Super Resolution
tags:
- Deep Learning
- Super Resolution
- Computer Vision
---

This is an unofficial implementation of "Deep Recursive Residual Network for Super Resolution (DRRN)", CVPR 2017 in Pytorch ([Source Code](https://github.com/jt827859032/DRRN-pytorch)). [[Paper]](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) 

<!-- more -->

You can get the official Caffe implementation [here](https://github.com/tyshiwo/DRRN_CVPR17).

This implementation is modified from the implementation of [VDSR](https://cv.snu.ac.kr/research/VDSR/) by [@Jiu XU](https://github.com/twtygqyy/pytorch-vdsr).

## Training

```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]
               
optional arguments:
  -h, --help            Show this help message and exit
  --batchSize           Training batch size
  --nEpochs             Number of epochs to train for
  --lr                  Learning rate. Default=0.1
  --step                Learning rate decay, Default: n=5 epochs
  --cuda                Use cuda?
  --resume              Path to checkpoint
  --clip                Clipping Gradients. Default=0.01
  --threads             Number of threads for data loader to use Default=1
  --momentum            Momentum, Default: 0.9
  --weight-decay        Weight decay, Default: 1e-4
  --pretrained          Path to the pretrained model, used for weight initialization (default: none)
```



## Evaluation



```
usage: eval.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
               [--scale SCALE]

PyTorch DRRN Evaluation

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --dataset DATASET  dataset name, Default: Set5
```
An example of training usage is shown as follows:
```
python eval.py --cuda
```



## Prepare Training dataset



  - the training data is generated with Matlab Bicubic Interpolation, please refer [Code for Data Generation](/data/generate_trainingset_x234.m) for creating training files.



## Performance



  - We provide a ***rough*** pre-trained DRRN_B1U25 [model](/model) trained on [291](/data/Train_291) images with data augmentation. The model can achieve a better performance with a smart optimization strategy. For the DRRN_B1U9 implementation, you can manually modify the number of recursive blocks [here](/drrn.py#L26:18).
  - The same adjustable gradient clipping's implementation as original paper.
  - No bias is used in this implementation.
  - No batch normalization is used in this implementation.
  - Performance in PSNR on Set5 

| Scale | DRRN_B1U25 Paper | DRRN_B1U25 PyTorch |
| ----: | ---------------: | -----------------: |
|    x2 |            37.74 |              37.69 |
|    x3 |            34.03 |              34.02 |
|    x4 |            31.68 |              31.70 |
