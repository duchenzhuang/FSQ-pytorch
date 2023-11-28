# FSQ-pytorch
A unoffical Pytorch Implementation of Finite Scalar Quantization (https://arxiv.org/abs/2309.15505)

<img width="1700" alt="image" src="https://github.com/duchenzhuang/FSQ-pytorch/blob/main/figures/fsq_vis.png">

In our view, FSQ is a great idea, and we manage to quickly implement a reproduction on a minimal framework. We are impressed by how FSQ is not only simple and effective in its concept but also highly optimizable during actual training.

## Experimental settings
We use the ImageNet dataset (128*128) for our experiments. The encoder we employe is a simple neural network with four convolutional layers, and the decoder is symmetric to the encoder. This network architecture is highly similar to the structure of [CogView's](https://arxiv.org/abs/2105.13290) VQ-VAE. The implementation of the FSQ quantizer is mainly adapted from another [GitHub repository](https://github.com/lucidrains/vector-quantize-pytorch).

## Training
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train.py --quantizer fsq --levels 8 8 8 5 5 5

The levels can also take on other values, as shown in the figure below.

<img width="1700" alt="image" src="https://github.com/duchenzhuang/FSQ-pytorch/blob/main/figures/fsq_levels.png">

## Quantitative Results
We evaluate several metrics on the validation set, and the results are shown in the table below.

| Codebook Size | L1 loss | Perceptual loss | Codebook Usage |
| :----:| :----: | :----: | :----: |
| 1k | 0.2319 | 0.2597 | 100% |
| 4k | 0.2135 | 0.2299 | 100% |
| 16k | 0.1917 | 0.1931 | 100% |
| 64k | 0.1807 | 0.1761 | 99.94% |

## Qualitative Results

<img width="1700" alt="image" src="https://github.com/duchenzhuang/FSQ-pytorch/blob/main/figures/quality.png">

## Acknowledgement

Our code draws heavily from [CogView's](https://arxiv.org/abs/2105.13290) first stage (VQVAE training) and  another [GitHub repository](https://github.com/lucidrains/vector-quantize-pytorch), and we would like to thank these teams for their selfless sharing. And we also thank Wendi Zheng and Ming Ding for their very constructive suggestions.
