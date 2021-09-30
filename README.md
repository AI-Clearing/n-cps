# nCPS

## Docker
To run the image (needs `"default-runtime": "nvidia"` on `/etc/docker/daemon.json` due to Apex compilation during the build)
```bash
sudo docker image build --file Dockerfile --tag ncps:0.0.1 ../ && scripts/run-voc-101.sh
```
(Please set the proper paths in `docker` directory)

-----

Original readme below.

-----

# TorchSemiSeg
<br>

> [[CVPR 2021] Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226)
>
> by [Xiaokang Chen](https://charlescxk.github.io)<sup>1</sup>, [Yuhui Yuan](https://scholar.google.com/citations?user=PzyvzksAAAAJ&hl=zh-CN)<sup>2</sup>, [Gang Zeng](https://www.cis.pku.edu.cn/info/1177/1378.htm)<sup>1</sup>, [Jingdong Wang](https://jingdongwang2017.github.io/)<sup>2</sup>.
> 
> <sup>1</sup> Key Laboratory of Machine Perception (MOE), Peking University
><sup>2</sup> Microsoft Research Asia.
> 
> [[Poster](https://charlescxk.github.io/papers/CVPR2021_CPS/00446-poster.pdf)] [[Video (YouTube)](https://www.youtube.com/watch?v=5HKitm0O27w)]
>
> ***Simpler Is Better !***

<br>

<img src=ReadmePic/cps.png width="600">

## News
- **[July 9  2021] We have released some SOTA methods (Mean Teacher, CCT, GCT).**  
- **[June 3 2021] Please check our paper in [Arxiv](https://arxiv.org/abs/2106.01226). Data and code have been released.**  


## Installation
Please refer to the [Installation](./docs/installation.md) document.

## Getting Started
Please follow the [Getting Started](./docs/getting_started.md) document.


## Citation

Please consider citing this project in your publications if it helps your research.

```bibtex
@inproceedings{chen2021-CPS,
  title={Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision},
  author={Chen, Xiaokang and Yuan, Yuhui and Zeng, Gang and Wang, Jingdong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

#### TODO
- [x] Dataset release
- [x] Code for CPS + CutMix
- [x] Code for Cityscapes dataset
- [x] Other SOTA semi-supervised segmentation methods
