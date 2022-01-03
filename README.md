# n-CPS
<br>

> [n-CPS: Generalising Cross Pseudo Supervision to n networks for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2106.01226)
>
> by [Dominik Filipiak](http://dfilipiak.com)<sup>1,2</sup>, [Piotr Tempczyk](https://ptempczyk.github.io)<sup>1,3</sup>, [Marek Cygan](https://www.mimuw.edu.pl/~cygan/)<sup>3</sup>.
> 
> <sup>1</sup> AI Clearing, Inc. <br/>
> <sup>2</sup> Semantic Technology Institute, Department of Computer Science, University of Innsbruck <br/>
> <sup>3</sup> Institute of Informatics, University of Warsaw

We present n-CPS – a generalisation of the recent state-of-the-art [cross pseudo supervision (CPS)]() approach for the task of semi-supervised semantic segmentation. In n-CPS, there are n simultaneously trained subnetworks that learn from each other through one-hot encoding perturbation and consistency regularisation. We also show that ensembling techniques applied to subnetworks outputs can significantly improve the performance. To the best of our knowledge, n-CPS paired with Cut-Mix outperforms CPS and sets the new state-of-the-art for Pascal VOC 2012 with (1/16, 1/8, 1/4, and 1/2 supervised regimes) and Cityscapes (1/16 supervised).

The code in this repository is based mostly on the original [CPS repository](https://github.com/charlesCXK/TorchSemiSeg) and [NVidia Apex](https://github.com/NVIDIA/apex).
<br>

<img src=ReadmePic/ncps.png width="600">

## Running the code
Please refer to the [Installation](./docs/installation.md) document.

## Citation

Please consider citing this project in your publications if it helps your research.

```bibtex
@misc{filipiak2021ncps,
      title={n-CPS: Generalising Cross Pseudo Supervision to n Networks for Semi-Supervised Semantic Segmentation}, 
      author={Dominik Filipiak and Piotr Tempczyk and Marek Cygan},
      year={2021},
      eprint={2112.07528},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
