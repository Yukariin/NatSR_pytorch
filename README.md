# NatSR_pytorch
Pytorch implementation of Natural and Realistic Single Image Super-Resolution

## Citation
```
@InProceedings{Soh_2019_CVPR,
author = {Soh, Jae Woong and Park, Gu Yong and Jo, Junho and Cho, Nam Ik},
title = {Natural and Realistic Single Image Super-Resolution With Explicit Natural Manifold Discrimination},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Acknowledgments
[ImageSplitter](https://github.com/Yukariin/NatSR_pytorch/blob/master/utils.py#L12) and [SQLite-based](https://github.com/Yukariin/NatSR_pytorch/blob/master/data.py#L59) dataset are based on [yu45020](https://github.com/yu45020)'s waifu2x [re-implementation](https://github.com/yu45020/Waifu2x).

[SQLite-based](https://github.com/Yukariin/NatSR_pytorch/blob/master/gen_data.py) data generator based on reference NatSR [implementation](https://github.com/JWSoh/NatSR).


[NMD](https://github.com/Yukariin/NatSR_pytorch/blob/master/train_nmd.py) training codes heavily based on reference NatSR [implementation](https://github.com/JWSoh/NatSR).
