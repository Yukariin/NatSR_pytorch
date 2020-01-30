# NatSR_pytorch
Pytorch implementation of Natural and Realistic Single Image Super-Resolution

## Pre-trained Models

### Data
All models trained on the same anime-themed datased based on [Danbooru2019](https://www.gwern.net/Danbooru2019).
Final dataset consists of 800/100 (train/val) images.
Original images are all PNG images at least 2K x 2K. Images downsampled to 2K by LANCZOS, that is they have 2K pixels on at least one of the axes (vertical or horizontal), and then cropped to multiple of 12 pixels on both axes.
All images splitted into 96x96/192x192 (x2/x4) HR and 48x48 LR (with jpeg noise) overlapping patches. All HR patches filtered by it's gradient and variance, and stored in SQLite database.

Image noise are from JPEG format only. Same as for [waifu2x](https://github.com/yu45020/Waifu2x).
Noise level 1 means quality ranges uniformly from [75, 95]; level 2 means quality ranges uniformly from [50, 75].

### Scores
Scores calculated on validation dataset which consists of ~14K HR/LR patches for scale factor of 2.

| Model | Noise level | L1(-)  | PSNR(+) | SSIM(+) |
| ----- | ----------- | ------ | ------- | ------- |
| FRSR  | 1           | 0.0091 | 35.0785 | 0.9802  |
| NSR   | 1           | 0.0081 | 35.7897 | 0.9817  |

### Models
[FRSR](https://www.dropbox.com/s/q8vzkzp1b8ndbsn/G_50000.pth) - Scale factor x2 - Noise level 1

[NatSR](https://www.dropbox.com/s/7fxygrc24jvjh14/G_220000.pth) - Scale factor x2 - Noise level 1

### Usage
```
python test.py --scale 2 --checkpoint path/to/model.pth --input path/to/image.jpg --output SR_output_x2.png
```

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
