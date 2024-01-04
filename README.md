# A Generalized Framework for Video Instance Segmentation (CVPR 2023)
[Miran Heo](https://sites.google.com/view/miranheo), [Sukjun Hwang](https://sukjunhwang.github.io), [Jeongseok Hyun](https://sites.google.com/view/jshyun/home), [Hanjung Kim](https://kimhanjung.github.io), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh), [Joon-Young Lee](https://joonyoung-cv.github.io), [Seon Joo Kim](https://sites.google.com/site/seonjookim/home)

[[`arXiv`](https://arxiv.org/abs/2211.08834)] [[`BibTeX`](#CitingGenVIS)]

<div align="center">
  <img src="https://user-images.githubusercontent.com/24949098/212600182-90721a1e-aa4c-452c-86ed-ab1149a16b8f.gif"  width="30%"/>
  <img src="https://user-images.githubusercontent.com/24949098/212599620-082b9604-49f1-4f21-bf8e-01885cd38e82.gif"  width="30%"/>

  <img src="https://user-images.githubusercontent.com/24949098/213493785-27312f33-dbae-4d44-8036-69e597366ab9.gif"  width="60%"/>
</div><br/>

## Updates
* **`Feb 28, 2023`:** GenVIS is accepted to CVPR 2023!
* **`Jan 20, 2023`:** Code is now available!

## Installation
GenVIS is built upon VITA.
See [installation instructions](https://github.com/sukjunhwang/VITA/blob/main/INSTALL.md).

## Getting Started

We provide a script `train_net_genvis.py`, that is made to train all the configs provided in GenVIS.

To train a model with "train_net_genvis.py" on VIS, first
setup the corresponding datasets following
[Preparing Datasets](https://github.com/sukjunhwang/VITA/blob/main/datasets/README.md).

Then run with pretrained weights on target VIS dataset in [VITA's Model Zoo](https://github.com/sukjunhwang/VITA#model-zoo):
```
python train_net_genvis.py --num-gpus 4 \
  --config-file configs/genvis/ovis/genvis_R50_bs8_online.yaml \
  MODEL.WEIGHTS vita_r50_ovis.pth
```

To evaluate a model's performance, use
```
python train_net_genvis.py --num-gpus 4 \
  --config-file configs/genvis/ovis/genvis_R50_bs8_online.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## <a name="ModelZoo"></a>Model Zoo
**Additional weights will be updated soon!**
### YouTubeVIS-2019
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 50.0 | 71.5 | 54.6 | 49.5 | 59.7 | [model](https://drive.google.com/file/d/1WdDsE4EGAuYQ1hqLB4XtZoYO0iSehnZo/view?usp=share_link) |
| R-50 | semi-online | 51.3 | 72.0 | 57.8 | 49.5 | 60.0 | [model](https://drive.google.com/file/d/1yQVzuFFrHsRDd96ywMsGLTDwVqKShFZt/view?usp=share_link) |
| Swin-L | online | 64.0 | 84.9 | 68.3 | 56.1 | 69.4 | [model](https://drive.google.com/file/d/1TZvH5qlhTnZ6WXk1oNmCmYz_cq1m5AuO/view?usp=share_link) |
| Swin-L | semi-online | 63.8 | 85.7 | 68.5 | 56.3 | 68.4 | [model](https://drive.google.com/file/d/1PTtkH-Angrw92D7P7-BXvtAQZ8nWmJ6Q/view?usp=share_link) |

### YouTubeVIS-2021
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 47.1 | 67.5 | 51.5 | 41.6 | 54.7 | [model](https://drive.google.com/file/d/1-WcWxoBRBIAyxhH0-1X2ywe1bquOWjkO/view?usp=share_link) |
| R-50 | semi-online | 46.3 | 67.0 | 50.2 | 40.6 | 53.2 | [model](https://drive.google.com/file/d/1AMqKe9OX-wsr39RUxggTwPY25cvABoub/view?usp=share_link) |
| Swin-L | online | 59.6 | 80.9 | 65.8 | 48.7 | 65.0 | [model](https://drive.google.com/file/d/1cHEfYb6QLGllR1i2xvL-AZnrthKx3wbV/view?usp=share_link) |
| Swin-L | semi-online | 60.1 | 80.9 | 66.5 | 49.1 | 64.7 | [model](https://drive.google.com/file/d/1Nl8bE5JXFdLSoABrvNax_rrnLrt0ZSNc/view?usp=share_link) |

### OVIS
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 35.8 | 60.8 | 36.2 | 16.3 | 39.6 | [model](https://drive.google.com/file/d/15Iitl2sSmAxFXT-PJCYfY37vcc7_iEO7/view?usp=share_link) |
| R-50 | semi-online | 34.5 | 59.4 | 35.0 | 16.6 | 38.3 | [model](https://drive.google.com/file/d/1Y8d0ETmW3XoD-zGxvZNRVvlz1jTsXY5a/view?usp=share_link) |
| Swin-L | online | 45.2 | 69.1 | 48.4 | 19.1 | 48.6 | [model](https://drive.google.com/file/d/11aqfoqDoyEIDcDmYqcWDEX3FK7ChIRks/view?usp=share_link) |
| Swin-L | semi-online | 45.4 | 69.2 | 47.8 | 18.9 | 49.0 | [model](https://drive.google.com/file/d/17uErrcAZ6-5ewdzUy9CxDK6tjOe5Xp93/view?usp=share_link) |

## License
The majority of GenVIS is licensed under a
[Apache-2.0 License](LICENSE).
However portions of the project are available under separate license terms: Detectron2([Apache-2.0 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)), IFC([Apache-2.0 License](https://github.com/sukjunhwang/IFC/blob/master/LICENSE)), Mask2Former([MIT License](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE)), Deformable-DETR([Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE)), and VITA([Apache-2.0 License](https://github.com/sukjunhwang/VITA/blob/main/LICENSE)).

## <a name="CitingGenVIS"></a>Citing GenVIS

If you use GenVIS in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@inproceedings{GenVIS,
  title={A Generalized Framework for Video Instance Segmentation},
  author={Heo, Miran and Hwang, Sukjun and Hyun, Jeongseok and Kim, Hanjung and Oh, Seoung Wug and Lee, Joon-Young and Kim, Seon Joo},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{VITA,
  title={VITA: Video Instance Segmentation via Object Token Association},
  author={Heo, Miran and Hwang, Sukjun and Oh, Seoung Wug and Lee, Joon-Young and Kim, Seon Joo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknowledgement

Our code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [IFC](https://github.com/sukjunhwang/IFC), [Mask2Former](https://github.com/facebookresearch/MaskFormer), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and [VITA](https://github.com/sukjunhwang/VITA). We are truly grateful for their excellent work.
