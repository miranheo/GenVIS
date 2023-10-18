# A Generalized Framework for Video Instance Segmentation (CVPR 2023)
[Miran Heo](https://sites.google.com/view/miranheo), [Sukjun Hwang](https://sukjunhwang.github.io), [Jeongseok Hyun](https://sites.google.com/view/jshyun/home), Hanjung Kim, [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh), [Joon-Young Lee](https://joonyoung-cv.github.io), [Seon Joo Kim](https://sites.google.com/site/seonjookim/home)

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
| R-50 | online | 50.0 | 71.5 | 54.6 | 49.5 | 59.7 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EbxTvAZ_o0dHiUBaw1CYM7cB8pZqFP5Sf7E3LCYyrXNIZw?e=cYAZBt) |
| R-50 | semi-online | 51.3 | 72.0 | 57.8 | 49.5 | 60.0 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EQn84SkT22NHn5K2EIfLQsMBy6keLAlIAD8Bhq5zsEKj1Q?e=MRANHG) |
| Swin-L | online | 64.0 | 84.9 | 68.3 | 56.1 | 69.4 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EWWtxwxuQ3tAuixVcZCPf9YB9kdE2JjF554RGEy0OyepVA?e=gEhMJo) |
| Swin-L | semi-online | 63.8 | 85.7 | 68.5 | 56.3 | 68.4 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/Ecb9sr84mcJPl2hxiEYrF2gBnMKiHFvS-pcKJdUgAG67Wg?e=tuhLa7) |

### YouTubeVIS-2021
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 47.1 | 67.5 | 51.5 | 41.6 | 54.7 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EYSdkV0C-cxMlDr54iv4_IsBtN76_JcqyyilWHfWXmNKcg?e=1OnZcG) |
| R-50 | semi-online | 46.3 | 67.0 | 50.2 | 40.6 | 53.2 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/ERTULYUDogZKqf2PFXH56SMBmgX3sVGh489_qFQMIduYVw?e=qDZogy) |
| Swin-L | online | 59.6 | 80.9 | 65.8 | 48.7 | 65.0 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/Efl0WKxpeW5ApZTRzh9StxYB8T63Lqq3xZXUBTRjDbPyOw) |
| Swin-L | semi-online | 60.1 | 80.9 | 66.5 | 49.1 | 64.7 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EXfDA0hNnV9Br9czvylfPNwBzRPgTAy09GSMXuPaKLczEg) |

### OVIS
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 35.8 | 60.8 | 36.2 | 16.3 | 39.6 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EaoKJ6bj99xFgv30nz0QOJkB0usiAnwpPmeoavUqLZeTxw?e=dCHOax) |
| R-50 | semi-online | 34.5 | 59.4 | 35.0 | 16.6 | 38.3 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/ESU6iD7LX4hPv_UAIFiBCmUB2PHcH7rrUtfp0qYndZ3Y2g?e=Zst877) |
| Swin-L | online | 45.2 | 69.1 | 48.4 | 19.1 | 48.6 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EfAYQjnt0AFKrFN2wpE6Ei8B-Nn7nRpSUWkaHyhOn4H-Gg?e=uMuuWN) |
| Swin-L | semi-online | 45.4 | 69.2 | 47.8 | 18.9 | 49.0 | [model](https://yonsei-my.sharepoint.com/:u:/g/personal/miran_o365_yonsei_ac_kr/EV4HRAHBnZdOjuW9p0FSb1wBOK79Dqq3DGJcfOof7Xi2ZA?e=eb5FA3) |

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
