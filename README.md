<h2 align="center">FactSeg: Foreground Activation Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery</h2>

<h5 align="right">by Ailong Ma, <a href="https://junjue-wang.github.io/homepage/">Junjue Wang*</a>, <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong*</a> and <a href="http://zhuozheng.top/">Zhuo Zheng</a></h5>

<div align="center">
  <img src="https://github.com/Junjue-Wang/FactSeg/blob/master/imgs/framework.png"><br><br>
</div>
<div align="center">
  <img src="https://github.com/Junjue-Wang/FactSeg/blob/master/imgs/result.png"><br><br>
</div>

This is an official implementation of FactSeg in our TGRS paper "
FactSeg: Foreground Activation Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery
"


## Citation
If you use FactSeg in your research, please cite our coming TGRS paper.
```text
@ARTICLE{FacSeg,
  author={Ma Ailong, Wang Junjue, Zhong Yanfei and Zheng Zhuo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FactSeg: Foreground Activation Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery}, 
  year={2021},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2021.3097148}}
```
This is follow-up work of our FarSeg (CVPR2020).
```text
@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
}
```

## Getting Started
### Install SimpleCV

```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```

#### Requirements:
- pytorch >= 1.1.0
- python >=3.6

### Prepare iSAID Dataset

```bash
ln -s </path/to/iSAID> ./isaid_segm
```

### Evaluate Model
#### 1. download pretrained weight in [Google Drive](https://drive.google.com/file/d/19cCWD3uSZJX_h_carMI1aW6lgAl32qZZ/view?usp=sharing)

#### 2. move weight file to log directory
```bash
mkdir -vp ./log/
mv ./factseg50.pth ./log/model-60000.pth
```
#### 3. inference on iSAID val
```bash
bash ./scripts/eval_factseg.sh
```

### Train Model
```bash
bash ./scripts/train_factseg.sh
```
