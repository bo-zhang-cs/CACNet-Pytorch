# CACNet-Pytorch
This is an unofficial PyTorch implementation of [Composing Photos Like a Photographer](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_Composing_Photos_Like_a_Photographer_CVPR_2021_paper.html), in which the anchor-point regressors is based on Anchor-to-Joint Regression Network and more details about this can be found in their [paper](https://arxiv.org/pdf/1908.09999.pdf) & [code](https://github.com/zhangboshen/A2J).

# Results

## 
| Test set | FCDB | FLMS | KU-PCP |
|:--:|:--:|:--:|:--:|
| Original Paper | IoU=0.718 BDE=0.069 | IoU=0.854 BDE=0.033 | Accuracy=88.2% |
| This code      | IoU=0.702 BDE=0.074 | IoU=0.841 BDE=0.037 | Accuracy=88.4% |

Note that the accuracy is produced training composition classification branch alone.

# Datasets Preparation
+ KU-PCP [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S1047320318301147)
[[Download link]](http://mcl.korea.ac.kr/research/Submitted/jtlee_JVCIR2018/KU_PCP_Dataset.zip).
+ FLMS [[Download Images]](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_image/image.tar) [[Download Annotation]](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_gt/release_data.tar)
+ FCDB [[Download link]](https://github.com/yiling-chen/flickr-cropping-dataset)

1. Download these datasets and change the default dataset folder in ``config_*.py``. 
2. There are some images that unable to open in KU-PCP dataset, and you can fix this by calling ``check_jpg_file`` function in ``KUPCP_dataset.py``.

# Requirements
- PyTorch>=1.0
- torchvision
- tensorboardX
- opencv-python
- tqdm

You can also install packages using pip according to [``requirements.txt``](./requirements.txt): 

```bash
pip install -r requirements.txt
```

# Usage

## Testing

```bash
  # clone this repository
  git clone https://github.com/bo-zhang-cs/CACNet-Pytorch.git
  cd CACNet-Pytorch && mkdir pretrained_model
  ```
Download pretrained model (~75MB) from [[Google Drive]](https://drive.google.com/file/d/19LUhHK1viHu9TYqzk2te2orqzKMA_ZRQ/view?usp=sharing) to the folder ``pretrained_model``.
```
python test.py
```
This will produce a folder ``results`` where you can find the predicted best crops.

## Training

### Train composition classification model
```
python train_composition_classification.py
```

### Train image cropping model (CACNet)
```
python train_image_cropping.py
```

### Tracking training process
```
tensorboard --logdir=./experiments
```
The model performance for each epoch is also recorded in *.csv* file under the produced folder *./experiments*. 

# Citation
```
@inproceedings{hong2021composing,
  title={Composing Photos Like a Photographer},
  author={Hong, Chaoyi and Du, Shuaiyuan and Xian, Ke and Lu, Hao and Cao, Zhiguo and Zhong, Weicai},
  booktitle={CVPR},
  year={2021}
}
```

## More references about image cropping 
[Awesome Image Aesthetic Assessment and Cropping](https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping)
