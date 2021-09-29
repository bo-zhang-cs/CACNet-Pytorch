import os
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import random

from config_cropping import cfg

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def rescale_bbox(bbox, ratio_w, ratio_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, 0] = np.floor(bbox[:, 0] * ratio_w)
    bbox[:, 1] = np.floor(bbox[:, 1] * ratio_h)
    bbox[:, 2] = np.ceil(bbox[:, 2] * ratio_w)
    bbox[:, 3] = np.ceil(bbox[:, 3] * ratio_h)
    return bbox.astype(np.float32)

class FCDBDataset(Dataset):
    def __init__(self, split, keep_aspect_ratio=False):
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = cfg.FCDB_dir
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, 'data')
        assert os.path.exists(self.image_dir), self.image_dir
        self.annos = self.parse_annotations(split)
        self.image_list = list(self.annos.keys())
        self.data_augment = (cfg.data_augmentation and self.split == 'train')
        self.PhotometricDistort = transforms.ColorJitter(
            brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])

    def parse_annotations(self, split):
        if split == 'train':
            split_file = os.path.join(self.data_dir, 'cropping_training_set.json')
        else:
            split_file = os.path.join(self.data_dir, 'cropping_testing_set.json')
        assert os.path.exists(split_file), split_file
        origin_data = json.loads(open(split_file, 'r').read())
        annos = dict()
        for item in origin_data:
            url = item['url']
            image_name = os.path.split(url)[-1]
            if os.path.exists(os.path.join(self.image_dir, image_name)):
                x,y,w,h = item['crop']
                crop = [x,y,x+w,y+h]
                annos[image_name] = crop
        print('{} set, {} images'.format(split, len(annos)))
        return annos

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w, h), Image.ANTIALIAS)

        crop = self.annos[image_name]
        crop = np.array(crop).reshape(-1,4).astype(np.float32)
        if self.data_augment:
            if random.uniform(0, 1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                temp_x1 = crop[:, 0].copy()
                crop[:, 0] = im_width - crop[:, 2]
                crop[:, 2] = im_width - temp_x1
            resized_image = self.PhotometricDistort(resized_image)
        im = self.image_transformer(resized_image)
        # debug
        # plt.subplot(1, 2, 1)
        # plt.imshow(resized_image)
        # plt.title('input image')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # x1,y1,x2,y2 = crop[0].astype(np.int32)
        # best_crop = np.asarray(resized_image)[y1:y2,x1:x2]
        # plt.imshow(best_crop)
        # plt.title('best crop')
        # plt.axis('off')
        # plt.show()
        return im, crop, im_width, im_height, image_file

class FLMSDataset(Dataset):
    def __init__(self, split='test', keep_aspect_ratio=False):
        self.keep_aspect = keep_aspect_ratio
        self.data_dir = cfg.FLMS_dir
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, 'image')
        assert os.path.exists(self.image_dir), self.image_dir
        self.annos = self.parse_annotations()
        self.image_list = list(self.annos.keys())
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])

    def parse_annotations(self):
        image_crops_file = os.path.join(self.data_dir, '500_image_dataset.mat')
        assert os.path.exists(image_crops_file), image_crops_file
        import scipy.io as scio
        image_crops = dict()
        anno = scio.loadmat(image_crops_file)
        for i in range(anno['img_gt'].shape[0]):
            image_name = anno['img_gt'][i, 0][0][0]
            gt_crops = anno['img_gt'][i, 0][1]
            gt_crops = gt_crops[:, [1, 0, 3, 2]]
            keep_index = np.where((gt_crops < 0).sum(1) == 0)
            gt_crops = gt_crops[keep_index].tolist()
            image_crops[image_name] = gt_crops
        print('{} images'.format(len(image_crops)))
        return image_crops

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        im = self.image_transformer(resized_image)
        crop = self.annos[image_name]
        crop = np.array(crop).reshape(-1,4).astype(np.float32)
        return im, crop, im_width, im_height, image_file

if __name__ == '__main__':
    fcdb_testset = FCDBDataset(split='train')
    dataloader = DataLoader(fcdb_testset, batch_size=4, num_workers=1)
    for batch_idx, data in enumerate(dataloader):
        im, crop, im_width, im_height, image_file  = data
        print(crop.reshape(-1,4), im_width, im_height)
        # print(im.shape, crop.shape, im_width.shape, im_height.shape)

    # FLMS_testset = FLMSDataset()
    # print('FLMS testset has {} images'.format(len(FLMS_testset)))
    # dataloader = DataLoader(FLMS_testset, batch_size=1, num_workers=4)
    # for batch_idx, data in enumerate(dataloader):
    #     im, crop, w, h, file = data
    #     print(im.shape, crop.shape, w.shape, h.shape