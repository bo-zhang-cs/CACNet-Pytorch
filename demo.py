import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from config_cropping import cfg
from CACNet import CACNet
import warnings
from torchvision import transforms
from PIL import Image, ImageOps
import random
import time

warnings.filterwarnings("ignore")

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
# Normalize the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])
    

def rescale_bbox(bbox, ratio_w, ratio_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, 0] = np.floor(bbox[:, 0] * ratio_w)
    bbox[:, 1] = np.floor(bbox[:, 1] * ratio_h)
    bbox[:, 2] = np.ceil(bbox[:, 2] * ratio_w)
    bbox[:, 3] = np.ceil(bbox[:, 3] * ratio_h)
    return bbox.astype(np.float32)

def parse_arguments():
    parser = argparse.ArgumentParser(description='CACNet inference')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to image or folder of images')
    parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU ID')
    parser.add_argument('-w', '--weight', type=str, required=False, 
                        default="./pretrained_model/best-FLMS_iou.pth", 
                        help='Path to model weights')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory for results')
    return parser.parse_args()

def preprocess_image(image, keep_aspect_ratio):
    im_width, im_height = image.size
    if keep_aspect_ratio:
        scale = float(cfg.image_size[0]) / min(im_height, im_width)
        h = round(im_height * scale / 32.0) * 32
        w = round(im_width * scale / 32.0) * 32
    else:
        h = cfg.image_size[1]
        w = cfg.image_size[0]

    resized_image = image.resize((w, h), Image.ANTIALIAS)

    if random.uniform(0, 1) > 0.5:
        resized_image = ImageOps.mirror(resized_image)

    return transform(resized_image).unsqueeze(0).to(device)

def evaluate_on_images(model, input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    keep_aspect_ratio = cfg.keep_aspect_ratio

    if os.path.isfile(input_path):
        image_files = [input_path]
    else:
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('jpg', 'png'))]

    start = time.time()
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        width, height = image.size
        im = preprocess_image(image, keep_aspect_ratio)
        # model inference
        logits,kcm,crop = model(im, only_classify=False)
        # convert the crop to the original image size
        crop[:,0::2] = crop[:,0::2] / im.shape[-1] * width
        crop[:,1::2] = crop[:,1::2] / im.shape[-2] * height
        pred_crop = crop.detach().cpu()
        pred_crop[:,0::2] = torch.clip(pred_crop[:,0::2], min=0, max=width)
        pred_crop[:,1::2] = torch.clip(pred_crop[:,1::2], min=0, max=height)
        best_crop = pred_crop[0].numpy().tolist()
        best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2
        # save the best crop
        cropped_img = image.crop(best_crop)
        res_path = os.path.join(output_dir, os.path.basename(image_file))
        cropped_img.save(res_path)
        cost_time = time.time() - start
        print('cost time: {:.2f}s, cropped image saved to {}'.format(cost_time, res_path))

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu') 
    model  = CACNet(loadweights=False)
    model.load_state_dict(torch.load(args.weight))
    model = model.to(device).eval()
    evaluate_on_images(model, args.input, args.output)
