import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
import json
from KUPCP_dataset import CompositionDataset, composition_cls
from Cropping_dataset import FCDBDataset, FLMSDataset
from config_cropping import cfg
from CACNet import CACNet
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:{}'.format(cfg.gpu_id))


device = torch.device('cuda:{}'.format(cfg.gpu_id))
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    ''''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = gt_crop[gt_crop[:,0] >= 0]
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0])
    over_y1 = torch.maximum(gt_crop[:,1], pre_crop[:,1])
    over_x2 = torch.minimum(gt_crop[:,2], pre_crop[:,2])
    over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.maximum(zero_t, over_x2 - over_x1)
    over_h  = torch.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / im_w + \
              (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item()

def evaluate_on_FCDB_and_FLMS(model, dataset, save_results=False):
    model.eval()
    device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0

    if save_results:
        save_file = os.path.join(results_dir, dataset + '.json')
        crop_dir  = os.path.join(results_dir, dataset)
        os.makedirs(crop_dir, exist_ok=True)
        test_results = dict()

    print('=' * 5, f'Evaluating on {dataset}', '=' * 5)
    with torch.no_grad():
        if dataset == 'FCDB':
            test_set = [FCDBDataset]
        elif dataset == 'FLMS':
            test_set = [FLMSDataset]
        else:
            raise Exception('Undefined test set ', dataset)
        for dataset in test_set:
            test_dataset= dataset(split='test',
                                  keep_aspect_ratio=cfg.keep_aspect_ratio)
            test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=1,
                                                      shuffle=False, num_workers=cfg.num_workers,
                                                      drop_last=False)
            for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                im = batch_data[0].to(device)
                gt_crop = batch_data[1] # x1,y1,x2,y2
                width = batch_data[2].item()
                height = batch_data[3].item()
                image_file = batch_data[4][0]
                image_name = os.path.basename(image_file)

                logits,kcm,crop = model(im, only_classify=False)
                crop[:,0::2] = crop[:,0::2] / im.shape[-1] * width
                crop[:,1::2] = crop[:,1::2] / im.shape[-2] * height
                pred_crop = crop.detach().cpu()
                gt_crop = gt_crop.reshape(-1, 4)
                pred_crop[:,0::2] = torch.clip(pred_crop[:,0::2], min=0, max=width)
                pred_crop[:,1::2] = torch.clip(pred_crop[:,1::2], min=0, max=height)

                iou, disp = compute_iou_and_disp(gt_crop, pred_crop, width, height)
                if iou >= alpha:
                    alpha_cnt += 1
                accum_iou += iou
                accum_disp += disp
                cnt += 1

                if save_results:
                    best_crop = pred_crop[0].numpy().tolist()
                    best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2
                    test_results[image_name] = best_crop

                    # save the best crop
                    source_img = cv2.imread(image_file)
                    croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
                    cv2.imwrite(os.path.join(crop_dir, image_name), croped_img)
    if save_results:
        with open(save_file, 'w') as f:
            json.dump(test_results, f)
    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp

def visualize_com_prediction(image_path, logits, kcm, category, save_folder):
    _, predicted = torch.max(logits.data, 1)
    # print('Composition prediction', predicted)
    # print('Ground-truth composition', category)
    label = composition_cls[predicted[0].item()]
    gt_label = [composition_cls[c] for c in category[0].numpy().tolist()]
    im = cv2.imread(image_path[0])
    height,width,_ = im.shape
    dst = im.copy()
    gt_ss = 'gt:{}'.format(gt_label)
    dst = cv2.putText(dst, gt_ss, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    pr_ss = 'predict:{}'.format(label)
    dst = cv2.putText(dst, pr_ss, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    # h,w,1
    kcm = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
    # norm_kcm = np.zeros((height, width, 1))
    norm_kcm = cv2.normalize(kcm, None, 0, 255, cv2.NORM_MINMAX)
    norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
    heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
    # heat_im = cv2.cvtColor(heat_im, cv2.COLOR_BGR2RGB)
    heat_im = cv2.resize(heat_im, (width, height))
    fuse_im = cv2.addWeighted(im, 0.2, heat_im, 0.8, 0)
    fuse_im = np.concatenate([dst, fuse_im], axis=1)
    cv2.imwrite(os.path.join(save_folder, os.path.basename(image_path[0])), fuse_im)

def evaluate_composition_classification(model):
    model.eval()
    device = next(model.parameters()).device
    print('=' * 5, 'Evaluating on Composition Classification Dataset', '=' * 5)
    total = 0
    correct = 0
    cls_cnt = [0 for i in range(9)]
    cls_correct = [0 for i in range(9)]

    with torch.no_grad():
        test_dataset = CompositionDataset(split='test', keep_aspect_ratio=cfg.keep_aspect_ratio)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False, num_workers=cfg.num_workers,
                                                  drop_last=False)
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(device)
            labels = batch_data[1]
            image_path = batch_data[2]

            logits,kcm = model(im, only_classify=True)
            logits   = logits.cpu()
            _,predicted = torch.max(logits.data,1)
            total += labels.shape[0]
            pr = predicted[0].item()
            gt = labels[0].numpy().tolist()

            if pr in gt:
                correct += 1
                cls_cnt[pr] += 1
                cls_correct[pr] += 1
            else:
                cls_cnt[gt[0]] += 1
    acc = float(correct) / total
    print('Test on {} images, {} Correct, Acc {:.2%}'.format(total, correct, acc))
    for i in range(len(cls_cnt)):
        print('{}: total {} images, {} correct, Acc {:.2%}'.format(
            composition_cls[i], cls_cnt[i], cls_correct[i], float(cls_correct[i]) / cls_cnt[i]))
    return acc


if __name__ == '__main__':
    weight_file = "./pretrained_model/best-FLMS_iou.pth"
    model = CACNet(loadweights=False)
    model.load_state_dict(torch.load(weight_file))
    model = model.to(device).eval()
    evaluate_on_FCDB_and_FLMS(model, dataset='FCDB', save_results=True)
    evaluate_on_FCDB_and_FLMS(model, dataset='FLMS', save_results=True)
