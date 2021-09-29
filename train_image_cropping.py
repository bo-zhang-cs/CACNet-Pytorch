import os
import sys
import numpy as np
from tensorboardX import SummaryWriter
import torch
import time
import datetime
import csv
import shutil
import random
import torch.utils.data as data
import cv2

from KUPCP_dataset import CompositionDataset, composition_cls
from Cropping_dataset import FCDBDataset
from config_cropping import cfg
from test import evaluate_on_FCDB_and_FLMS, evaluate_composition_classification
from CACNet import CACNet
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:{}'.format(cfg.gpu_id))
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_dataloader():
    crop_dataset = FCDBDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
    crop_loader = torch.utils.data.DataLoader(crop_dataset, batch_size=cfg.crop_batch_size,
                                             shuffle=True, num_workers=cfg.num_workers,
                                             drop_last=False, worker_init_fn=random.seed(SEED))
    print('FCDB training set has {} samples, batch_size={}, total {} batches'.format(
        len(crop_dataset), cfg.crop_batch_size, len(crop_loader)))
    com_dataset = CompositionDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
    com_loader  = torch.utils.data.DataLoader(com_dataset, batch_size=cfg.com_batch_size,
                                              shuffle=True, num_workers=cfg.num_workers,
                                              drop_last=False, worker_init_fn=random.seed(SEED))
    print('Composition training set has {} samples, batch_size={}, total {} batches'.format(
        len(com_dataset), cfg.com_batch_size, len(com_loader)))
    return crop_loader, com_loader

class Trainer:
    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.iters = 0
        self.max_epoch = cfg.max_epoch
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.optimizer, self.lr_scheduler = self.get_optimizer()
        self.crop_loader, self.com_loader = create_dataloader()
        self.com_dataiter = iter(self.com_loader)
        self.eval_results = []
        self.best_results = {'FCDB_iou': 0., 'FCDB_disp':  1.,
                             'FLMS_iou': 0., 'FLMS_disp':  1.}
        self.crop_criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.com_criterion  = torch.nn.CrossEntropyLoss(reduction='mean')
        self.visual_path    = os.path.join(cfg.exp_path, 'visualized_results')

    def get_optimizer(self):
        # params = [
        #     {'params': self.model.cropping_module.parameters(), 'lr': cfg.lr},
        #     {'params': self.model.composition_module.parameters(), 'lr': 1e-4},
        #     {'params': self.model.backbone.parameters(), 'lr': 1e-4}
        # ]
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=cfg.lr_decay_epoch, gamma=cfg.lr_decay
        )
        return optim, lr_scheduler

    def run(self):
        print(("========  Begin Training  ========="))
        for epoch in range(1,self.max_epoch+1):
            self.epoch = epoch
            self.train()
            if epoch % cfg.eval_freq == 0:
                self.eval()
                self.record_eval_results()
            self.lr_scheduler.step()

    def fetch_com_batch(self):
        try:
            batch_data = next(self.com_dataiter)
        except:
            self.com_dataiter = iter(self.com_loader)
            batch_data = next(self.com_dataiter)
        return batch_data

    def visualize_com_prediction(self, image_path, logits, kcm, category):
        os.makedirs(self.visual_path, exist_ok=True)
        _, predicted = torch.max(logits.data, 1)
        # print('Composition prediction', predicted)
        # print('Ground-truth composition', category)
        label = composition_cls[predicted[0].item()]
        gt_label = composition_cls[category[0].item()]
        im = cv2.imread(image_path[0])
        height, width, _ = im.shape
        dst = im.copy()
        gt_ss = 'gt:{}'.format(gt_label)
        dst = cv2.putText(dst, gt_ss, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        pr_ss = 'pre:{}'.format(label)
        dst = cv2.putText(dst, pr_ss, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # h,w,1
        kcm = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        norm_kcm = cv2.normalize(kcm, None, 0, 255, cv2.NORM_MINMAX)
        norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
        heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
        heat_im = cv2.resize(heat_im, (width, height))
        fuse_im = cv2.addWeighted(im, 0.2, heat_im, 0.8, 0)
        fuse_im = np.concatenate([dst, fuse_im], axis=1)
        cv2.imwrite(os.path.join(self.visual_path, 'com-{}.jpg'.format(self.iters)), fuse_im)

    def train(self):
        self.model.train()
        start = time.time()
        batch_crop_loss = 0
        batch_com_loss  = 0
        total_batch = len(self.crop_loader)
        batch_com_cnt = 0
        batch_com_correct = 0

        for batch_idx, batch_data in enumerate(self.crop_loader):
            # ================ training on cropping task ===============
            self.iters += 1
            im = batch_data[0].to(device)
            crop  = batch_data[1].to(device).squeeze(1)
            width = batch_data[2].to(device)
            height = batch_data[3].to(device)
            image_path = batch_data[4]

            crop[:,0::2] = crop[:, 0::2] / width[:,None] *  im.shape[-1]
            crop[:,1::2] = crop[:, 1::2] / height[:,None] *  im.shape[-2]
            self.model.composition_module.eval() # freeze the BN params of classification branch
            logits, kcm, pre_crop = self.model(im, only_classify=False)
            crop_loss = self.crop_criterion(pre_crop, crop)
            # print('gt {} v.s. pre {}'.format(crop.shape, pre_crop.shape))
            # print(crop, pre_crop)
            batch_crop_loss += crop_loss.item()
            crop_loss *= cfg.crop_loss_factor
            self.optimizer.zero_grad()
            crop_loss.backward()

            # ================== training on composition classification task ==============
            batch_data = self.fetch_com_batch()
            im = batch_data[0].to(device)
            labels = batch_data[1].to(device).squeeze(1)
            image_path = batch_data[2]
            self.model.composition_module.train() # allow update BN classifier's params
            logits,kcm = self.model(im, only_classify=True)
            com_loss = self.com_criterion(logits, labels)
            batch_com_loss += com_loss.item()
            com_loss *= cfg.com_loss_factor
            com_loss.backward()
            self.optimizer.step()

            batch_com_cnt += labels.shape[0]
            _, predicted = torch.max(logits.data, 1)
            batch_com_correct += (predicted == labels).sum().item()

            # if self.iters % cfg.save_image_freq == 0:
            #     self.visualize_com_prediction(image_path, logits, kcm, labels)

            if batch_idx > 0 and batch_idx % cfg.display_freq == 0:
                avg_crop_loss = batch_crop_loss / (1 + batch_idx)
                avg_com_loss  = batch_com_loss  / (1 + batch_idx)
                avg_com_acc   = float(batch_com_correct) / batch_com_cnt
                batch_com_cnt = 0
                batch_com_correct = 0

                cur_lr   = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/crop_loss', avg_crop_loss, self.iters)
                self.writer.add_scalar('train/com_loss',  avg_com_loss,  self.iters)
                self.writer.add_scalar('train/lr',   cur_lr,   self.iters)

                time_per_batch = (time.time() - start) / (batch_idx + 1.)
                last_batches = (self.max_epoch - self.epoch - 1) * total_batch + (total_batch - batch_idx - 1)
                last_time = int(last_batches * time_per_batch)
                time_str = str(datetime.timedelta(seconds=last_time))

                print('=== epoch:{}/{}, step:{}/{} | Crop_Loss:{:.4f} | Com_Loss:{:.4f} | Com_acc:{:.2%} | lr:{:.6f} | estimated last time:{} ==='.format(
                    self.epoch, self.max_epoch, batch_idx, total_batch, avg_crop_loss, avg_com_loss, avg_com_acc, cur_lr, time_str
                ))

    def eval(self):
        FCDB_iou, FCDB_disp   = evaluate_on_FCDB_and_FLMS(self.model, dataset='FCDB')
        FLMS_iou, FLMS_disp   = evaluate_on_FCDB_and_FLMS(self.model, dataset='FLMS')
        com_acc               = evaluate_composition_classification(self.model)
        self.writer.add_scalar('test/accuracy', com_acc, self.epoch)
        self.eval_results.append([self.epoch, FCDB_iou, FCDB_disp, FLMS_iou, FLMS_disp])
        epoch_result = {'FCDB_iou': FCDB_iou, 'FCDB_disp': FCDB_disp,
                        'FLMS_iou': FLMS_iou, 'FLMS_disp': FLMS_disp}
        for m in self.best_results.keys():
            update = False
            if ('disp' not in m) and (epoch_result[m] > self.best_results[m]):
                update = True
            elif ('disp' in m) and (epoch_result[m] < self.best_results[m]):
                update = True
            if update:
                self.best_results[m] = epoch_result[m]
                checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best-{}.pth'.format(m))
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Update best {} model, best {}={:.4f}'.format(m, m, self.best_results[m]))
            if m in ['FCDB_iou', 'FLMS_iou']:
                self.writer.add_scalar('test/{}'.format(m), epoch_result[m], self.epoch)
                self.writer.add_scalar('test/best-{}'.format(m), self.best_results[m], self.epoch)

        if self.epoch > 0 and self.epoch % cfg.save_freq == 0:
            checkpoint_path = os.path.join(cfg.checkpoint_dir, 'epoch-{}.pth'.format(self.epoch))
            torch.save(self.model.state_dict(), checkpoint_path)

    def record_eval_results(self):
        csv_path = os.path.join(cfg.exp_path, '..', '{}.csv'.format(cfg.exp_name))
        header = ['epoch', 'FCDB_iou', 'FCDB_disp', 'FLMS_iou', 'FLMS_disp']
        rows = [header]
        for i in range(len(self.eval_results)):
            new_results = []
            for j in range(len(self.eval_results[i])):
                if header[j] == 'epoch':
                    new_results.append(self.eval_results[i][j])
                else:
                    new_results.append(round(self.eval_results[i][j], 4))
            self.eval_results[i] = new_results
        rows += self.eval_results
        metrics = [[] for i in header]
        for result in self.eval_results:
            for i, r in enumerate(result):
                metrics[i].append(r)
        for name, m in zip(header, metrics):
            if name == 'epoch':
                continue
            index = m.index(max(m))
            if 'disp' in name:
                index = m.index(min(m))
            title = 'best {}(epoch-{})'.format(name, index)
            row = [l[index] for l in metrics]
            row[0] = title
            rows.append(row)
        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ', csv_path)

if __name__ == '__main__':
    cfg.create_path()
    for file in os.listdir('./'):
        if file.endswith('.py'):
            shutil.copy(file, cfg.exp_path)
            print('backup', file)
    net = CACNet(loadweights=True).to(device)
    trainer = Trainer(net)
    trainer.run()