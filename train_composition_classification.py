import os
import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import shutil
import pickle
import torch.utils.data as data
import torch.optim as optim

from KUPCP_dataset import CompositionDataset, composition_cls
from config_classification import cfg
from test import evaluate_composition_classification
from CACNet import ComClassifier

cfg.create_path()
device = torch.device('cuda:{}'.format(cfg.gpu_id))
writer = SummaryWriter(log_dir=cfg.log_dir)

for file in ['train_composition_classification.py', 'KUPCP_dataset.py', 'CACNet.py', 'config_classification.py']:
    if file.endswith('.py'):
        shutil.copy(file, cfg.exp_path)
        print('backup', file)

batch_size = cfg.com_batch_size
com_dataset = CompositionDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
trainloader  = torch.utils.data.DataLoader(com_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=cfg.num_workers,
                                          drop_last=False)
print('Composition training set has {} samples, batch_size={}, total {} batches'.format(
    len(com_dataset), batch_size, len(trainloader)))

net = ComClassifier(loadweights=True)
net = net.train().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40], gamma=0.1)

display_step = 10
total_batch = 0
best_acc = 0.

for epoch in range(80):  # loop over the dataset multiple times
    running_loss = 0.0
    net.train()
    batch_total = 0
    batch_correct = 0

    for i, batch_data in enumerate(trainloader, 0):
        total_batch += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs = batch_data[0].to(device)
        labels = batch_data[1].to(device)

        if labels.dim() == 2:
            labels = labels.squeeze(1)

        image_path = batch_data[2]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs,_ = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _,pred = torch.max(outputs,dim=1)
        batch_total += labels.shape[0]
        batch_correct += (pred == labels).sum().item()
        # print statistics
        running_loss += loss.item()

        if i > 0 and i % 20 == 0:    # print every 2000 mini-batches
            running_loss /= display_step
            accuracy = float(batch_correct) / batch_total
            writer.add_scalar('train/loss', running_loss, total_batch)
            writer.add_scalar('train/accuracy', accuracy, total_batch)
            print('{},{} | loss: {:.6f} | acc: {:.2%} | lr: {:.6f}'.format(
                epoch + 1, i + 1, running_loss, accuracy, optimizer.param_groups[0]['lr']))
            running_loss = 0.0
            batch_correct = 0
            batch_total  = 0
    if epoch % 2 == 0:
        acc = evaluate_composition_classification(net)
        writer.add_scalar('test/accuracy', acc, epoch)
        if acc > best_acc:
            best_acc = acc
            checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best-acc.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print('='*5, 'update best checkpoint', '='*5)
    scheduler.step()
print('Finished Training')