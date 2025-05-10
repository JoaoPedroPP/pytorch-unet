import copy
import csv
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict
from Dataset.dataset import LIDCDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.amp.grad_scaler import GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchvision import transforms

from UNet.UNet import UNet

def dice(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))

    return loss.mean()

def calc_loss(pred, target, metrics):
    pred = F.sigmoid(pred)

    dice_coef = dice(pred, target)
    loss = 1 - dice_coef
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred_flatten = pred.flatten()
    target_flatten = target.flatten()

    acc = BinaryAccuracy(threshold=0.5)
    acc.update(pred_flatten, target_flatten)

    prec = BinaryPrecision(threshold=0.5)
    prec.update(pred_flatten, target_flatten)

    # recall = BinaryRecall(threshold=0.5)
    # recall.update(pred_flatten, target_flatten)
    # recall.update(torch.tensor([0.2, 0.7]), torch.tensor([0, 1]))

    f1 = BinaryF1Score(threshold=0.5)
    f1.update(pred_flatten, target_flatten)

    metrics['dice'] += dice_coef.data.cpu().numpy()
    metrics['loss'] += loss.data.cpu().numpy()
    metrics['bce'] += bce.data.cpu().numpy()
    metrics['accuracy'] += acc.compute()
    metrics['precision'] += prec.compute()
    # metrics['recall'] += recall.compute()
    metrics['f1'] += f1.compute()

    return loss

def print_metrics(metrics, epoch_samples, phase, epoch):
    outputs = []
    csv_metrics = []
    csv_header = ['epoch']
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        csv_metrics.append(metrics[k]/epoch_samples)
        if phase == 'train' and epoch == 0:
            csv_header.append(f'{k}_train')
            csv_header.append(f'{k}_test')

    if phase == 'train' and epoch == 0:
        write_csv(csv_header, 'w')

    print("{}: {}".format(phase, ", ".join(outputs)))

    return csv_metrics

def write_csv(data, mode='a'):
    with open('logs.csv',mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([data])

def get_data_loaders(dataset_path, mask_path, seed=None):
    # use the same transformations for train/val in this example
    raw = list(filter(lambda l: not l.endswith('edge_mask.png'), os.listdir(mask_path)))
    input_imgs = list(filter(lambda l: not l.endswith('mask.png'), raw))
    # input_imgs = list(filter(lambda l: l.endswith('.npy'), os.listdir(dataset_path)))
    mask_imgs = list(filter(lambda l: l.endswith('mask.png'), raw))

    input_imgs.sort()
    mask_imgs.sort()

    input_imgs_paths = list(map(lambda p: os.path.join(dataset_path, p), input_imgs))
    mask_imgs_paths = list(map(lambda p: os.path.join(mask_path, p), mask_imgs))

    train_inputs, validation_inputs, train_masks, validation_masks = train_test_split(input_imgs_paths, mask_imgs_paths, test_size=0.2, shuffle=True, random_state=seed)
    train_inputs, test_inputs, train_masks, test_masks = train_test_split(train_inputs, train_masks, test_size=0.3, shuffle=True, random_state=seed)
    print(len(train_inputs), len(test_inputs), len(validation_inputs))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = LIDCDataset(train_inputs,train_masks)
    test_set = LIDCDataset(test_inputs,test_masks)
    validation_set = LIDCDataset(validation_inputs,validation_masks)

    image_datasets = {
        'train': train_set, 'test': test_set, 'validation': validation_set
    }

    batch_size = 1

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'validation': DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders

def train_model(model, optimizer, scheduler, dataset_path, num_epochs=25, mask_path=None, seed=None):
    if mask_path == None:
        dataloaders = get_data_loaders(dataset_path=dataset_path, mask_path=dataset_path, seed=seed)
    else:
        dataloaders = get_data_loaders(dataset_path=dataset_path, mask_path=mask_path, seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10


    scaler = GradScaler(enabled=False)

    csv_metrics = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        csv_metrics = []
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, masks in dataloaders[phase]:
                inputs = inputs.float().to(device)
                masks = masks.float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    loss = calc_loss(outputs, masks, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # statistics
                epoch_samples += inputs.size(0)

            if phase == 'train':
                csv_metrics = print_metrics(metrics, epoch_samples, phase, epoch)
            else:
                test_metrics = print_metrics(metrics, epoch_samples, phase, epoch)
                csv_metrics = np.concatenate(([epoch], csv_metrics, test_metrics))
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        write_csv(csv_metrics)

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, dataloaders

def main():
    print("Starting the model")

    num_classes = 1
    epochs = 200
    dataset_path = './support_images/dataset/raw'
    mask_path = './support_images/dataset/raw'
    # dataset_path = '/run/media/jpolonip/JP2-HD/MestradoFiles/Dataset/raw2/train'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model, dataloaders = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs, dataset_path=dataset_path, mask_path=mask_path, seed=123)

    model.eval()
    i = 1
    for inputs, masks in dataloaders['validation']:
        input = inputs.float().to(device)
        mask = masks.float().to(device)

        pred = model(input)
        pred = F.sigmoid(pred)
        pred = pred.data.cpu().numpy()

        pred = (pred[0] * 255).astype(np.uint8)
        input = input.numpy()
        mask = mask.numpy()

        out = Image.fromarray(pred[0])
        inn = Image.fromarray(input[0][0].astype(np.uint8), 'L')
        mak = Image.fromarray((mask[0][0] * 255).astype(np.uint8), 'L')

        out.save(f'./support_images/preds/{i:05}_pred.png')
        inn.save(f'./support_images/preds/{i:05}_input.png')
        mak.save(f'./support_images/preds/{i:05}_mask.png')
        
        i += 1



main()
