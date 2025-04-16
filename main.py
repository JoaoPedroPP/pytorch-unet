import copy
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
from torchvision import transforms
# from torchvision.io import write_png

from UNet.UNet import UNet

def dice(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    
    pred[pred > 0] = 1
    pred[pred <= 0] = 0

    intersection = (pred * target).sum()

    dice = (2. * (intersection + smooth)) / (pred.sum() + target.sum() + smooth)

    return dice

def dice_original(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics):
    pred = F.sigmoid(pred)

    dice_coef = dice_original(pred, target)
    loss = 1 - dice_coef
    bce = F.binary_cross_entropy_with_logits(pred, target)
    # bce = F.binary_cross_entropy(pred, target)
    # loss = bce

    metrics['dice'] += dice_coef.data.cpu().numpy()
    metrics['loss'] += loss.data.cpu().numpy()
    metrics['bce'] += bce.data.cpu().numpy()

    return loss

def calc_loss_original(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = F.sigmoid(pred)
    dice_loss = 1 - dice(pred, target)

    loss = bce * bce_weight + dice_loss * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def get_data_loaders(dataset_path):
    # use the same transformations for train/val in this example
    raw = list(filter(lambda l: not l.endswith('edge_mask.png'), os.listdir(dataset_path)))
    input_imgs = list(filter(lambda l: not l.endswith('mask.png'), raw))
    mask_imgs = list(filter(lambda l: l.endswith('mask.png'), raw))

    input_imgs.sort()
    mask_imgs.sort()

    input_imgs_paths = list(map(lambda p: os.path.join(dataset_path, p), input_imgs))
    mask_imgs_paths = list(map(lambda p: os.path.join(dataset_path, p), mask_imgs))

    train_inputs, validation_inputs, train_masks, validation_masks = train_test_split(input_imgs_paths, mask_imgs_paths, test_size=0.2, shuffle=True)
    train_inputs, test_inputs, train_masks, test_masks = train_test_split(train_inputs, train_masks, test_size=0.3, shuffle=True)
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

def train_model(model, optimizer, scheduler, dataset_path, num_epochs=25):
    dataloaders = get_data_loaders(dataset_path=dataset_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10


    scaler = GradScaler(enabled=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, dataloaders

def main():
    print("Starting the model")

    num_classes = 1
    epochs = 5
    dataset_path = './support_images/dataset/raw'
    # dataset_path = '/run/media/jpolonip/JP2-HD/MestradoFiles/Dataset/raw2/train'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model, dataloaders = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs, dataset_path=dataset_path)

    model.eval()
    i = 1
    for inputs, masks in dataloaders['validation']:
        inputs = inputs.float().to(device)
        masks = masks.float().to(device)

        pred = model(inputs)
        pred = F.sigmoid(pred)
        # pred = F.softmax(pred)
        print(pred.shape)
        # print(pred)
        pred = pred.data.cpu().numpy()
        print(pred.min(), pred.max())
        # np.squeeze(pred, axis=0)
        print(pred.shape)

        pred = (pred[0] * 255).astype(np.uint8)
        # pred = (pred[:, :, 0] * 255.).astype(np.uint8)
        print(pred.shape)

        im = Image.fromarray(pred[0])
        print(pred.shape)
        # pred = transforms.functional.convert_image_dtype(pred, torch.uint8)
        im.save(f'./support_images/preds/pred_{i:05}.png')
        
        i += 1



main()
