import copy
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict
from Dataset.dataset import LIDCDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from UNet.UNet import UNet

def dice(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum()
    # print((pred*target).sum())

    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice
    # return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=1.0):
    dice_coef = dice(pred, target)

    loss = 1 - dice_coef

    metrics['dice'] += dice_coef.data.cpu().numpy()
    metrics['loss'] += loss.data.cpu().numpy()

    return loss

def calc_loss_original(pred, target, metrics, bce_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
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

def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    # train_set = SimDataset(100, transform = trans)
    # test_set = SimDataset(20, transform = trans)
    train_set = LIDCDataset('./support_images/dataset/raw')
    test_set = LIDCDataset('./support_images/dataset/raw')

    image_datasets = {
        'train': train_set, 'test': test_set
    }

    batch_size = 1

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders

def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

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
                    loss = calc_loss(outputs, masks, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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
    return model

def main():
    print("Starting the model")

    num_classes = 1
    epochs = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)

main()
