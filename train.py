import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models

import torch.optim as optim
from torch.nn import CrossEntropyLoss
import argparse
import os
import sys
import utils
import timm 
import time
from logging import getLogger
from utils import AverageMeter

WORKDIR = os.getcwd()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def train(args):
    # logger = getLogger()
    print(args.save_model_path)
    outdir = os.path.join(args.save_model_path, args.log)

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    print(outdir)
    log_file = open(outdir+'/'+ args.log+'_log' + '.txt', 'w')

    jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = utils.Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                              eigvec=[[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  # 将单通道转换为三通道
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # jittering,
        # lighting,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = ImageFolder(args.data, transform=transform)
    print(len(dataset))
    print(dataset.classes)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # pretrained_model_name = "resnet50"
    # model = timm.create_model(pretrained_model_name, pretrained=True)
    
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    # print(num_features) #2048
    # freeze
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(num_features, 2) 
    # print(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    # print(optimizer)
    criterion = CrossEntropyLoss()
    
    # train
    log_file.write("\n")
    log_file.flush()
    
    for epoch in range(args.epochs):
        end = time.time()
        data_time = AverageMeter()
        losses = AverageMeter()
        ttop1 = AverageMeter()
        model.train()
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            data_time.update(time.time() - end)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs[0].size(0))
            
            optimizer.zero_grad()
            
            prec1 = accuracy(outputs.squeeze(), labels, topk=(1,))
            ttop1.update(prec1[0], inputs.size(0)) 
            
            loss.backward()
            optimizer.step()
            
            print(
                "Epoch: [{0}]\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}\t"
                "Top1 {ttop1.avg:.4f}".format(
                    epoch,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                    ttop1=ttop1
                )

            )
            
            cur_lr = optimizer.param_groups[0]['lr']
            log_file.write('epoch:%d, lr=%f, loss= %.4f, top1= %.4f' % (epoch + 1, cur_lr, losses.avg, ttop1.avg))
            log_file.write("\n")
            log_file.flush()
            
        log_file.write('epoch:%d, lr=%f, train_epoch_loss= %.4f' % (epoch + 1, cur_lr, losses.avg))
        log_file.write("\n")
        log_file.flush()    
        
        if (epoch + 1) % args.save_freq == 0:
            model_name = f"resnet50_{epoch+1}.pt"
            # save_dict = {
            # 'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'epoch': epoch + 1,
            # 'loss': losses.avg,
            #  }
            
            torch.save(model.state_dict(), os.path.join(outdir, model_name))
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=WORKDIR+"/data/train_data_concentrateA_norm_m3", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--save_model_path", default=WORKDIR+"/models", type=str)
    parser.add_argument("--log", default="resnet50_merged3channels", type=str)
    
    
    args = parser.parse_args()
    
    train(args)