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
import numpy as np
import time
from utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    accuracy,
)
from logging import getLogger
import torch.backends.cudnn as cudnn

logger = getLogger()
WORKDIR = os.getcwd()
print(f"Working directory is: {WORKDIR}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

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

def main(args):
    logger = getLogger()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    print(f"Data path is mounted; {args.data_path}")
    print(f"Output path is {args.output_path}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
        
    outdir = os.path.join(args.save_model_path, args.logname)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    print(f"Experiment output path: {outdir}")

    ### Data initialize
    mount_datapath = args.data_path
    print(os.path.isdir(os.path.join(mount_datapath, "train_data")))
    print(len(os.listdir(os.path.join(mount_datapath, "train_data"))))

    # build data
    train_dataset = ImageFolder(os.path.join(mount_datapath, "train_data"))
    val_dataset = ImageFolder(os.path.join(mount_datapath, "validation_data"))


    jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = utils.Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                              eigvec=[[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )

    train_dataset.transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        jittering,
        lighting,
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    ### Build model
    model = models.resnet34(pretrained=True)
    num_features = model.fc.in_features
    # print(num_features) #2048
    # freeze
    # for param in model.parameters():
    #     param.requires_grad = False
    
    model.fc = nn.Linear(num_features, 2) 
    logger.info("Load pretrained model")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    # print(optimizer)
    # set scheduler
    if args.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )

    # criterion = CrossEntropyLoss()
    
    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": (0., 0.)}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True


    for epoch in range(start_epoch, args.epochs):
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        scores = train(model, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model)
        training_stats.update(scores + scores_val)

        scheduler.step()
        # save checkpoint
        if epoch % args.save_freq == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(outdir, "checkpoint.pth.tar"))
    logger.info("Training completed.\n"
                "Test accuracies: top-1 {acc1:.1f}, top-5 {acc5:.1f}".format(acc1=best_acc[0], acc5=best_acc[1]))

def train(model, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inp.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inp.size(), lam)
            inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.size()[-1] * inp.size()[-2]))

            output = model(inp)
            # compute cross entropy loss
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(inp)
            loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec top1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec top5 {top5.val:.3f} ({top5.avg:.3f})\t"
                "LR {lr_W}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return epoch, losses.avg, top1.avg.item(), top5.avg.item()
            
def validate_network(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(inp)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc[0]:
        best_acc = (top1.avg.item(), top5.avg.item())

    if 1:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Acc@5 {top5.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, top5=top5, acc=best_acc[0]))

    return losses.avg, top1.avg.item(), top5.avg.item()      

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == "__main__":
    
    # mount dataset
    from azureml.core import Dataset
    from azureml.core import Workspace
    ws = Workspace.from_config()
    print(ws.name, ws.location, ws.resource_group, sep='\t')

    # data_folder = './data/cytoData_v2'
    ws.set_default_datastore('workspaceblobcompound')
    ds = ws.get_default_datastore()
    print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)
    ds_paths = [(ds, 'reference_dataset_compounds_v2/')]
    dataset = Dataset.File.from_files(path = ds_paths)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=dataset.as_mount(), type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--output_path", default=WORKDIR+"/models", type=str)
    parser.add_argument("--logname", default="resnet34_singlechl_cutmix", type=str)
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--cutmix_prob", default=0, type=float, help='cutmix probability')
    parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
    
    
    args = parser.parse_args()
    
    main(args)