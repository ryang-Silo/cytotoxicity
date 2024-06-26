import os
import sys
import numpy as np 
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from cyto_loader_local import cytoDataset, Normaliztion, ToTensor, Rescale, RandomHorizontalFlip, RandomCrop, CenterCrop #, worker_init_fn #, RandomHorizontalFlip2CHNL, Rescale2CHNL, RandomCrop2CHNL,Rescale2CHNL, CenterCrop2CHNL 
from i3d import InceptionI3d
import cv2
import matplotlib.pyplot as plt 
import random
from azureml.fsspec import AzureMachineLearningFileSystem
import joblib
from azureml.core import Run
from azureml.core import Workspace
from azureml.core import Dataset


channel_minmax = {1: [179, 65535],
                2: [345, 65535],
                3: [0, 38383],
                4: [144, 10705],
                5: [3621, 65535]}
curdir = os.path.join(os.getcwd(), 'workspace/cytotoxicity')
print(curdir)
root_dir = os.path.join(curdir, 'data/mnt2_cytoData_v2') 
outmodel_dir = os.path.join(curdir, 'models')
os.makedirs(outmodel_dir, exist_ok=True)

# # mound dataset
# ws = Workspace.from_config()
# print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
# ws.set_default_datastore('workspaceblobcompound')
# ds = ws.get_default_datastore()
# print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)
# ds_paths = [(ds, 'reference_dataset_compounds_v2/')]
# dataset = Dataset.File.from_files(path = ds_paths)
# mount_context = dataset.mount(root_dir)
# #dataset.download(target_path='.', overwrite=True)
# mount_context.start()

# print(len(os.listdir(root_dir)))
# print (root_dir)



# uri = 'azureml://subscriptions/25130c3f-778b-4637-bfb8-3b1b885b45e7/resourcegroups/rg-silo-dev-003/workspaces/aml-silo-dev-003/datastores/workspaceblobcompound/paths'
# create the filesystem
# fs = AzureMachineLearningFileSystem(uri)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 

def show_cam_on_image(img, mask, outdir, val=False):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    if val:
        cv2.imwrite(outdir+'/'+args.log + "cam_val.jpg", np.uint8(255 * cam))
    else:
        cv2.imwrite(outdir+'/'+args.log + "cam.jpg", np.uint8(255 * cam))

# feature  -->   [ batch, channel, temporal, height, width ]
def FeatureMap2Heatmap( img, zfeat, index, outdir, attmap, val):
    att = attmap.cpu()
    attheatmap = torch.zeros(att.size(1), att.size(2))
    for i in range(att.size(0)):
        attheatmap += torch.pow(att[i,:,:],2).view(att.size(1),att.size(2))

    attheatmap = attheatmap.data.numpy()

    cam = np.maximum(attheatmap, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    img = img.permute(1,2,0).cpu()
    show_cam_on_image(img, cam, outdir, val)

    if not val:
        name = 'train'
    else:
        name = 'val'
 

    feature_first_frame = zfeat[0,:,4,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(outdir+'/'+args.log + '_z_feat_' + name  + str(index) + '.jpg')
    plt.close()
    
    

    return heatmap

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * float(n)
        self.count += n
        self.avg = self.sum / self.count


def smart_resume(ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True):
    """Resumes training from a checkpoint, updating optimizer, ema, and epochs, with optional resume verification."""
    best_top1 = 0.0
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
        best_top1 = ckpt["best_top1"]
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        print(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")
    if epochs < start_epoch:
        print(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt["epoch"]  # finetune additional epochs
    return best_top1, start_epoch, epochs

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return model, optimizer, checkpoint['epoch']

def train_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Use {device} for training")
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))  
    #     torch.cuda.set_device(args.gpu)
    # else:
    #     print("Use CPU")

    set_seed(10659)

    print_freq = args.print_freq    
    fold = args.fold
    num_classes = 2
    
    for ik in (fold, fold+1): #range(fold): #(fold, fold+1):
        index = ik + 1
        print("cross-validation: ",index)
        outdir = os.path.join(outmodel_dir, args.log)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        log_file = open(outdir+'/'+ args.log+'_log' + str(index) + '.txt', 'w')
        
        trainval_list = os.path.join(curdir, 'data/train_test_folds/train_fold_'+'%d' % (index)+'_info.txt')
        test_list = os.path.join(curdir, 'data/train_test_folds/test_fold_'+'%d' % (index)+'_info.txt') 
       

        log_file.write('cross-valid : %d'%(index))
        log_file.write("\n")
        log_file.flush()
        


        finetune = args.finetune
        if finetune==True:
            print('finetune!\n')
            log_file.write('finetune!\n')
            log_file.flush()

            model = InceptionI3d(2, num_classes=2)
            param = list(model.parameters())
            optimizer = optim.Adam(param, lr=args.lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  
            if args.resume:
                checkpoint_path = '/home/azureuser/cloudfiles/code/workspace/cytotoxicity/models/cyto_5fold/*.pt'
                model, optimizer, scheduler, start_epoch = load_ckp(checkpoint_path, model, optimizer, scheduler)
        else:
            print('train from scratch!\n')
            log_file.write('train from scratch!\n')
            log_file.flush()
            model = InceptionI3d(2, in_channels=1)
            param = list(model.parameters())
            optimizer = optim.Adam(param, lr=args.lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  
            start_epoch = 0     
        
        model = model.to(device) #.cuda()
        criterion_CE = nn.CrossEntropyLoss() 
        
        Epochs_results = np.empty([args.epochs, 1], dtype=float) * np.nan
        best_top1 = 0.0
        
        # train
        for epoch in range(start_epoch, args.epochs): 
            
            #only update lr for i3d, not for attention
#             if (epoch + 1) % args.step_size == 0:
#                 optimizer.param_groups[0]['lr'] *= args.gamma
#                 #optimizer.param_groups[1]['lr'] *= 0.1

            running_loss_total = 0.0
            train_loss = AverageMeter()
            ttop1 = AverageMeter()
            model.train()
    
            # load random 16-frame clip data every epoch
            cyto_train = cytoDataset(cv_info=trainval_list, root_dir=root_dir, transform=transforms.Compose([Normaliztion(), Rescale((384,384)),RandomCrop((256,256)),RandomHorizontalFlip(), ToTensor()]))
            dataloader_train = DataLoader(cyto_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

            for i, sample_batched in enumerate(dataloader_train):
                # print(i, sample_batched['video'].shape, sample_batched['label'].shape)
                inputs, label = sample_batched['video'].to(device), sample_batched['label'].to(device)
                # # repeat to 3 channels
                # inputs = inputs.repeat(1, 3, 1, 1, 1)
                optimizer.zero_grad()
                
                predict, share_feat = model(inputs)
                # print(predict.shape) #[2, 2, 2, 2, 2])
                # print(label.shape)
                # print(share_feat.shape) #[2, 1024, 10, 8, 8]
                
                loss1 = criterion_CE(predict, label)
                # cls_loss = F.binary_cross_entropy_with_logits(torch.max(predict, dim=2)[0], torch.max(label, dim=2)[0])
                
                loss = loss1
                
                prec1 = accuracy(predict.squeeze(), label, topk=(1,))
                ttop1.update(prec1[0], inputs.size(0)) 
                      
                loss.backward()
                optimizer.step()
                
                
                running_loss_total += loss.item()    
                train_loss.update(loss.item(), 1)  

                if (i+1) % print_freq == 0:    
                    batches_loss_total = running_loss_total / print_freq            

                    running_loss_total = 0.0
                    # visualization #[1, 1, 80, 256, 256]
                    # visual = FeatureMap2Heatmap(inputs[0,:,31,:,:], share_feat, index, outdir, share_feat[0,:,4,:,:], val=False)
                    
                    # log written
                    cur_lr = optimizer.param_groups[0]['lr']
                    #cur_lr_nl = optimizer.param_groups[1]['lr']
                    print('\n epoch:%d, mini-batch:%3d, lr=%f, loss= %.4f, top1= %.4f \n' % (epoch + 1, i + 1, cur_lr, batches_loss_total, ttop1.avg))
                    log_file.write('epoch:%d, mini-batch:%3d, lr=%f, loss= %.4f, top1= %.4f' % (epoch + 1, i + 1, cur_lr, batches_loss_total, ttop1.avg))
                    log_file.write("\n")
                    log_file.flush()

                    # x = np.arange(0, frames)
                    # y1 = 2*rPPG[0].cpu().data.numpy()
                    # y2 = ecg[0].cpu().data.numpy()  # +1 all positive
                    # fig = plt.figure() 
                    # ax = fig.add_subplot(111)
                    # predicted, = ax.plot(x, y1, color='red', label='predicted')
                    # label, = ax.plot(x, y2, color='blue', label='label')
                    # plt.savefig(outdir+'/'+args.log + str(index) + 'train_rPPG.jpg')
                    # plt.close()
                # if i == 1:
                #     break

            scheduler.step()
            log_file.write('epoch:%d, lr=%f, train_epoch_loss= %.4f' % (epoch + 1, cur_lr, train_loss.avg))
            log_file.write("\n")
            log_file.flush()
                    
            #### validation/test  
            model.eval()
            
            running_loss =0.0                
            top1 = AverageMeter()
            count_5 = 0

            cyto_test = cytoDataset(cv_info=test_list, root_dir=root_dir, transform=transforms.Compose([Normaliztion(), Rescale((384,384)),RandomCrop((256,256)), ToTensor()]))
            dataloader_test = DataLoader(cyto_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

            with torch.no_grad():               
                for i, sample_batched in enumerate(dataloader_test):
                    print('Testing...')
                    print(i, sample_batched['video'].shape, sample_batched['label'].shape)
                    count_5 += 1
                              
                    inputs, label = sample_batched['video'].to(device), sample_batched['label'].to(device)                      
                    # inputs = inputs.repeat(1, 3, 1, 1, 1)
                    predict, share_feat = model(inputs)
                                        
                    loss1 = criterion_CE(predict, label)
                    loss = loss1 
                    prec1 = accuracy(predict.squeeze(-1), label, topk=(1,))
                    top1.update(prec1[0], inputs.size(0))          
                    
                    running_loss += loss.item()
                    
                    # if (i+1) % print_freq == 0: 
                        # visual = FeatureMap2Heatmap(inputs[0,:,31,:,:], share_feat, index, outdir, share_feat[0,:,4,:,:], val=True)


                    # if i == 1:
                    #     break

                val_loss =running_loss / (count_5)
                      
                print('\n epoch:%d, test--> pain_loss= %.4f,  top1= %.4f \n' % (epoch + 1, val_loss, top1.avg))
                log_file.write('\n epoch:%d, test--> top1_Acc= %.4f, pain_loss= %.4f \n' % (epoch + 1, top1.avg, val_loss))
                log_file.write('\n')
                log_file.flush()
                
                Epochs_results[epoch, 0] = top1.avg  
                                  
            if top1.avg >= best_top1:
                best_top1 = top1.avg
                ckpt = {
                'fold': index,
                'best_top1': best_top1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                }
                torch.save(ckpt, outdir+'/'+args.log+'_%d.pt' % (index)) 

        
        best_Acc = np.max(Epochs_results)
        epind = np.argmax(Epochs_results)
        print('Best epoch results is ep:{}, top1:{}'.format(epind, best_Acc))
        
        log_file.write('\n Fold: %d,-----> best epoch results is ep:%d, top1= %.4f \n' % (index, epind+1, best_Acc))
        log_file.write('\n ----------------------------------------\n')
        log_file.flush() 
    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--step_size', type=int, default=10, help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.8, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--print_freq', type=int, default=1, help='how many batches display once')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--fold', type=int, default=4, help='fold')
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--log', type=str, default="cyto_5fold", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test(args)
