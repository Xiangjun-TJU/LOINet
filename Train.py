"""
 @Time    : 2021/7/6 14:36
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : infer.py
 @Function: Inference
 
"""
import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict

import os, torch, random, pickle, time
from argparse import ArgumentParser
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel.scatter_gather import gather
import load_data as ld
import datasets
import transforms
from skimage import io
import scipy.io as sio
from torch.nn import init
import matplotlib.image as mp
#from dual_path_unet import dual_path_unet as Dunet
from parallel import DataParallelModel, DataParallelCriterion
from utils1 import SalEval, AverageMeterSmooth, Logger, plot_training_process
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


parser = ArgumentParser()
parser.add_argument('--data_dir', default='', type=str, help='data directory')
parser.add_argument('--width', default=256, type=int, help='width of RGB image')
parser.add_argument('--height', default=256, type=int, help='height of RGB image')
parser.add_argument('--max_epochs', default=100, type=int, help='max number of epochs')
parser.add_argument('--num_workers', default=8, type=int, help='No. of parallel threads')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--warmup', default=0, type=int, help='lr warming up epoches')
parser.add_argument('--scheduler', default='cos', type=str, choices=['step', 'poly', 'cos'],
                    help='Lr scheduler (valid: step, poly, cos)')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for multi-step lr decay')
parser.add_argument('--milestones', default='[30, 60, 90]', type=str, help='milestones for multi-step lr decay')
parser.add_argument('--print_freq', default=50, type=int, help='frequency of printing training info')
parser.add_argument('--savedir', default='', type=str, help='Directory to save the results')
parser.add_argument('--resume', default= '', type=str, help='use this checkpoint to continue training')
parser.add_argument('--cached_data_file', default='duts_train.p', type=str, help='Cached file name')
parser.add_argument('--pretrained', default='', type=str, help='path for the ImageNet pretrained backbone model')
parser.add_argument('--seed', default=666, type=int, help='Random Seed')
parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to run on the GPU')
parser.add_argument('--model', default='lib.Network_Res2Net_GRA_NCD', type=str, help='which model to test')

args = parser.parse_args()

exec('from {} import Network as net'.format(args.model))

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def adjust_lr(optimizer, epoch):
    if epoch < args.warmup:
        lr = args.lr * (epoch + 1) / args.warmup
    else:
        if args.scheduler == 'cos':
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.max_epochs))
        elif args.scheduler == 'poly':
            lr = args.lr * (1 - epoch * 1.0 / args.max_epochs) ** 0.9
        elif args.scheduler == 'step':
            lr = args.lr
            for milestone in eval(args.milestones):
                if epoch >= milestone:
                    lr *= args.gamma
        else:
            raise ValueError('Unknown lr mode {}'.format(args.scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)
class W_BCE_IOU_loss(nn.Module):
    def __init__(self):
        super(W_BCE_IOU_loss, self).__init__()
    def forward(self, pred, mask):
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

        pred  = torch.sigmoid(pred)
        inter = ((pred*mask)*weit).sum(dim=(2,3))
        union = ((pred+mask)*weit).sum(dim=(2,3))
        wiou  = 1-(inter+1)/(union-inter+1)
        return (wbce+wiou).mean()

class BCE_IOU_loss(nn.Module):
    def __init__(self):
        super(BCE_IOU_loss, self).__init__()

    def forward(self, inputs, target):
       # target = target.float()
        loss = F.binary_cross_entropy(inputs, target) + iou_loss(inputs, target) 
        return loss.mean()
class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()
def mae_loss(pred, mask):
    return torch.mean(torch.abs(pred-mask))
def load_meas_matrix():
    WL = np.zeros((500,256,1))
    WR = np.zeros((620,256,1))
    d = sio.loadmat('/home/yxj001/sp0722/Flatnet_quantization-master/flatcam_prototype2_calibdata.mat') ##Initialize the weight matrices with transpose
    phil = np.zeros((500,256,1))
    phir = np.zeros((620,256,1))

    pl = sio.loadmat('/home/yxj001/sp0722/Flatnet_quantization-master/phil_toep_slope22.mat')
    pr = sio.loadmat('/home/yxj001/sp0722/Flatnet_quantization-master/phir_toep_slope22.mat')
    WL[:,:,0] = pl['phil'][:,:,0]
    WR[:,:,0] = pr['phir'][:,:,0] 
    #if args.init_matrix_type=='':

    WL = WL.astype('float32')   #  Pseudo inverse   WL
    WR = WR.astype('float32')   #  Pseudo inverse   WR  

    phil[:,:,0] = d['P1gb']
    phir[:,:,0] = d['Q1gb']
    phil = phil.astype('float32')   #  phiL
    phir = phir.astype('float32')   #  phiR

    return phil,phir,WL,WR

class initial_inversion2(nn.Module):
    def __init__(self):
        super(initial_inversion2, self).__init__()
    def forward(self,meas,WL,WR):
        x0=F.leaky_relu(torch.matmul(torch.matmul(meas[:,0,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x1=F.leaky_relu(torch.matmul(torch.matmul(meas[:,1,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x2=F.leaky_relu(torch.matmul(torch.matmul(meas[:,2,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        X_init=torch.cat((x0,x1,x2),3)
        X_init = X_init.permute(0,3,1,2)
        return X_init

# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, phil, phir, WL, WR, LayerNo):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        self.WL = nn.Parameter(torch.tensor(WL))
        self.WR = nn.Parameter(torch.tensor(WR))
        self.ini_inversion = initial_inversion2()
    def forward(self, meas):
        x = self.ini_inversion(meas,self.WL,self.WR)
        savename = 'phil_epoch' 
        np.save(savename, self.WL.detach().cpu().numpy())
        savename = 'phir_epoch'
        np.save(savename, self.WR.detach().cpu().numpy())   
        x_init = x
        
        return [x_init, x]


@torch.no_grad()
def val(val_loader, epoch):
    # switch to evaluation mode
    model.eval()
    Inversion.eval()
    salEvalVal = SalEval()

    total_batches = len(val_loader)
    for iter, (gt, target, meas, body, detail) in enumerate(val_loader):
        if args.gpu:
            gt = gt.cuda()
            target = target.cuda()
            meas = meas.cuda()
            body = body.cuda()
            detail = detail.cuda()
        gt = torch.autograd.Variable(gt)
        target = torch.autograd.Variable(target)
        meas = torch.autograd.Variable(meas)
        body = torch.autograd.Variable(body)
        detail = torch.autograd.Variable(detail)

        start_time = time.time()
        # run the mdoel
        x_init, x_final = Inversion(meas)
       # output_b,output_d,output_mask = model(x_final,1)
       # _, _, _, output_mask = model(img_var)
        output_p, output_f1, output_f2,  output_mask = model(x_final)
#        if epoch==100:
#            x0 = x_final[0, 0, :, :].squeeze().cpu().detach().numpy()
#            x0 = (x0 - np.min(x0))/(np.max(x0) - np.min(x0))
#            io.imsave('/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/Results02b/Img_rec/'+str(epoch)+'--'+str(iter)+'x_re.png',x0)        
#            x2 = output_mask[0, 0, :, :].squeeze().cpu().detach().numpy()
#            io.imsave('/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/Results02b/Sal/'+str(epoch)+'--'+str(iter)+'Sal-Maps.png',x2)
#            x3 = target[0, 0, :, :].squeeze().cpu().detach().numpy()
#            io.imsave('/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/Results02b/label/'+str(epoch)+'--'+str(iter)+'Sal-Mask.png',x3)
            


        if epoch>=0:
            x0 = x_final[0, 0, :, :].squeeze().cpu().detach().numpy()
            x0 = (x0 - np.min(x0))/(np.max(x0) - np.min(x0))
            output_img_dir = '/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/Img_rec/'
            if not os.path.exists(output_img_dir):
                os.mkdir(output_img_dir)
            io.imsave(output_img_dir+str(iter)+'x_re.png',x0)  
                       
            x1 = output_mask[0, 0, :, :].squeeze().cpu().detach().numpy()
            output_mask_dir = '/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/Sal-mask/'
            if not os.path.exists(output_mask_dir):
                os.mkdir(output_mask_dir)
            io.imsave(output_mask_dir+str(iter)+'Sal-Maps.png',x1)    

            x2 = output_f2[0, 0, :, :].squeeze().cpu().detach().numpy()
            output_f2_dir = '/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/Sal-f2/'
            if not os.path.exists(output_f2_dir):
                os.mkdir(output_f2_dir)
            io.imsave(output_f2_dir+str(iter)+'Sal-f2.png',x2)  
            
            x21 = output_f1[0, 0, :, :].squeeze().cpu().detach().numpy()
            output_f1_dir = '/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/Sal-f1/'
            if not os.path.exists(output_f1_dir):
                os.mkdir(output_f1_dir)
            io.imsave(output_f1_dir+str(iter)+'Sal-f1.png',x2)
            
            x3 = output_p[0, 0, :, :].squeeze().cpu().detach().numpy()
            output_p_dir = '/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/Sal-p/'
            if not os.path.exists(output_p_dir):
                os.mkdir(output_p_dir)
            io.imsave(output_p_dir+str(iter)+'Sal-p.png',x3)  
            x4 = target[0, 0, :, :].squeeze().cpu().detach().numpy()
            output_gt_dir = '/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/gt/'
            if not os.path.exists(output_gt_dir):
                os.mkdir(output_gt_dir)
            io.imsave(output_gt_dir+str(iter)+'Sal-label.png',x4)




        torch.cuda.synchronize()
        val_times.update(time.time() - start_time)

        loss1 = criterion1(output_p, target)   #BCELoss
        loss2 = criterion(output_f1, target)

        loss3 = criterion(output_f2, target) 
        
        
        loss5 = criterion(output_mask, target) 
        
        loss =  loss1 + loss2 + loss3 + loss5
        val_losses.update(loss.item())

        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output_mask = gather(output_mask, 0, dim=0)
        salEvalVal.addBatch(output_mask[:, 0, :, :], target[:,0,:,:].bool())
        print()

        if iter % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Time: %.3f loss: %.3f (avg: %.3f)' %
                        (epoch, args.max_epochs, iter, total_batches, val_times.avg,
                        val_losses.val, val_losses.avg))

    F_beta, MAE = salEvalVal.getMetric()
    record['val']['F_beta'].append(F_beta)
    record['val']['MAE'].append(MAE)

    return F_beta, MAE


def train(train_loader, epoch, cur_iter=0, fig = 0, verbose=True):
    # switch to train mode
    model.train()
    Inversion.train()
    if verbose:
        salEvalTrain = SalEval()

    total_batches = len(train_loader)
    scale = cur_iter // total_batches
    end = time.time()
    for iter, (gt, target, meas, body, detail) in enumerate(train_loader):
        if args.gpu == True:
            gt = gt.cuda()
            target = target.cuda()
            meas = meas.cuda()
            body =body.cuda()
            detail = detail.cuda()            
        gt = torch.autograd.Variable(gt)
        target = torch.autograd.Variable(target)
        meas = torch.autograd.Variable(meas)
        body = torch.autograd.Variable(body)
        detail = torch.autograd.Variable(detail)    
        start_time = time.time()
      #  for param in Inversion.parameters():
     #       param.requires_grad = False     
        [x_init, x_final] = Inversion(meas)

        loss_discrepancy1 = torch.mean(torch.pow((x_final - gt), 2))  #MSEloss
        loss_discrepancy2 = torch.mean(torch.abs(x_final - gt))  #MSEloss
        loss_TV = criterion2(x_final)   ##TV
        
        # run the mdoel
        output_p, output_f1, output_f2, output_mask = model(x_final)
#        print('target',target.shape) 
        loss1 = criterion1(output_p, target) + criterion3(output_p, target)  #BCELoss
        loss2 = criterion(output_f1, target) + criterion3(output_f1, target)

        loss3 = criterion(output_f2, target) + criterion3(output_f2, target)
        
        loss5 = criterion(output_mask, target) + criterion3(output_mask, target)

        loss =  0.5*(loss1 + loss2 + loss3 + loss5) + 0.5*loss_discrepancy1  + 0.25*loss_discrepancy2 + 0.5*loss_TV
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses[scale].update(loss.item())
        train_batch_times[scale].update(time.time() - start_time)
        train_data_times[scale].update(start_time - end)
        record[scale].append(train_losses[scale].avg)

        if verbose:
            # compute the confusion matrix
            if args.gpu and torch.cuda.device_count() > 1:
                output = gather(output_mask, 0, dim=0)

            salEvalTrain.addBatch(output_mask[:, 0, :, :], target[:, 0, :, :].bool())

        if iter % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Batch time: %.3f Data time: %.3f ' \
                        'loss: %.4f (avg: %.4f)  mse_final %.4f loss_TV %.4f   mae_final %.6f lr: %.1e' % \
                        (epoch, args.max_epochs, iter + cur_iter, total_batches + cur_iter, \
                         train_batch_times[scale].avg, train_data_times[scale].avg, \
                         train_losses[scale].val, train_losses[scale].avg, loss_discrepancy1.item(), loss_TV.item(), loss_discrepancy2.item(), lr))
        end = time.time()

    if verbose:
        F_beta, MAE = salEvalTrain.getMetric()
        record['train']['F_beta'].append(F_beta)
        record['train']['MAE'].append(MAE)

        return F_beta, MAE


# create the directory if not exist
if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)

log_name = 'log_' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + '.txt'
logger = Logger(os.path.join(args.savedir, log_name))
logger.info('Called with args:')
for (key, value) in vars(args).items():
    logger.info('{0:16} | {1}'.format(key, value))

# check if processed data file exists or not
if not os.path.isfile(args.cached_data_file):
    data_loader = ld.LoadData(args.data_dir, 'DUTS-TR', args.cached_data_file)
    data = data_loader.process()
    if data is None:
        logger.info('Error while pickling data. Please check.')
        exit(-1)
else:
    data = pickle.load(open(args.cached_data_file, 'rb'))

# ImageNet statistics
mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
#Wei_L = sio.loadmat('/home/yxj001/sp0722/paper_three/FastSaliency-master/FastSaliency-master/Pretrained/WL.mat')
#Wei_R = sio.loadmat('/home/yxj001/sp0722/paper_three/FastSaliency-master/FastSaliency-master/Pretrained/WR.mat')
#WL = Wei_L['WL']
#WR = Wei_R['WR']

PhiL, PhiR, WL, WR = load_meas_matrix()
# load the model
Inversion = ISTANet(PhiL, PhiR, WL, WR, 4)
model = net()
if args.gpu and torch.cuda.device_count() > 1:
    model = DataParallelModel(model)
    Inversion = DataParallelModel(Inversion)
if args.gpu:
    model = model.cuda()
    Inversion = Inversion.cuda()
logger.info('Model Architecture:\n' + str(model))
logger.info('Model Architecture (Inversion):\n' + str(Inversion))
total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
logger.info('Total network parameters: ' + str(total_paramters))

total_paramters1 = sum([np.prod(p.size()) for p in Inversion.parameters()])
logger.info('Total network parameters: ' + str(total_paramters1))

logger.info('Data statistics:')
logger.info('mean: [%.5f, %.5f, %.5f], std: [%.5f, %.5f, %.5f]' % (*data['mean'], *data['std']))

criterion = W_BCE_IOU_loss()
criterion1 = BCE_IOU_loss()
criterion2 = L_TV()
criterion3 = CEL()

if args.gpu and torch.cuda.device_count() > 1:
    criterion = DataParallelCriterion(criterion)
    criterion1 = DataParallelCriterion(criterion1)
    criterion2 = DataParallelCriterion(criterion2)
    criterion3 = DataParallelCriterion(criterion3)
if args.gpu:
    criterion = criterion.cuda()
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()
    criterion3 = criterion3.cuda()
train_losses = [AverageMeterSmooth() for _ in range(5)]
train_batch_times = [AverageMeterSmooth() for _ in range(5)]
train_data_times = [AverageMeterSmooth() for _ in range(5)]
val_losses = AverageMeterSmooth()
val_times = AverageMeterSmooth()

record = {
        0: [], 1: [], 2: [], 3: [], 4: [], 'lr': [],
        'val': {'F_beta': [], 'MAE': []},
        'train': {'F_beta': [], 'MAE': []}
        }
bests = {'F_beta_tr': 0., 'F_beta_val': 0., 'MAE_tr': 1., 'MAE_val': 1.}

# compose the data with transforms
trainTransform_main = transforms.Compose([
#        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(int(args.width), int(args.height)),
    #    transforms.RandomCropResize(int(7./256.*args.width)),
    #    transforms.RandomFlip(),
        transforms.ToTensor()
        ])

valTransform = transforms.Compose([
#        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(args.width, args.height),
        transforms.ToTensor()
        ])

# since we training from scratch, we create data loaders at different scales
# so that we can generate more augmented data and prevent the network from overfitting
train_set = datasets.Dataset(args.data_dir, 'Lensless_Train', transform=None)
val_set = datasets.Dataset(args.data_dir, 'Lensless_Test', transform=valTransform)
train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )
max_batches = len(train_loader) * 4

optimizer = torch.optim.Adam(list(model.parameters())+list(Inversion.parameters()), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, eval(args.milestones), args.gamma)
logger.info('Optimizer Info:\n' + str(optimizer))

start_epoch = 38
#Inversion.load_state_dict(checkpoint['state_dict'])
#Inversion.load_state_dict(torch.load('/home/yxj001/sp0722/Paper_first/model/MLP2_Net_layer_DUnet_4_group_1_ratio_25_lr_0.0001/net_params_80.pth'))
Inversion.load_state_dict(torch.load('/home/yxj001/sp0722/paper_three/FastSaliency-master/CVPR2021_PFNet-main/ablation_study/Basic_NCD_mGRA_old1/Inversion_epoch99.pth'))
if args.resume is not None:
    if os.path.isfile(args.resume):
        logger.info('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
      #  start_epoch = checkpoint['epoch']
       # optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint)
       # model.load_state_dict(checkpoint)
        logger.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, start_epoch))
    else:
        logger.info('=> no checkpoint found at {}'.format(args.resume))


for epoch in range(start_epoch, args.max_epochs):
    # train for one epoch
    lr = adjust_lr(optimizer, epoch)
    record['lr'].append(lr)
    length = len(train_loader)
   # train_set.transform = trainTransform_scale4
   # train(train_loader, epoch, length * 3, verbose=False)
    train_set.wh = 256
    F_beta_val, MAE_val = val(val_loader, epoch)
    F_beta_tr, MAE_tr = train(train_loader, epoch, 0, 0, verbose=True)

    # evaluate on validation set
    F_beta_val, MAE_val = val(val_loader, epoch)
    if F_beta_tr > bests['F_beta_tr']: bests['F_beta_tr'] = F_beta_tr
    if MAE_tr < bests['MAE_tr']: bests['MAE_tr'] = MAE_tr
    if F_beta_val > bests['F_beta_val']: bests['F_beta_val'] = F_beta_val
    if MAE_val < bests['MAE_val']: bests['MAE_val'] = MAE_val

    scheduler.step()
    torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_F_beta': bests['F_beta_val'],
            'best_MAE': bests['MAE_val']
            }, os.path.join(args.savedir, 'checkpoint.pth'))

    # save the model also
    model_file_name = os.path.join(args.savedir, 'model_epoch' + str(epoch + 1) + '.pth')
    torch.save(model.state_dict(), model_file_name)
    Inv_file_name = os.path.join(args.savedir, 'Inversion_epoch' + str(epoch + 1) + '.pth')
    torch.save(Inversion.state_dict(), Inv_file_name)
    
    logger.info('Epoch %d: F_beta (tr) %.4f (Best: %.4f) MAE (tr) %.4f (Best: %.4f) ' \
                'F_beta (val) %.4f (Best: %.4f) MAE (val) %.4f (Best: %.4f)' % \
                (epoch, F_beta_tr, bests['F_beta_tr'], MAE_tr, bests['MAE_tr'], \
                F_beta_val, bests['F_beta_val'], MAE_val, bests['MAE_val']))
    plot_training_process(record, args.savedir, bests)

logger.close()