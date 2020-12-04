# -*- coding: utf-8 -*-
#
# This script can be used to train any deep learning model on the BigEarthNet.
#
# To run the code, you need to provide a json file for configurations of the training.
#
# Author: Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Email: jian.kang@tu-berlin.de

import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil

import argparse
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import sys
sys.path.append('../')

from utils.ResNet import ResNet50, ResNet101, ResNet152
from utils.VGG import VGG16, VGG19
from utils.KBranchCNN import KBranchCNN

from utils.dataGenBigEarth import dataGenBigEarthLMDB, ToTensor, Normalize

from utils.metrics import MetricTracker, Precision_score, Recall_score, F1_score, F2_score, Hamming_loss, Subset_accuracy, \
    Accuracy_score, One_error, Coverage_error, Ranking_loss, LabelAvgPrec_score

model_choices = ['resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'KBranchCNN']

parser = argparse.ArgumentParser(description='PyTorch multi-label classification')
parser.add_argument('--bigEarthLMDBPth', metavar='DATA_DIR',
                        help='path to the saved big earth LMDB dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--epochs', type=int, default=500, help='epoch number')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--resume', '-r', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch')
parser.add_argument('--model', default='resnet50', type=str, metavar='M',
                    choices=model_choices,
                        help='choose model for training, choices are: ' \
                         + ' | '.join(model_choices) + ' (default: resnet50)')
parser.add_argument('--BigEarthNet19', dest='BigEarthNet19', action='store_true',
                    help='use the BigEarthNet19 class nomenclature')
parser.add_argument('--train_csv', metavar='CSV_PTH',
                        help='path to the csv file of train patches')
parser.add_argument('--val_csv', metavar='CSV_PTH',
                        help='path to the csv file of val patches')
parser.add_argument('--test_csv', metavar='CSV_PTH',
                        help='path to the csv file of test patches')

args = parser.parse_args()

checkpoint_dir = os.path.join('./', args.model, 'checkpoints')
logs_dir = os.path.join('./', args.model, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))


def main():
    global args

    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    print('saving file name is ', sv_name)

    write_arguments_to_file(args, os.path.join(logs_dir, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    bands_mean = {
                    'bands10_mean': [ 429.9430203 ,  614.21682446,  590.23569706, 2218.94553375],
                    'bands20_mean': [ 950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
                    'bands60_mean': [ 340.76769064, 2246.0605464 ],
                }

    bands_std = {
                    'bands10_std': [ 572.41639287,  582.87945694,  675.88746967, 1365.45589904],
                    'bands20_std': [ 729.89827633, 1096.01480586, 1273.45393088, 1356.13789355, 1079.19066363,  818.86747235],
                    'bands60_std': [ 554.81258967, 1302.3292881 ]
                }
    upsampling = True

    if args.BigEarthNet19:
        numCls = 19
    else:
        numCls = 43

    if args.model == "resnet50":
        model = ResNet50(numCls)
    elif args.model == "resnet101":
        model = ResNet101(numCls)
    elif args.model == "resnet152":
        model = ResNet152(numCls)
    elif args.model == 'vgg16':
        model = VGG16(numCls)
    elif args.model == 'vgg19':
        model = VGG19(numCls)
    elif args.model == 'KBranchCNN':
        model = KBranchCNN(numCls)
        upsampling = False
    else:
        raise NameError("no model")

    train_dataGen = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=args.bigEarthLMDBPth,
                    state='train',
                    imgTransform=transforms.Compose([
                        ToTensor(),
                        Normalize(bands_mean, bands_std)
                    ]),
                    upsampling=upsampling,
                    train_csv=args.train_csv,
                    val_csv=args.val_csv,
                    test_csv=args.test_csv
    )

    val_dataGen = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=args.bigEarthLMDBPth,
                    state='val',
                    imgTransform=transforms.Compose([
                        ToTensor(),
                        Normalize(bands_mean, bands_std)
                    ]),
                    upsampling=upsampling,
                    train_csv=args.train_csv,
                    val_csv=args.val_csv,
                    test_csv=args.test_csv
    )

    train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    if use_cuda:
        model.cuda()

    multiLabelLoss = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        train(train_data_loader, model, optimizer, multiLabelLoss, epoch, use_cuda, train_writer)
        micro_f1 = val(val_data_loader, model, optimizer, epoch, use_cuda, val_writer)

        is_best_acc = micro_f1 > best_acc
        best_acc = max(best_acc, micro_f1)

        save_checkpoint({
            'epoch': epoch,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec': best_acc,
        }, is_best_acc, sv_name)

def train(trainloader, model, optimizer, multiLabelLoss, epoch, use_cuda, train_writer):

    lossTracker = MetricTracker()
    
    model.train()

    for idx, data in enumerate(tqdm(trainloader, desc="training")):
        
        numSample = data["bands10"].size(0)

        if args.model not in ['KBranchCNN']:
            bands = torch.cat((data["bands10"], data["bands20"]), dim=1).to(torch.device("cuda"))
        else:
            bands10 = data["bands10"].to(torch.device("cuda"))
            bands20 = data["bands20"].to(torch.device("cuda"))
        
        labels = data["label"].to(torch.device("cuda"))

        optimizer.zero_grad()

        if args.model in ['KBranchCNN']:
            logits = model(bands10, bands20)
        else:
            logits = model(bands)
        
        loss = multiLabelLoss(logits, labels)

        loss.backward()
        optimizer.step()

        lossTracker.update(loss.item(), numSample)

    train_writer.add_scalar("loss", lossTracker.avg, epoch)

    print('Train loss: {:.6f}'.format(lossTracker.avg))


def val(valloader, model, optimizer, epoch, use_cuda, val_writer):

    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score()
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()

    model.eval()

    y_true = []
    predicted_probs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            if args.model not in ['KBranchCNN']:
                bands = torch.cat((data["bands10"], data["bands20"]), dim=1).to(torch.device("cuda"))
            else:
                bands10 = data["bands10"].to(torch.device("cuda"))
                bands20 = data["bands20"].to(torch.device("cuda"))
            
            labels = data["label"].to(torch.device("cuda"))

            if args.model in ['KBranchCNN']:
                logits = model(bands10, bands20)
            else:
                logits = model(bands)
            
            probs = torch.sigmoid(logits).cpu().numpy()

            predicted_probs += list(probs)
            y_true += list(labels)
    
    predicted_probs = np.asarray(predicted_probs)
    y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    y_true = np.asarray(y_true)

    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)

    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    info = {
        "macroPrec" : macro_prec,
        "microPrec" : micro_prec,
        "samplePrec" : sample_prec,
        "macroRec" : macro_rec,
        "microRec" : micro_rec,
        "sampleRec" : sample_rec,
        "macroF1" : macro_f1,
        "microF1" : micro_f1,
        "sampleF1" : sample_f1,
        "macroF2" : macro_f2,
        "microF2" : micro_f2,
        "sampleF2" : sample_f2,
        "HammingLoss" : hamming_loss,
        "subsetAcc" : subset_acc,
        "macroAcc" : macro_acc,
        "microAcc" : micro_acc,
        "sampleAcc" : sample_acc,
        "oneError" : one_error,
        "coverageError" : coverage_error,
        "rankLoss" : rank_loss,
        "labelAvgPrec" : labelAvgPrec
    }

    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)

    print('Validation microPrec: {:.6f} microF1: {:.6f} sampleF1: {:.6f} microF2: {:.6f} sampleF2: {:.6f}'.format(
            micro_prec,
            micro_f1,
            sample_f1,
            micro_f2,
            sample_f2
            ))
    return micro_f1


if __name__ == "__main__":
    main()