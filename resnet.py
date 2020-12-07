from torchvision import models
from bigearthnet_pytorch.utils.dataGenBigEarth import dataGenBigEarthLMDB, ToTensor, Normalize
from bigearthnet_pytorch.utils.metrics import MetricTracker, Precision_score, F1_score, F2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import os
import shutil
import argparse


parser = argparse.ArgumentParser(
    description='PyTorch Resnet 50 BigEarthNet multi-label classification')
parser.add_argument('--bigEarthLMDBPth', metavar='DATA_DIR',
                    help='path to the saved big earth LMDB dataset')
parser.add_argument('--train_csv', metavar='CSV_PTH',
                    help='path to the csv file of train patches')
parser.add_argument('--val_csv', metavar='CSV_PTH',
                    help='path to the csv file of val patches')
args = parser.parse_args()


def save_checkpoint(state, is_best, name, checkpoint_dir):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            checkpoint_dir, name + '_model_best.pth.tar'))


def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)


def train(trainloader, model, optimizer, multiLabelLoss, epoch):

    lossTracker = MetricTracker()

    model.train()

    for data in tqdm(trainloader, desc="training"):

        numSample = data["bands10"].size(0)

        bands = torch.cat((data["bands10"], data["bands20"]), dim=1).to(
            torch.device("cuda"))

        labels = data["label"].to(torch.device("cuda"))

        optimizer.zero_grad()

        logits = model(bands)

        loss = multiLabelLoss(logits, labels)

        loss.backward()
        optimizer.step()

        lossTracker.update(loss.item(), numSample)
    print('Train loss: {:.6f}'.format(lossTracker.avg))


def val(valloader, model, optimizer, epoch):
    prec_score_ = Precision_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    model.eval()

    y_true = []
    predicted_probs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            bands = torch.cat((data["bands10"], data["bands20"]), dim=1).to(
                torch.device("cuda"))

            labels = data["label"].to(torch.device("cuda"))

            logits = model(bands)

            probs = torch.sigmoid(logits).cpu().numpy()

            predicted_probs += list(probs)
            y_true += list(labels)

    predicted_probs = np.asarray(predicted_probs)
    y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    y_true = np.asarray(y_true)

    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(
        y_predicted, y_true)

    print('Validation microPrec: {:.6f} microF1: {:.6f} sampleF1: {:.6f} microF2: {:.6f} sampleF2: {:.6f}'.format(
        micro_prec,
        micro_f1,
        sample_f1,
        micro_f2,
        sample_f2
    ))
    return micro_f1


class ResNet50(nn.Module):
    def __init__(self, numCls=19):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        self.conv1 = nn.Conv2d(10, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits


best_acc = 0
start_epoch = 0
bands_mean = {
    'bands10_mean': [429.9430203,  614.21682446,  590.23569706, 2218.94553375],
    'bands20_mean': [950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
    'bands60_mean': [340.76769064, 2246.0605464],
}
bands_std = {
    'bands10_std': [572.41639287,  582.87945694,  675.88746967, 1365.45589904],
    'bands20_std': [729.89827633, 1096.01480586, 1273.45393088, 1356.13789355, 1079.19066363,  818.86747235],
    'bands60_std': [554.81258967, 1302.3292881]
}
model = ResNet50(43)

train_dataGen = dataGenBigEarthLMDB(
    bigEarthPthLMDB=args.bigEarthLMDBPth,
    state='train',
    imgTransform=transforms.Compose([
        ToTensor(),
        Normalize(bands_mean, bands_std)
    ]),
    upsampling=True,
    train_csv=args.train_csv,
)

val_dataGen = dataGenBigEarthLMDB(
    bigEarthPthLMDB=args.bigEarthLMDBPth,
    state='val',
    imgTransform=transforms.Compose([
        ToTensor(),
        Normalize(bands_mean, bands_std)
    ]),
    upsampling=True,
    val_csv=args.val_csv,
)


train_data_loader = DataLoader(train_dataGen, batch_size=256,
                               num_workers=8, shuffle=True, pin_memory=True)
val_data_loader = DataLoader(val_dataGen, batch_size=256,
                             num_workers=8, shuffle=False, pin_memory=True)


model.cuda()
multiLabelLoss = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(start_epoch, 500):

    print('Epoch {}/{}'.format(epoch, 500 - 1))
    print('-' * 10)

    train(train_data_loader, model, optimizer,
          multiLabelLoss, epoch)
    micro_f1 = val(val_data_loader, model, optimizer,
                   epoch)

    is_best_acc = micro_f1 > best_acc
    best_acc = max(best_acc, micro_f1)

    save_checkpoint({
        'epoch': epoch,
        'arch': "resnet50",
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_prec': best_acc,
    }, is_best_acc, "1", "./checkpoints/")
