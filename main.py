from cnn_nets import LENET, RESNET34
from custom_dataloader import load_split_train_test, uncor_selecter
from train import train_model
from federated_train_algorithms import train_fedavg_model
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import torch

parser = argparse.ArgumentParser(
    description='Run speech recognition on Google or Azure. Followed by WER for the generated hypothesis.')
parser.add_argument('--cnn_model', '-c', type=str, required=True,
                    help='Specify which CNN to use. "lenet" or "resnet34"')

parser.add_argument('--client_nr', '-n', type=int, required=False, default=3,
                    help='number of clients to split the data on')
parser.add_argument('--skewness', '-s', type=int, required=False, default=40,
                    help='the percentage of label skewness, when data splittd on')

parser.add_argument('--data_dir', '-d', type=str, required=True,
                    help='Specify path to images folder of UCMerced_LandUse dataset. Eg. ./UCMerced_LandUse/Images')
parser.add_argument('--multilabel_excelfilepath', type=str, default='multilabels/LandUse_Multilabeled.xlsx',
                    help='Specify path to images folder of UCMerced_LandUse dataset. Eg. ./UCMerced_LandUse/Images')
args = parser.parse_args()

df = pd.read_excel(Path(args.multilabel_excelfilepath).resolve())
df_label = np.array(df)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = np.array(["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
                        "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"])
data_dir = Path(args.data_dir).resolve()
trainloaders, valloader = load_split_train_test(data_dir, df_label, args.client_nr, args.skewness, .2)
if args.cnn_model == "lenet":
    print("Using Lenet")
    model = LENET(len(class_names))
elif args.cnn_model == "resnet34":
    print("Using Resnet 34")
    model = RESNET34(len(class_names))
else:
    print("Unknown CNN")
    exit()

model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=7, gamma=0.1)

C_FRACTION = 0.7
model = train_fedavg_model(model, device, trainloaders, valloader, optimizer_ft,
                           criterion, exp_lr_scheduler, C_FRACTION, len(class_names), epochs=1)
