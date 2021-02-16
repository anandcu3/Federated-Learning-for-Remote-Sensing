from cnn_nets import LENET, RESNET34, ALEXNET
from custom_dataloader import load_split_train_test, uncor_selecter
from custom_loss_fns import BasicLoss_wrapper
from train import train_model
from federated_train_algorithms import FedAvg, FedProx, BSP
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import torch
import datetime
import copy
import time

parser = argparse.ArgumentParser(
    description='Run speech recognition on Google or Azure. Followed by WER for the generated hypothesis.')
parser.add_argument('--cnn_model', '-c', type=str, required=True,
                    help='Specify which CNN to use. "lenet", "alexnet" or "resnet34"')
parser.add_argument('--client_nr', '-n', type=int, required=False, default=3,
                    help='number of clients to split the data on')
parser.add_argument('--federated_algo', '-f', type=str, required=False, default="FedAvg",
                    help='Specify the federated algorithm if not running on centralised mode. FedAvg (default), FedProx or BSP')
parser.add_argument('--skewness', '-s', type=int, required=False, default=40,
                    help='the percentage of label skewness, when data splittd on')
parser.add_argument('--epochs', '-e', type=int, required=False, default=15,
                    help='the number of epochs')
parser.add_argument('--client_epochs', '-ce', type=int, required=False, default=5,
                    help='the number of epochs per client')
parser.add_argument('--lr', type=float, required=False, default=0.001,
                    help='Learning Rate')
parser.add_argument('--vs', type=float, required=False, default=.2,
                    help='validation split')
parser.add_argument('--centralised', dest='centralised', action='store_true',
                    default=False, help="Use the flag if centralised learning is required")
parser.add_argument('--small_skew', action='store_true',
                    default=False, help="Use the flag to skew the small represented label classes")
parser.add_argument('--data_dir', '-d', type=str, required=True,
                    help='Specify path to images folder of UCMerced_LandUse dataset. Eg. ./UCMerced_LandUse/Images')
parser.add_argument('--multilabel_excelfilepath', type=str, default='multilabels/LandUse_Multilabeled.xlsx',
                    help='Specify path to label file of UCMerced_LandUse dataset. Eg. ./labelfolder/LandUse_Multilabeled.xlsx')
args = parser.parse_args()

class_names = np.array(["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
                        "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df = pd.read_excel(Path(args.multilabel_excelfilepath).resolve())
df_label = np.array(df)

criterion = nn.BCEWithLogitsLoss()
data_dir = Path(args.data_dir).resolve()
C_FRACTION = 0.1  # For FedAvg and FedProx
MU = 1  # For FedProx

if args.centralised:
    args.client_nr = 1

if args.cnn_model == "lenet":
    print("Using Lenet")
    model = LENET(len(class_names))
elif args.cnn_model == "resnet34":
    print("Using Resnet 34")
    model = RESNET34(len(class_names))
elif args.cnn_model == "alexnet":
    print("Using alexnet")
    model = ALEXNET(len(class_names))
else:
    print("Unknown CNN")
    exit()

trainloaders, valloader, train_dataset_len = load_split_train_test(
    data_dir, df_label, args.client_nr, args.skewness, args.small_skew, args.vs)

if args.centralised:
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)
    criterion = BasicLoss_wrapper(criterion)

    best_model = copy.deepcopy(model)
    best_acc = 0.0
    stats = []
    for e in range(args.epochs):
        model, _ = train_model(model, device, trainloaders[0], criterion, optimizer_ft, exp_lr_scheduler, len(
            class_names), num_epochs=1, phase='train')
        model, statistics = train_model(model,  device, valloader,  criterion, optimizer_ft,
                                        exp_lr_scheduler,  len(class_names),  num_epochs=1, phase='val')

        if statistics[3][0] > best_acc:
            best_acc = statistics[3][0]
            best_model = copy.deepcopy(model)
        print(
            f"centralised at epoch {e} with validation acc {statistics[3][0]}")
        stats.append([statistics[2][0], statistics[3][0]])

    torch.save(
        model, f'latest_centralised_{args.cnn_model}_with_{args.client_nr}_clients_{args.skewness}.pt')
    torch.save(
        best_model, f'best_centralised_{args.cnn_model}_with_{args.client_nr}_clients_{args.skewness}.pt')
    np.savetxt(
        f'centralised_acc&loss_for_{args.cnn_model}_with_{args.client_nr}_clients_{args.skewness}.csv', np.array(stats).T, delimiter=",")


else:
    if args.federated_algo == "FedAvg":
        federated_algo = FedAvg(model, device, trainloaders, valloader, optim.SGD, criterion,
                                optim.lr_scheduler.StepLR, len(class_names), train_dataset_len, C_FRACTION, epochs=args.epochs, client_epochs=args.client_epochs)
    elif args.federated_algo == "FedProx":
        federated_algo = FedProx(model, device, trainloaders, valloader, optim.SGD, criterion,
                                 optim.lr_scheduler.StepLR, len(class_names), train_dataset_len, C_FRACTION, MU, epochs=args.epochs, client_epochs=args.client_epochs)
    elif args.federated_algo == "BSP":
        federated_algo = BSP(model, device, trainloaders, valloader, optim.SGD, criterion,
                             optim.lr_scheduler.StepLR, len(class_names), train_dataset_len, epochs=args.epochs)
    else:
        print("Specify a valid federated algorithm")
        exit()
    last_model, best_model, loss_acc_stats = federated_algo.train_federated_model()

    torch.save(
        last_model, f'latest_{args.federated_algo}_{args.cnn_model}_with_{args.client_nr}_clients_{args.skewness}_clientsepox_{args.client_epochs}_vsplit_{args.vs}_lr_{args.lr}.pt')
    torch.save(
        best_model, f'best_{args.federated_algo}_{args.cnn_model}_with_{args.client_nr}_clients_{args.skewness}_clientsepox_{args.client_epochs}_vsplit_{args.vs}_lr_{args.lr}.pt')
    np.savetxt(
        f'{args.federated_algo}_acc&loss_for_{args.cnn_model}_with_{args.client_nr}_clients_{args.skewness}_clientsepox_{args.client_epochs}_vsplit_{args.vs}_lr_{args.lr}_{time.time()}.csv', loss_acc_stats.T, delimiter=",")
