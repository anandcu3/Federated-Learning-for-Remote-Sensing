from cnn_nets import LENET, RESNET34
from custom_dataloader import load_split_train_test, uncor_selecter
from train import train_model
from federated_train_algorithms import train_fedavg_model
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch

df = pd.read_excel(Path('multilabels/LandUse_Multilabeled.xlsx').resolve())
df_label = np.array(df)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = np.array(["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
                        "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"])
data_dir = Path('./UCMerced_LandUse/Images').resolve()
trainloaders, valloader = load_split_train_test(data_dir, df_label, .2)
model = LENET(len(class_names))
model = RESNET34(len(class_names))

model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=7, gamma=0.1)

C_FRACTION = 0.7
model = train_fedavg_model(model, device, trainloaders, valloader, optimizer_ft,
                           criterion, exp_lr_scheduler, C_FRACTION, len(class_names), epochs=1)
