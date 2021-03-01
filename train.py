import numpy as np
import torch
import time
import copy
from sklearn.metrics import f1_score


def train_model(model, device, dataloaders, criterion, optimizer, scheduler, n_classes, num_epochs=1, phase='train', valloader_for_train=None):
    tloss, tacc = [], []
    vloss, vacc = [], []
    f1score = []

    since = time.time()
    model.to(device)
    initial_model = copy.deepcopy(model)
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print('Client Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        f1_running_score = 0
        b_nr = 0
        # Iterate over data.
        for inputs, labels in dataloaders['data']:
            b_nr += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            if phase == 'train':
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                #_, preds = torch.max(outputs, 1)
                outputcpu = outputs.cpu()
                preds = np.heaviside(outputcpu.detach().numpy(), 0)
                loss = criterion.loss_calculate(outputs, labels.type(
                    torch.float), model, initial_model)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
                #outputsnp = outputs.cpu().numpy()
                #preds = np.array(outputsnp > 0.5, dtype=float)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += ((torch.sum(torch.from_numpy(preds).to(device)
                                            == labels.data)).item() / n_classes)
            f1_running_score += f1_score(preds,
                                         labels.data.to("cpu").numpy(), average="samples")

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataloaders['size']
        epoch_acc = (running_corrects) / dataloaders['size']
        f1all = f1_running_score / b_nr

        if phase == 'train':
            tloss.append(epoch_loss)
            tacc.append(epoch_acc)

        if phase == 'val':
            vloss.append(epoch_loss)
            vacc.append(epoch_acc)
            f1score.append(f1all)

        # print(dataset_sizes[phase],epoch_acc)
        # print(type(epoch_loss),type(epoch_acc))
        print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, f1all))
        print('-' * 10)

        if valloader_for_train and phase == "train":
            # Iterate over data.
            for inputs, labels in valloader_for_train['data']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                outputcpu = outputs.cpu()
                preds = np.heaviside(outputcpu.detach().numpy(), 0)
                loss = criterion.loss_calculate(outputs, labels.type(
                    torch.float), model, initial_model)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += ((torch.sum(torch.from_numpy(preds).to(device)
                                                == labels.data)).item() / n_classes)

            epoch_loss = running_loss / valloader_for_train['size']
            epoch_acc = (running_corrects) / valloader_for_train['size']
            print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('-' * 10)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
# model.load_state_dict(best_model_wts)

    return model, [tloss, tacc, vloss, vacc, f1score]
