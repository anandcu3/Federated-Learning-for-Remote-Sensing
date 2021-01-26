import numpy as np
import torch
import time


def train_model(model, device, dataloaders, criterion, optimizer, scheduler, n_classes, num_epochs=1, phase='train'):
    tloss, tacc = [], []
    vloss, vacc = [], []

    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        if True:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    outputcpu = outputs.cpu()
                    preds = np.heaviside(outputcpu.detach().numpy(), 0)
                    #print(outputs, preds)
                    loss = criterion(outputs, labels.type(torch.float))

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
                # print("running_corrects",running_corrects)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = (running_corrects) / len(dataloaders.dataset)

            if phase == 'train':
                tloss.append(epoch_loss)
                tacc.append(epoch_acc)

            if phase == 'val':
                vloss.append(epoch_loss)
                vacc.append(epoch_acc)

            # print(dataset_sizes[phase],epoch_acc)
            # print(type(epoch_loss),type(epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, [tloss, tacc, vloss, vacc]
