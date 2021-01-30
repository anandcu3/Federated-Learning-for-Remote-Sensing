from train import train_model
import copy
import random
import torch
from cnn_nets import LENET, RESNET34
import torch.optim as optim


def train_fedavg_model(model, device, clients, valloader, optimizer, criterion, scheduler, c_fraction, n_classes, train_dataset_len, epochs=10):
    # iterate through epochs
    for i in range(epochs):
        init_model = copy.deepcopy(model)

        # get random subset of clients
        fraction = int(c_fraction * float(len(clients)))
        client_subset = random.sample(clients, fraction)

        # train each of the clients
        model_client_list = []
        print("Running epoch numero " + str(i))
        for ind, client in enumerate(client_subset):
            #model_for_client = RESNET34(n_classes)
            model_for_client = copy.deepcopy(model)
            #model_for_client.load_state_dict(model.state_dict())
            model_for_client = model_for_client.to(device)
            optimizer_ft = optim.SGD(model_for_client.parameters(), lr=0.001, momentum=0.9)
            exp_scheduler = optim.lr_scheduler.StepLR(
                                optimizer_ft, step_size=7, gamma=0.1)

            client_model, statistics = train_model(
                model_for_client, device, client, criterion, optimizer_ft, exp_scheduler,  n_classes, num_epochs=10, phase='train')
            model_client_list.append(client_model)
            print(f"Done with clientelo numero {ind} with stats: {statistics}")
            #del model_for_client
            #torch.cuda.empty_cache()

        # first initializer
        model_state = model_client_list[0].state_dict()
        client_data_size = client_subset[0]['size']
        for key in model_state:
            model_state[key] = (client_data_size /
                                train_dataset_len) * model_state[key]

        for c in range(1, len(model_client_list)):

            client_model_state = model_client_list[c].state_dict()
            client_new_data_size = client_subset[c]['size']

            for key in model_state:
                model_state[key] += (client_new_data_size /
                                     train_dataset_len) * client_model_state[key]

        averagedModel = copy.deepcopy(init_model)
        averagedModel.load_state_dict(model_state)
        model = copy.deepcopy(averagedModel)

        model = model.to(device)
        model, statistics = train_model(
            model,  device, valloader,  criterion, optimizer, scheduler,  n_classes,  num_epochs=1, phase='val')
        print("Done with validation", statistics)

    return model


def train_fedprox_model():
    # TODO : Implemenet the other algorithm
    pass
