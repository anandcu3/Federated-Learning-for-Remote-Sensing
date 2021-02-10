from train import train_model
import copy
import random
import torch
from cnn_nets import LENET, RESNET34
import torch.optim as optim
import numpy as np
from custom_loss_fns import BasicLoss_wrapper, FedProxLoss


class FedAvg():
    def __init__(self, model, device, clients, valloader, optimizer, criterion, scheduler, n_classes, train_dataset_len, c_fraction, epochs=10):
        self.best_model_wts = copy.deepcopy(model)
        self.init_model = copy.deepcopy(model)
        self.model = model
        self.device = device
        self.trainloaders = clients
        self.valloader = valloader
        self.optimizer = optimizer
        self.criterion = BasicLoss_wrapper(criterion)
        self.scheduler = scheduler
        self.n_classes = n_classes
        self.train_dataset_len = train_dataset_len
        self.c_fraction = c_fraction
        self.epochs = epochs

    def train_federated_model(self):
        best_acc = 0.0
        stats = []

        # iterate through epochs
        for i in range(self.epochs):

            # get random subset of clients
            fraction = int(self.c_fraction * float(len(self.trainloaders)))
            client_subset = random.sample(self.trainloaders, fraction)

            # train each of the clients
            model_client_list = []
            print("----------------------------------")
            print("Running epoch number " + str(i + 1))
            for ind, client in enumerate(client_subset):
                #model_for_client = RESNET34(n_classes)
                model_for_client = copy.deepcopy(self.model)
                # model_for_client.load_state_dict(model.state_dict())
                model_for_client = model_for_client.to(self.device)
                optimizer_ft = self.optimizer(
                    model_for_client.parameters(), lr=0.001, momentum=0.9)
                exp_scheduler = self.scheduler(
                    optimizer_ft, step_size=7, gamma=0.1)

                client_model, statistics = train_model(
                    model_for_client, self.device, client, self.criterion, optimizer_ft, exp_scheduler,  self.n_classes, num_epochs=5, phase='train')
                model_client_list.append(client_model)
                print(
                    f"Done with client number {ind + 1} with stats: {statistics}")
                #del model_for_client
                # torch.cuda.empty_cache()

            # first initializer
            model_state = model_client_list[0].state_dict()
            client_data_size = client_subset[0]['size']
            for key in model_state:
                model_state[key] = (client_data_size /
                                    self.train_dataset_len) * model_state[key]

            for c in range(1, len(model_client_list)):

                client_model_state = model_client_list[c].state_dict()
                client_new_data_size = client_subset[c]['size']

                for key in model_state:
                    model_state[key] += (client_new_data_size /
                                         self.train_dataset_len) * client_model_state[key]

            averagedModel = copy.deepcopy(self.init_model)
            averagedModel.load_state_dict(model_state)
            self.model = copy.deepcopy(averagedModel)

            self.model = self.model.to(self.device)
            self.model, statistics = train_model(
                self.model,  self.device, self.valloader,  self.criterion, None, None,  self.n_classes,  num_epochs=1, phase='val')

            # deep copy the model
            if statistics[3][0] > best_acc:
                best_acc = statistics[3][0]
                self.best_model_wts = copy.deepcopy(self.model)

            print("Done with validation", statistics)
            stats.append([statistics[2][0], statistics[3][0]])

        return self.model, self.best_model_wts, np.array(stats)


class FedProx(FedAvg):
    def __init__(self, model, device, clients, valloader, optimizer, criterion, scheduler, n_classes, train_dataset_len, c_fraction, mu=0, epochs=10):
        super(FedProx, self).__init__(model, device, clients, valloader, optimizer,
                                      criterion, scheduler, n_classes, train_dataset_len, c_fraction, epochs)
        self.mu = mu
        self.criterion = FedProxLoss(criterion, mu)

class BSP():
    def __init__(self, model, device, clients, valloader, optimizer, criterion, scheduler, n_classes, train_dataset_len, epochs=10):
        self.model = model
        self.device = device
        self.clients = clients
        self.valloader = valloader
        self.optimizer = optimizer
        self.criterion = BasicLoss_wrapper(criterion)
        self.scheduler = scheduler
        self.n_classes = n_classes
        self.train_dataset_len = train_dataset_len
        self.epochs = epochs

    def train_federated_model(self):
        best_acc = 0.0
        stats = []

        # iterate through epochs
        for i in range(self.epochs):
            print("----------------------------------")
            print("Running epoch number " + str(i + 1))
            for ind, client in enumerate(self.clients):
                optimizer_ft = self.optimizer(self.model.parameters(), lr=0.001, momentum=0.9)
                exp_scheduler = self.scheduler(optimizer_ft, step_size=7, gamma=0.1)
                self.model = self.model.to(self.device)
                self.model, statistics = train_model(
                    self.model, self.device, client, self.criterion, optimizer_ft, exp_scheduler, self.n_classes, num_epochs=1, phase='train')
                print(f"Done with client number {ind + 1} with stats: {statistics}")

            self.model = self.model.to(self.device)
            self.model, statistics = train_model(
                self.model, self.device, self.valloader, self.criterion, None, None, self.n_classes,  num_epochs=1, phase='val')
            
            # deep copy the model
            if statistics[3][0] > best_acc:
                best_acc = statistics[3][0]
                self.best_model_wts = copy.deepcopy(self.model)

            print("Done with validation", statistics)
            stats.append([statistics[2][0], statistics[3][0]])
        
        return self.model, self.best_model_wts, np.array(stats)