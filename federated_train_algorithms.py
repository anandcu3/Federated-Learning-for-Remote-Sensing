from train import train_model
import copy


def train_fedavg_model(model, device, clients, valloader, optimizer, criterion, scheduler, c_fraction, n_classes, epochs=10):
    # iterate through epochs
    for i in range(epochs):
        # get random subset of clients
        #fraction = int( c_fraction * float(len(clients_t)) )
        #client_subset = random.sample(clients, fraction)

        # train each of the clients
        models_client_list = []
        print("Running epoch numero " + str(i))
        for dataloaders in clients:
            model_for_client = copy.deepcopy(model)
            client_model, statistics = train_model(
                model_for_client, device, dataloaders, criterion, optimizer, scheduler, n_classes, num_epochs=10, phase='train')
            client_model = copy.deepcopy(client_model)
            models_client_list.append(client_model)
            print("Done with clientelo numero whateva", statistics)

        # still need to average
        # average clients params
        # model = sum(k for 1 - num_clients): ( data_client / total_num_data ) * model_client_k

        _, statistics = train_model(
            model, device, valloader, criterion, optimizer, scheduler, n_classes, num_epochs=1, phase='val')
        print("Done with validation", statistics)

    return 0


def train_fedprox_model():
    # TODO : Implemenet the other algorithm
    pass
