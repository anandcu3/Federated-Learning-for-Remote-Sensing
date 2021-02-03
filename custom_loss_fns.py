import torch


class BasicLoss_wrapper():
    def __init__(self, base_criterion):
        self.base_criterion = base_criterion

    def loss_calculate(self, outputs, preds, new_weight=None, central_weight=None, mu=None):
        return self.base_criterion(outputs, preds)


class FedProxLoss():
    def __init__(self, base_criterion, mu):
        self.base_criterion = base_criterion
        self.mu = mu

    def loss_calculate(self, outputs, preds, new_weight=None, central_weight=None):
        weight_diff = sum((torch.pow((x - y), 2).sum() for x, y in zip(
            new_weight.state_dict().values(), central_weight.state_dict().values())))
        return self.base_criterion(outputs, preds) + self.mu / 2 * weight_diff
