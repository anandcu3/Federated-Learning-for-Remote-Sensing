# The Impact of Federated Learning on Distributed Remote Sensing Archives

The repository consists of the code for Federated Learning Experiments for Remote Sensing image data using convolution neural networks. It contains the implementation of three Federated Learning models: 
* [FedAVG](https://arxiv.org/abs/1602.05629)
* [FedProx](https://arxiv.org/abs/1812.06127)
* [BSP](https://dl.acm.org/doi/10.1145/79173.79181)
The implementation is specifically made for the multi-label *UCMerced Landuse* [dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html). To apply other datasets it requires some modification.

## Table of Contents
* Description of the files & Usage
* Implementation of the Federated Learning Models
    * FedAVG
    * FedProx
    * BSP
* Data Preparation and Splitting
 
## Description of the files & Usage

-  `main.py` : The file to run to start training based on the required experiment setup. Required Arguments are data path and the CNN model to use. For more details on how to use the arguments use `python main.py -h`. E.g.  `python main.py -c resnet34 -d ./UCMerced_LandUse/Images` trains FedAVG using resnet34 with the images in the provided path. The following parameters can be chosen for training:
 * CNN Model
 * Number iof clients
 * Federated Learning Model / Centralized
 * Percentage of label skewness
 * Number of Epochs
 * Number of local Epochs (FedAVG and FedProx)
 * Learning Rate
 * Validation Split
 * Data directory and multilabel excelfile path
- `visualize.py`: This file is used to plot the results from training. When training the FL models using  `main.py` a `csv` is generated containing *loss*, *accuracy* and *F1-Score*. For more details on how to use the arguments use `python visualize.py -h`.
- `legacy_notebooks` : Code before combining all notebooks to a single project. Initial work was done almost individually.
- `multilabels` : Contains the multilabel excel files for the UCMerced_LandUse dataset.
- `cnn_nets.py` : Contains the CNN architectures that can be used: `ResNet34`, `LeNet` and `AlexNet`
- `custom_dataloader.py` : Includes the functions to split data across multiple clients in both IID and non-IID distributions. Furthermore it checks which classes are least correlated.
- `custom_loss_fns.py` : Custom loss functions can be found here. One loss function is specifically for the FedProx Federated Learning algorithm. The other one is a wrapper to the Pytorch loss function. This was included to have a generic train function that supports custom loss functions.
- `CustomDataSet.py` : Inherits the abstract class `torch.utils.data.Dataset` and overrides `__len__` and `__getitem__` method. This custom dataset class supports multilabel for each image.
- `federated_train_algorithms.py` : Includes the implemented Federated Learning models. Supported FL algorithms:
  * FedAvg (Federated Averaging) ([Paper](https://arxiv.org/abs/1602.05629))
  * FedProx ([Paper](https://arxiv.org/abs/1812.06127))
  * Bulk Synchronous Processing (BSP) ([Paper](https://dl.acm.org/doi/10.1145/79173.79181))
- `FL_with_pytorch_only.ipynb` : The final notebook version from which the modular code was written from.
-  `train.py` : The file has the training loop. This training loop is used by all federated algorithms.
- `requirements.txt` : Can be used directly with pip/conda to setup the required packages.

## Implementation of the Federated Learning Models

### FedAVG (Federated Averaging)
// tbd

### FedProx
// tbd

### BSP
// tbd

## Data Preparation and Splitting
// tbd
