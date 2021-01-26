The repository consists of the code for Federated Learning Experiments for Remote Sensing image data using convolution neural networks.

#### Description of the files & folders

- `bigearthnet_pytorch` : code borrowed from RSIM repository and contains files to manage bigearthnet from pytorch. Since the dataset was eventually moved to UCMerced_LandUse this is not very relevant now.

- `legacy_notebooks` : code before combining them to a single project. initial work was done almost individually
- `multilabels` : contains the label files for the images for UCMerced_LandUse dataset for the multilabel case.
- `cnn_nets.py` : Contains the CNN architectures. Currently has LENET and RESNET34
- `custom_dataloader.py` : Has functions to split data across multiple client in both IID and non-IID distributions. Also has function to check which classes are least correlated, etc
- `CustomDataSet.py` : Inherits the abstract class `torch.utils.data.Dataset` and overrides `__len__` and `__getitem__` method. This custom dataset class supports Multilabel for each image.
- `federated_train_algorithms.py` : Has the federated training algotihms implemented in it. Supports FedAvg Currently.
- `FL_with_pytorch_only.ipynb` : The final notebook version from which the modular code was written from.
-  `main.py` : The file to run to start training based on the required experiment setup. Required Arguments are data path and the CNN model to use. For more details on how to use the arguments use `python main.py -h ` and use `python main.py -c resnet34 -d ./UCMerced_LandUse/Images` for example to use resnet34 and with the images in the provided path.
-  `train.py` : The file has the training loop. This training loop will be common for different federated algorithms.
