The repository consists of the code for Federated Learning Experiments for Remote Sensing image data using convolution neural networks.

#### Description of the files & folders

-  `main.py` : The file to run to start training based on the required experiment setup. Required Arguments are data path and the CNN model to use. For more details on how to use the arguments use `python main.py -h ` and use `python main.py -c resnet34 -d ./UCMerced_LandUse/Images` for example to use resnet34 and with the images in the provided path. The python module using `argparse` package to acheive this. Here is a list of all the options available for execution.

```
usage: main.py [-h] --cnn_model CNN_MODEL [--client_nr CLIENT_NR] [--federated_algo FEDERATED_ALGO]\
[--skewness SKEWNESS][--epochs EPOCHS] [--client_epochs CLIENT_EPOCHS] [--lr LR] [--vs VS]\
[--centralised] [--small_skew] --data_dir DATA_DIR \
[--multilabel_excelfilepath MULTILABEL_EXCELFILEPATH]

Run federated algorithms for remote sensing data like UCMercedLandUse dataset.

optional arguments:
  -h, --help            show this help message and exit
  --cnn_model CNN_MODEL, -c CNN_MODEL
                        Specify which CNN to use. "lenet", "alexnet" or "resnet34"
  --client_nr CLIENT_NR, -n CLIENT_NR
                        number of clients to split the data on
  --federated_algo FEDERATED_ALGO, -f FEDERATED_ALGO
                        Specify the federated algorithm if not running on centralised mode. FedAvg (default), FedProx or BSP
  --skewness SKEWNESS, -s SKEWNESS
                        the percentage of label skewness, when data splittd on
  --epochs EPOCHS, -e EPOCHS
                        the number of epochs
  --client_epochs CLIENT_EPOCHS, -ce CLIENT_EPOCHS
                        the number of epochs per client
  --lr LR               Learning Rate
  --vs VS               validation split
  --centralised         Use the flag if centralised learning is required
  --small_skew          Use the flag to skew the small represented label classes
  --data_dir DATA_DIR, -d DATA_DIR
                        Specify path to images folder of UCMerced_LandUse dataset. Eg. ./UCMerced_LandUse/Images
  --multilabel_excelfilepath MULTILABEL_EXCELFILEPATH
                        Specify path to label file of UCMerced_LandUse dataset. Eg. ./labelfolder/LandUse_Multilabeled.xlsx

```

- `legacy_notebooks` : code before combining them to a single project. initial work was done almost individually
- `multilabels` : contains the label files for the images for UCMerced_LandUse dataset for the multilabel case.
- `cnn_nets.py` : Contains the CNN architectures. Currently has LENET and RESNET34
- `custom_dataloader.py` : Has functions to split data across multiple client in both IID and non-IID distributions. Also has function to check which classes are least correlated, etc
- `custom_loss_fns.py` : Has a couple of classes for loss function. One loss function is specific to FedProx federated algorithm. The other is just a wrapper to the Pytorch loss function. This is done to have a generic train function that supports custom loss functions.
- `CustomDataSet.py` : Inherits the abstract class `torch.utils.data.Dataset` and overrides `__len__` and `__getitem__` method. This custom dataset class supports Multilabel for each image.
- `federated_train_algorithms.py` : Has the federated training algorithms implemented in it. Supports
  1. FedAvg (Federated Averaging)
     > [Paper](https://arxiv.org/abs/1602.05629)
  2. FedProx
      >[Paper](https://arxiv.org/abs/1812.06127)
  3. Bulk Synchronous Processing (BSP)
      >[Paper](https://dl.acm.org/doi/10.1145/79173.79181)

- `FL_with_pytorch_only.ipynb` : The final notebook version from which the modular code was written from.
-  `train.py` : The file has the training loop. This training loop will be common for different federated algorithms.
- `requirements.txt` : Can be used directly with pip/conda to setup the required packages.
