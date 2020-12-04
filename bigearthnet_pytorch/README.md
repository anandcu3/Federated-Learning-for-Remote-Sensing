# BigEarthNet Deep Learning Models With PyTorch
This repository contains code to use deep learning models, pre-trained on the [BigEarthNet archive](http://bigearth.net/) with PyTorch, to train new models, and to evaluate pre-trained models. Note that in addition to original BigEarthNet labels (will be called `original` in the code), there is a new class nomenclature (will be called `BigEarthNet-19` in the code). This repository is compatible with both options.

* For original BigEarthNet labels, it is highly recommended to first check the [BigEarthNet Deep Learning Models repository](https://gitlab.tu-berlin.de/rsim/bigearthnet-models). 

* For BigEarthNet-19 labels, it is highly recommended to first check the [BigEarthNet-19 Deep Learning Models repository](https://gitlab.tu-berlin.de/rsim/bigearthnet-19-models).


## Prerequisites
* The python package requirement for running the code is in `requirements.txt`
* For installing `gdal`, it is recommended to install it from `conda`, i.e., `conda install -c conda-forge gdal`
* The `prep_splits.py` script from [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-models/blob/master/prep_splits.py) for original labels and the `prep_splits_BigEarthNet-19.py` script from [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-19-models/blob/master/prep_splits.py) for BigEarthNet-19 labels generate `LMDB` files for train, validation and test sets from the BigEarthNet archive. To train or evaluate any model, required LMDB files should be first prepared. 
  
## Training
The script `train/main.py` is for training the CNN models. This file excepts the following parameters:

* `--bigEarthLMDBPth`: LMDB file path previously created for the BigEarthNet
* `-b`: Batch size used during training
* `--epochs`: The number of epochs for the training
* `--lr`: The initial learning rate
* `--resume`: The file path of a pre-trained model snapshot (i.e., checkpoint)
* `--num_workers`: The number of workers for data loading
* `--model`: The name of the CNN model to be trained
* `--BigEarthNet19`: A flag to indicate whether to use the new class nomenclature (BigEarthNet-19) during training
* `--train_csv`: The path to the csv file of train patches
* `--val_csv`: The path to the csv file of val patches
* `--test_csv`: The path to the csv file of test patches


## Evaluation
The script `test/test.py` is for the evaluation of the pre-trained CNN models.
* `--bigEarthLMDBPth`: LMDB file path previously created for the BigEarthNet
* `-b`: Batch size used during evaluation
* `--checkpoint_pth`: The file path of a pre-trained model snapshot (i.e., checkpoint)
* `--num_workers`: The number of workers for data loading
* `--model`: The name of the CNN model to be evaluated
* `--BigEarthNet19`: A flag to indicate whether to use the new class nomenclature (BigEarthNet-19) during evaluation
* `--train_csv`: The path to the csv file of train patches
* `--val_csv`: The path to the csv file of val patches
* `--test_csv`: The path to the csv file of test patches

Authors
-------

**Jian Kang**

[**Gencer Sümbül**](http://www.user.tu-berlin.de/gencersumbul/)

Maintained by
-------

[**Mahdyar Ravanbakhsh**](https://www.rsim.tu-berlin.de/menue/team/dr_sayyed_mahdyar_ravanbakhsh/)

# License
The code in this repository to facilitate the use of the archive is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2019 The BigEarthNet Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```