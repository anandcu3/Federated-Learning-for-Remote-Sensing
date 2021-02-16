#Sanity Check run : 1 EPOCH Skew 40 resnet34 Fedprox
python main.py -c resnet34 --vs 0.3 -e 1 -n 100 -s 40 -f FedProx -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 resnet34 Fedprox 1 ce 100 clients
python main.py -c resnet34 --vs 0.3 -e 25 -n 100 -s 40 -f FedProx -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 alexnet Fedprox 1 ce 100 clients
python main.py -c alexnet --vs 0.3 -e 25 -n 100 -s 40 -f FedProx -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet Fedprox 1 ce 100 clients
python main.py -c lenet --vs 0.3 -e 25 -n 100 -s 40 -f FedProx -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 resnet34 FedAvg 1 ce 100 clients
python main.py -c resnet34 --vs 0.3 -e 25 -n 100 -s 40 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 alexnet FedAvg 1 ce 100 clients
python main.py -c alexnet --vs 0.3 -e 25 -n 100 -s 40 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet FedAvg 1 ce 100 clients
python main.py -c lenet --vs 0.3 -e 25 -n 100 -s 40 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 resnet34 FedAvg 1 ce 100 clients
python main.py -c resnet34 --vs 0.3 -e 25 -n 100 -s 0 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 alexnet FedAvg 1 ce 100 clients
python main.py -c alexnet --vs 0.3 -e 25 -n 100 -s 0 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet FedAvg 1 ce 100 clients
python main.py -c lenet --vs 0.3 -e 25 -n 100 -s 0 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 resnet34 Fedprox 1 ce 100 clients
python main.py -c resnet34 --vs 0.3 -e 25 -n 200 -s 40 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 alexnet Fedprox 1 ce 100 clients
python main.py -c alexnet --vs 0.3 -e 25 -n 200 -s 40 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet Fedprox 1 ce 100 clients
python main.py -c lenet --vs 0.3 -e 25 -n 200 -s 40 -f FedAvg -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 20 resnet34 Fedprox
#python main.py -c alexnet -e 25 -n 100 -s 20 -f FedProx -ce 5 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 alexnet Fedprox
#python main.py -c alexnet -n 3 -s 40 -f FedProx -ce 5 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 20 alexnet Fedprox
#python main.py -c alexnet -n 3 -s 20 -f FedProx -ce 5 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet Fedprox
#python main.py -c lenet -n 3 -s 40 -f FedProx -ce 5 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 20 lenet Fedprox
#python main.py -c lenet -n 3 -s 20 -f FedProx -ce 5 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 resnet34 Fedprox ce 1
#python main.py -c resnet34 -n 3 -s 20 -f FedProx -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet Fedprox ce 1
#python main.py -c lenet -n 3 -s 20 -f FedProx -ce 1 -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 lenet BSP ce 1 25 epochs
#python main.py -e 25 -c lenet -n 3 -s 40 -f BSP -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 resnet34 BSP ce 1 25 epochs
#python main.py -e 25 -c resnet34 -n 3 -s 40 -f BSP -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx

#Skew 40 alexnet BSP ce 1 25 epochs
#python main.py -e 25 -c alexnet -n 3 -s 40 -f BSP -d UCMerced_LandUse_augment/Images --multilabel_excelfilepath UCMerced_LandUse_augment/new_LandUse_Multilabeled.xlsx
