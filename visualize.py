import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse
from PIL import Image

# parser
parser = argparse.ArgumentParser(description='Visualization of Results')
parser.add_argument('--type', '-t', type=str, required=True,
                    help='Existing types: \'ACC_F1\' | \'ACC_MULTI\'')
parser.add_argument('--data_dir', '-d', nargs='+', required=True,
                    help='Specify path to csv file contain results, e.g. \'name_of_res1.csv\' \'name_of_res2.csv\'')
args = parser.parse_args()

# VISUALIZE LOSS / ACCURACY
#redo
def read_csv(path):
    res_data = np.genfromtxt(path, delimiter=',')
    #loss = res_data[0,:]
    acc = res_data[1,:]
    f1 = res_data[2,:]
    return f1, acc

def visualize_acc_f1(path: str):
    if path[-3:] != "csv":
        print("You need to reference a CSV file, e.g. myfile.csv")
        exit()

    f1, acc = read_csv(path)

    # plot
    f1_hist = np.array(f1)
    acc_hist = np.array(acc)

    # get parameters
    # FedProx_acc&loss_for_lenet_with_8_clients_20_smallskew_False_clientsepox_5_vsplit_0.2_lr_0.001_cfraction_0.8_bs_1_1613863528.4838111
    path = path[:-4]
    chars = path.split("_")
    fed_alg = chars[0]
    model = chars[3]
    num_clients = chars[5]
    skew = chars[7]
    title = "Alg: " + fed_alg + "; Model: " +  model + "; Number of clients: " + num_clients + "; skewness: " + skew + "%"
    
    plt.title(title)
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy/F1-Score")
    plt.plot(range(1,len(f1)+1),f1_hist,label="F1-Score")
    plt.plot(range(1,len(acc)+1),acc_hist,label="Accuracy")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, len(acc)+1, 1.0))
    plt.legend()
    plt.show(block=True)

# PLOT ACC OF MULTIPLE TRAININGS
def get_acc_from_csv(paths):
    accs = {}
    for path in paths:
        # get data
        res_data = np.genfromtxt(path, delimiter=',')

        # get title
        file_name = path[:-4]
        chars = file_name.split("_")
        fed_alg = chars[0]
        model = chars[3]
        num_clients = chars[5]
        skew = chars[7]
        title = fed_alg + " with " +  model + " for " + num_clients + " clients and " + skew + "%" + " skewness"

        # add to list
        accs[title] = res_data[1,:]

    return accs

def plot_acc_curves(acclist, title):
    eporange = np.arange(len(list(acclist.values())[0]))
    
    plt.figure(figsize=(10, 8))
    plt.suptitle(title,fontsize=20)
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Accuracy',fontsize=14)
    plt.ylim(bottom=.6)
    for al in acclist:
        plt.plot(eporange, acclist[al], 'o-', label=str(al))
    
    plt.legend(loc='lower right',fontsize=14)
    plt.show(block=True)

# CONTROLLER
if args.type != "ACC_F1" and args.type != "ACC_MULTI": # and args.type != "ACC_MULTI_BARS":
    print("Please set the plotting type to one of the following: \'LOSS_ACC\' | \'ACC_MULTI\''")
    exit()

if args.type == "ACC_F1":
    path = args.data_dir[0]
    visualize_acc_f1(path)

if args.type == "ACC_MULTI":
    accs = get_acc_from_csv(args.data_dir)
    plot_acc_curves(accs, args.title)
