import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse

# parser
parser = argparse.ArgumentParser(description='Visualization of Results')
parser.add_argument('--data_dir', '-d', type=str, required=True,
                    help='Specify path to csv file contain results (not (!!) labeled 2), e.g. name_of_res.csv')
args = parser.parse_args()

def read_csv(path):
    loss = []
    acc = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            loss.append(float(row[0]))
            acc.append(float(row[1]))
        return loss, acc

def visualize(path: str):
    if path[-3:] != "csv":
        print("You need to reference a CSV file, e.g. myfile.csv")
        exit()

    loss, acc = read_csv(path)

    # plot
    loss_hist = np.array(loss)
    acc_hist = np.array(acc)

    # get parameters
    path = path[:-4]
    chars = path.split("_")
    fed_alg = chars[0]
    model = chars[3]
    num_clients = chars[5]
    skew = chars[7]
    title = "Alg: " + fed_alg + "; Model: " +  model + "; Number of clients: " + num_clients + "; skewness: 0." + skew
    
    plt.title(title)
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.plot(range(1,len(loss)+1),loss_hist,label="Loss")
    plt.plot(range(1,len(acc)+1),acc_hist,label="Accuracy")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, len(acc)+1, 1.0))
    plt.legend()
    plt.show(block=True)

visualize(args.data_dir)