from CustomDataSet import CustomDataSet
from torchvision import transforms
import numpy as np
import random
import torch


def uncor_selecter(df_label, nr_label=4, min_img=300):
    """retrun a list with the least correlated labels """
    image_perlabel = np.sum(df_label[:, 1:], axis=0)
    biggest_label = np.where(np.any([image_perlabel > min_img], axis=0))[0]
    print(biggest_label, image_perlabel[biggest_label])

    selected_list = []
    allcor_lost = np.array([0, 0, 0])
    for i in range(0, len(biggest_label) - 1):
        it = biggest_label[i]
        for j in range(i + 1, len(biggest_label)):
            jt = biggest_label[j]

            colxor = np.sum(np.logical_xor(df_label[:, it].astype(bool), df_label[:, jt].astype(
                bool))) - np.sum(np.logical_and(df_label[:, it], df_label[:, jt]))
            allcor_lost = np.vstack((allcor_lost, np.array([colxor, it, jt])))
    sorted_list = allcor_lost[allcor_lost[:, 0].argsort()]
    selected_list.append(sorted_list[-1, 1])
    selected_list.append(sorted_list[-1, 2])
    #print(sorted_list, selected_list)

    while len(selected_list) < nr_label:
        biggest_label = np.setdiff1d(biggest_label, np.array(selected_list))
        largestxor = 0
        largestind = 0
        for i in biggest_label:
            overall_xor = 0
            for j in (selected_list):
                overall_xor += np.sum(np.logical_xor(df_label[:, i].astype(bool), df_label[:, j].astype(
                    bool))) - np.sum(np.logical_and(df_label[:, i], df_label[:, j]))

            if overall_xor >= largestxor:
                largestxor = overall_xor
                largestind = i

        selected_list.append(largestind)

    return selected_list


def sampler_split_for_client(cdata, idxs, df_label, nr_client=4, minimum_skew_percentage=.4):
    selected_labels = uncor_selecter(df_label, nr_client, 500)

    splitlists = []
    for sb in selected_labels:
        splitlists.append([])

    for i in idxs:
        nplabel = cdata.__getlabel__(i)
        #nplabel = label.numpy()

        if np.any(nplabel[selected_labels] == 1):
            if random.random() < minimum_skew_percentage:

                flip = np.random.randint(np.sum(nplabel[selected_labels] == 1))
                mask = np.where(nplabel[selected_labels] == 1)[0][flip]
                splitlists[mask].append(i)

            else:
                flip = np.random.randint(nr_client)
                splitlists[flip].append(i)

        else:
            flip = np.random.randint(nr_client)
            splitlists[flip].append(i)

    for alist in splitlists:
        print(len(alist))
    return splitlists


def load_split_train_test(datadir, labelmat, client_nr, skewness_percent, valid_size=.2):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = CustomDataSet(
        datadir, transform=train_transforms, labelmat=labelmat)
    test_data = CustomDataSet(
        datadir, transform=train_transforms, labelmat=labelmat)
    print(train_data.__len__())
    indices = list(range(train_data.__len__()))
    split = int(np.floor(valid_size * train_data.__len__()))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]

    lists = sampler_split_for_client(
        train_data, train_idx, labelmat, client_nr, skewness_percent/100)

    test_sampler = SubsetRandomSampler(test_idx)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              sampler=test_sampler, batch_size=4)

    test_loader_dict = {'data': test_loader, 'size': len(test_sampler)}

    dataloaders = []
    for client_sampler in lists:
        train_sampler = SubsetRandomSampler(client_sampler)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=4
        )
        dataloaders.append({'data': train_loader, 'size': len(client_sampler)})

    return dataloaders, test_loader_dict, len(train_idx)
