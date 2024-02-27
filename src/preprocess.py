from sklearn.model_selection import train_test_split
import numpy as np


def split(data):
    # get rid of rows without gender
    data = data[~data["gender"].isnull()]
    # create a unique list of client IDs
    # this avoids data leakage
    client_ids = list(set(list(data.client_id)))
    # split the client IDs
    train, test = train_test_split(client_ids, test_size=0.15, random_state=0)
    train, val = train_test_split(train, test_size=0.15, random_state=0)
    # get the rows by client ID
    train = data[data.client_id.isin(train)]
    test = data[data.client_id.isin(test)]
    val = data[data.client_id.isin(val)]
    # pull out path and gender for each set
    train_path = list(train.path)
    train_gender = list(train.gender)
    val_path = list(val.path)
    val_gender = list(val.gender)
    test_path = list(test.path)
    test_gender = list(test.gender)
    # return
    return (train_path, train_gender), (val_path, val_gender), (test_path, test_gender)


# a function to normalise the data
def normalise(data, train_mean=None, train_std=None):
    # create a flag for if this is the training set
    train_set = False
    if not train_mean and not train_std:
        train_mean = []
        train_std = []
        train_set = True
    res = []
    # iterate through features
    for i, x in enumerate(np.array(data).T):
        # skip features that should not be normalised
        if i in range(0, 18) or i in range(54, 200):
            # if this is the training set
            if train_set:
                # find the mean and std
                mean = np.mean(x)
                std = np.std(x)
                # add to array
                train_mean.append(mean)
                train_std.append(std)
            # normalise result
            temp = (x-train_mean[i])/train_std[i]
        else:
            temp = x
            if train_set:
                train_mean.append(0)
                train_std.append(0)
        res.append(temp)
    res = np.array(res).T
    if train_set:
        return res, train_mean, train_std
    else:
        return res