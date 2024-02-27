from . import util
from . import preprocess
from . import engineer
from . import classify
from . import evaluate
import pandas as pd
import keras

def prepare():

    ## load data
    config = util.loadConfig()
    data = pd.read_csv(config["valided_tsv"])
    # select relevant columns
    data = data[["client_id", "path", "gender"]]

    ## split data
    output = preprocess.split(data)
    # save output
    util.saveObject(output, "raw_output.pkl")
    train_path, train_gender = output[0]
    val_path, val_gender = output[1]
    test_path, test_gender = output[2]

    ## feature engineer
    train_content = engineer.get_xy(train_path, train_gender)
    val_content = engineer.get_xy(val_path, val_gender)
    test_content = engineer.get_xy(test_path, test_gender)
    engineered_output = (train_content, val_content, test_content)
    util.saveObject(engineered_output, "engineered_output.pkl")

    ## normalise
    train_normalised, train_mean, train_std = preprocess.normalise(train_content[0])
    val_normalised = preprocess.normalise(val_content[0], train_mean, train_std)
    test_normalised = preprocess.normalise(test_content[0], train_mean, train_std)
    # save noramlised constanted
    util.saveObject((train_mean, train_std), "normalise_objects.pkl")

    ## encode labels
    y_train_encoded = [0 if y=="male" else 1 for y in train_content[1]]
    y_val_encoded = [0 if y=="male" else 1 for y in val_content[1]]
    y_test_encoded = [0 if y=="male" else 1 for y in test_content[1]]

    # save data
    all_data = (train_normalised, val_normalised, test_normalised, y_train_encoded, y_val_encoded, y_test_encoded)
    util.saveObject(all_data, "all_data.pkl")

    return


def train(clf="xgb"):
    config = util.loadConfig()

    # load data for trianing
    X_train, X_val, _, y_train, y_val, _ = util.loadObject("all_data")

    if clf == "dl":
        ## train
        model = classify.create_model(X_train.shape[1])
        model_res = classify.train_dl_model(model, X_train, X_val, 
                                        y_train, y_val)
        # save the model and results
        model.save(config["output_dir"])
        util.saveObject(model_res, "dl_training_results.pkl")

    elif clf == "xgb":
        #train
        clf = classify.train_xgb(X_train, X_val, y_train, y_val, verbose=1)
        # save output
        util.saveObject(clf, "xgb_model.pkl")
    
    return


def evaluate(set="val", model="xgb"):
    config = util.loadConfig()
    _, X_val, X_test, _, y_val, y_test = util.loadObject("all_data.pkl")
    _, val_content, test_content = util.loadObject("engineered_output.pkl")

    # get the correct object
    if set == "test":
        X = X_test
        y = y_test
        users = test_content[3]
    else:
        X = X_val
        y = y_val
        users = val_content[3]
    
    if model=="dl":
        model = keras.models.load_model(config["output_dir"])
    else:
        model = util.loadObject("xgb_model.pkl")

    report = evaluate.evaluate(X, y, model, users)
    print(report)

