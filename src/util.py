import json
import pickle

def readConfig():
    config = json.load(open("config.json", "rb"))
    return config


def loadObject(name, location=None):
    # if unspecified go to output location
    if not location:
        location = readConfig()["output_dir"]
    with open(location+name, "rb") as fp:
        obj = pickle.load(fp)
    return obj


def saveObject(object, name, location=None):
    # if unspecified go to output location
    if not location:
        location = readConfig()["output_dir"]
    with open(location+name, "wb") as fp:
        pickle.dump(object, fp)
    return