from . import preprocess
from . import util
from . import engineer
from sklearn.cluster import KMeans
import tensorflow as tf
import keras
import pickle
import numpy as np 
import os


def predict_data(fp, model="xgb"):
    config = util.loadConfig()
    # process data
    filename = os.path.basename(fp)
    dir = os.path.dirname(fp) +"/"
    X = engineer.split_encode_mp3(filename, dir)

    # normalise
    train_mean, train_std = util.loadObject("normalise_objects.pkl")

    X = preprocess.normalise(X, train_mean, train_std)

    if model=="cnn":
        model = keras.models.load_model(config["output_dir"])
        X = tf.convert_to_tensor(X)
        pred = model.predict(X, verbose=0)
        pred = np.mean(np.argmax(pred, axis=1))
    else:
        model = util.loadObject("xgb_model.pkl")
        pred = model.predict(X)
        pred = np.mean(pred)
    
    boundary = 0.75

    if pred <= boundary:
        adj_pred = (pred/boundary)/2
    elif pred > boundary:
        adj_pred = ((pred-boundary)/(1-boundary))/2+0.5

    array = ["-"]*11
    idx = min(int(round(adj_pred,1)*11), 10)
    array[idx] = "+"
    output = "M|"+"".join(array)+"|F"
    if pred <= boundary:
        guess = "male"
    elif pred > boundary:
        guess = "female"
    
    print(output)
    print(guess, round(pred, 2))
    return output, pred, guess

