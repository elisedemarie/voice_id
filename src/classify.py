import keras
from keras import Sequential, layers, activations
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import classification_report

def create_model(input_shape):
    input_shape = (input_shape,)
    dr = 0.1
    model = Sequential()
    model.add(layers.Dense(128, input_shape=input_shape, activation="relu"))
    model.add(layers.Dropout(rate=dr))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(rate=dr))
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dropout(rate=dr))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(rate=dr))
    model.add(layers.Dense(2, activation="softmax"))

    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, 
                loss='mse',
                metrics="binary_accuracy")
    return model

def train_dl_model(model, X_train, X_val, y_train, y_val, verbose=1):
    X_train = tf.convert_to_tensor(X_train)
    X_val= tf.convert_to_tensor(X_val)

    y_train = tf.convert_to_tensor(y_train)
    y_train_binary = keras.utils.to_categorical(y_train, 2)

    y_val = tf.convert_to_tensor(y_val)
    y_val_binary = keras.utils.to_categorical(y_val, 2)

    model_res= model.fit(X_train, y_train_binary, 
                validation_data=[X_val, y_val_binary],
                batch_size=16, 
                epochs=20,
                shuffle=True,
                verbose=verbose)
    
    return model_res

def train_xgb(X_train, X_val, y_train, y_val, verbose=1):
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    if verbose:
        print(classification_report(y_val,pred))
    return clf