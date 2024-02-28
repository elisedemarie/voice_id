import keras
from keras import Sequential, layers, activations
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

def create_model(input_shape):
    input_shape = (int(input_shape/3),3,)
    dr = 0.1
    model = Sequential()    
    model.add(layers.Conv1D(3, 8))
    model.add(layers.Dropout(rate=dr))
    model.add(layers.Conv1D(16, 4))
    model.add(layers.Dropout(rate=dr))
    model.add(layers.Conv1D(64, 4))
    model.add(layers.Dropout(rate=dr))
    model.add(layers.Conv1D(32, 4))
    model.add(layers.Dropout(rate=dr))
    model.add(layers.Conv1D(16, 4))
    model.add(layers.GlobalMaxPooling1D())
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
    X_train = tf.convert_to_tensor(X_train.reshape(X_train.shape[0], int(X_train.shape[1]/3), 3))
    X_val= tf.convert_to_tensor(X_val.reshape(X_val.shape[0], int(X_val.shape[1]/3), 3))

    y_train = tf.convert_to_tensor(y_train)
    y_train_binary = keras.utils.to_categorical(y_train, 2)

    y_val = tf.convert_to_tensor(y_val)
    y_val_binary = keras.utils.to_categorical(y_val, 2)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model_res= model.fit(X_train, y_train_binary, 
                validation_data=[X_val, y_val_binary],
                batch_size=16, 
                epochs=20,
                shuffle=True,
                verbose=verbose,
                callbacks=[early_stopping]
                )
    
    return model_res



def train_xgb(X_train, X_val, y_train, y_val, verbose=1):
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    if verbose:
        print(classification_report(y_val,pred))
    return clf