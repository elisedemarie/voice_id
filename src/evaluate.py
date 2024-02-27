import numpy as np
from sklearn.metrics import classification_report

def evaluate(X_val, y_val, model, users, boundary=0.25):
    # make predictions on the validation set
    pred = model.predict(X_val)

    ## group by the user
    user_data = {x:{"data":[], "gender":None} for x in users}
    # iter through users
    for i,x in enumerate(users):
        user_data[x]["data"].append(pred[i])
        user_data[x]['gender']=y_val[i]

    # make a user prediction
    boundary = boundary
    for x in user_data:
        data = user_data[x]["data"]
        data = np.array(data)
        if len(data.shape)==2:
            pred = np.mean(np.argmax(data, axis=1))
        else:
            pred = np.mean(data)
        if pred >= boundary:
            guess = 1
        else:
            guess = 0
        user_data[x]["prediction"] = guess

    # evaluate
    obs = []
    pred = []
    for x in user_data:
        pred.append(user_data[x]["prediction"])
        obs.append(user_data[x]["gender"])

    return classification_report(obs, pred), user_data