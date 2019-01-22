from flask import Flask, request, jsonify, render_template
import os
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('readme.html')

# Model development
import pandas as pd, math, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_excel('C:/Users/kelechi.emenike/Downloads/MEGA/iPyNB/plant_model.xlsx', skiprows=1)
data['Power Generated'] = (data['Power Generated']*1000).apply(math.log)
data['Fuel Consumed'] = data['Fuel Consumed'].apply(math.log)

df = data[['Power Generated', 'Fuel Consumed']]
X = df[['Power Generated']]
y = df['Fuel Consumed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)

import pickle
# Write pickle
try:
    with open("ML_model.pkl", "wb") as file_handler:
        pickle.dump(lr, file_handler)
    # Read pickle
    with open("ML_model.pkl","rb") as file_handler:
        loaded_pickle = pickle.load(file_handler)
except:
    pass

from sklearn.externals import joblib
joblib.dump(lr,"ML_model.pkl") # dump linear regression model object into pickle
joblib.dump(X_train, "X_train.pkl") # dump training set into pickle
joblib.dump(y_train, "y_train.pkl") # dump training labels into pickle

# Predict model with parameters
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json() # get data from json string, save as "data"
            powerGen = float(data["powerGen"]) # select the "powerGen" key from the data and save as float value "floatGen"
            powerGenLN = np.array(math.log(powerGen*24*1000)).reshape(-1,1) # convert to array

            lr = joblib.load("./ML_model.pkl")
        except ValueError:
            return jsonify("Please enter a number")

        return jsonify(math.exp(lr.predict(powerGenLN)[0]))
# Retrain model
@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            X_train = joblib.load("./X_train.pkl")
            y_train = joblib.load("./y_train.pkl")
            print(f"old X train shape: {X_train.shape}")
            print(f"old X train shape: {y_train.shape}")

            df = pd.read_json(data)
            df['Power Generated'] = (df['Power Generated']*1000).apply(math.log)
            df['Fuel Consumed'] = df['Fuel Consumed'].apply(math.log)
            df = df[['Power Generated', 'Fuel Consumed']]
            X = df[['Power Generated']]
            y = df['Fuel Consumed']

            new_X = pd.concat([X_train,X])
            new_y = pd.concat([y_train,y])
            print(new_X[:1])
            print(new_y[:1])
            print(f"new X train shape: {new_X.shape}")
            print(f"new X train shape: {new_y.shape}")

            new_lr = LinearRegression()
            new_lr.fit(new_X, new_y)

            os.remove("./ML_model.pkl")
            os.remove("./X_train.pkl")
            os.remove("./y_train.pkl")

            joblib.dump(new_lr, "ML_model.pkl")
            joblib.dump(new_X, "X_train.pkl")
            joblib.dump(new_y, "y_train.pkl")

            lr = joblib.load("./ML_model.pkl")
        except ValueError as e:
            return jsonify("Error while retraining - {}.format(e)")

        return jsonify("Retrained model successfully")
# Get parameters
@app.route("/getParameters",methods=['GET'])
def getParameters():
    if request.method == 'GET':
        try:
            lr = joblib.load("./ML_model.pkl")
            X_train = joblib.load("./X_train.pkl")
            y_train = joblib.load("./y_train.pkl")

            return jsonify({"score(%)":r2_score(y_train, lr.predict(X_train)),\
                            "RegressionCoefficients":lr.coef_.tolist(),\
                            "RegressionIntercept":lr.intercept_,\
                            })
        except (ValueError, TypeError) as e:
            return jsonify("Error when getting details - {}".format(e))

if __name__ == "__main__":
    app.run(debug=True)