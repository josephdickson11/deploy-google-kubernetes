from flask import Flask, render_template, url_for, redirect, jsonify, request
from sklearn.pipeline import  Pipeline
import pandas as pd
from joblib import load
import numpy as np

# name of flask app
app = Flask(__name__)

# create model object
model = load('HOR_model_pipeline.pkl')
pedictor_col = ['can_off','can_off_sta','can_off_dis','can_inc_cha_ope_sea','net_ope_exp','can_par_aff','campaign_duration']


# run the server
# this line of code is not needed when running the flask run command in terminal
# app.run(debug= True)


# defining static app routes
@app.route("/")  # decorator
def home():  # route handler function
    # returning a response
    return render_template('index.html')


# create prediction function
@app.route("/predict", methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=pedictor_col)
    prediction = model.predict(data_unseen)
    if prediction == "Y":
        prediction = "you will win the election"
    else:
        prediction = "you will not win the election given these circumstances"
        return  prediction
    return render_template('index.html', pred=prediction)
