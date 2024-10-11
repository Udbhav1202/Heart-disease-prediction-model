from flask import Flask, request, render_template  
import joblib  
import sklearn  
import pickle, gzip  
import pandas as pd  
import numpy as np  
import os

print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

app = Flask(__name__)  
model = joblib.load("Heart_Disease_Prediction.pkl")
@app.route('/')  
def home():  
   return render_template("index.html")  
@app.route("/predict", methods=["POST"])  
def predict():  
    # Collect and convert form data
    age = int(request.form["age"])  
    sex = int(request.form["sex"])  
    trestbps = int(request.form["trestbps"])  
    chol = int(request.form["chol"])  
    oldpeak = float(request.form["oldpeak"])  
    thalach = int(request.form["thalach"])  
    fbs = int(request.form["fbs"])  
    exang = int(request.form["exang"])  
    slope = int(request.form["slope"])  
    cp = int(request.form["cp"])  
    thal = int(request.form["thal"])  
    ca = int(request.form["ca"])  
    restecg = int(request.form["restecg"])  

    # Prepare the input array
    arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])  

    # Predict
    pred = model.predict(arr)  
    res_val = "NO HEART PROBLEM" if pred == 0 else "HEART PROBLEM"  
   
    return render_template('index.html', prediction_text='PATIENT HAS {}'.format(res_val))  
 
if __name__ == "__main__":  
   app.run(debug=True)