import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import pickle

app = Flask(__name__)

if not os.path.exists("model.pkl"):
    raise FileNotFoundError("❌ model.pkl not found. Please run train_model.py first.")

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)

