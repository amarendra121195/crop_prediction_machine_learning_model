# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)


from flask import Flask, render_template, request
import joblib  # ✅ required to load your model
import numpy as np

app = Flask(__name__)
model = joblib.load('crop_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    try:
        values = [float(request.form[f]) for f in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        pred = model.predict([values])
        prediction = pred[0]
    except Exception as e:
        prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

