from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print(request.form)
    age = int(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)

    return render_template(
    "index.html",
    prediction_text=f"₹ {prediction[0]:,.2f}"
)



if __name__ == "__main__":
    app.run(debug=True)