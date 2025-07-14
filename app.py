from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    try:
        # Get user input
        q = float(request.form.get("q"))
        username = request.form.get("username")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    try:
        # Get the input from the form and convert to float
        q = float(request.form.get("q"))

        # Load the trained model
        model = joblib.load("dbs.jl")

        # Make prediction and format the result
        pred_value = round(float(model.predict([[q]])[0]), 2)

        return render_template("prediction.html", r=pred_value)
    
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
