from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        username = request.form.get("username")
        if not username:
            return render_template("index.html", error="Please enter your name.")
        return render_template("main.html", username=username)
    else:
        return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            q = float(request.form.get("q"))
            username = request.form.get("username")
            model = joblib.load("dbs.jl")
            pred_value = round(float(model.predict([[q]])[0]), 2)
            return render_template("prediction.html", r=pred_value, username=username)
        except Exception as e:
            return f"Error: {e}", 400
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
