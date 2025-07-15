from flask import Flask, render_template, request
import joblib
from groq import Groq

# NOTE: Do NOT set your GROQ_API_KEY in code.
# Instead, set the GROQ_API_KEY as an environment variable in your Render.com dashboard:
# - Go to your service > Environment > Add Environment Variable
# - Key: GROQ_API_KEY, Value: <your_actual_api_key>
# The Groq client will automatically use this environment variable.

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    q = request.form.get("q")
    # db
    return(render_template("main.html"))

@app.route("/deepseek",methods=["GET","POST"])
def deepseek():
    return render_template("deepseek.html")

@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    q = request.form.get("q")
    client = Groq()
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
           }
        ]
    )
    return(render_template("deepseek_reply.html", r=completion_ds.choices[0].message.content))

@app.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@app.route("/llama_reply",methods=["GET","POST"])
def llama_reply():
    q = request.form.get("q")
    # load model
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content))

@app.route("/dbs",methods=["GET","POST"])
def dbs():
    return(render_template("dbs.html"))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    q = float(request.form.get("q"))
    # Load the trained model
    model = joblib.load("dbs.jl")
    pred_value = round(float(model.predict([[q]])[0]), 2)
    return render_template("prediction.html", r=pred_value)

if __name__ == "__main__":
    app.run()

