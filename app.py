import os
from flask import Flask, render_template, request, jsonify
import joblib
from groq import Groq
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler

# NOTE: Do NOT set your GROQ_API_KEY in code.
# Instead, set the GROQ_API_KEY as an environment variable in your Render.com dashboard:
# - Go to your service > Environment > Add Environment Variable
# - Key: GROQ_API_KEY, Value: <your_actual_api_key>
# The Groq client will automatically use this environment variable.

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_BOT = Bot(token=TELEGRAM_BOT_TOKEN)
# We'll use a global dispatcher for all updates
telegram_dispatcher = Dispatcher(TELEGRAM_BOT, None, workers=0, use_context=True)
# In-memory user_data for session context per user
user_data = {}

def get_llama_reply(messages: list) -> str:
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return completion.choices[0].message.content

def get_deepseek_reply(messages: list) -> str:
    client = Groq()
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=messages
    )
    return completion_ds.choices[0].message.content

def predict_dbs(usdsgd: float) -> str:
    try:
        # Get absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'dbs.jl')
        
        # Log the path for debugging
        print(f"Looking for model at: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            return "Prediction model not found. Please check server logs."
            
        # Try to load the model
        print("Loading model...")
        dbs_model = joblib.load(model_path)
        
        # Make prediction
        print(f"Making prediction with USD/SGD rate: {usdsgd}")
        pred = dbs_model.predict([[usdsgd]])[0]
        # Convert numpy array value to Python float before formatting
        pred_float = float(pred)
        return f"Predicted DBS share price: {pred_float:.2f} SGD"
        
    except Exception as e:
        print(f"ERROR in predict_dbs: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error making prediction: {str(e)}"

# Telegram command handlers

def get_user_data(user_id):
    if user_id not in user_data:
        user_data[user_id] = {}
    return user_data[user_id]

def start(update, context):
    update.message.reply_text(
        "Welcome to GroqSeeker_Bot!\n\n" +
        "Use /llama <your question> to chat with LLAMA,\n" +
        "Use /deepseek <your question> to chat with Deepseek,\n" +
        "Use /predict <usdsgd> to predict DBS share price."
    )

def help_command(update, context):
    update.message.reply_text(
        "Commands:\n" +
        "/llama <question> - Chat with LLAMA AI\n" +
        "/deepseek <question> - Chat with Deepseek AI\n" +
        "/predict <usdsgd> - Predict DBS share price\n"
    )

def llama_command(update, context):
    user_id = update.effective_user.id
    q = ' '.join(context.args)
    udata = get_user_data(user_id)
    if 'llama_history' not in udata:
        udata['llama_history'] = []
    if not q:
        update.message.reply_text("Please provide a question after /llama.")
        return
    udata['llama_history'].append({"role": "user", "content": q})
    reply = get_llama_reply(udata['llama_history'])
    udata['llama_history'].append({"role": "assistant", "content": reply})
    update.message.reply_text(reply)

def deepseek_command(update, context):
    user_id = update.effective_user.id
    q = ' '.join(context.args)
    udata = get_user_data(user_id)
    if 'deepseek_history' not in udata:
        udata['deepseek_history'] = []
    if not q:
        update.message.reply_text("Please provide a question after /deepseek.")
        return
    udata['deepseek_history'].append({"role": "user", "content": q})
    reply = get_deepseek_reply(udata['deepseek_history'])
    udata['deepseek_history'].append({"role": "assistant", "content": reply})
    update.message.reply_text(reply)

def predict_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Debug print
    print(f"Predict command received from user {user_id}")
    print(f"Args: {context.args}")
    
    if not context.args:
        last_rate = udata.get('last_usdsgd')
        if last_rate:
            update.message.reply_text(f"Your last USD/SGD rate was: {last_rate}")
        else:
            update.message.reply_text("Please provide the USD/SGD rate after /predict.")
        return
    try:
        usdsgd = float(context.args[0])
        print(f"Valid USD/SGD rate provided: {usdsgd}")
        udata['last_usdsgd'] = usdsgd
        reply = predict_dbs(usdsgd)
    except ValueError as e:
        print(f"Invalid input: {context.args[0]} - {str(e)}")
        reply = "Invalid input. Please provide a valid number for USD/SGD."
    except Exception as e:
        print(f"Error in predict_command: {str(e)}")
        reply = f"Error processing prediction: {str(e)}"
    update.message.reply_text(reply)

def reset_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    udata.pop('llama_history', None)
    udata.pop('deepseek_history', None)
    update.message.reply_text("Your chat history has been reset.")

# Register handlers with the dispatcher
telegram_dispatcher.add_handler(CommandHandler("start", start))
telegram_dispatcher.add_handler(CommandHandler("help", help_command))
telegram_dispatcher.add_handler(CommandHandler("llama", llama_command))
telegram_dispatcher.add_handler(CommandHandler("deepseek", deepseek_command))
telegram_dispatcher.add_handler(CommandHandler("predict", predict_command))
telegram_dispatcher.add_handler(CommandHandler("reset", reset_command))

@app.route("/telegram_webhook", methods=["POST"])
def telegram_webhook():
    try:
        data = request.get_json(force=True)
        print("Received Telegram update:", data)  # Debug incoming updates
        
        # Check if we have a valid token
        if not TELEGRAM_BOT_TOKEN:
            print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set!")
            return jsonify(success=False, error="Bot token not configured"), 500
            
        update = Update.de_json(data, TELEGRAM_BOT)
        if update:
            print(f"Processing update ID: {update.update_id}, type: {'message' if update.message else 'callback_query' if update.callback_query else 'unknown'}")
            telegram_dispatcher.process_update(update)
        else:
            print("WARNING: Received invalid update format from Telegram")
            
        return jsonify(success=True)
    except Exception as e:
        print(f"ERROR in telegram_webhook: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify(success=False, error=str(e)), 500

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

