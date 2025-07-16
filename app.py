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

# Maximum tokens to allow in conversation history before truncating
MAX_TOKENS = 4000  # Conservative limit below Groq's 6000 token limit

def get_llama_reply(messages: list) -> str:
    try:
        client = Groq()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_str = str(e)
        print(f"Error in get_llama_reply: {error_str}")
        
        # Handle token limit errors
        if "413" in error_str and "Request too large" in error_str:
            # Conversation history too long
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."

def get_deepseek_reply(messages: list) -> str:
    try:
        client = Groq()
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages
        )
        return completion_ds.choices[0].message.content
    except Exception as e:
        error_str = str(e)
        print(f"Error in get_deepseek_reply: {error_str}")
        
        # Handle token limit errors
        if "413" in error_str and "Request too large" in error_str:
            # Conversation history too long
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."
        
        # Handle other API errors
        return f"⚠️ Error from Groq API: {error_str}"

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

def truncate_conversation(messages, max_tokens=MAX_TOKENS):
    """
    Automatically truncate conversation history to stay within token limits.
    Uses a simple heuristic of ~4 chars per token for estimation.
    """
    if not messages:
        return messages
        
    # Simple estimation: ~4 chars per token
    total_chars = sum(len(msg["content"]) for msg in messages)
    estimated_tokens = total_chars // 4
    
    # If we're under the limit, no need to truncate
    if estimated_tokens <= max_tokens:
        return messages
        
    print(f"Truncating conversation: {estimated_tokens} tokens (estimated) exceeds {max_tokens} limit")
    
    # Keep truncating from the beginning until we're under the limit
    # Always keep at least the most recent exchange (2 messages)
    while estimated_tokens > max_tokens and len(messages) > 2:
        # If first message is system, remove the second message instead
        if messages and messages[0]["role"] == "system":
            if len(messages) <= 2:  # Only system + 1 message left
                break
            removed = messages.pop(1)
        else:
            removed = messages.pop(0)
            
        estimated_tokens -= len(removed["content"]) // 4
        print(f"Removed message: {removed['role']} ({len(removed['content']) // 4} tokens)")
    
    # Add a system message indicating truncation if we removed messages
    if estimated_tokens > max_tokens:
        truncation_notice = {"role": "system", "content": "[Some earlier messages were removed to stay within token limits]"}
        if messages and messages[0]["role"] == "system":
            # Insert after existing system message
            messages.insert(1, truncation_notice)
        else:
            # Insert at beginning
            messages.insert(0, truncation_notice)
            
    return messages

def send_telegram_message(update, text):
    """Split long messages into smaller chunks to avoid Telegram's 4096 character limit"""
    # Maximum message length for Telegram
    MAX_MESSAGE_LENGTH = 4000  # Slightly less than 4096 to be safe
    
    # If message is short enough, send it as is
    if len(text) <= MAX_MESSAGE_LENGTH:
        update.message.reply_text(text)
        return
        
    # Split long message into chunks
    chunks = []
    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        chunks.append(text[i:i + MAX_MESSAGE_LENGTH])
    
    # Send each chunk as a separate message
    for i, chunk in enumerate(chunks):
        # Add part number if there are multiple chunks
        if len(chunks) > 1:
            prefix = f"Part {i+1}/{len(chunks)}:\n\n"
            update.message.reply_text(prefix + chunk)
        else:
            update.message.reply_text(chunk)

def start(update, context):
    send_telegram_message(update,
        "Welcome to GroqSeeker_Bot!\n\n" +
        "Use /llama <your question> to chat with LLAMA,\n" +
        "Use /deepseek <your question> to chat with Deepseek,\n" +
        "Use /predict <usdsgd> to predict DBS share price."
    )

def help_command(update, context):
    send_telegram_message(update,
        "Commands:\n" +
        "/llama <question> - Chat with LLAMA AI\n" +
        "/deepseek <question> - Chat with Deepseek AI\n" +
        "/predict <usdsgd> - Predict DBS share price\n"
    )

def llama_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    if 'llama_history' not in udata:
        udata['llama_history'] = []
    if not context.args:
        send_telegram_message(update, "Please provide a question after /llama.")
        return
        
    # Get the user's question from arguments
    q = ' '.join(context.args)
    print(f"LLAMA query from user {user_id}: {q}")
    
    # Add user message to history
    udata['llama_history'].append({"role": "user", "content": q})
    
    # Truncate conversation if needed before sending to API
    udata['llama_history'] = truncate_conversation(udata['llama_history'])
    
    # Get reply from LLAMA
    reply = get_llama_reply(udata['llama_history'])
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['llama_history'].append({"role": "assistant", "content": reply})
        
    send_telegram_message(update, reply)

def deepseek_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    if 'deepseek_history' not in udata:
        udata['deepseek_history'] = []
    if not context.args:
        send_telegram_message(update, "Please provide a question after /deepseek.")
        return
        
    # Get the user's question from arguments
    q = ' '.join(context.args)
    print(f"Deepseek query from user {user_id}: {q}")
    
    # Add user message to history
    udata['deepseek_history'].append({"role": "user", "content": q})
    
    # Truncate conversation if needed before sending to API
    udata['deepseek_history'] = truncate_conversation(udata['deepseek_history'])
    
    # Get reply from Deepseek
    reply = get_deepseek_reply(udata['deepseek_history'])
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['deepseek_history'].append({"role": "assistant", "content": reply})
        
    send_telegram_message(update, reply)

def predict_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Debug print
    print(f"Predict command received from user {user_id}")
    print(f"Args: {context.args}")
    
    if not context.args:
        last_rate = udata.get('last_usdsgd')
        if last_rate:
            send_telegram_message(update, f"Your last USD/SGD rate was: {last_rate}")
        else:
            send_telegram_message(update, "Please provide the USD/SGD rate after /predict.")
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
    send_telegram_message(update, "Your chat history has been reset.")

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

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/telegram')
def telegram_info():
    # Check webhook status
    webhook_status = "Active"
    try:
        # This is a placeholder - in a real implementation, you might want to
        # actually check with the Telegram API if the webhook is properly set
        pass
    except Exception as e:
        webhook_status = f"Error: {str(e)}"
    
    return render_template('telegram.html', 
                          status="GroqSeeker_Bot is ready to use in Telegram", 
                          webhook_status=webhook_status)

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

