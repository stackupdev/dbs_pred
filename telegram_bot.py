import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from groq import Groq
import joblib

# Load DBS model (if needed for prediction)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dbs.jl')
dbs_model = None
if os.path.exists(MODEL_PATH):
    dbs_model = joblib.load(MODEL_PATH)

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
    if dbs_model is None:
        return "Prediction model not loaded."
    pred = dbs_model.predict([[usdsgd]])[0]
    return f"Predicted DBS share price: {pred:.2f} SGD"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to GroqSeeker_Bot!\n\n" +
        "Use /llama <your question> to chat with LLAMA,\n" +
        "Use /deepseek <your question> to chat with Deepseek,\n" +
        "Use /predict <usdsgd> to predict DBS share price."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n" +
        "/llama <question> - Chat with LLAMA AI\n" +
        "/deepseek <question> - Chat with Deepseek AI\n" +
        "/predict <usdsgd> - Predict DBS share price\n"
    )

async def llama_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = ' '.join(context.args)
    if 'llama_history' not in context.user_data:
        context.user_data['llama_history'] = []
    if not q:
        await update.message.reply_text("Please provide a question after /llama.")
        return
    # Add user message to history
    context.user_data['llama_history'].append({"role": "user", "content": q})
    # Get reply using full history
    reply = get_llama_reply(context.user_data['llama_history'])
    # Add assistant reply to history
    context.user_data['llama_history'].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)

async def deepseek_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = ' '.join(context.args)
    if 'deepseek_history' not in context.user_data:
        context.user_data['deepseek_history'] = []
    if not q:
        await update.message.reply_text("Please provide a question after /deepseek.")
        return
    # Add user message to history
    context.user_data['deepseek_history'].append({"role": "user", "content": q})
    # Get reply using full history
    reply = get_deepseek_reply(context.user_data['deepseek_history'])
    # Add assistant reply to history
    context.user_data['deepseek_history'].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        last_rate = context.user_data.get('last_usdsgd')
        if last_rate:
            await update.message.reply_text(f"Your last USD/SGD rate was: {last_rate}")
        else:
            await update.message.reply_text("Please provide the USD/SGD rate after /predict.")
        return
    try:
        usdsgd = float(context.args[0])
        context.user_data['last_usdsgd'] = usdsgd
        reply = predict_dbs(usdsgd)
    except Exception:
        reply = "Invalid input. Please provide a valid number for USD/SGD."
    await update.message.reply_text(reply)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop('llama_history', None)
    context.user_data.pop('deepseek_history', None)
    await update.message.reply_text("Your chat history has been reset.")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"Echoing message from {update.effective_user.id}: {update.message.text}")
    await update.message.reply_text(f"You said: {update.message.text}")

def main():
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        return
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("llama", llama_command))
    app.add_handler(CommandHandler("deepseek", deepseek_command))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    print("GroqSeeker_Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
