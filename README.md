# DBS Prediction Web Service

This is a Flask-based web application for making predictions using a trained model (`dbs.jl`) and interacting with the Groq API (LLM). The app features a simple web interface for users to input data, get model predictions, and interact with an LLM via the Groq API.

## Features

- **Prediction Endpoint:** Enter a value and receive a model prediction.
- **LLM Integration:** Ask questions and get responses from the Groq Llama model.
- **Ready for Cloud Deployment:** Designed for easy deployment on Render.com.

## Requirements

- Python 3.11
- gunicorn
- flask
- joblib
- scikit-learn
- groq

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd dbs_pred
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your trained model file (`dbs.jl`) in the project root.

4. Set your Groq API key as an environment variable:
    ```bash
    export GROQ_API_KEY=your_groq_api_key
    ```

5. Run the app:
    ```bash
    python app.py
    ```

6. Visit `http://localhost:5000` in your browser.

### Deployment on Render.com

1. **Connect your repository** to Render.com and create a new Web Service.
2. **Set the environment variable** in the Render dashboard:
    - Key: `GROQ_API_KEY`
    - Value: _your actual Groq API key_
3. **Ensure your `dbs.jl` model file** is present in the root directory of the repo.
4. Render will install dependencies from `requirements.txt` and run your app.

## Project Structure

```
dbs_pred/
├── app.py
├── dbs.jl
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── main.html
│   ├── llama.html
│   ├── llama_reply.html
│   ├── dbs.html
│   └── prediction.html
└── README.md
```

## API Endpoints

- `/` — Home page
- `/main` — Main interface
- `/llama` — LLM input page
- `/llama_reply` — LLM response page
- `/dbs` — DBS info page
- `/prediction` — Model prediction page

## Security

- **Do NOT hardcode your API keys** in the codebase.
- Set `GROQ_API_KEY` as an environment variable, especially for cloud deployments.

## License

[MIT](LICENSE) (or your preferred license)