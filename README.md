# AquaCheckAI

Predicts whether a water sample is safe to drink from 9 chemical measurements.
Machine learning in Python, website in HTML/CSS/JavaScript, plus an AI assistant (AquaBot).

```
aquacheckai/
  backend/
    app.py                  Flask server: website + /api/predict, /api/chat, /api/meta
    train_model.py          YOU run this to train the model
    chatbot.py              AquaBot (offline knowledge base, optional LLM mode)
    generate_sample_data.py makes a synthetic CSV if you have no dataset yet
    data/water_potability.csv
    model/                  model.joblib + meta.json (written by train_model.py)
  frontend/                 index.html, style.css, app.js
```

## Run it

```bash
cd backend
pip install -r requirements.txt
python train_model.py      # trains the model (see below)
python app.py              # open http://127.0.0.1:5000
```

## Train it yourself

1. Download the Kaggle "Water Potability" dataset and save it as `backend/data/water_potability.csv`
   (columns: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon,
   Trihalomethanes, Turbidity, Potability).
   The file currently in that folder is **synthetic sample data** so the app runs out of the box.
   Replace it with the real one before reporting any numbers.
2. Run `python train_model.py`. It fills missing values with the median, splits 80/20,
   compares Logistic Regression and Random Forest, and saves the better one by F1 score.
3. Restart `python app.py`. The website's sliders, metrics and charts all update from `model/meta.json`.

Ideas to experiment with in `train_model.py`: different models (Gradient Boosting, SVM),
`GridSearchCV` for hyperparameters, feature engineering, SMOTE for class balance.

## AquaBot

- Works offline by default (keyword knowledge base + knowledge of your latest test result).
- To make it a ChatGPT-style assistant, set an API key before starting the server:
  `export ANTHROPIC_API_KEY=your_key` (Windows: `set ANTHROPIC_API_KEY=your_key`).
  It then answers with an LLM and is given your latest result as context.
  If the call fails, it falls back to offline mode.

_AquaCheckAI is an educational project, not a substitute for a certified laboratory test._
