# 📩 Spam Message Classifier

A beginner-friendly machine learning project that tells whether a message is **spam** or **not spam (ham)**, with a simple web page where you can paste any message and check it.

## How it works
1. Messages are converted to numbers with **TF-IDF**
2. A **Multinomial Naive Bayes** model learns which words/phrases appear in spam
3. A small **Flask** app loads the trained model and shows the result with a spam probability

## Tech stack
Python · scikit-learn · pandas · Flask · HTML/CSS/JS

## Run locally
```bash
pip install -r requirements.txt
python train.py      # trains the model, prints accuracy, creates model.pkl
python app.py        # open http://127.0.0.1:5000
```

## Project structure
```
spam.csv              dataset (label, message)
train.py              training + evaluation
app.py                Flask web app + /predict API
templates/index.html  web page
```

## API
`POST /predict` with `{"message": "You won a prize, click here"}`
→ `{"label": "spam", "spam_probability": 99.8}`

## Using the real dataset (recommended)
The included `spam.csv` is a small sample dataset. For better real-world accuracy, download the
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI / Kaggle),
rename its columns to `label,message`, replace `spam.csv`, and run `python train.py` again.

## Deploy (Render, free)
- Build command: `pip install -r requirements.txt && python train.py`
- Start command: `gunicorn app:app`
