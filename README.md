# Financial News Sentiment Analyzer (MLOps Project)

## Overview
This project builds a Naive Bayes classifier to predict sentiment
(positive, neutral, negative) from financial news headlines.

## Dataset
Financial PhraseBank dataset stored in /data directory.

## Tech Stack
- Python
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- GitHub Actions
- Jenkins CI/CD

## Run Locally

Install dependencies:
pip install -r requirements.txt

Train model:
python src/train.py

Predict sentiment:
python src/predict.py "Markets rally after earnings"