# Spam Detection using Machine Learning

This project classifies SMS messages as Spam or Ham.  
It is developed as part of my Machine Learning Internship – Week 2 Task.

## Features
- Text preprocessing
- TF-IDF vectorization
- Multinomial Naive Bayes classifier
- Saved ML pipeline for predictions

## Project Structure
spam-detection/
│── data/
│     └── spam.csv
│── model/
│     └── spam_detector.pkl
│── src/
│     ├── train.py
│     └── predict.py

## Install Dependencies
pip install pandas scikit-learn joblib

## Train the Model
cd src  
python train.py

## Run Predictions
python predict.py

## Dataset
Uses the SMS Spam Collection dataset (v1 → label, v2 → message).

