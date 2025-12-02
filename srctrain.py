import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding="latin-1")

# Standardize columns
df = df.rename(columns={"v1": "label", "v2": "message"})
df = df[["label", "message"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("../model", exist_ok=True)
joblib.dump(pipeline, "../model/spam_detector.pkl")

print("\nModel saved at: ../model/spam_detector.pkl")

