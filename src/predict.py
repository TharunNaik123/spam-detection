import joblib

# Load saved model
model = joblib.load("../model/spam_detector.pkl")

def predict_message(text):
    return model.predict([text])[0]

if __name__ == "__main__":
    print("Sample Predictions:\n")

    examples = [
        "Congratulations! You have won a free lottery ticket.",
        "Please call me when you are free.",
        "Your bank account has been locked. Verify immediately."
    ]

    for msg in examples:
        print("Message:", msg)
        print("Prediction:", predict_message(msg))
        print("-" * 40)
