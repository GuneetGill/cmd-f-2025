import joblib
import os 
import urllib.parse  # for safe URL encoding
import json


# Paths to the saved models in the train/ folder
TRAIN_DIR = "train"
VECTORIZER_PATH = os.path.join(TRAIN_DIR, "tfidf_vectorizer.pkl")
SVM_PATH = os.path.join(TRAIN_DIR, "svm_model.pkl")
LR_PATH = os.path.join(TRAIN_DIR, "logistic_regression_model.pkl")
RF_PATH = os.path.join(TRAIN_DIR, "random_forest_model.pkl")
NB_PATH = os.path.join(TRAIN_DIR, "naive_bayes_model.pkl")

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

# Load models
svm_model = joblib.load(SVM_PATH)
lr_model = joblib.load(LR_PATH)
rf_model = joblib.load(RF_PATH)
nb_model = joblib.load(NB_PATH)

def analyze_prompt_complexity(prompt, model=svm_model):
    """
    SVM-based prompt complexity analyzer compatible with version 1 return type.
    Returns a JSON string.
    """
    X_new = vectorizer.transform([prompt])
    prediction = model.predict(X_new)[0].upper()  # SIMPLE or COMPLEX

    if prediction == "SIMPLE":
        result = {"classification": prediction, "google_search_url": ""}
    else:
        query = urllib.parse.quote(prompt)
        result = {"classification": prediction, "google_search_url": f"https://www.google.com/search?q={query}"}

    return json.dumps(result)
