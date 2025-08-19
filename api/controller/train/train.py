import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# ----------------------------
#  Load and prepare data
# ----------------------------
def load_and_prepare_data(simple_path, complex_path):
    simple = pd.read_csv(simple_path, header=None, names=["prompt", "label"])
    complex_ = pd.read_csv(complex_path, header=None, names=["prompt", "label"])
    joint_df = pd.concat([simple, complex_], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return joint_df

joint_df = load_and_prepare_data('../data/prompt_data_simple.csv', '../data/prompt_data_complex.csv')

# Split into training and test sets
train_df, test_df_ans_key = train_test_split(
    joint_df,
    test_size=0.2,
    random_state=42,
    stratify=joint_df['label']
)

X_train_text = train_df['prompt']
y_train = train_df['label']
X_test_text = test_df_ans_key['prompt']
y_test = test_df_ans_key['label']

# ----------------------------
#  TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ----------------------------
# Model training & evaluation function
# ----------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    return model

# ----------------------------
# Define models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Naive Bayes": MultinomialNB()
}

# Train and evaluate all models
trained_models = {}
for name, model in models.items():
    trained_models[name] = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)

# ----------------------------
#  Predict new prompts
# ----------------------------
new_prompts = [
    # Simple prompts
    "What is the capital of France?",
    "How many ounces are in a cup?",
    "Weather forecast for New York tomorrow",
    "Who won the 2024 Olympic 100m race?",
    "How to boil an egg?",
    "Thai food restaurent near me",
    "Google office Vancouver",

    # Complex prompts
    "Develop an algorithm for autonomous drone navigation",
    "Explain the role of mitochondria in cellular metabolism using biochemistry terms",
    "Build a machine learning model to predict stock prices",
    "Design a protocol for secure multiparty computation",
    "Create a simulation of climate change impacts on crop yield",
    "Code me a IOS app about different dog breeds",
    "Help me write this email to my doctor"
]

X_new = vectorizer.transform(new_prompts)

for name, model in trained_models.items():
    preds = model.predict(X_new)
    print(f"\n{name} predictions on new prompts:")
    for prompt, label in zip(new_prompts, preds):
        print(f"{prompt} → {label}")
