import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib
import csv

# Load the CSV file into a DataFrame
simple = pd.read_csv('../data/prompt_data_simple.csv', header=None, names=["prompt", "label"])
complex_ = pd.read_csv('../data/prompt_data_complex.csv', header=None, names=["prompt", "label"])


#combine both together
joint_df = pd.concat([simple, complex_], ignore_index=True)
#shuffle the dataset
joint_df = joint_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Split into training and test sets (stratified to keep class proportions)
train_df, test_df_ans_key = train_test_split(
    joint_df,
    test_size=0.2,
    random_state=42,
    stratify=joint_df['label']
)

# Test set WITHOUT labels
test_df = test_df_ans_key.drop(columns=['label']).copy()

train_df = train_df.reset_index(drop=True)
test_df_ans_key = test_df_ans_key.reset_index(drop=True)
test_df = test_df_ans_key.drop(columns=['label']).copy().reset_index(drop=True)


# Initialize TF-IDF vectorizer which converts text to numbers
#TF (Term Frequency): how often a word appears in this document.
#IDF (Inverse Document Frequency): downweights words that appear everywhere.
vectorizer = TfidfVectorizer()

# Fit the vectorizer on training prompts and transform them into vectors
X_train = vectorizer.fit_transform(train_df['prompt'])
# Target labels
y_train = train_df['label']

# Transform test prompts (without fitting again!)
X_test = vectorizer.transform(test_df['prompt'])


# --- logstic regression classifier ---
model_lr = LogisticRegression(max_iter=1000)  # increase iterations if needed
model_lr.fit(X_train, y_train)

#evalute the model 
y_pred_lr = model_lr.predict(X_test)
print("logstic regression Report:")
print(classification_report(test_df_ans_key['label'], y_pred_lr))


# --- Random Forest classifier ---
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

#evalute the model 
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Report:")
print(classification_report(test_df_ans_key['label'], y_pred_rf))


# --- Support Vector Machine (SVM) classifier ---
svm_model = SVC(kernel='linear', random_state=42)  # linear kernel for text
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(test_df_ans_key['label'], y_pred_svm))


# --- Multinomial Naive Bayes classifier ---
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(test_df_ans_key['label'], y_pred_nb))



new_prompts = [
    # Simple (easy Google-search-style queries)
    "What is the capital of France?",                        # simple
    "How many ounces are in a cup?",                          # simple
    "Weather forecast for New York tomorrow",                 # simple
    "Who won the 2024 Olympic 100m race?",                    # simple
    "How to boil an egg?",                                    # simple

    # Complex (requires design, computation, or technical reasoning)
    "Develop an algorithm for autonomous drone navigation",  # complex
    "Explain the role of mitochondria in cellular metabolism using biochemistry terms", # complex
    "Build a machine learning model to predict stock prices", # complex
    "Design a protocol for secure multiparty computation",   # complex
    "Create a simulation of climate change impacts on crop yield", # complex
]


# Transform with the same vectorizer
X_new = vectorizer.transform(new_prompts)
preds = model_lr.predict(X_new)

for prompt, label in zip(new_prompts, preds):
    print(f"{prompt} → {label}")
print("\n")

# Transform with the same vectorizer
X_new = vectorizer.transform(new_prompts)
preds = rf_model.predict(X_new)

for prompt, label in zip(new_prompts, preds):
    print(f"{prompt} → {label}")
print("\n")
    
# Transform with the same vectorizer
X_new = vectorizer.transform(new_prompts)
preds = svm_model.predict(X_new)

for prompt, label in zip(new_prompts, preds):
    print(f"{prompt} → {label}")
print("\n")
    
# Transform with the same vectorizer
X_new = vectorizer.transform(new_prompts)
preds = nb_model.predict(X_new)

for prompt, label in zip(new_prompts, preds):
    print(f"{prompt} → {label}")
print("\n")