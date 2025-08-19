import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import csv

# Load the CSV file into a DataFrame
simple = pd.read_csv('../data/prompt_data_simple.csv', header=None, names=["prompt", "label"])
complex_ = pd.read_csv('../data/prompt_data_complex.csv', header=None, names=["prompt", "label"])

print("simple set:")
print(simple.head())
print("complex set:")
print(complex_.head())

#combine both together
joint_df = pd.concat([simple, complex_], ignore_index=True)
#shuffle the dataset
joint_df = joint_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("joint:")
print(joint_df.head())

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


# Print first 5 rows to verify
print("Train set:")
print(train_df.head())
print("\nTest set with labels:")
print(test_df_ans_key.head())
print("\nTest set without labels:")
print(test_df.head())


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

#use logstic regression
model = LogisticRegression(max_iter=1000)  # increase iterations if needed
model.fit(X_train, y_train)

#evalute the model 
y_pred = model.predict(X_test)
print(classification_report(test_df_ans_key['label'], y_pred))


# 7. Save model and vectorizer
joblib.dump(model, '../data/logreg_model.pkl')
joblib.dump(vectorizer, '../data/tfidf_vectorizer.pkl')


# new_prompts = [
#     # Simple (easy Google-search-style queries)
#     "What is the capital of France?",                        # simple
#     "How many ounces are in a cup?",                          # simple
#     "Weather forecast for New York tomorrow",                 # simple
#     "Who won the 2024 Olympic 100m race?",                    # simple
#     "How to boil an egg?",                                    # simple

#     # Complex (requires design, computation, or technical reasoning)
#     "Develop an algorithm for autonomous drone navigation",  # complex
#     "Explain the role of mitochondria in cellular metabolism using biochemistry terms", # complex
#     "Build a machine learning model to predict stock prices", # complex
#     "Design a protocol for secure multiparty computation",   # complex
#     "Create a simulation of climate change impacts on crop yield", # complex
# ]


# # Transform with the same vectorizer
# X_new = vectorizer.transform(new_prompts)
# preds = model.predict(X_new)

# for prompt, label in zip(new_prompts, preds):
#     print(f"{prompt} → {label}")
