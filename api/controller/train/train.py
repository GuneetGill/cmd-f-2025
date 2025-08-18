import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Function to load CSV and fix quotes
def load_fixed_csv(path):
    # Read everything as a single column
    df = pd.read_csv(path, header=None, names=['raw'])
    
    # Split 'raw' column into 'prompt' and 'label'
    df[['prompt', 'label']] = df['raw'].str.split(',', n=1, expand=True)
    
    # Remove quotes and extra spaces
    df['prompt'] = df['prompt'].str.strip().str.replace('"', '')
    df['label'] = df['label'].str.strip().str.replace('"', '')
    
    # Drop the raw column
    df = df.drop(columns=['raw'])
    
    return df

# Load both datasets into panda dataframe 
simple = load_fixed_csv('../data/prompt_data_simple.csv')
complex_ = load_fixed_csv('../data/prompt_data_complex.csv')

#combine both together
joint_df = pd.concat([simple, complex_], ignore_index=True)
#randomize the order 
joint_df = joint_df.sample(frac=1, random_state=42).reset_index(drop=True)

#split training and test data
# 20% size for testing 
test_size = int(0.2 * len(joint_df))

# Test set WITH labels (answer key)
test_df_ans_key = joint_df.iloc[:test_size].copy()
print(test_df_ans_key.head())

# Test set WITHOUT labels (just prompts)
test_df = test_df_ans_key.drop(columns=['label']).copy()
print(test_df.head())

# Remaining training data
train_df = joint_df.iloc[test_size:].copy()


# # 4. TF-IDF Vectorization
# vectorizer = TfidfVectorizer()
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # 5. Train Logistic Regression
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# # 6. Evaluate
# y_pred = model.predict(X_test_tfidf)
# print(classification_report(y_test, y_pred))

# # 7. Save model and vectorizer
# joblib.dump(model, '../data/logreg_model.pkl')
# joblib.dump(vectorizer, '../data/tfidf_vectorizer.pkl')
