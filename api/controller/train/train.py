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
    df[['prompt', 'label']] = df['raw'].str.rsplit(',', n=1, expand=True)
    
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


# # 7. Save model and vectorizer
# joblib.dump(model, '../data/logreg_model.pkl')
# joblib.dump(vectorizer, '../data/tfidf_vectorizer.pkl')
