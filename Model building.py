import pandas as pd
data_cleaned = pd.read_csv(r"C:\Users\ADMIN\Desktop\Project_276\Cleaned_Transcript_Data.csv")

from transformers import pipeline

#pip install torch
#import torch
#print(torch.version)

# Load the sentiment analysis pipeline (using PyTorch as the backend)
sentiment_analyzer = pipeline('sentiment-analysis', framework='pt', model='distilbert-base-uncased-finetuned-sst-2-english')

# Function to apply sentiment analysis with truncation and padding
def analyze_sentiment_with_truncation(text):
    # Tokenize the text with truncation and padding to ensure it fits within the model's input size
    result = sentiment_analyzer(text, truncation=True, padding=True)
    return result[0]['score']

# Apply sentiment analysis with truncation to the 'Transcript' column
data_cleaned['Sentiment_Score_Transformers'] = data_cleaned['Transcript'].apply(
    lambda x: analyze_sentiment_with_truncation(x) if isinstance(x, str) else 0.0
)

# Show the first few rows of the cleaned data with sentiment scores from Transformers
print(data_cleaned[['Transcript', 'Sentiment_Score_Transformers']].head())

# Function to classify sentiment based on Transformers sentiment score
def classify_sentiment_transformers(score):
    if score >= 0.9:
        return 'Positive'
    elif score >= 0.7:
        return 'Neutral'
    else:
        return 'Negative'

# Apply the classification to the 'Sentiment_Score_Transformers' column and create a new 'Sentiment' column
data_cleaned['Sentiment'] = data_cleaned['Sentiment_Score_Transformers'].apply(classify_sentiment_transformers)

# Display the result
print(data_cleaned[['Transcript', 'Sentiment_Score_Transformers', 'Sentiment']].head())

# Save the cleaned dataset to a CSV file
output_file = r"C:\Users\ADMIN\Desktop\Project_276\Seniment_Analysis.csv"
data_cleaned.to_csv(output_file, index=False)

#BERT_CODE_Summarization

#pip install transformers pandas


import pandas as pd
import re
from transformers import pipeline

# Load data
df=data_cleaned

# Step 1: Clean transcript
def clean_transcript(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\b(interviewer|candidate):', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['Cleaned_Transcript'] = df['Transcript'].apply(clean_transcript)

# Step 2: Load summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 3: Post-process summary to clean dialogue, speaker tags, and questions
def clean_summary(summary):
    summary = re.sub(r'\b\w+:\s*', '', summary)  # Remove speaker tags like "Manish:"
    summary = re.sub(r'\b(i|i\'m|my|me)\b', 'the candidate', summary, flags=re.IGNORECASE)  # Convert 1st person
    summary = re.sub(r'\b(can|could|would|will)\s+you\b.*?\?', '', summary, flags=re.IGNORECASE)  # Remove questions
    summary = re.sub(r'\s+', ' ', summary).strip()  # Final clean
    return summary

# Step 4: Generate and clean summaries
def generate_summary(text):
    if not text or len(text.split()) < 30:
        return "Text too short to summarize."
    try:
        raw_summary = summarizer(text, max_length=110, min_length=55, do_sample=False)[0]['summary_text']
        return clean_summary(raw_summary)
    except Exception as e:
        return f"Error: {e}"

df['Summary'] = df['Cleaned_Transcript'].apply(generate_summary)

# Step 5: Save results
#df.to_csv("Transcript_Summarized_Cleaned2.csv", index=False)

# Optional: View sample
print(df[['Transcript', 'Summary']].head())

#JD_RESUME_SCORE
#pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer, util

# Load pretrained BERT model (suitable for semantic textual similarity)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and accurate

# Define a function to compute cosine similarity using BERT embeddings
def semantic_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2) or not text1.strip() or not text2.strip():
        return 0.0
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score, 4)

# Compute BERT-based similarity
df['JD_Resume_Score_BERT'] = df.apply(lambda row: semantic_similarity(row['Job Description'], row['Resume']), axis=1)
df['JD_Transcript_Score_BERT'] = df.apply(lambda row: semantic_similarity(row['Job Description'], row['Transcript']), axis=1)


# Preview
print(df[["JD_Resume_Score_BERT", "JD_Transcript_Score_BERT"]].describe())


#skill extraction
# Importing required library
from transformers import pipeline

# Initialize Zero-Shot Classifier with GPU
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)  # device=0 ensures GPU usage

# Define skill labels (you can extend this list)
skill_labels = [
    'python', 'java', 'sql', 'tableau', 'excel', 'power bi', 'react', 'node.js', 'javascript',
    'html', 'css', 'aws', 'docker', 'jenkins', 'mongodb', 'mysql', 'pandas', 'numpy',
    'data analysis', 'data visualization', 'machine learning', 'deep learning', 'nlp',
    'fastapi', 'jira', 'confluence', 'mixpanel', 'google analytics', 'matplotlib', 'seaborn',
    'sklearn', 'git', 'bash', 'linux'
]

# Function to extract skills using BERT Zero-Shot
def extract_skills_zeroshot(text, skill_labels):
    result = classifier(text, candidate_labels=skill_labels, multi_label=True)
    # Filter out the skills with a low score (set threshold to 0.5)
    return [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]

# Assuming the dataframe 'df' with 'Resume' and 'Job Description' columns is already loaded
# Example of using the function on Resume and Job Description columns
df['skills_extracted_resume_zeroshot'] = df['Resume'].apply(lambda x: extract_skills_zeroshot(x, skill_labels))
df['skills_extracted_jd_zeroshot'] = df['Job Description'].apply(lambda x: extract_skills_zeroshot(x, skill_labels))

# Save the output
df.to_csv('/content/Skills_Extracted_BERT_ZeroShot.csv', index=False)

# Display the first few rows of extracted skills
print(df[['Resume', 'skills_extracted_resume_zeroshot', 'Job Description', 'skills_extracted_jd_zeroshot']].head())

#scoring

# Function to compute skill matching score between resume and JD based on required skills
def compute_skill_matching_score(resume_skills, jd_skills):
    # Convert both lists of skills into sets for better comparison (avoiding duplicates)
    resume_skills_set = set(resume_skills)
    jd_skills_set = set(jd_skills)

    # Calculate number of matching skills (intersection)
    matching_skills = len(resume_skills_set.intersection(jd_skills_set))

    # If there are no required skills, return a matching score of 0 (avoid division by zero)
    if len(jd_skills_set) == 0:
        return 0.0

    # Calculate the matching score as the percentage of matched skills
    match_score = (matching_skills / len(jd_skills_set)) * 100
    return match_score

# Apply the skill matching score calculation for each row based on required skills from JD and resume skills
df['JD_Resume_Skill_Match_Score'] = df.apply(
    lambda row: compute_skill_matching_score(
        row['skills_extracted_resume_zeroshot'], row['skills_extracted_jd_zeroshot']), axis=1)


# Display the updated dataframe with matching scores
print(df[['Position', 'JD_Resume_Skill_Match_Score']].head())

# Save the cleaned dataset to a CSV file
output_file = r"C:\Users\ADMIN\Desktop\Project_276\Final_Generated_Dataset.csv"
data_cleaned.to_csv(output_file, index=False)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np

# Load the dataset
data = pd.read_csv(r"C:\Users\ADMIN\Desktop\Project_276\Final_Generated_Dataset.csv")

# Select relevant features for the model
features = ['Age', 'Position', 'JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score', 'Sentiment']
target = 'Selected/Not-selected'  # Correct target column name

# Check for missing values
print(data[features].isnull().sum())

# Handle missing values by imputing or dropping rows
# Impute numerical columns with their mean
numerical_cols = data[features].select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Impute categorical columns like 'Position' with a placeholder
data['Position'].fillna('Unknown', inplace=True)

# Encode categorical data (e.g., 'Position' column)
label_encoder = LabelEncoder()
data['Position'] = label_encoder.fit_transform(data['Position'])

# Encode Sentiment column (Positive -> 1, Neutral -> 0, Negative -> -1)
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)

# Scale numerical features (e.g., Age, BERT scores, Sentiment)
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split dataset into train and test sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the preprocessing was successful
print(X_train.head())
print(y_train.head())


# Check the column names to verify if 'Selected' exists
print(data.columns)

target = 'Selected/Not-selected'


#Logistic regression
from sklearn.linear_model import LogisticRegression

# Initialize the model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

import joblib

# Save the trained XGBoost model
model_filename = 'LogisticRegression_model.pkl'
joblib.dump(logreg, model_filename)

print(f"Model saved as {model_filename}")

#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

import joblib

# Save the trained XGBoost model
model_filename = 'Randomforest_model.pkl'
joblib.dump(rf, model_filename)

print(f"Model saved as {model_filename}")

#SVM

from sklearn.svm import SVC

# Initialize the model
svm = SVC(random_state=42)

# Train the model
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

import joblib

# Save the trained XGBoost model
model_filename = 'SVM_model.pkl'
joblib.dump(svm, model_filename)

print(f"Model saved as {model_filename}")

# XG Boost

from xgboost import XGBClassifier

# Initialize the model
xgb = XGBClassifier(random_state=42)

# Train the model
xgb.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb.predict(X_test)

# Evaluate the model
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

import joblib

# Save the trained XGBoost model
model_filename = 'xgboost_model.pkl'
joblib.dump(xgb, model_filename)

print(f"Model saved as {model_filename}")

#KNN

from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)

# Evaluate the model
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

import joblib

# Save the trained XGBoost model
model_filename = 'KNN_model.pkl'
joblib.dump(knn, model_filename)

print(f"Model saved as {model_filename}")

#MLP (Multi Level Perceptron

from keras.models import Sequential
from keras.layers import Dense

# Initialize the model
mlp = Sequential()

# Add layers
mlp.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
mlp.add(Dense(32, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
mlp.fit(X_train, y_train, epochs=35, batch_size=32)

# Evaluate the model
_, accuracy = mlp.evaluate(X_test, y_test)
print(f"MLP Accuracy: {accuracy}")


import joblib

# Save the trained XGBoost model
model_filename = 'mlp_model.pkl'
joblib.dump(mlp, model_filename)

print(f"Model saved as {model_filename}")

#RNN or **LSTM**

import torch
import torch.nn as nn
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Input, Concatenate, Flatten
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv(r'C:\Users\ADMIN\Desktop\Project_276\Final_Generated_Dataset.csv')

import torch
import torch.nn as nn
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Input, Concatenate, Flatten
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('/content/Final_Generated_Dataset.csv')

# Preprocess data
df['Selected/Not-selected'] = df['Selected/Not-selected'].apply(lambda x: 1 if x == 'Selected' else 0)
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['Sentiment_map'] = df['Sentiment'].map(sentiment_mapping)

# Separate numerical features and text features
numerical_features = ['Age', 'Experience', 'Sentiment_map']  # Numerical columns
text_features = ['Resume', 'Job Description']  # Text columns (Resume + Job Description)

# Extract numerical features
X_numerical = df[numerical_features]

# Concatenate Resume and Job Description for text features
X_text = df['Resume'] + " " + df['Job Description']  # Concatenate text data

# Target variable
y = df['Selected/Not-selected']

# Train-test split
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(X_text, X_numerical, y, test_size=0.2, random_state=42)

# Tokenizer for BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and pad the text data
X_train_text_seq = tokenizer(X_train_text.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
X_test_text_seq = tokenizer(X_test_text.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Pad sequences for LSTM input (the length can be adjusted based on your preference)
X_train_text_pad = pad_sequences(X_train_text_seq['input_ids'], maxlen=100)  # Adjust padding if needed
X_test_text_pad = pad_sequences(X_test_text_seq['input_ids'], maxlen=100)

# Scale numerical features
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Define model using the Functional API (to handle both text and numerical features)
input_text = Input(shape=(100,))  # Input shape for the padded text sequences
#embedding = Embedding(input_dim=10000, output_dim=128, input_length=100)(input_text)
embedding = Embedding(input_dim=tokenizer.vocab_size, output_dim=128, input_length=100)(input_text)
dropout = SpatialDropout1D(0.2)(embedding)
lstm_out = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(dropout)

# Create an input layer for numerical data
input_num = Input(shape=(X_train_num_scaled.shape[1],))  # Input shape for numerical features

# Concatenate LSTM output and numerical features
concatenated = Concatenate()([lstm_out, input_num])

# Dense layer for binary classification output
dense_out = Dense(1, activation='sigmoid')(concatenated)

# Final model definition
model = Model(inputs=[input_text, input_num], outputs=dense_out)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([X_train_text_pad, X_train_num_scaled], y_train, epochs=25, batch_size=32)

# Evaluate the model
_, accuracy = model.evaluate([X_test_text_pad, X_test_num_scaled], y_test)
print(f"LSTM Accuracy: {accuracy}")
