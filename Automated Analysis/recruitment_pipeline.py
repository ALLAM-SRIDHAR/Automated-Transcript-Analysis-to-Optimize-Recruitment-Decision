# recruitment_pipeline.py
'''
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load all models at once using Streamlit's cache
@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentiment_analyzer, summarizer, classifier, bert_model

sentiment_analyzer, summarizer, classifier, bert_model = load_models()

skill_labels = [
    'python', 'java', 'sql', 'tableau', 'excel', 'power bi', 'react', 'node.js', 'javascript',
    'html', 'css', 'aws', 'docker', 'jenkins', 'mongodb', 'mysql', 'pandas', 'numpy',
    'data analysis', 'data visualization', 'machine learning', 'deep learning', 'nlp',
    'fastapi', 'jira', 'confluence', 'mixpanel', 'google analytics', 'matplotlib', 'seaborn',
    'sklearn', 'git', 'bash', 'linux'
]

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def clean_resume(resume):
    if not isinstance(resume, str):
        return ""
    resume = re.sub(r'\b(Name|Email|Phone|Contact|Address|Date)\s*:', '', resume, flags=re.IGNORECASE)
    resume = re.sub(r'\bReferences\s*:(.|\n)*', '', resume, flags=re.IGNORECASE)
    resume = re.sub(r'\S+@\S+', '', resume)
    resume = re.sub(r'\+?\d[\d\s.-]{8,}', '', resume)
    resume = re.sub(r'[^a-zA-Z0-9\s,.:\-()/%]', '', resume)
    resume = re.sub(r':\s*:', '', resume)
    resume = re.sub(r'\s+', ' ', resume).strip().lower()
    return resume

def clean_transcript_with_punctuation(transcript):
    transcript = re.sub(r'\[\d{1,2}:\d{2}\]', '', transcript)
    transcript = re.sub(r'\[[^\]]*\]', '', transcript)
    transcript = transcript.replace("â€™", "'").replace("’", "'").replace("â€”", "'")
    transcript = re.sub(r':\s*:', '', transcript)
    transcript = re.sub(r'::', ':', transcript)
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    transcript = transcript.lower().capitalize()
    transcript = transcript.replace('interviewer', 'Interviewer:').replace('candidate', 'Candidate:')
    return transcript

def clean_job_description(jd):
    if not isinstance(jd, str):
        return ""
    jd = re.sub(r'how to apply:(.|\n)*', '', jd, flags=re.IGNORECASE)
    jd = re.sub(r'\b(Job Title|Company|Location|Job Type)\s*:\s*.*', '', jd, flags=re.IGNORECASE)
    jd = re.sub(r'\S+@\S+', '', jd)
    jd = re.sub(r'\+?\d[\d\s.-]{8,}', '', jd)
    jd = jd.replace("â€™", "'").replace("â€“", "-")
    jd = re.sub(r'[^a-zA-Z0-9\s,.:\-/()%]', '', jd)
    jd = re.sub(r':\s*:', '', jd)
    jd = re.sub(r'\s+', ' ', jd).strip().lower()
    return jd

def standardize_experience(exp):
    exp = re.sub(r'[^0-9.]+', '', str(exp))
    return int(float(exp)) if exp else 0

def clean_data(df):
    df = df.copy()
    df['Resume'] = df['Resume'].apply(clean_resume)
    df['Transcript'] = df['Transcript'].apply(clean_transcript_with_punctuation)
    df['Job Description'] = df['Job Description'].apply(clean_job_description)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Experience'] = df['Experience'].apply(standardize_experience)
    df.drop_duplicates(inplace=True)
    return df

def analyze_sentiment_with_truncation(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    try:
        result = sentiment_analyzer(text[:512])
        return result[0]['score']
    except:
        return 0.0

def classify_sentiment(score):
    if score >= 0.9:
        return 'Positive'
    elif score >= 0.7:
        return 'Neutral'
    else:
        return 'Negative'

def generate_summary(text):
    if not text or len(text.split()) < 30:
        return "Too short to summarize."
    try:
        return summarizer(text, max_length=110, min_length=55, do_sample=False)[0]['summary_text']
    except:
        return "Error summarizing"

def semantic_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2) or not text1.strip() or not text2.strip():
        return 0.0
    embeddings = bert_model.encode([text1, text2], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())

def extract_skills_zeroshot(text, labels):
    result = classifier(text, candidate_labels=labels, multi_label=True)
    return [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]

def compute_skill_matching_score(skills1, skills2):
    return len(set(skills1) & set(skills2)) / max(len(skills2), 1) * 100

def engineer_features(df):
    df['Sentiment_Score'] = df['Transcript'].apply(analyze_sentiment_with_truncation)
    df['Sentiment'] = df['Sentiment_Score'].apply(classify_sentiment)
    df['Summary'] = df['Transcript'].apply(generate_summary)
    df['JD_Resume_Score_BERT'] = df.apply(lambda row: semantic_similarity(row['Job Description'], row['Resume']), axis=1)
    df['JD_Transcript_Score_BERT'] = df.apply(lambda row: semantic_similarity(row['Job Description'], row['Transcript']), axis=1)
    df['skills_resume'] = df['Resume'].apply(lambda x: extract_skills_zeroshot(x, skill_labels))
    df['skills_jd'] = df['Job Description'].apply(lambda x: extract_skills_zeroshot(x, skill_labels))
    df['JD_Resume_Skill_Match_Score'] = df.apply(lambda row: compute_skill_matching_score(row['skills_resume'], row['skills_jd']), axis=1)
    return df

def final_preprocessing(df):
    if df['Sentiment'].dtype == object:
        df['Sentiment'] = LabelEncoder().fit_transform(df['Sentiment'])
    for col in ['JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score']:
        df[col] = df[col].round(2)
    scaler = StandardScaler()
    scaled_features = ['Age', 'JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score']
    df[scaled_features] = scaler.fit_transform(df[scaled_features])
    if 'Selected/Not-selected' in df.columns:
        df['Selected/Not-selected'] = df['Selected/Not-selected'].map({'Selected': 1, 'Not Selected': 0})
    return df

def preprocess_for_prediction(df):
    df = clean_data(df)
    df = engineer_features(df)
    df = final_preprocessing(df)
    return df

def make_prediction(df, model_path):
    model = joblib.load(model_path)
    features = ['Age', 'JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score', 'Sentiment']
    return model.predict(df[features])
'''


import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import streamlit as st

@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentiment_analyzer, summarizer, classifier, bert_model
# Load once
sentiment_analyzer, summarizer, classifier, model = load_models()

# Define skill labels (you can extend this list)
skill_labels = [
    'python', 'java', 'sql', 'tableau', 'excel', 'power bi', 'react', 'node.js', 'javascript',
    'html', 'css', 'aws', 'docker', 'jenkins', 'mongodb', 'mysql', 'pandas', 'numpy',
    'data analysis', 'data visualization', 'machine learning', 'deep learning', 'nlp',
    'fastapi', 'jira', 'confluence', 'mixpanel', 'google analytics', 'matplotlib', 'seaborn',
    'sklearn', 'git', 'bash', 'linux'
]

# ==================== STEP 1: DATA LOADING ====================
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

# ==================== STEP 2: CLEANING ====================
def clean_resume(resume):
    if not isinstance(resume, str):
        return ""

    # Remove 'Name:', 'Email:', 'Phone:', etc.
    resume = re.sub(r'\b(Name|Email|Phone|Contact|Address|Date)\s*:', '', resume, flags=re.IGNORECASE)

    # Remove 'References:' section and everything after it
    resume = re.sub(r'\bReferences\s*:(.|\n)*', '', resume, flags=re.IGNORECASE)

    # Remove email addresses
    resume = re.sub(r'\S+@\S+', '', resume)

    # Remove phone numbers
    resume = re.sub(r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', resume)

    # Replace 'â€“' with 'to' if it's part of a date range
    resume = re.sub(r'(\b\w+\s+\d{4})\s+â€“\s+(\w+|\bPresent\b)', r'\1 to \2', resume)

    # Keep punctuation: .,:- and remove all other special characters
    resume = re.sub(r'[^a-zA-Z0-9\s,.:\-()/%]', '', resume)

    # Remove redundant colons like ": :"
    resume = re.sub(r':\s*:', '', resume)

    # Collapse multiple spaces into one
    resume = re.sub(r'\s+', ' ', resume).strip()

    # Convert to lowercase
    resume = resume.lower()

    return resume

def clean_transcript_with_punctuation(transcript):
    # Step 1: Remove timestamps in the format [0:02], [1:05], etc.
    transcript = re.sub(r'\[\d{1,2}:\d{2}\]', '', transcript)

    # Step 2: Remove non-verbal cues like [Mouse click heard], [Background: Pen tapping], etc.
    transcript = re.sub(r'\[[^\]]*\]', '', transcript)

    # Step 3: Replace characters like "â€™" with "'"
    transcript = transcript.replace("â€™", "'")
    transcript = transcript.replace("’", "'")  # Another common variation of the apostrophe
    transcript = transcript.replace("â€”", "'")  # Replace em dash with apostrophe

    # Remove redundant colons like ": :"
    transcript = re.sub(r':\s*:', '', transcript)
    # Remove redundant colons like "::"
    transcript = re.sub(r'::', ':', transcript)
    transcript = re.sub(r': :', ':', transcript)

    # Step 4: Clean multiple spaces
    transcript = re.sub(r'\s+', ' ', transcript).strip()

    # Step 5: Add punctuation:
    # Add a question mark after interviewer questions (if missing)
    transcript = re.sub(r'(interviewer[^:])(?=\s[^:]*\?)', r'\1?', transcript)

    # Add periods after candidate statements if they don't already end with one
    transcript = re.sub(r'(candidate[^:])(?=\s[^.]*$)', r'\1.', transcript)

    # Step 6: Convert all text to lowercase
    transcript = transcript.lower()

    # Step 7: Capitalize the first letter of each sentence after removing unwanted characters
    transcript = transcript.capitalize()

    # Step 8: Replace interviewer and candidate labels with more readable structure
    transcript = transcript.replace('interviewer', 'Interviewer:').replace('candidate', 'Candidate:')

    return transcript

def clean_job_description(jd):
    if not isinstance(jd, str):
        return ""

    # Remove section: "How to Apply:" and anything after it
    jd = re.sub(r'how to apply:(.|\n)*', '', jd, flags=re.IGNORECASE)

    # Remove labels and associated content for Job Title, Company, Location, Job Type
    jd = re.sub(r'\b(Job Title|Company|Company Name|Location|Job Type)\s*:\s*.*', '', jd, flags=re.IGNORECASE)

    # Remove email addresses
    jd = re.sub(r'\S+@\S+', '', jd)

    # Remove phone numbers
    jd = re.sub(r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', jd)

    # Replace "â€™" with apostrophe (')
    jd = jd.replace("â€™", "'")

    # Replace "â€“" between numbers or dates with "to"
    jd = re.sub(r'(\d+)\s*â€“\s*(\d+)', r'\1 to \2', jd)
    jd = re.sub(r'(\b\w+\s+\d{4})\s*â€“\s*(\b\w+\s+\d{4}|\bPresent\b)', r'\1 to \2', jd)

    # Replace any remaining "â€“" with "-"
    jd = jd.replace("â€“", "-")

    # Remove unwanted keywords (optional)
    jd = re.sub(r'\b(email|phone|contact|address|date|references)\b', '', jd, flags=re.IGNORECASE)

    # Remove unwanted characters but preserve commas, periods, colons, hyphens
    jd = re.sub(r'[^a-zA-Z0-9\s,.:\-/()%]', '', jd)

    # Remove multiple colons (e.g., ": :")
    jd = re.sub(r':\s*:', '', jd)

    # Normalize whitespace
    jd = re.sub(r'\s+', ' ', jd).strip()

    # Lowercase the text
    jd = jd.lower()

    return jd

def standardize_experience(experience):
    # Remove any non-numeric characters after the first number
    experience = re.sub(r'[^0-9.]+', '', str(experience))  # Remove any non-numeric characters (except for dot)
    # Convert to float and round down (int truncates decimals)
    if experience:
        return int(float(experience))  # Convert to integer after rounding down
    return 0  # In case of missing or incorrect data

def clean_data(df):
    df = df.copy()
    df['Resume'] = df['Resume'].apply(clean_resume)
    df['Transcript'] = df['Transcript'].apply(clean_transcript_with_punctuation)
    df['Job Description'] = df['Job Description'].apply(clean_job_description)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Experience']=df['Experience'].apply(standardize_experience)
    df.drop_duplicates(inplace=True)
    return df

# ==================== STEP 3: FEATURE ENGINEERING ====================
def analyze_sentiment_with_truncation(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0  # Neutral baseline for empty or non-string

    try:
        result = sentiment_analyzer(text, truncation=True, padding=True, max_length=512)
        return result[0]['score'] if 'score' in result[0] else 0.0
    except Exception as e:
        print(f"Sentiment error: {e}")
        return 0.0

def classify_sentiment_transformers(score):
    if score >= 0.9:
        return 'Positive'
    elif score >= 0.7:
        return 'Neutral'
    else:
        return 'Negative'

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

def semantic_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2) or not text1.strip() or not text2.strip():
        return 0.0
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score, 4)

def extract_skills_zeroshot(text, skill_labels):
    result = classifier(text, candidate_labels=skill_labels, multi_label=True)
    # Filter out the skills with a low score (set threshold to 0.5)
    return [label for label, score in zip(result["labels"], result["scores"]) if score > 0.5]

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


def engineer_features(df):
    df['Sentiment_Score'] = df['Transcript'].apply(analyze_sentiment_with_truncation)
    df['Sentiment'] = df['Sentiment_Score'].apply(classify_sentiment_transformers)
    df['Summary'] = df['Transcript'].apply(generate_summary)
    df['JD_Resume_Score_BERT'] = df.apply(lambda row: semantic_similarity(row['Job Description'], row['Resume']), axis=1)
    df['JD_Transcript_Score_BERT'] = df.apply(lambda row: semantic_similarity(row['Job Description'], row['Transcript']), axis=1)
    #df['skills_resume'] = df['Resume'].apply(extract_skills_zeroshot)
    #df['skills_jd'] = df['Job Description'].apply(extract_skills_zeroshot)
    df['skills_resume'] = df['Resume'].apply(lambda x: extract_skills_zeroshot(x, skill_labels))
    df['skills_jd'] = df['Job Description'].apply(lambda x: extract_skills_zeroshot(x, skill_labels))
    #df['JD_Resume_Skill_Match_Score'] = df.apply(lambda row: compute_skill_matching_score(row['skills_resume'], row['skills_jd']), axis=1)
    df['JD_Resume_Skill_Match_Score'] = df.apply(lambda row: compute_skill_matching_score(row['skills_resume'], row['skills_jd']), axis=1)
    return df



# ==================== STEP 4: FINAL PREPROCESSING ====================

def final_preprocessing(df):
    # 1. Encode Sentiment column (if not already numerical)
    if df['Sentiment'].dtype == object:
        sentiment_encoder = LabelEncoder()
        df['Sentiment'] = sentiment_encoder.fit_transform(df['Sentiment'])
    
    # 2. Round scores to 2 decimal places (safe to do here)
    for col in ['JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # 3. Scale numeric features
    # Save original scores for display
    df['Orig_JD_Resume_Score_BERT'] = df['JD_Resume_Score_BERT']
    df['Orig_JD_Transcript_Score_BERT'] = df['JD_Transcript_Score_BERT']
    df['Orig_JD_Resume_Skill_Match_Score'] = df['JD_Resume_Skill_Match_Score']
    df['Orig_Age'] = df['Age']

    scaler = StandardScaler()
    features_to_scale = ['Age', 'JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score']
    for col in features_to_scale:
        if col not in df.columns:
            df[col] = 0.0  # fill missing features if not yet generated
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # 4. Map target label
    if 'Selected/Not-selected' in df.columns:
        df['Selected/Not-selected'] = df['Selected/Not-selected'].map({'Selected': 1, 'Not Selected': 0})

    return df

def preprocess_for_prediction(df):
    df = clean_data(df)
    df = engineer_features(df)
    df = final_preprocessing(df)
    return df

def make_prediction(df, model_path):
    model = joblib.load(model_path)
    features = ['Age', 'JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT', 'JD_Resume_Skill_Match_Score', 'Sentiment']
    return model.predict(df[features])
 