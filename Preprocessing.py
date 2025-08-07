##### Understanding the dataset #####

import pandas as pd

# Load the dataset
#data = pd.read_csv(r"C:\Users\ADMIN\Desktop\Project_276\Transcript Dataset.csv")
data = pd.read_excel(r"C:\Users\ADMIN\Desktop\Project_276\Syntheic Generated Dataset.xlsx")
# 1.1 Inspect the dataset
print("First few rows of the dataset:")
data_head = data.head()

# 1.2 Identify the columns
columns = data.columns

# 1.3 Check data types
data_types = data.dtypes

# 1.4 Check for missing values
missing_values = data.isnull().sum()

# 1.5 Get summary statistics for numerical columns
summary_stats = data.describe()

# 1.6 Examine textual data (check first 5 transcript entries if available)
# Assuming there's a column for interview transcripts, change 'transcript_column' to the actual name
transcript_sample = data['Transcript'].head() if 'Transcript' in data.columns else "No transcript column found."

# 1.7 Identify target variables (Assuming 'skill_match' and 'sentiment' are target columns, adjust accordingly)
target_columns = ['skill_match', 'sentiment'] if 'skill_match' in data.columns and 'sentiment' in data.columns else "No target columns identified."

data_head, columns, data_types, missing_values, summary_stats, transcript_sample, target_columns


#####Preprocessing the dataset or cleaning dataset #####

# 1. Handle missing data (if any)
data_cleaned = data.copy()

# Check for missing values again just in case (although we have none)
missing_values_cleaned = data_cleaned.isnull().sum()

# If there are missing values in a column, you can fill them as follows
# For numerical columns (e.g., Age), fill with the median or mean
data_cleaned['Age'].fillna(data_cleaned['Age'].median(), inplace=True)

# For textual columns (e.g., Transcript, Resume), you can fill missing values with 'No information provided'
data_cleaned['Transcript'].fillna('No information provided', inplace=True)
data_cleaned['Resume'].fillna('Not available',inplace=True)
data_cleaned['Job Description'].fillna('Not available',inplace=True)

# 2. Remove duplicates

#to check duplicate
data_cleaned.duplicated().sum()

#to view duplicate
# Show the duplicate rows
duplicate_rows = data_cleaned[data_cleaned.duplicated()]
print(duplicate_rows)

#to remove duplicate
data_cleaned.drop_duplicates(inplace=True)

# 3. Standardize Data Formatting
# Standardize experience entries (e.g., "2+ years" to "2 years")
#data_cleaned['Experience'] = data_cleaned['Experience'].replace(r'\+.*years', 'years', regex=True)

import re
# Function to standardize and convert experience to numeric form
def standardize_experience(experience):
    # Remove any non-numeric characters after the first number
    experience = re.sub(r'[^0-9.]+', '', str(experience))  # Remove any non-numeric characters (except for dot)
    # Convert to float and round down (int truncates decimals)
    if experience:
        return int(float(experience))  # Convert to integer after rounding down
    return 0  # In case of missing or incorrect data

# Apply the function to the 'Experience' column
data_cleaned['Experience'] = data_cleaned['Experience'].apply(standardize_experience)
# Show the cleaned 'Experience' column
data_cleaned['Experience'].head()


# Convert Interview Date to a standard format (e.g., YYYY-MM-DD)
data_cleaned['Interview Date'] = pd.to_datetime(data_cleaned['Interview Date'], errors='coerce')

# 4. Handle Inconsistent Data
# Standardize text columns like 'Position' and 'Technical Skills'
data_cleaned['Position'] = data_cleaned['Position'].str.lower()
data_cleaned['Technical Skills'] = data_cleaned['Technical Skills'].str.lower()

#data_cleaned['Transcript'] = data_cleaned['Transcript'].str.lower()

# 5. Remove special characters and unwanted text from textual columns

#import re

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

# Apply text cleaning to Resume column
data_cleaned['Resume'] = data_cleaned['Resume'].apply(clean_resume)


# Function to clean the Job Description text

#import re

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

# Apply text cleaning to Job Description column
data_cleaned['Job Description'] = data_cleaned['Job Description'].apply(clean_job_description)

# Display the cleaned columns to verify
data_cleaned[['Resume', 'Job Description']].head()


#For transcript column
# Function to clean the Transcript text while adding punctuation and structure

#import re

# Function to clean the Transcript text while adding punctuation and structure
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
    transcript = re.sub(r'(interviewer[^:]*)(?=\s*[^:]*\?)', r'\1?', transcript)
    
    # Add periods after candidate statements if they don't already end with one
    transcript = re.sub(r'(candidate[^:]*)(?=\s*[^.]*$)', r'\1.', transcript)

    # Step 6: Convert all text to lowercase
    transcript = transcript.lower()

    # Step 7: Capitalize the first letter of each sentence after removing unwanted characters
    transcript = transcript.capitalize()

    # Step 8: Replace interviewer and candidate labels with more readable structure
    transcript = transcript.replace('interviewer', 'Interviewer:').replace('candidate', 'Candidate:')

    return transcript

# Apply the new cleaning function to the Transcript column
data_cleaned['Transcript'] = data_cleaned['Transcript'].apply(clean_transcript_with_punctuation)

# Display the cleaned Transcript column to verify
data_cleaned['Transcript'].head()


# 6. Handle Categorical Data (e.g., Position, Experience)
# You can apply label encoding or one-hot encoding depending on your model choice
# For simplicity, here’s an example of label encoding for 'Position' (assuming Position has many categories)
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data_cleaned['Position'] = le.fit_transform(data_cleaned['Position'])
'''

'''
# Add a catch-all "Unseen" category for new positions that will be encountered in the future
data_cleaned['Position_Unseen'] = 0

# Apply one-hot encoding
encoded_positions = pd.get_dummies(data_cleaned['Position'], prefix='Position')

# If a new category appears in the future, it will be mapped to the 'Position_Unseen' column
# Concatenate the new columns
data_cleaned = pd.concat([data_cleaned, encoded_positions], axis=1)

# Drop the original 'Position' column
data_cleaned.drop('Position', axis=1, inplace=True)
'''

# Outlier Detection (for numerical columns like Age)
import matplotlib.pyplot as plt

# Visualize outliers using a boxplot
plt.boxplot(data_cleaned['Age'])
plt.title("Boxplot for Age")
plt.show()


# Visualize outliers using a boxplot
plt.boxplot(data_cleaned['Experience'])
plt.title("Boxplot for Experience")
plt.show()

# If you decide to remove outliers, you can filter out values beyond certain thresholds:
# For example, removing ages greater than 35 or less than 20
data_cleaned = data_cleaned[(data_cleaned['Age'] >= 20) & (data_cleaned['Age'] <= 35)]

# Finally, inspect the cleaned dataset
data_cleaned.head()


# Save the cleaned dataset to a CSV file
output_file = r"C:\Users\ADMIN\Desktop\Project_276\Cleaned_Transcript_Data.csv"
data_cleaned.to_csv(output_file, index=False)
# Confirmation message
print(f"Cleaned data saved to {output_file}")

###########################################################################

#Feature Engineering


#######Sentiment Analysis#############
## Models used:  Vader(from nltk), text blob, transformers(BERT),Spacy Textblob, etc..


#######Sentiment Analysis#############
#vader(nltk)
#pip install nltk

import nltk
nltk.download('vader_lexicon')

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score (polarity) from the transcript
def get_sentiment_score(text):
    # If the text is empty or NaN, return 0 (neutral sentiment)
    if not isinstance(text, str) or not text.strip():
        return 0.0
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score ranges from -1 (negative) to +1 (positive)

# Apply sentiment analysis to the 'Transcript' column
data_cleaned['Sentiment_Score'] = data_cleaned['Transcript'].apply(get_sentiment_score)

# Show the first few rows of the cleaned data with sentiment scores
print(data_cleaned[['Transcript', 'Sentiment_Score']].head())

# Save the cleaned dataset to a CSV file
output_file = r"C:\Users\ADMIN\Desktop\Project_276\Cleaned_Transcript_Data1.csv"
data_cleaned.to_csv(output_file, index=False)

## Textblob
#pip install textblob

from textblob import TextBlob

# Function to get sentiment polarity from TextBlob
def get_sentiment_textblob(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0  # Neutral if no valid text
    sentiment = TextBlob(text).sentiment.polarity  # Range: -1 (negative) to +1 (positive)
    return sentiment

# Apply sentiment analysis using TextBlob to the 'Transcript' column
data_cleaned['Sentiment_Score_TextBlob'] = data_cleaned['Transcript'].apply(get_sentiment_textblob)

# Show the first few rows of the cleaned data with sentiment scores from TextBlob
print(data_cleaned[['Transcript', 'Sentiment_Score_TextBlob']].head())


#transformers(BERT)
#pip install transformers
#pip install tf-keras
#pip install tensorflow

from transformers import pipeline

#pip install torch
#import torch
#print(torch.__version__)

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

# Save the cleaned dataset to a CSV file
output_file = r"C:\Users\ADMIN\Desktop\Project_276\Cleaned_Transcript_Data.csv"
data_cleaned.to_csv(output_file, index=False)

###############################

# Function to classify sentiment based on Transformers sentiment score
def classify_sentiment_transformers(score):
    if score >= 0.7:
        return 'Positive'
    elif score >= 0.4:
        return 'Neutral'
    else:
        return 'Negative'

#########################
#import pandas as pd
#data_cleaned=pd.read_csv(r"C:\Users\ADMIN\Desktop\Project_276\Cleaned_Transcript_Data4.csv")

'''
#pip install spacy
import spacy
#print(spacy.__version__)
#pip install spacytextblob
import spacytextblob
print(spacytextblob.__version__)  # Check the version of spacytextblob

#from spacytextblob import SpacyTextBlob
##!python -m spacy download en_core_web_sm     ##in console
# Load a spaCy model
nlp = spacy.load('en_core_web_sm')

# Adding TextBlob to the pipeline for sentiment analysis
nlp.add_pipe("spacytextblob")

# Function to get sentiment
def spacy_sentiment(text):
    doc = nlp(text)
    return doc._.sentiment.polarity

# Apply spaCy sentiment analysis to the 'Transcript' column
data_cleaned['Sentiment_Score_SpaCy'] = data_cleaned['Transcript'].apply(spacy_sentiment)

# Show the first few rows of the cleaned data with sentiment scores
print(data_cleaned[['Transcript', 'Sentiment_Score_SpaCy']].head())
'''
######################################################
############################
