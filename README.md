# Automated-Transcript-Analysis-to-Optimize-Recruitment-Decision


### Project Statement

**Client:** One of the leading HR-tech companies that provides end-to-end recruitment solutions for top enterprises globally. They conduct thousands of interviews weekly across different geographies and roles, leading to high manual effort in evaluating and shortlisting candidates..

**Business Problem:**
The recruitment team spends considerable time manually reviewing and summarizing lengthy interview transcripts. This process is slow, inconsistent, and often overlooks key details like skills, sentiments, and red flags. As interview volumes grow, scaling becomes a major challenge. An automated solution is needed to generate concise summaries and extract critical insights for quicker, better hiring decisions.

**Business Objective:**
Maximize the efficiency and clarity of interview evaluations.

**Business Constraint:**
Minimize computational cost and processing time while maintaining performance.

**Business Success Criteria:**
Minimize candidate screening time by at least 50%.

**ML Success Criteria:**
Achieve model accuracy of at least 90%.

**Economic Success Criteria:**
Achieve at least 40% reduction in candidate evaluation cost.



# 📌 Automated Candidate Screening and Selection System

## 📖 Overview

This project is a **Streamlit-based web application** that automates candidate screening by analyzing resumes, interview transcripts, and job descriptions. It uses **LLMs**, **NLP techniques**, and a trained ML model to predict whether a candidate is likely to be selected or not.

---

## 🎯 Objectives

- Clean and process resumes, transcripts, and job descriptions.
- Extract key features using **sentiment analysis**, **summarization**, **skill matching**, and **semantic similarity**.
- Predict candidate selection using a trained **XGBoost classification model**.
- Provide an intuitive Streamlit UI for HR teams to upload candidate files and get instant predictions.

---

## 🧠 Technologies Used

| Category         | Tools / Libraries |
|------------------|------------------|
| Programming      | Python            |
| Web App          | Streamlit         |
| Data Handling    | Pandas, NumPy     |
| NLP Models       | HuggingFace Transformers, Sentence Transformers |
| ML Model         | XGBoost, Scikit-learn |
| Environment      | Google Colab (for training), Local system (for deployment) |
| Model Deployment | Streamlit         |

---

## 🛠️ Features

- 📂 Upload `.csv` or `.xlsx` with resume, transcript, JD
- ✨ Automated cleaning of raw text (removal of timestamps, noise, formatting)
- 💬 Sentiment analysis of transcript
- 📝 Summarization of interview content
- 📊 Semantic similarity between Resume/Transcript and JD
- 💡 Skill extraction and skill-match scoring
- ✅ Final prediction: **Selected** or **Not Selected**
- ⬇️ Download predictions as CSV

---

## 🧪 Dataset

- 🔧 **Synthetic dataset** generated using **Grok** and **ChatGPT**
- ~272 candidate profiles
- Features:
  - `Resume`, `Transcript`, `Job Description`
  - `Age`, `Experience`, `Selected/Not-selected`

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ALLAM-SRIDHAR/Automated-Transcript-Analysis-to-Optimize-Recruitment-Decision.git
cd Automated-Transcript-Analysis-to-Optimize-Recruitment-Decision

# Install dependencies
pip install -r requirements.txt
