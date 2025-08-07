import streamlit as st
import pandas as pd
import joblib

from recruitment_pipeline import clean_data, engineer_features, final_preprocessing

# ==================== Streamlit UI ====================
st.title("Candidate Selection Predictor 🎯")
st.write("Upload a CSV or Excel file to predict candidate selection using the trained model.")

uploaded_file = st.file_uploader("Upload file (.csv or .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Step 1: Read file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Step 2: Clean + Feature Engineering
    with st.spinner("Cleaning and processing data..."):
        df_cleaned = clean_data(df)
        df_features = engineer_features(df_cleaned)
        df_final = final_preprocessing(df_features)

    # Step 3: Load model
    try:
        model = joblib.load("final_xgb_model.pkl")
        st.success("Model loaded!")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    # Step 4: Make predictions
    try:
        input_features = ['Age', 'JD_Resume_Score_BERT', 'JD_Transcript_Score_BERT',
                          'JD_Resume_Skill_Match_Score', 'Sentiment']
        predictions = model.predict(df_final[input_features])
        df_final['Prediction'] = predictions
        df_final['Prediction_Label'] = df_final['Prediction'].map({1: "Selected", 0: "Not Selected"})
        st.success("Predictions completed!")
        st.dataframe(df_final[['Prediction_Label', 'Orig_Age', 
                       'Orig_JD_Resume_Score_BERT', 
                       'Orig_JD_Transcript_Score_BERT', 
                       'Orig_JD_Resume_Skill_Match_Score','Summary']])

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Step 5: Download option
    csv_output = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", data=csv_output, file_name='predicted_candidates.csv', mime='text/csv')