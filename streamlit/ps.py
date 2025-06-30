import streamlit as st
import os
import fitz  # PyMuPDF
import pandas as pd
import google.generativeai as genai
import json
from datetime import datetime
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Paper Screening Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Default Screening Criteria (from your original script) ---
DEFAULT_INCLUSION_CRITERIA = """
- Empirical-based observational studies (prospective and retrospective cohorts, case-control, cross-sectional)
- Quantitative studies and mixed studies
- Randomized Controlled Trials (RCTs)
- Pre-design-post design studies
- Modeling studies
- Studies involving E-healthcare systems
- Studies conducted in English
- Studies published from year 2000 onwards
"""

DEFAULT_EXCLUSION_CRITERIA = """
- Review studies (systematic reviews, narrative reviews, review of reviews)
- Conference abstracts, proceedings, and theses
- Studies in languages other than English
- Studies involving animal models
- Studies published before the year 2000
- Opinion-based grey literature (commentaries, editorials, brief reports, perspectives, analyses)
"""

# --- Core Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts all text from an uploaded PDF file object."""
    text = ""
    try:
        # Open the PDF from the in-memory file object
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""
    # Truncate text to a manageable size to save costs and time
    return text.strip()[:20000]

def get_screening_analysis_from_gemini(text, inclusion_criteria, exclusion_criteria, model):
    """
    Asks Gemini to extract data and make a decision based on dynamic rules.
    Returns a Python dictionary parsed from the model's JSON output.
    """
    prompt = f"""
    You are an expert academic research assistant performing a systematic literature review screening.
    Analyze the text from the research paper provided below.

    **Your Tasks:**
    1.  Extract the required bibliographic information.
    2.  Based on the rules provided, decide whether to "Include" or "Exclude" the paper.
    3.  Provide a brief reason for your decision.

    **Screening Rules:**
    **Inclusion Criteria:**
    {inclusion_criteria}

    **Exclusion Criteria:**
    {exclusion_criteria}


    **Output Format:**
    You MUST return your response as a single, valid JSON object. Do not add any text before or after the JSON.
    Use the following structure. If a field is not found, use "N/A" or an empty list [].

    {{
      "item_type": "e.g., Journal Article, Conference Paper",
      "publication_year": "YYYY",
      "authors": ["Author One", "Author Two"],
      "title": "Full Title of the Paper",
      "isbn": "ISBN if available, else N/A",
      "issn": "ISSN if available, else N/A",
      "doi": "DOI if available, else N/A",
      "url": "URL if available, else N/A",
      "abstract": "The full abstract of the paper.",
      "language": "e.g., English",
      "final_decision": "Include or Exclude",
      "reason": "Brief justification based on the rules."
    }}

    --- TEXT TO ANALYZE ---
    {text}
    """

    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_text)
        return data
    except json.JSONDecodeError:
        st.warning(f"⚠️ Gemini did not return valid JSON. Trying to extract content anyway...")
        st.text_area("Gemini's Raw Response:", response.text, height=150)
        return {"error": "Failed to parse JSON response from API."}
    except Exception as e:
        st.error(f"⚠️ An API error occurred: {e}")
        return {"error": f"API Error: {str(e)}"}

# --- Streamlit UI ---

st.title("📄 AI-Powered Paper Screening Assistant")
st.markdown("Upload your PDF research papers and let Gemini AI perform an initial screening based on your criteria.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key Input
    api_key = st.text_input("Enter your Gemini API Key", type="password")

    # Dynamic Screening Criteria
    st.subheader("Screening Rules")
    inclusion_criteria = st.text_area(
        "Inclusion Criteria",
        value=DEFAULT_INCLUSION_CRITERIA,
        height=250
    )
    exclusion_criteria = st.text_area(
        "Exclusion Criteria",
        value=DEFAULT_EXCLUSION_CRITERIA,
        height=250
    )

# --- Main Page for File Upload and Processing ---
uploaded_files = st.file_uploader(
    "Upload PDF files for screening",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Start Screening Process"):
    if not api_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            st.info(f"🚀 Found {len(uploaded_files)} files. Starting the screening process...")

            # Define the exact column order for the final CSV
            csv_columns = [
                'Sl No.', 'Item Type', 'Publication Year', 'Author', 'Title', 'ISBN',
                'ISSN', 'DOI', 'Url', 'Abstract', 'Note', 'Date', 'Language',
                'Final decision', 'Include/exclude reason'
            ]
            results = []

            # Progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, pdf_file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.info(f"Processing ({idx+1}/{len(uploaded_files)}): {pdf_file.name}")

                text = extract_text_from_pdf(pdf_file)
                if not text:
                    row_data = {col: 'N/A' for col in csv_columns}
                    row_data['Sl No.'] = idx + 1
                    row_data['Title'] = pdf_file.name
                    row_data['Include/exclude reason'] = "Failed to read text from PDF file."
                    results.append(row_data)
                    continue

                analysis_data = get_screening_analysis_from_gemini(text, inclusion_criteria, exclusion_criteria, model)

                if "error" in analysis_data:
                    row_data = {col: 'N/A' for col in csv_columns}
                    row_data['Sl No.'] = idx + 1
                    row_data['Title'] = pdf_file.name
                    row_data['Include/exclude reason'] = analysis_data["error"]
                    results.append(row_data)
                    continue

                # Map the JSON data to our CSV columns
                row_data = {
                    'Sl No.': idx + 1,
                    'Item Type': analysis_data.get('item_type', 'N/A'),
                    'Publication Year': analysis_data.get('publication_year', 'N/A'),
                    'Author': ', '.join(analysis_data.get('authors', [])),
                    'Title': analysis_data.get('title', pdf_file.name),
                    'ISBN': analysis_data.get('isbn', 'N/A'),
                    'ISSN': analysis_data.get('issn', 'N/A'),
                    'DOI': analysis_data.get('doi', 'N/A'),
                    'Url': analysis_data.get('url', 'N/A'),
                    'Abstract': analysis_data.get('abstract', 'N/A'),
                    'Note': '',
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Language': analysis_data.get('language', 'N/A'),
                    'Final decision': analysis_data.get('final_decision', 'N/A'),
                    'Include/exclude reason': analysis_data.get('reason', 'N/A')
                }
                results.append(row_data)

            status_text.success("✅ Analysis complete!")
            
            if results:
                st.header("Screening Results")
                df = pd.DataFrame(results, columns=csv_columns)
                st.dataframe(df)

                # Convert DataFrame to CSV for download
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')

                csv_data = convert_df_to_csv(df)
                
                st.download_button(
                   label="📥 Download Results as CSV",
                   data=csv_data,
                   file_name=f"Paper_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                   mime="text/csv",
                )

        except Exception as e:
            st.error(f"An unexpected error occurred during setup or processing: {e}")
            st.info("This could be due to an invalid API key or a problem with the Gemini service.")

st.markdown(
    "<hr style='margin-top:2em;margin-bottom:0.5em;border:1px solid #eee'>"
    "<div style='text-align:center; color:gray; font-size:0.95em;'>"
    "All rights reserved by <a href='https://www.linkedin.com/in/ekram-asif/' style='color:white; text-decoration:none;' target='_blank'>Ekram Asif</a>"
    "</div>",
    unsafe_allow_html=True
)