import streamlit as st
import pickle
import re
import nltk
from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean the resume text
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Web app
def main():
    st.set_page_config(page_title="Resume Screening App", page_icon=":briefcase:", layout="wide")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ['Home', 'About'])
    
    # Main title and header image
    st.title("💼 Resume Screening App")
    image = Image.open("C:\\Users\\HP\\Downloads\\interview-8467386_1280.webp")
    st.image(image, width=600, caption="Resume Screening", use_column_width=False)

    # Apply consistent styling across all sections
    st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: #00ffbf; /* Mint green text */
        font-family: 'Helvetica Neue', sans-serif;
    }
    .title {
        text-align: center;
        font-weight: bold;
        font-size: 32px;
        color: #00ffbf;
    }
    .sidebar .sidebar-content {
        background: #1E1E1E;
        color: #00ffbf;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, st, ul, li {
        color: #00ffbf !important;
    }
    .reportview-container .main .block-container {
        padding-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Home section
    if options == 'Home':
        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader('Upload Resume (PDF or TXT)', type=['txt', 'pdf'])

        if uploaded_file is not None:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

            with st.spinner("Processing your resume..."):
                cleaned_resume = clean_resume(resume_text)
                input_features = tfidfd.transform([cleaned_resume])
                prediction_id = clf.predict(input_features)[0]

            st.success("Resume processed successfully!")

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and Fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            st.subheader(f"Predicted Category: **{category_name}**")

    # About section
    elif options == 'About':
        st.subheader("About the App")
        st.write("""
        This Resume Screening App is designed to automatically classify resumes into various job categories.
        Upload a resume and the app will predict the most likely job category for the candidate.
        """)

        st.write("""
        **Features:**
        - Automatic resume text extraction and cleaning
        - Resume classification using a trained ML model
        - Simple and intuitive UI
        """)

if __name__ == "__main__":
    main()
