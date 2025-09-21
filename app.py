import streamlit as st
import sqlite3
import json
import os
import shutil
import chromadb
from pydantic import BaseModel, Field
from typing import List, Union
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from dotenv import load_dotenv
import fitz
import docx

# --- Configuration & DB Setup ---
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = "my-secret-key"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
DB_NAME = "analyzer.db"
CHROMA_PATH = "chroma_db"
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT NOT NULL,
            resume_filename TEXT NOT NULL,
            final_score REAL NOT NULL,
            verdict TEXT,
            identified_skills TEXT,
            found_skills TEXT,
            analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- ChromaDB Setup ---
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

chroma_client = get_chroma_client()
resume_collection = chroma_client.get_or_create_collection(name="resumes")

# --- Authentication Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- Data Extraction ---
def extract_text(file_path):
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        return "".join(page.get_text() for page in doc)
    elif file_path.lower().endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    raise ValueError("Unsupported file type")

# --- Scoring Logic ---
@st.cache_resource
def get_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_sentence_model()

def calculate_hard_match(resume_text, required_skills):
    score = 0
    found_skills = []
    resume_lower = resume_text.lower()
    for skill in required_skills:
        skill_lower = skill.lower()
        if skill_lower in resume_lower or fuzz.partial_ratio(skill_lower, resume_lower) > 85:
            score += 1
            found_skills.append(skill)
    return ((score / len(required_skills)) * 100 if required_skills else 100), found_skills

def calculate_semantic_match(resume_text, jd_text):
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)
    cos_sim = np.dot(resume_embedding, jd_embedding) / (np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding))
    return (cos_sim.item() + 1) / 2 * 100

def calculate_final_score(hard_match_score, semantic_match_score, weights=(0.4, 0.6)):
    return (weights[0] * hard_match_score) + (weights[1] * semantic_match_score)

# --- Skill Extractor (Simplified) ---
def get_skills_from_jd(jd_text: str) -> List[str]:
    # This is a simplified extraction since the original LLM is not directly available
    keywords = ["SQL", "Python", "Power BI", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Excel", "Data Visualization", "Data Cleaning", "Machine Learning", "NLP", "DAX", "Web Scraping", "Scikit-learn", "Statistical Analysis", "Team Collaboration", "Problem Solving", "Business Intelligence", "Analytics"]
    jd_lower = jd_text.lower()
    found = [kw for kw in keywords if kw.lower() in jd_lower]
    return list(set(found))

# --- Application UI and Logic ---
def login_page():
    st.title("Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and verify_password(password, user['hashed_password']):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Incorrect username or password")

def analyzer_page():
    st.title("Resume Analyzer")

    st.subheader("Select Analysis Type")
    col_single, col_bulk = st.columns(2)
    with col_single:
        if st.button("Single Analysis", use_container_width=True, type="secondary"):
            st.session_state.analysis_type = "Single"
    with col_bulk:
        if st.button("Bulk Analysis", use_container_width=True, type="secondary"):
            st.session_state.analysis_type = "Bulk"
    
    if "analysis_type" not in st.session_state:
        st.session_state.analysis_type = "Single"

    st.markdown(f"---")
    st.subheader(f"{st.session_state.analysis_type} Analysis")

    jd_file = st.file_uploader("Upload Job Description (JD)", type=['pdf', 'docx'])
    
    if st.session_state.analysis_type == "Single":
        resume_files = st.file_uploader("Upload Resume", type=['pdf', 'docx'], accept_multiple_files=False)
        resume_files = [resume_files] if resume_files else []
    else:
        resume_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True)

    if st.button(f"Analyze {len(resume_files)} Resume{'s' if len(resume_files) > 1 else ''}", use_container_width=True, type="primary"):
        if not jd_file or not resume_files:
            st.warning("Please upload a JD and at least one resume.")
            return

        with st.spinner("Analyzing..."):
            jd_path = os.path.join("temp_uploads", jd_file.name)
            with open(jd_path, "wb") as f:
                f.write(jd_file.getbuffer())
            jd_content = extract_text(jd_path)
            required_skills = get_skills_from_jd(jd_content)

            results_list = []
            conn = sqlite3.connect(DB_NAME)
            conn.row_factory = sqlite3.Row
            
            for resume_file in resume_files:
                resume_path = os.path.join("temp_uploads", resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getbuffer())
                
                try:
                    resume_content = extract_text(resume_path)
                    hard_score, found_skills = calculate_hard_match(resume_content, required_skills)
                    semantic_score = calculate_semantic_match(resume_content, jd_content)
                    final_score = calculate_final_score(hard_score, semantic_score)
                    score_val = float(f"{final_score:.2f}")
                    verdict = "High" if score_val > 80 else "Medium" if score_val > 60 else "Low"
                    
                    conn.execute(
                        "INSERT INTO analysis_results (job_description, resume_filename, final_score, verdict, identified_skills, found_skills) VALUES (?, ?, ?, ?, ?, ?)",
                        (jd_file.name, resume_file.name, score_val, verdict, json.dumps(required_skills), json.dumps(found_skills))
                    )
                    conn.commit()
                    
                    results_list.append({
                        "resume_filename": resume_file.name,
                        "final_score": score_val,
                        "verdict": verdict,
                        "found_skills": found_skills,
                        "hard_match_percent": f"{hard_score:.2f}",
                        "semantic_fit_percent": f"{semantic_score:.2f}"
                    })
                except Exception as e:
                    st.error(f"Error processing {resume_file.name}: {e}")
                finally:
                    os.remove(resume_path)
            
            conn.close()
            os.remove(jd_path)

            if st.session_state.analysis_type == "Single" and results_list:
                result = results_list[0]
                st.subheader("Analysis Results")
                
                # --- UI IMPROVEMENT: Displaying metrics in columns with color ---
                cols = st.columns(3)
                cols[0].markdown(
                    f"""<div style='border: 1px solid #4b5563; border-radius: 0.75rem; padding: 1rem; text-align: center;'>
                    <p style='color: #cbd5e1; font-size: 0.875rem;'>Hard Match</p>
                    <p style='font-size: 2.25rem; font-weight: bold; color: #5eead4;'>{result['hard_match_percent']}%</p>
                    </div>""", unsafe_allow_html=True
                )
                cols[1].markdown(
                    f"""<div style='border: 1px solid #4b5563; border-radius: 0.75rem; padding: 1rem; text-align: center;'>
                    <p style='color: #cbd5e1; font-size: 0.875rem;'>Semantic Fit</p>
                    <p style='font-size: 2.25rem; font-weight: bold; color: #5eead4;'>{result['semantic_fit_percent']}%</p>
                    </div>""", unsafe_allow_html=True
                )
                cols[2].markdown(
                    f"""<div style='background-color: #0d9488; border-radius: 0.75rem; padding: 1rem; text-align: center; color: white;'>
                    <p style='font-size: 0.875rem;'>Final Score / Verdict</p>
                    <p style='font-size: 2.25rem; font-weight: bold;'>{result['final_score']}% ({result['verdict']})</p>
                    </div>""", unsafe_allow_html=True
                )

                st.markdown("---")
                
                # --- UI IMPROVEMENT: Skills display in a cleaner format ---
                st.subheader("Skills Analysis")
                cols_skills = st.columns(2)
                with cols_skills[0]:
                    st.write("Required Skills (from JD):")
                    skills_list = ", ".join(required_skills)
                    st.markdown(f"**{skills_list}**")
                with cols_skills[1]:
                    st.write("Found Skills (in Resume):")
                    if result['found_skills']:
                        found_list = ", ".join(result['found_skills'])
                        st.markdown(f"**{found_list}**")
                    else:
                        st.write("No direct matches found.")
            
            else: # Bulk Analysis
                results_list.sort(key=lambda x: x['final_score'], reverse=True)
                st.subheader("Bulk Analysis Results")
                table_data = [{"Rank": i + 1, "Resume": r['resume_filename'], "Final Score": f"{r['final_score']:.2f}%", "Verdict": r['verdict']} for i, r in enumerate(results_list)]
                st.table(table_data)

def bulk_indexer():
    st.subheader("📚 Bulk Resume Indexing")
    st.write("Add multiple resumes to the Vector Database at once to enable search features.")
    files = st.file_uploader("Select Resumes to Index", type=['pdf', 'docx'], accept_multiple_files=True)
    
    if st.button(f"Index {len(files)} Resumes" if files else "Index Resumes", disabled=not files):
        if not files:
            st.warning("Please select at least one resume file.")
            return

        with st.spinner("Indexing resumes..."):
            progress = {}
            for file in files:
                file_path = os.path.join("temp_uploads", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                try:
                    resume_text = extract_text(file_path)
                    resume_embedding = model.encode(resume_text).tolist()
                    resume_collection.add(
                        embeddings=[resume_embedding],
                        documents=[resume_text],
                        metadatas=[{"filename": file.name}],
                        ids=[file.name]
                    )
                    progress[file.name] = "Success"
                except Exception as e:
                    progress[file.name] = f"Error: {e}"
                finally:
                    os.remove(file_path)
            
            st.success("Indexing complete!")
            st.json(progress)

def keyword_search_page():
    st.title("Keyword Search")
    st.write("Search the indexed resumes for specific skills or keywords.")
    
    keywords = st.text_area("Enter keywords separated by commas (e.g., Python, React, Data Science)")
    
    if st.button("Search Resumes", type="primary"):
        if not keywords.strip():
            st.warning("Please enter at least one keyword.")
            return

        with st.spinner("Searching..."):
            search_query = " ".join([kw.strip() for kw in keywords.split(',') if kw.strip()])
            if not search_query:
                st.warning("Please enter valid keywords.")
                return

            query_embedding = model.encode(search_query).tolist()
            results = resume_collection.query(query_embeddings=[query_embedding], n_results=10)
            
            matches = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity_score = 100 / (1 + distance)
                    matches.append({"resume_filename": doc_id, "score": f"{similarity_score:.2f}%"})

            if matches:
                st.subheader("Top Matches")
                st.table(matches)
            else:
                st.info("No similar resumes found for the given keywords.")

    st.markdown("---")
    bulk_indexer()

def admin_page():
    st.title("Admin Dashboard")
    
    # Analytics
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    total_analyses = conn.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    total_indexed_resumes = resume_collection.count()
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Analyses", total_analyses)
    with cols[1]:
        st.metric("Indexed Resumes", total_indexed_resumes)
    with cols[2]:
        st.metric("Registered Users", total_users)

    # Tabs for management
    tab1, tab2, tab3, tab4 = st.tabs(["User Management", "Resume Vector Store", "Analysis History", "Add User"])

    with tab1:
        st.subheader("User Management")
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        users = conn.execute("SELECT id, username FROM users").fetchall()
        conn.close()
        
        for user in users:
            col1, col2 = st.columns([0.7, 0.3])
            col1.write(user['username'])
            # Check if the logged-in user is not the one being deleted
            if col2.button("Delete User", key=f"del_user_{user['username']}", disabled=(user['username'] == st.session_state.username)):
                try:
                    conn = sqlite3.connect(DB_NAME)
                    conn.execute("DELETE FROM users WHERE username = ?", (user['username'],))
                    conn.commit()
                    conn.close()
                    st.success(f"User '{user['username']}' deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete user: {e}")

    with tab2:
        st.subheader("Resume Vector Store")
        resumes = resume_collection.get()
        if resumes['ids']:
            for filename in resumes['ids']:
                col1, col2 = st.columns([0.7, 0.3])
                col1.write(filename)
                if col2.button("Delete Resume", key=f"del_res_{filename}"):
                    try:
                        resume_collection.delete(ids=[filename])
                        st.success(f"Resume '{filename}' deleted from vector store.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete resume: {e}")
        else:
            st.info("No resumes indexed yet.")

    with tab3:
        st.subheader("Detailed Analysis History")
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        jobs = conn.execute("SELECT DISTINCT job_description FROM analysis_results ORDER BY job_description").fetchall()
        conn.close()
        
        job_options = [job['job_description'] for job in jobs]
        selected_job = st.selectbox("Select a Job to see full analysis:", [""] + job_options)

        if st.button("Clear History for Job", disabled=not selected_job):
            try:
                conn = sqlite3.connect(DB_NAME)
                conn.execute("DELETE FROM analysis_results WHERE job_description = ?", (selected_job,))
                conn.commit()
                conn.close()
                st.success(f"History for job '{selected_job}' has been deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete history: {e}")
        
        if selected_job:
            conn = sqlite3.connect(DB_NAME)
            conn.row_factory = sqlite3.Row
            results = conn.execute(
                "SELECT * FROM analysis_results WHERE job_description = ? ORDER BY final_score DESC", (selected_job,)
            ).fetchall()
            conn.close()
            
            if results:
                st.dataframe([dict(row) for row in results], use_container_width=True)
            else:
                st.info("No results found for this job.")

    with tab4:
        st.subheader("Add New User")
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            submitted = st.form_submit_button("Add User")
            
            if submitted:
                if new_username and new_password:
                    conn = sqlite3.connect(DB_NAME)
                    conn.row_factory = sqlite3.Row
                    hashed_password = get_password_hash(new_password)
                    try:
                        conn.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (new_username, hashed_password))
                        conn.commit()
                        st.success(f"User '{new_username}' created successfully!")
                    except sqlite3.IntegrityError:
                        st.error("Username already registered.")
                    finally:
                        conn.close()
                else:
                    st.error("Username and password cannot be empty.")

# --- Main App Logic ---
setup_database()

# State management for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Analyzer"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar Navigation buttons
with st.sidebar:
    st.title("Navigation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyzer", use_container_width=True):
            st.session_state.page = "Analyzer"
    with col2:
        if st.button("Search", use_container_width=True):
            st.session_state.page = "Keyword Search"
    
    if st.button("Admin", use_container_width=True):
        st.session_state.page = "Admin"

    st.markdown("---")
    
    if st.session_state.logged_in:
        st.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.page = "Analyzer"
            st.rerun()

# Page routing based on session state
if st.session_state.page == "Analyzer":
    analyzer_page()
elif st.session_state.page == "Keyword Search":
    keyword_search_page()
elif st.session_state.page == "Admin":
    if st.session_state.logged_in:
        admin_page()
    else:
        login_page()
elif st.session_state.page == "Admin Login": # Handle the login page state
    login_page()
    
# Inject custom CSS
st.markdown("""
<style>
/* Streamlit-specific overrides */
[data-testid="stSidebar"] {
    background: #111827;
    color: white;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

.st-emotion-cache-1cypcdb {
    background-color: #111827;
    border-radius: 0.5rem;
    border-color: #374151;
}

/* Aurora Background */
body {
    background-color: #111827; /* Dark gray fallback */
    background-image: 
        radial-gradient(at 20% 20%, hsla(210, 80%, 40%, 0.3) 0px, transparent 50%),
        radial-gradient(at 80% 20%, hsla(28, 100%, 53%, 0.2) 0px, transparent 50%),
        radial-gradient(at 80% 80%, hsla(340, 90%, 60%, 0.2) 0px, transparent 50%),
        radial-gradient(at 20% 80%, hsla(190, 90%, 50%, 0.3) 0px, transparent 50%);
    background-attachment: fixed;
}

/* Main content container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
    margin: 0 auto;
}

/* Title styling */
h1 {
    font-size: 3rem;
    font-weight: bold;
    background-image: linear-gradient(to right, #22d3ee, #d9f99d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

/* Subheader styling */
h3 {
    color: #cbd5e1;
}

/* Button styling */
.stButton>button {
    background-color: #1f2937;
    border: 1px solid #4b5563;
    color: white;
    font-weight: bold;
    border-radius: 0.5rem;
}
.stButton>button:hover {
    background-color: #374151;
}
.stButton>button:disabled {
    background-color: #4b5563;
    color: #94a3b8;
}

/* Primary buttons */
.stButton>button.primary {
    background-color: #0d9488;
    border: none;
}
.stButton>button.primary:hover {
    background-color: #0f766e;
}

/* Tabs */
.st-emotion-cache-1cypcdb {
    background-color: #1f2937;
    border-radius: 0.5rem;
}

/* Metric cards */
.st-emotion-cache-pkb1o {
    background-color: #1f2937;
    border: 1px solid #4b5563;
    border-radius: 0.75rem;
    padding: 1rem;
    color: #cbd5e1;
}
.st-emotion-cache-pkb1o div[data-testid="stMetricValue"] {
    font-size: 2.25rem;
    font-weight: bold;
    color: #5eead4;
}

/* Dataframe Styling */
.st-emotion-cache-1k4w16m {
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)