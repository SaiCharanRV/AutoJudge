import streamlit as st
import joblib
import numpy as np
import sys
import os

# Link to your source folder for modularity
sys.path.append(os.path.abspath("src"))
from features import get_extra_features

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚖️",
    layout="wide"
)

# --- ADVANCED CSS FOR UI STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&family=Poppins:wght@600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Result card styling */
    .result-card {
        padding: 24px;
        border-radius: 20px;
        background: #ffffff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #f0f0f0;
    }

    .metric-label {
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
        letter-spacing: 1.5px;
        color: #888;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
        margin: 0;
    }

    /* Gradient Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        height: 12px;
    }
    
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #eee !important;
        background-color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    tfidf = joblib.load("data/processed/tfidf.pkl")
    clf = joblib.load("data/processed/classifier.pkl")
    reg = joblib.load("data/processed/regressor.pkl")
    return tfidf, clf, reg

tfidf, clf, reg = load_assets()

# --- SESSION STATE & CLEAR LOGIC ---
# Initialize session state for all input areas if they don't exist
if 'desc_val' not in st.session_state:
    st.session_state.desc_val = ""
if 'inp_val' not in st.session_state:
    st.session_state.inp_val = ""
if 'out_val' not in st.session_state:
    st.session_state.out_val = ""

def clear_all_inputs():
    """Callback function to reset all session state values."""
    st.session_state.desc_val = ""
    st.session_state.inp_val = ""
    st.session_state.out_val = ""
    # Clear the text area components via their internal keys
    st.session_state.desc_key = ""
    st.session_state.inp_key = ""
    st.session_state.out_key = ""

# --- SIDEBAR ENGINE PANEL ---
with st.sidebar:
    st.markdown("### 🏎️ AutoJudge Engine")
    st.info("System uses Natural Language Processing to judge complexity based on descriptions and constraints.")
    
    st.divider()
    
    # st.caption("v1.0.0 | Developer Mode")
    # st.markdown("""
    #     <div style="display: flex; align-items: center;">
    #         <span style="font-size: 24px; margin-right: 10px;">🎯</span>
    #         <span style="font-size: 18px; font-weight: 700;">Accuracy: <span style="color: #2e7d32;">57.59%</span></span>
    #     </div>
    # """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### ⚙️ Engine Controls")
    analyze_btn = st.button("🚀 Analyze Problem", use_container_width=True, type="primary")
    # THE FIX: Added on_click callback to reset the fields
    st.button("🧹 Clear All", use_container_width=True, on_click=clear_all_inputs)

# --- MAIN DASHBOARD ---
st.markdown("<h1 style='font-family: Poppins; font-weight: 800; margin-bottom: 0;'>AutoJudge Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; font-size: 1.1rem; font-style: italic;'>Automated Programming Difficulty Classification & Scoring</p>", unsafe_allow_html=True)

# Container for results to appear at the top
result_placeholder = st.container()

st.divider()

# INPUT AREAS
st.markdown("### 📋 Problem Specifications")

# Use 'key' and 'value' bound to session state to allow clearing
desc = st.text_area("Detailed Problem Description", 
                    value=st.session_state.desc_val, 
                    height=150, 
                    placeholder="Enter main problem statement here...",
                    key="desc_key")

col_spec1, col_spec2 = st.columns(2)
with col_spec1:
    inp_desc = st.text_area("Input Specifications", 
                            value=st.session_state.inp_val, 
                            height=100, 
                            placeholder="Constraints, data types...",
                            key="inp_key")
with col_spec2:
    out_desc = st.text_area("Output Specifications", 
                             value=st.session_state.out_val, 
                             height=100, 
                             placeholder="Return format...",
                             key="out_key")

# Sync session state with current input on every run
st.session_state.desc_val = desc
st.session_state.inp_val = inp_desc
st.session_state.out_val = out_desc

# --- PREDICTION & CLAMPING LOGIC ---
if analyze_btn:
    if desc.strip():
        # 1. Feature Extraction
        full_text = f"{desc} {inp_desc} {out_desc}"
        X_t = tfidf.transform([full_text]).toarray()
        X_m = get_extra_features(full_text).reshape(1, -1)
        X_f = np.hstack([X_t, X_m])
        
        # 2. Get AI Predictions
        label = clf.predict(X_f)[0]
        raw_score = float(reg.predict(X_f)[0])
        
        # 3. Apply Professional Clamping Rules
        if label == "easy":
            final_score, display_label, color = min(raw_score, 3.33), "EASY", "#00C851"
        elif label == "medium":
            final_score, display_label, color = max(3.33, min(raw_score, 6.66)), "MEDIUM", "#ffbb33"
        else: # hard
            final_score, display_label, color = max(6.67, raw_score), "HARD", "#ff4444"

        # 4. SHOW RESULTS
        with result_placeholder:
            st.markdown("### 📊 Evaluation Results")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"""
                    <div class="result-card">
                        <p class="metric-label">Classification</p>
                        <p class="metric-value" style="color: {color};">{display_label}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                    <div class="result-card">
                        <p class="metric-label">Difficulty Score</p>
                        <p class="metric-value">{final_score:.2f} <span style="font-size: 16px; color: #aaa;">/ 10</span></p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(final_score / 10)
    else:
        st.warning("⚠️ Please enter a problem description before analyzing.")