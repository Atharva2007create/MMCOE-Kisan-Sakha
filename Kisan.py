import streamlit as st
import pandas as pd
import io
import google.generativeai as genai
from datetime import datetime

# ══════════════════════════════════════════════════
# 1. SYSTEM CONFIG & API BACKEND
# ══════════════════════════════════════════════════
st.set_page_config(page_title="Kisan Sakha v5.0", page_icon="🌾", layout="wide")

# BACKEND API KEY (Place your key here for internal use)
BACKEND_API_KEY = "" 

# ══════════════════════════════════════════════════
# 2. DATASETS (MAHARASHTRA RAG CONTEXT)
# ══════════════════════════════════════════════════
SOIL_DATA = """District,SoilType,pH,MajorCrops,Varieties
Pune,Black Cotton,7.5-8.2,Sugarcane,Co-86032;CoM-0265
Nashik,Red Laterite,6.0-6.8,Grapes,Thompson Seedless;Sharad Seedless
Nagpur,Black Soil,7.2-8.5,Oranges,Nagpur Mandarin;Nagpuri Santra
Solapur,Shallow Black,7.8-8.3,Pomegranate,Bhagwa;Mridula
"""

# Market data simulates daily updates via prompt grounding
MSP_2025_26 = """Commodity,MSP_INR_Quintal
Paddy (Common),2300
Soybean (Yellow),4892
Cotton (Long Staple),7521
Tur (Arhar),7550
"""

# ══════════════════════════════════════════════════
# 3. TRANSLATION & STATE
# ══════════════════════════════════════════════════
if "lang" not in st.session_state: st.session_state.lang = "English"
if "page" not in st.session_state: st.session_state.page = "home"
if "chat_history" not in st.session_state: st.session_state.chat_history = {"growth": [], "maint": [], "sales": []}

IS_MR = st.session_state.lang == "मराठी"

T = {
    "title": {"English": "Kisan Sakha v5.0", "मराठी": "किसान सखा v5.0"},
    "tagline": {"English": "Official Maharashtra Farming Intelligence", "मराठी": "अधिकृत महाराष्ट्र कृषी बुद्धिमत्ता"},
    "nav_growth": {"English": "Crop Growth", "मराठी": "पीक वाढ"},
    "nav_maint": {"English": "Crop Maintenance", "मराठी": "पीक देखभाल"},
    "nav_sales": {"English": "Crop Sales", "मराठी": "पीक विक्री"},
    "chat_input": {"English": "Ask your farming doubt...", "मराठी": "तुमची शंका विचारा..."},
    "sidebar_key": {"English": "Gemini API Key", "मराठी": "Gemini API की"},
    "back": {"English": "← Home", "मराठी": "← मुख्यपृष्ठ"}
}

def t(key): return T[key]["मराठी"] if IS_MR else T[key]["English"]

# ══════════════════════════════════════════════════
# 4. CUSTOM UI STYLING
# ══════════════════════════════════════════════════
st.markdown(f"""
<style>
    .stApp {{ background: #fdfdfd; }}
    /* Three-Toggle Dashboard */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {{ background-color: #FF8C00 !important; color: white !important; border: none; height: 100px; font-weight: bold; font-size: 1.2rem; }}
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {{ background-color: #007BFF !important; color: white !important; border: none; height: 100px; font-weight: bold; font-size: 1.2rem; }}
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {{ background-color: #28A745 !important; color: white !important; border: none; height: 100px; font-weight: bold; font-size: 1.2rem; }}
    
    .stButton>button:hover {{ filter: brightness(1.1); transform: translateY(-2px); }}
    .chat-card {{ padding: 1.5rem; border-radius: 10px; background: white; border-left: 5px solid #28A745; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# 5. RAG AI LOGIC (GEMINI 2.5 FLASH)
# ══════════════════════════════════════════════════
def call_ai(query, section):
    api_key = st.session_state.get("user_api_key") or BACKEND_API_KEY
    if not api_key: return "⚠️ Please enter API Key in Sidebar."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Section-specific RAG instructions
        contexts = {
            "growth": f"Expert in soil chemistry and planting. Ground truth data: {SOIL_DATA}. Mention varieties like Bhagwa for Pomegranate and Co-86032 for Sugarcane. Cover all soil types in Maharashtra.",
            "maint": "Expert in pest control and irrigation. Refer to Maharashtra Dept of Agriculture schedules. Provide precise organic and chemical solutions.",
            "sales": f"Market Analyst. Refer to MSP 2025-26: {MSP_2025_26}. Use RAG to fetch live APMC prices for Pune, Nashik, and Vashi markets."
        }
        
        full_prompt = f"System: {contexts[section]} Language: {st.session_state.lang}.\nUser: {query}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ══════════════════════════════════════════════════
# 6. APP WORKFLOW
# ══════════════════════════════════════════════════

# SIDEBAR
with st.sidebar:
    st.title("🌾 Settings")
    st.session_state.lang = st.radio("Select Language / भाषा निवडा", ["English", "मराठी"])
    st.session_state.user_api_key = st.text_input(t("sidebar_key"), type="password")
    if st.button(t("back")): st.session_state.page = "home"

# --- HOME PAGE ---
if st.session_state.page == "home":
    st.title(t("title"))
    st.subheader(t("tagline"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("nav_growth")): st.session_state.page = "growth"
    with col2:
        if st.button(t("nav_maint")): st.session_state.page = "maint"
    with col3:
        if st.button(t("nav_sales")): st.session_state.page = "sales"

# --- MODULAR TOGGLE PAGES ---
else:
    section = st.session_state.page
    labels = {"growth": "nav_growth", "maint": "nav_maint", "sales": "nav_sales"}
    
    st.markdown(f"### {t(labels[section])}")
    
    # 11. AI CHATBAR IN EVERY TOGGLE
    for msg in st.session_state.chat_history[section]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
    user_q = st.chat_input(t("chat_input"))
    if user_q:
        st.session_state.chat_history[section].append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)
        
        with st.spinner("🔄"):
            ans = call_ai(user_q, section)
            st.session_state.chat_history[section].append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"): st.markdown(ans)
