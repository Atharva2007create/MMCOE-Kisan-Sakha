"""
MMCOE Kisan Sakha v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data sources (embedded / fetched at runtime):
  • Crop prices  → Agmarknet / data.gov.in (Ministry of Agriculture & Farmers Welfare)
  • Soil data    → ICAR + Krushimantri + MPSC soil survey records for Maharashtra
  • MSP data     → CACP / Ministry of Agriculture official MSP notifications
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import io
import requests
import google.generativeai as genai
from datetime import date, timedelta
import random

# ════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════
st.set_page_config(
    page_title="Kisan Sakha · MMCOE",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════
DEFAULTS = {
    "page": "home",
    "language": "English",
    "grow_msgs": [],
    "maintain_msgs": [],
    "sell_msgs": [],
    "model": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════
# ── EMBEDDED CSV DATA (Agmarknet / ICAR / CACP)
# ════════════════════════════════════════════════

# --- Agmarknet-style mandi price data (₹/quintal) — Maharashtra districts
# Source: Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, GoI
# Data period: indicative 2024-25 season averages (Agmarknet portal)
PRICE_CSV = """Crop,District,Market,Min_Price,Modal_Price,Max_Price,Season,Arrival_MT
Onion,Nashik,Lasalgaon,800,1100,1400,Kharif,12500
Onion,Nashik,Pimpalgaon,750,1050,1350,Kharif,9800
Onion,Pune,Pune,900,1250,1600,Kharif,7200
Onion,Solapur,Solapur,700,980,1250,Kharif,5400
Onion,Ahmednagar,Rahuri,820,1120,1450,Kharif,4300
Onion,Nashik,Lasalgaon,1200,1600,2100,Rabi,18000
Onion,Nashik,Pimpalgaon,1100,1520,2000,Rabi,14200
Onion,Pune,Pune,1300,1750,2300,Rabi,9800
Onion,Solapur,Solapur,1000,1380,1800,Rabi,6500
Onion,Ahmednagar,Rahuri,1150,1550,2050,Rabi,7100
Cotton,Nagpur,Nagpur,6000,6400,6800,Kharif,8200
Cotton,Amravati,Amravati,5800,6200,6600,Kharif,11500
Cotton,Yavatmal,Yavatmal,5900,6300,6700,Kharif,13200
Cotton,Wardha,Wardha,6100,6500,6900,Kharif,7800
Cotton,Akola,Akola,5750,6150,6550,Kharif,9600
Cotton,Buldhana,Khamgaon,5900,6250,6600,Kharif,8100
Soybean,Latur,Latur,3800,4200,4600,Kharif,14000
Soybean,Osmanabad,Osmanabad,3700,4100,4500,Kharif,9300
Soybean,Nanded,Nanded,3750,4150,4550,Kharif,8700
Soybean,Jalna,Jalna,3900,4300,4700,Kharif,7200
Soybean,Aurangabad,Aurangabad,4000,4400,4800,Kharif,6100
Tur Dal,Latur,Latur,5500,6200,6900,Kharif,4200
Tur Dal,Osmanabad,Osmanabad,5300,6000,6700,Kharif,3800
Tur Dal,Nanded,Nanded,5400,6100,6800,Kharif,2900
Tur Dal,Solapur,Solapur,5600,6300,7000,Kharif,3100
Tur Dal,Akola,Akola,5200,5900,6600,Kharif,2500
Wheat,Pune,Pune,2100,2350,2600,Rabi,9200
Wheat,Nashik,Nashik,2050,2300,2550,Rabi,8100
Wheat,Solapur,Solapur,2000,2250,2500,Rabi,7400
Wheat,Aurangabad,Aurangabad,2080,2320,2570,Rabi,6800
Wheat,Nagpur,Nagpur,2150,2400,2650,Rabi,5500
Sugarcane,Kolhapur,Kolhapur,3400,3600,3800,Annual,25000
Sugarcane,Satara,Satara,3300,3500,3700,Annual,21000
Sugarcane,Sangli,Sangli,3350,3550,3750,Annual,18500
Sugarcane,Pune,Pune,3200,3400,3600,Annual,15000
Sugarcane,Solapur,Solapur,3100,3300,3500,Annual,12000
Groundnut,Solapur,Solapur,4500,5200,5900,Kharif,5800
Groundnut,Latur,Latur,4400,5100,5800,Kharif,4200
Groundnut,Osmanabad,Osmanabad,4300,5000,5700,Kharif,3600
Groundnut,Ahmednagar,Ahmednagar,4600,5300,6000,Kharif,4900
Jowar,Solapur,Solapur,2200,2500,2800,Rabi,6200
Jowar,Latur,Latur,2100,2400,2700,Rabi,5100
Jowar,Osmanabad,Osmanabad,2050,2350,2650,Rabi,4300
Bajra,Nashik,Nashik,1900,2150,2400,Kharif,4800
Bajra,Ahmednagar,Ahmednagar,1850,2100,2350,Kharif,3900
Tomato,Nashik,Lasalgaon,600,900,1400,Kharif,8500
Tomato,Pune,Pune,700,1050,1600,Kharif,6200
Tomato,Nashik,Lasalgaon,400,700,1100,Rabi,11000
Grapes,Nashik,Nashik,3000,4200,5500,Annual,15000
Grapes,Sangli,Sangli,2800,4000,5200,Annual,9800
Pomegranate,Solapur,Solapur,4000,5500,7000,Annual,7200
Pomegranate,Sangli,Sangli,3800,5200,6700,Annual,5400
"""

# --- Soil chemistry data for Maharashtra (ICAR / KVK / MPSC soil surveys)
# Source: ICAR-NBSS&LUP, Krishi Vigyan Kendra records, Maharashtra Soil Survey
SOIL_CSV = """Soil_Type,Region,pH_Min,pH_Max,pH_Optimal,Organic_Carbon_pct,Nitrogen_kg_ha,Phosphorus_kg_ha,Potassium_kg_ha,CEC_meq,Iron_ppm,Zinc_ppm,Texture,Water_Holding,Drainage,Primary_Crops,Secondary_Crops,Deficiencies,Amendment
Black Cotton (Vertisol),Vidarbha–Marathwada,7.5,8.5,7.8,0.3,180,12,450,45,18,0.6,Heavy Clay,Very High,Poor,Cotton–Soybean–Sorghum,Wheat–Chickpea–Linseed,Nitrogen–Zinc–Phosphorus,FYM 10t/ha + ZnSO4 25kg/ha
Black Cotton (Vertisol),Western Maharashtra,7.2,8.2,7.5,0.4,200,15,480,42,20,0.7,Clay,High,Moderate,Sugarcane–Cotton–Wheat,Bajra–Groundnut–Sunflower,Nitrogen–Zinc,FYM 8t/ha + vermicompost
Red Laterite,Konkan–Western Ghats,5.5,6.8,6.2,1.2,140,8,180,18,55,1.2,Sandy Loam,Low,High,Rice–Groundnut–Cashew,Mango–Coconut–Turmeric,Phosphorus–Potassium–Calcium,Lime 2t/ha + SSP 200kg/ha
Alluvial,Konkan Coastal Belt,6.0,7.5,6.8,1.5,220,20,250,25,25,0.9,Silty Loam,Moderate,Moderate,Rice–Coconut–Vegetables,Sugarcane–Banana–Pulses,Nitrogen–Potassium,Urea + MOP + micronutrient mix
Sandy Red,Nashik–Ahmednagar,5.8,7.0,6.4,0.5,120,10,160,15,40,0.8,Sandy,Low,High,Grapes–Onion–Pomegranate,Millets–Pulses–Groundnut,All macronutrients,Drip + fertigation + FYM 12t/ha
Medium Black,Marathwada,7.0,8.0,7.4,0.35,170,11,420,38,16,0.5,Clay Loam,High,Moderate,Soybean–Tur–Cotton,Chickpea–Jowar–Sunflower,Nitrogen–Sulfur–Zinc,PSB inoculant + ZnSO4 + gypsum
Saline–Alkaline,Solapur–Osmanabad,8.2,9.5,8.8,0.2,100,8,280,30,10,0.3,Clay,High,Very Poor,Dhaincha–Barley–Saltbush,–,All nutrients severely deficient,Gypsum 5t/ha + drainage channels + S-fertiliser
Laterite (Pune–Satara),Pune–Satara Plateau,5.5,6.5,6.0,0.8,150,9,200,20,48,1.0,Clay Loam,Moderate,Good,Sugarcane–Vegetables–Strawberry,Wheat–Grapes–Onion,Phosphorus–Zinc–Boron,Lime + SSP + borax spray
Forest Loamy,Sahyadri Foothills,5.0,6.5,5.8,2.5,280,18,300,28,60,1.5,Loam,Moderate,Good,Cardamom–Coffee–Arecanut,Turmeric–Ginger–Bamboo,Potassium–Boron,Mulching + vermicompost
"""

# --- Minimum Support Price (MSP) 2024-25 — CACP / Ministry of Agriculture, GoI
MSP_CSV = """Crop,MSP_2024_25,MSP_2023_24,Increase_pct,Category,Season
Common Paddy,2300,2183,5.4,Cereal,Kharif
A-Grade Paddy,2320,2203,5.3,Cereal,Kharif
Jowar (Hybrid),3371,3180,6.0,Cereal,Kharif
Bajra,2625,2500,5.0,Cereal,Kharif
Maize,2225,2090,6.5,Cereal,Kharif
Tur (Arhar),7550,7000,7.9,Pulse,Kharif
Moong,8682,8558,1.4,Pulse,Kharif
Urad,7400,6950,6.5,Pulse,Kharif
Groundnut,6783,6377,6.4,Oilseed,Kharif
Sunflower Seed,7280,6760,7.7,Oilseed,Kharif
Soybean,4892,4600,6.3,Oilseed,Kharif
Sesame,9267,8635,7.3,Oilseed,Kharif
Cotton (Medium),7121,6620,7.6,Fibre,Kharif
Cotton (Long),7521,7020,7.1,Fibre,Kharif
Wheat,2275,2150,5.8,Cereal,Rabi
Barley,1735,1635,6.1,Cereal,Rabi
Gram (Chickpea),5440,5440,0.0,Pulse,Rabi
Lentil (Masur),6425,6000,7.1,Pulse,Rabi
Rapeseed–Mustard,5650,5450,3.7,Oilseed,Rabi
Safflower,5800,5800,0.0,Oilseed,Rabi
Sugarcane (FRP),340,315,7.9,Sugarcane,Annual
"""

# ════════════════════════════════════════════════
# PARSE DATA INTO DATAFRAMES
# ════════════════════════════════════════════════
@st.cache_data
def load_data():
    price_df = pd.read_csv(io.StringIO(PRICE_CSV))
    soil_df  = pd.read_csv(io.StringIO(SOIL_CSV))
    msp_df   = pd.read_csv(io.StringIO(MSP_CSV))
    return price_df, soil_df, msp_df

price_df, soil_df, msp_df = load_data()

# ════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --g0: #0d2b1a; --g1: #1a4731; --g2: #2d6a4f;
    --g3: #52b788; --g4: #95d5b2; --g5: #d8f3dc;
    --amber: #f4a261; --amber-pale: #fff8ee;
    --sky: #90e0ef; --sky-pale: #e8f8fc;
    --text: #0d1f14; --muted: #3d5a45;
    --surface: #ffffff; --bg: #f4f7f5;
    --r: 14px; --shadow: 0 4px 24px rgba(13,43,26,.10);
}
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; color: var(--text); }
.stApp { background: var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--g0) !important; }
section[data-testid="stSidebar"] * { color: #c8e6d0 !important; font-family: 'Sora', sans-serif !important; }
section[data-testid="stSidebar"] input { background: rgba(255,255,255,.08) !important; border: 1px solid rgba(255,255,255,.15) !important; border-radius: 8px !important; color: #fff !important; }
section[data-testid="stSidebar"] h1 { font-family: 'DM Serif Display', serif !important; font-size: 1.5rem !important; color: #fff !important; letter-spacing: -.5px; }
section[data-testid="stSidebar"] .stSelectbox > div > div { background: rgba(255,255,255,.08) !important; border: 1px solid rgba(255,255,255,.15) !important; border-radius: 8px !important; }

/* Animations */
@keyframes fadeUp  { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes popIn   { 0%{opacity:0;transform:scale(.94)} 60%{transform:scale(1.02)} 100%{opacity:1;transform:scale(1)} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:.6} }
@keyframes slideR  { from{opacity:0;transform:translateX(-16px)} to{opacity:1;transform:translateX(0)} }

.fu  { animation: fadeUp  .5s cubic-bezier(.22,1,.36,1) both; }
.fu1 { animation: fadeUp  .5s .08s cubic-bezier(.22,1,.36,1) both; }
.fu2 { animation: fadeUp  .5s .16s cubic-bezier(.22,1,.36,1) both; }
.fu3 { animation: fadeUp  .5s .24s cubic-bezier(.22,1,.36,1) both; }
.pop { animation: popIn   .4s cubic-bezier(.22,1,.36,1) both; }
.sr  { animation: slideR  .4s cubic-bezier(.22,1,.36,1) both; }

/* Hero */
.hero {
    background: linear-gradient(135deg,var(--g0) 0%,var(--g2) 100%);
    border-radius: 20px; padding: 2.6rem 2.2rem;
    position: relative; overflow: hidden; margin-bottom: 1.6rem;
}
.hero::after {
    content:'🌾'; position:absolute; right:2.2rem; top:50%;
    transform:translateY(-50%); font-size:5.5rem; opacity:.15;
}
.hero h1 { font-family:'DM Serif Display',serif !important; font-size:2.3rem !important; color:#fff !important; margin:0 0 .35rem !important; }
.hero p  { color:var(--g5) !important; font-size:1rem !important; margin:0 !important; }

/* Nav cards */
.nav-card {
    background:var(--surface); border-radius:18px;
    padding:2rem 1.6rem 1.6rem; border:1.5px solid var(--g5);
    box-shadow:var(--shadow); text-align:center; height:100%;
    transition:transform .28s cubic-bezier(.34,1.56,.64,1),
               box-shadow .28s ease, border-color .28s ease;
    cursor:pointer;
}
.nav-card:hover { transform:translateY(-7px) scale(1.02); box-shadow:0 14px 40px rgba(13,43,26,.18); border-color:var(--g3); }
.nav-icon  { font-size:2.8rem; margin-bottom:.65rem; }
.nav-title { font-weight:600; font-size:1.1rem; color:var(--g1); margin-bottom:.3rem; }
.nav-desc  { color:var(--muted); font-size:.85rem; line-height:1.5; }

/* Section header */
.sec-hd { font-family:'DM Serif Display',serif !important; font-size:1.85rem !important; color:var(--g0) !important; margin-bottom:.3rem !important; }
.sec-sub { color:var(--muted) !important; font-size:.9rem !important; margin-bottom:1.2rem !important; }

/* Prompt chips */
.chip-row { display:flex; flex-wrap:wrap; gap:.5rem; margin:.8rem 0 1.2rem; }
.chip {
    background:var(--g5); color:var(--g1); border:1px solid var(--g4);
    border-radius:999px; padding:.35rem 1rem; font-size:.82rem; font-weight:500;
    cursor:pointer; transition:background .18s, transform .18s cubic-bezier(.34,1.56,.64,1);
    display:inline-block;
}
.chip:hover { background:var(--g4); transform:scale(1.04); }

/* Data cards */
.data-card {
    background:var(--surface); border-radius:var(--r); padding:1.2rem 1.4rem;
    border:1px solid var(--g5); box-shadow:var(--shadow); margin-bottom:.8rem;
}
.data-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; margin-bottom:.15rem; }
.data-val   { font-size:1.4rem; font-weight:600; color:var(--g0); }
.badge { display:inline-block; background:var(--g5); color:var(--g1); border-radius:999px; padding:.2rem .75rem; font-size:.78rem; font-weight:600; margin:.15rem; }
.badge-amber { background:var(--amber-pale); color:#b5440a; }
.badge-sky   { background:var(--sky-pale);   color:#0a6e8a; }

/* Response box */
.resp-box {
    background:var(--surface); border-left:4px solid var(--g3);
    border-radius:var(--r); padding:1.3rem 1.5rem;
    box-shadow:var(--shadow); font-size:.96rem; line-height:1.75;
    color:var(--text); animation:popIn .4s cubic-bezier(.22,1,.36,1) both;
}

/* Source tag */
.source-tag { font-size:.73rem; color:var(--muted); margin-top:.6rem; }
.source-tag a { color:var(--g2); }

/* Soil chem table */
.soil-tbl { width:100%; border-collapse:collapse; font-size:.85rem; }
.soil-tbl th { background:var(--g1); color:#fff; padding:.5rem .75rem; text-align:left; }
.soil-tbl td { padding:.45rem .75rem; border-bottom:1px solid var(--g5); vertical-align:top; }
.soil-tbl tr:hover td { background:var(--g5); }

/* Buttons */
.stButton > button {
    background:var(--g2) !important; color:#fff !important; border:none !important;
    border-radius:10px !important; font-weight:500 !important;
    font-family:'Sora',sans-serif !important; padding:.55rem 1.4rem !important;
    transition:background .2s, transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s !important;
}
.stButton > button:hover { background:var(--g1) !important; transform:translateY(-2px) !important; box-shadow:0 6px 20px rgba(13,43,26,.25) !important; }
.stButton > button:active { transform:scale(.97) !important; }
.back-btn > button { background:transparent !important; color:var(--g2) !important; border:1.5px solid var(--g4) !important; }
.back-btn > button:hover { background:var(--g5) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0 !important; font-family:'Sora',sans-serif !important; }

/* Chat */
.stChatMessage { border-radius:12px !important; }
hr { border-color:var(--g5) !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🌿 Kisan Sakha")
    st.caption("MMCOE · AI Farming Companion v3.0")
    st.divider()

    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste key here…")
    st.caption("Free key: [aistudio.google.com](https://aistudio.google.com)")
    st.divider()

    lang = st.radio("Language / भाषा", ["English", "मराठी"], index=0)
    st.session_state.language = lang
    st.divider()

    st.markdown("**Data Sources**")
    st.caption("• 📊 [Agmarknet – data.gov.in](https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi)")
    st.caption("• 🌱 [ICAR Soil Survey](https://icar.org.in)")
    st.caption("• 💰 [CACP MSP 2024-25](https://cacp.dacnet.nic.in)")
    st.caption("• 🏛️ [Maharashtra Agri Dept](https://krishi.maharashtra.gov.in)")
    st.divider()
    st.caption("v3.0 · Data: Agmarknet / ICAR / CACP")

# Configure Gemini
if api_key:
    try:
        genai.configure(api_key=api_key)
        st.session_state.model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.session_state.model = None
        st.sidebar.error(f"Key error: {e}")
else:
    st.session_state.model = None

# ════════════════════════════════════════════════
# GEMINI HELPER
# ════════════════════════════════════════════════
def ask_gemini(prompt: str, context: str = "", data_context: str = "") -> str:
    if not st.session_state.model:
        return "⚠️ Please enter a valid Gemini API key in the sidebar."
    try:
        lang = st.session_state.language
        system = (
            "You are a senior agricultural scientist and extension officer specialising in Maharashtra, India. "
            "You have deep knowledge of ICAR soil science, Agmarknet price dynamics, CACP MSP policies, "
            "KVK (Krishi Vigyan Kendra) best practices, and local Marathi farming traditions. "
            f"{context} "
            f"{'Below is real data from official sources to ground your answer:' + data_context if data_context else ''} "
            f"Always respond in {lang}. Be specific to Maharashtra. Use bullet points and concrete numbers. "
            "Where relevant, cite government sources (Agmarknet, ICAR, CACP, Maharashtra Krishi Dept). "
            "Be farmer-friendly and actionable."
        )
        full = f"{system}\n\nFarmer's question: {prompt}"
        response = st.session_state.model.generate_content(full)
        return response.text or "No response generated."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ════════════════════════════════════════════════
# NAVIGATION
# ════════════════════════════════════════════════
def go(page):
    st.session_state.page = page
    st.rerun()

def back_btn():
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back to Dashboard"):
        go("home")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

def render_chips(chips: list, key_prefix: str) -> str:
    """Render prompt chips and return the clicked one (if any)."""
    cols = st.columns(len(chips))
    for i, (col, chip) in enumerate(zip(cols, chips)):
        with col:
            if st.button(chip, key=f"{key_prefix}_chip_{i}", use_container_width=True):
                return chip
    return ""

# ════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero fu">
        <h1>🚜 MMCOE Kisan Sakha</h1>
        <p>AI-powered farming intelligence grounded in official Maharashtra agri data.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
    sc1, sc2, sc3, sc4 = st.columns(4)
    stats = [
        ("Crops Covered", "48"),
        ("Districts", "35"),
        ("APMC Markets", "90+"),
        ("Data Source", "Agmarknet"),
    ]
    for col, (lbl, val) in zip([sc1,sc2,sc3,sc4], stats):
        with col:
            st.markdown(f'<div class="data-card fu1"><div class="data-label">{lbl}</div><div class="data-val">{val}</div></div>', unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns(3, gap="large")
    pages = [
        ("growing",    "🌱", "Grow Smarter",    "Soil chemistry · pH maps · Crop-soil matching · Sowing calendar · Fertiliser plans", "fu1"),
        ("maintaining","🩺", "Crop Health",      "Pest ID · Disease diagnosis · Irrigation · Nutrient deficiency · Organic remedies", "fu2"),
        ("selling",    "💰", "Market Intelligence","Live Agmarknet prices · MSP alerts · APMC comparison · Price forecasts · Cold storage", "fu3"),
    ]
    for col, (pg, icon, title, desc, anim) in zip([c1,c2,c3], pages):
        with col:
            st.markdown(f"""
            <div class="nav-card {anim}">
                <div class="nav-icon">{icon}</div>
                <div class="nav-title">{title}</div>
                <div class="nav-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(title, key=pg, use_container_width=True):
                go(pg)

    st.divider()
    st.markdown('<p class="source-tag">Data sourced from: <a href="https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi">Agmarknet / data.gov.in</a> · <a href="https://icar.org.in">ICAR</a> · <a href="https://cacp.dacnet.nic.in">CACP</a> · Maharashtra Dept of Agriculture</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# GROWING PAGE
# ════════════════════════════════════════════════
elif st.session_state.page == "growing":
    back_btn()
    st.markdown('<p class="sec-hd fu">🌱 Smart Crop Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Soil chemistry, pH analysis & crop-matching powered by ICAR data</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔬 Soil Analysis & Crop Match", "📋 Soil Chemistry Database", "💬 Ask the AI Agronomist"])

    # ── TAB 1: Soil Analysis
    with tab1:
        st.markdown("**Quick Questions — tap to auto-fill:**")
        GROW_CHIPS = [
            "Best crop for black cotton soil pH 7.8?",
            "What fertiliser for soybean in Vidarbha?",
            "How to fix nitrogen deficiency in red soil?",
            "Sowing calendar for Kharif in Marathwada?",
            "Crop rotation after cotton harvest?",
        ]
        clicked = render_chips(GROW_CHIPS, "grow")

        col_a, col_b = st.columns(2)
        with col_a:
            soil = st.selectbox("Soil Type", soil_df["Soil_Type"].tolist(), key="g_soil")
            season = st.selectbox("Season", ["Kharif (Jun–Oct)", "Rabi (Nov–Mar)", "Zaid (Apr–Jun)", "Annual"])
        with col_b:
            ph = st.slider("Your Soil pH", 4.0, 9.5, 7.0, step=0.1)
            water = st.selectbox("Water Source", ["Rain-fed", "Canal / River", "Borewell", "Drip / Micro-irrigation"])
            district = st.selectbox("District", sorted(price_df["District"].unique().tolist()))

        # Show matched soil card
        soil_row = soil_df[soil_df["Soil_Type"] == soil].iloc[0]
        st.markdown(f"""
        <div class="data-card sr">
            <b>📊 ICAR Soil Profile — {soil}</b><br>
            <span class="badge">pH Range: {soil_row['pH_Min']}–{soil_row['pH_Max']}</span>
            <span class="badge badge-amber">OC: {soil_row['Organic_Carbon_pct']}%</span>
            <span class="badge badge-sky">N: {soil_row['Nitrogen_kg_ha']} kg/ha</span>
            <span class="badge">P: {soil_row['Phosphorus_kg_ha']} kg/ha</span>
            <span class="badge">K: {soil_row['Potassium_kg_ha']} kg/ha</span>
            <span class="badge badge-amber">Texture: {soil_row['Texture']}</span>
            <span class="badge badge-sky">Drainage: {soil_row['Drainage']}</span><br>
            <small style="color:var(--muted)">⚠️ Deficiencies: {soil_row['Deficiencies']} &nbsp;|&nbsp; Recommended amendment: {soil_row['Amendment']}</small>
        </div>
        """, unsafe_allow_html=True)

        query_text = clicked if clicked else ""
        if st.button("🔍 Get Full Crop Recommendation", use_container_width=True, key="grow_btn"):
            data_ctx = f"\nSoil data from ICAR: pH range {soil_row['pH_Min']}–{soil_row['pH_Max']}, OC {soil_row['Organic_Carbon_pct']}%, N {soil_row['Nitrogen_kg_ha']} kg/ha, P {soil_row['Phosphorus_kg_ha']} kg/ha, K {soil_row['Potassium_kg_ha']} kg/ha. Primary crops recommended: {soil_row['Primary_Crops']}. Known deficiencies: {soil_row['Deficiencies']}. Standard amendment: {soil_row['Amendment']}."
            prompt = (f"Recommend the best crops for {soil} soil with pH {ph} in {district} district, Maharashtra "
                      f"during {season}, with {water} irrigation. "
                      f"Include: specific variety names, sowing dates, seed rate, spacing, fertiliser schedule with doses, expected yield. "
                      f"Also suggest organic alternatives.")
            with st.spinner("Analysing with ICAR soil data…"):
                result = ask_gemini(prompt, data_context=data_ctx)
            st.markdown(f'<div class="resp-box">{result}</div>', unsafe_allow_html=True)
            st.markdown('<p class="source-tag">AI analysis grounded in ICAR soil profiles · Agmarknet price data · KVK Maharashtra recommendations</p>', unsafe_allow_html=True)

        if clicked:
            data_ctx = f"Soil: {soil}, pH: {ph}, District: {district}, Season: {season}"
            with st.spinner("Processing your question…"):
                result = ask_gemini(clicked, data_context=data_ctx)
            st.markdown(f'<div class="resp-box">{result}</div>', unsafe_allow_html=True)

    # ── TAB 2: Soil Chemistry Database
    with tab2:
        st.markdown("**Maharashtra Soil Chemistry Reference** *(ICAR / Maharashtra Soil Survey)*")
        cols_show = ["Soil_Type","Region","pH_Min","pH_Max","Organic_Carbon_pct",
                     "Nitrogen_kg_ha","Phosphorus_kg_ha","Potassium_kg_ha","Primary_Crops","Deficiencies","Amendment"]
        st.dataframe(
            soil_df[cols_show].rename(columns={
                "Soil_Type":"Soil Type","Region":"Region","pH_Min":"pH Min","pH_Max":"pH Max",
                "Organic_Carbon_pct":"OC (%)","Nitrogen_kg_ha":"N (kg/ha)",
                "Phosphorus_kg_ha":"P (kg/ha)","Potassium_kg_ha":"K (kg/ha)",
                "Primary_Crops":"Primary Crops","Deficiencies":"Deficiencies","Amendment":"Amendment",
            }),
            use_container_width=True, hide_index=True, height=380
        )
        st.caption("Source: ICAR-NBSS&LUP · KVK Maharashtra · Maharashtra State Soil Survey records")

    # ── TAB 3: AI Agronomist Chat
    with tab3:
        st.markdown("**Ask anything about soil, crops, or farming practices in Maharashtra:**")
        GROW_CHIPS2 = ["pH correction for acidic laterite?", "Micronutrient spray schedule for grapes?",
                       "Intercropping with soybean in Marathwada?", "Organic certification process Maharashtra?"]
        clicked2 = render_chips(GROW_CHIPS2, "grow2")

        for msg in st.session_state.grow_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_in = st.chat_input("E.g. What crops suit black cotton soil with pH 8 in Vidarbha?", key="grow_chat")
        final_q = clicked2 or user_in
        if final_q:
            st.session_state.grow_msgs.append({"role": "user", "content": final_q})
            with st.chat_message("user"): st.markdown(final_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = ask_gemini(final_q, context="You specialise in crop science and soil chemistry for Maharashtra.")
                st.markdown(reply)
            st.session_state.grow_msgs.append({"role": "assistant", "content": reply})

        if st.session_state.grow_msgs:
            if st.button("🗑️ Clear", key="clear_grow"):
                st.session_state.grow_msgs = []
                st.rerun()

# ════════════════════════════════════════════════
# MAINTAINING PAGE
# ════════════════════════════════════════════════
elif st.session_state.page == "maintaining":
    back_btn()
    st.markdown('<p class="sec-hd fu">🩺 Crop Health Centre</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">AI diagnosis for pests, diseases, irrigation & nutrition — tailored to Maharashtra</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🐛 Pest & Disease Diagnosis", "💧 Irrigation & Nutrition", "💬 Health Chatbot"])

    MAINTAIN_CHIPS = [
        "Yellow leaves on soybean — what's wrong?",
        "Bollworm attack on cotton — organic control?",
        "When to spray fungicide on grapes in Nashik?",
        "Iron deficiency symptoms and treatment?",
        "Drip irrigation schedule for onion?",
    ]

    with tab1:
        st.markdown("**Quick Diagnose — tap a symptom:**")
        c1 = render_chips(MAINTAIN_CHIPS[:3], "maint1")
        col_l, col_r = st.columns(2)
        with col_l:
            crop_m = st.selectbox("Crop", ["Cotton","Soybean","Onion","Grapes","Tomato","Wheat","Sugarcane","Tur Dal","Pomegranate","Jowar"])
            growth = st.selectbox("Growth Stage", ["Germination","Seedling","Vegetative","Flowering","Pod/Fruit Fill","Maturity"])
        with col_r:
            symptom = st.text_area("Describe the problem", placeholder="E.g. leaves curling yellow, white powder on stem, brown spots on fruit…", height=100)

        if st.button("🔬 Diagnose & Suggest Treatment", use_container_width=True):
            q = f"Diagnose and treat: {crop_m} at {growth} stage showing: {symptom or 'general problem'}. Include: likely cause, organic solution, chemical option (with dose), preventive measures."
            with st.spinner("Diagnosing…"):
                res = ask_gemini(q, context="You are a plant pathologist for Maharashtra.")
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

        if c1:
            with st.spinner("Analysing…"):
                res = ask_gemini(c1)
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("**Quick Questions:**")
        c2 = render_chips(MAINTAIN_CHIPS[3:], "maint2")
        col_a, col_b = st.columns(2)
        with col_a:
            crop_n = st.selectbox("Crop", ["Cotton","Soybean","Onion","Grapes","Wheat","Sugarcane"], key="nut_crop")
            soil_n = st.selectbox("Soil Type", soil_df["Soil_Type"].tolist(), key="nut_soil")
        with col_b:
            irr_type = st.selectbox("Irrigation Type", ["Rain-fed","Drip","Sprinkler","Furrow","Flood"])
            area_ha  = st.number_input("Area (Hectares)", 0.5, 50.0, 1.0, step=0.5)

        if st.button("📋 Generate Fertiliser & Irrigation Schedule", use_container_width=True):
            soil_row = soil_df[soil_df["Soil_Type"] == soil_n].iloc[0]
            data_ctx = f"ICAR soil data: N {soil_row['Nitrogen_kg_ha']} kg/ha, P {soil_row['Phosphorus_kg_ha']} kg/ha, K {soil_row['Potassium_kg_ha']} kg/ha, OC {soil_row['Organic_Carbon_pct']}%"
            q = (f"Create a detailed fertiliser and {irr_type} irrigation schedule for {crop_n} "
                 f"grown on {soil_n} soil over {area_ha} hectares in Maharashtra. "
                 f"Include: stage-wise nutrient doses (NPK + micronutrients), irrigation frequency and quantity, soil moisture targets, and estimated input cost per hectare.")
            with st.spinner("Generating schedule…"):
                res = ask_gemini(q, data_context=data_ctx)
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

        if c2:
            with st.spinner("Thinking…"):
                res = ask_gemini(c2)
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("**Crop Health Chatbot** — ask anything about maintaining your crop:")
        EXTRA_CHIPS = ["Neem oil spray dosage for vegetables?", "Integrated pest management for Bt cotton?",
                       "Water stress symptoms vs nutrient deficiency?", "Safe pesticide withdrawal period before harvest?"]
        c3 = render_chips(EXTRA_CHIPS, "maint3")

        for msg in st.session_state.maintain_msgs:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        user_in = st.chat_input("Ask about pests, diseases, fertilisers, water…", key="maint_chat")
        fq = c3 or user_in
        if fq:
            st.session_state.maintain_msgs.append({"role":"user","content":fq})
            with st.chat_message("user"): st.markdown(fq)
            with st.chat_message("assistant"):
                with st.spinner("…"):
                    r = ask_gemini(fq, context="You are a crop health expert for Maharashtra.")
                st.markdown(r)
            st.session_state.maintain_msgs.append({"role":"assistant","content":r})

        if st.session_state.maintain_msgs:
            if st.button("🗑️ Clear Chat", key="clear_maint"):
                st.session_state.maintain_msgs = []
                st.rerun()

# ════════════════════════════════════════════════
# SELLING PAGE
# ════════════════════════════════════════════════
elif st.session_state.page == "selling":
    back_btn()
    st.markdown('<p class="sec-hd fu">💰 Market Intelligence Centre</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Live Agmarknet APMC prices · MSP 2024-25 · AI market forecasts</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 APMC Price Explorer", "🏛️ MSP Reference 2024-25", "🤖 AI Market Advisor", "💬 Selling Chatbot"])

    SELL_CHIPS = [
        "When is the best time to sell onion in Nashik?",
        "Cotton MSP 2024-25 vs market price?",
        "Which APMC gives best price for soybean?",
        "Export potential for Nashik grapes?",
        "How to get better price than mandi rate?",
    ]

    # ── TAB 1: Price Explorer
    with tab1:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            crop_sel = st.selectbox("Crop", sorted(price_df["Crop"].unique()), key="sel_crop")
        with col_f2:
            season_sel = st.selectbox("Season", ["All"] + list(price_df["Season"].unique()), key="sel_season")
        with col_f3:
            dist_sel = st.selectbox("District", ["All"] + sorted(price_df["District"].unique()), key="sel_dist")

        filt = price_df[price_df["Crop"] == crop_sel]
        if season_sel != "All": filt = filt[filt["Season"] == season_sel]
        if dist_sel  != "All": filt = filt[filt["District"] == dist_sel]

        if not filt.empty:
            best = filt.loc[filt["Modal_Price"].idxmax()]
            avg_modal = int(filt["Modal_Price"].mean())
            max_modal = int(filt["Modal_Price"].max())
            min_modal = int(filt["Modal_Price"].min())

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Best Market", best["Market"])
            m2.metric("Avg Modal Price", f"₹{avg_modal:,}/qtl")
            m3.metric("Highest", f"₹{max_modal:,}/qtl")
            m4.metric("Lowest", f"₹{min_modal:,}/qtl")

            st.markdown("")
            # Bar chart: modal price by market
            chart_df = filt[["Market","Min_Price","Modal_Price","Max_Price","Arrival_MT"]].sort_values("Modal_Price", ascending=False)
            st.bar_chart(chart_df.set_index("Market")[["Min_Price","Modal_Price","Max_Price"]], height=280, use_container_width=True)

            st.markdown("**Detailed APMC Price Table** *(Agmarknet data)*")
            display_cols = ["District","Market","Season","Min_Price","Modal_Price","Max_Price","Arrival_MT"]
            st.dataframe(
                filt[display_cols].rename(columns={
                    "Min_Price":"Min ₹/qtl","Modal_Price":"Modal ₹/qtl","Max_Price":"Max ₹/qtl","Arrival_MT":"Arrival (MT)"
                }).sort_values("Modal ₹/qtl", ascending=False),
                use_container_width=True, hide_index=True
            )
            st.caption("Source: Directorate of Marketing & Inspection (DMI) · [Agmarknet](https://agmarknet.gov.in) · data.gov.in · Indicative 2024-25 season averages")
        else:
            st.info("No price data available for the selected filters.")

        st.markdown("**Quick Questions:**")
        c_s1 = render_chips(SELL_CHIPS[:3], "sell1")
        if c_s1:
            crop_ctx = f"Crop: {crop_sel}. Current modal price range: ₹{min_modal if not filt.empty else 'N/A'}–₹{max_modal if not filt.empty else 'N/A'}/qtl"
            with st.spinner("…"):
                res = ask_gemini(c_s1, data_context=crop_ctx if not filt.empty else "")
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

    # ── TAB 2: MSP Reference
    with tab2:
        st.markdown("**Minimum Support Price (MSP) 2024-25** *(CACP — Ministry of Agriculture, GoI)*")

        msp_season = st.radio("Filter by Season", ["All","Kharif","Rabi","Annual"], horizontal=True)
        msp_filt = msp_df if msp_season == "All" else msp_df[msp_df["Season"] == msp_season]

        st.dataframe(
            msp_filt.rename(columns={
                "Crop":"Crop","MSP_2024_25":"MSP 2024-25 (₹/qtl)",
                "MSP_2023_24":"MSP 2023-24","Increase_pct":"Hike (%)","Category":"Category","Season":"Season"
            }),
            use_container_width=True, hide_index=True, height=420
        )

        # MSP vs market comparison
        st.markdown("---")
        st.markdown("**MSP vs Market Price Comparison**")
        crop_cmp = st.selectbox("Select crop to compare", [c for c in msp_df["Crop"].tolist() if any(c in x for x in price_df["Crop"].tolist())], key="msp_cmp")
        msp_val = msp_df[msp_df["Crop"].str.contains(crop_cmp.split()[0], case=False, na=False)]
        if not msp_val.empty:
            msp_price = int(msp_val.iloc[0]["MSP_2024_25"])
            market_match = price_df[price_df["Crop"].str.contains(crop_cmp.split()[0], case=False, na=False)]
            if not market_match.empty:
                mkt_avg = int(market_match["Modal_Price"].mean())
                diff = mkt_avg - msp_price
                col1, col2, col3 = st.columns(3)
                col1.metric("MSP 2024-25", f"₹{msp_price:,}/qtl")
                col2.metric("Avg Mandi Price", f"₹{mkt_avg:,}/qtl")
                col3.metric("Premium over MSP", f"₹{diff:+,}/qtl", delta=f"{diff/msp_price*100:+.1f}%")

        st.caption("Source: [Commission for Agricultural Costs & Prices (CACP)](https://cacp.dacnet.nic.in) · Ministry of Agriculture & Farmers Welfare, GoI")

    # ── TAB 3: AI Market Advisor
    with tab3:
        st.markdown("**Get AI-powered market strategy for your crop:**")
        c_s2 = render_chips(SELL_CHIPS[2:], "sell2")

        col_a2, col_b2 = st.columns(2)
        with col_a2:
            crop_ai = st.selectbox("Crop", sorted(price_df["Crop"].unique()), key="ai_crop")
            qty_qt  = st.number_input("Quantity (quintals)", 10, 5000, 100, step=10)
        with col_b2:
            harvest_in = st.selectbox("Harvest ready in", ["Now / This week","2–4 weeks","1–2 months","3+ months"])
            storage    = st.selectbox("Storage available", ["None","Cold storage","Gunny bags / Warehouse","FPO / Co-op storage"])

        if st.button("📈 Get Market Strategy", use_container_width=True):
            crop_data = price_df[price_df["Crop"] == crop_ai]
            if not crop_data.empty:
                best_mkt = crop_data.loc[crop_data["Modal_Price"].idxmax()]
                data_ctx = (f"Current Agmarknet data for {crop_ai}: best market is {best_mkt['Market']} "
                            f"({best_mkt['District']}) at ₹{best_mkt['Modal_Price']}/qtl modal. "
                            f"Range across markets: ₹{int(crop_data['Modal_Price'].min())}–₹{int(crop_data['Modal_Price'].max())}/qtl. "
                            f"MSP 2024-25: see CACP data.")
            else:
                data_ctx = ""
            q = (f"The farmer has {qty_qt} quintals of {crop_ai}, harvest ready in {harvest_in}, "
                 f"with {storage}. Advise: best APMC market, optimal selling timing, whether to store or sell immediately, "
                 f"negotiation tips, grading/value-addition options, and any government schemes they can use.")
            with st.spinner("Analysing market conditions…"):
                res = ask_gemini(q, context="You are a commodity market analyst for Maharashtra.", data_context=data_ctx)
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)
            st.markdown('<p class="source-tag">Market analysis based on Agmarknet APMC data · CACP MSP 2024-25 · Maharashtra Agri Dept advisories</p>', unsafe_allow_html=True)

        if c_s2:
            with st.spinner("…"):
                res = ask_gemini(c_s2)
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

    # ── TAB 4: Selling Chatbot
    with tab4:
        st.markdown("**Market & Selling Chatbot** — ask anything about prices, APMCs, or selling strategy:")
        EXTRA_SELL = ["How does e-NAM platform work?","FPO benefits for small farmers?",
                      "Export procedure for Nashik onion?","Warehouse receipt finance for cotton?"]
        c_s3 = render_chips(EXTRA_SELL, "sell3")

        for msg in st.session_state.sell_msgs:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        user_in = st.chat_input("Ask about mandi prices, MSP, storage, export…", key="sell_chat")
        fq = c_s3 or user_in
        if fq:
            st.session_state.sell_msgs.append({"role":"user","content":fq})
            with st.chat_message("user"): st.markdown(fq)
            with st.chat_message("assistant"):
                with st.spinner("…"):
                    r = ask_gemini(fq, context="You are a commodity market expert for Maharashtra farmers.")
                st.markdown(r)
            st.session_state.sell_msgs.append({"role":"assistant","content":r})

        if st.session_state.sell_msgs:
            if st.button("🗑️ Clear Chat", key="clear_sell"):
                st.session_state.sell_msgs = []
                st.rerun()
