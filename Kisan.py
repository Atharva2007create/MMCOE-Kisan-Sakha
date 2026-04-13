"""
MMCOE Kisan Sakha — deployment-ready Streamlit app.
Set GOOGLE_API_KEY in the environment (from Google AI Studio). Optional: DATA_GOV_IN_API_KEY for live mandi merge.
Optional: GOOGLE_WEATHER_API_KEY or GOOGLE_MAPS_API_KEY with Google Maps Platform Weather API enabled (live panel).
Uses Gemini models via google-generativeai; RAG cache (24h); embedded baselines as fallback.
"""

import os
import re
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import requests

# Backend: GOOGLE_API_KEY only (create at https://aistudio.google.com/apikey).
_ENV_GOOGLE_KEY = (os.environ.get("AIzaSyBOYvUD1IDosANVf1r6s0--_ym8UvwuwcA") or "").strip()
_GEMINI_MODEL = "gemini-2.5-flash"  # model id for the Generative AI API (not a second API key)

# data.gov.in API key (optional) for daily Agmarknet-style mandi rows for Maharashtra.
_ENV_DATA_GOV_KEY = (os.environ.get("DATA_GOV_IN_API_KEY") or "").strip()

# Weather is now powered via Gemini RAG — no separate Weather API key needed.
# The same GOOGLE_API_KEY used for AI answers also powers the weather panel.

_RAG_URLS = (
    "https://krishi.maharashtra.gov.in/",
    "https://agmarknet.gov.in/",
    "https://www.maharashtra.gov.in/",
)

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Kisan Sakha · MMCOE",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
DEFAULTS = {
    "page": "home",
    "lang": "English",
    "grow_msgs": [],
    "maintain_msgs": [],
    "sell_msgs": [],
    "model": None,
    "weather_open": False,
    "weather_district": "Pune",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

IS_MR = st.session_state.lang == "मराठी"

# ══════════════════════════════════════════════════
# TRANSLATION DICTIONARY
# ══════════════════════════════════════════════════
T = {
    "app_title":        {"English": "🚜 MMCOE Kisan Sakha", "मराठी": "🚜 एमएमसीओई किसान सखा"},
    "app_subtitle":     {"English": "AI-powered farming intelligence for Maharashtra's farmers", "मराठी": "महाराष्ट्रातील शेतकऱ्यांसाठी AI-आधारित कृषी माहिती"},
    "settings":         {"English": "⚙️ Settings", "मराठी": "⚙️ सेटिंग्ज"},
    "api_label":        {"English": "Google AI API key", "मराठी": "Google AI API की"},
    "api_placeholder":  {"English": "Paste key here…", "मराठी": "इथे की पेस्ट करा…"},
    "lang_label":       {"English": "Language / भाषा", "मराठी": "भाषा"},
    "data_sources":     {"English": "Data Sources", "मराठी": "माहिती स्रोत"},
    "back":             {"English": "← Back to Dashboard", "मराठी": "← मुख्य पानावर परत"},
    "crops_covered":    {"English": "Crops Covered", "मराठी": "पिके"},
    "districts":        {"English": "Districts", "मराठी": "जिल्हे"},
    "markets":          {"English": "APMC Markets", "मराठी": "APMC बाजार"},
    "data_src_lbl":     {"English": "Data Source", "मराठी": "माहिती स्रोत"},
    # Nav cards
    "grow_title":       {"English": "Grow Smarter", "मराठी": "हुशारीने पिकवा"},
    "grow_sub":         {"English": "पिक निवड व माती विश्लेषण", "मराठी": "पिक निवड व माती विश्लेषण"},
    "grow_desc":        {"English": "Soil chemistry · pH maps · Crop-soil matching · Sowing calendar · Fertiliser plans", "मराठी": "माती रसायनशास्त्र · pH नकाशे · पिक-माती जुळणी · पेरणी दिनदर्शिका · खत योजना"},
    "maintain_title":   {"English": "Crop Health", "मराठी": "पिकाचे आरोग्य"},
    "maintain_sub":     {"English": "रोग व कीड व्यवस्थापन", "मराठी": "रोग व कीड व्यवस्थापन"},
    "maintain_desc":    {"English": "Pest ID · Disease diagnosis · Irrigation · Nutrient deficiency · Organic remedies", "मराठी": "कीड ओळख · रोग निदान · सिंचन · पोषक कमतरता · सेंद्रिय उपाय"},
    "sell_title":       {"English": "Market Intelligence", "मराठी": "बाजार माहिती"},
    "sell_sub":         {"English": "भाव व विक्री धोरण", "मराठी": "भाव व विक्री धोरण"},
    "sell_desc":        {"English": "Live Agmarknet prices · MSP alerts · APMC comparison · Price forecasts · Cold storage", "मराठी": "Agmarknet भाव · MSP सूचना · APMC तुलना · भाव अंदाज · शीतगृह"},
    # Growing page
    "grow_header":      {"English": "🌱 Smart Crop Advisor", "मराठी": "🌱 हुशार पिक सल्लागार"},
    "grow_subhd":       {"English": "Soil chemistry, pH analysis & crop-matching powered by ICAR data", "मराठी": "ICAR डेटावर आधारित माती रसायनशास्त्र, pH विश्लेषण व पिक जुळणी"},
    "tab_soil":         {"English": "🔬 Soil Analysis & Crop Match", "मराठी": "🔬 माती विश्लेषण व पिक जुळणी"},
    "tab_soildb":       {"English": "📋 Soil Chemistry Database", "मराठी": "📋 माती रसायन डेटाबेस"},
    "tab_aichat":       {"English": "💬 Ask AI Agronomist", "मराठी": "💬 AI कृषितज्ञाला विचारा"},
    "soil_type":        {"English": "Soil Type", "मराठी": "मातीचा प्रकार"},
    "season":           {"English": "Season", "मराठी": "हंगाम"},
    "soil_ph":          {"English": "Your Soil pH", "मराठी": "आपल्या मातीचा pH"},
    "water_src":        {"English": "Water Source", "मराठी": "पाण्याचा स्रोत"},
    "district":         {"English": "District", "मराठी": "जिल्हा"},
    "get_rec":          {"English": "🔍 Get Full Crop Recommendation", "मराठी": "🔍 संपूर्ण पिक शिफारस मिळवा"},
    # Maintaining page
    "maint_header":     {"English": "🩺 Crop Health Centre", "मराठी": "🩺 पिक आरोग्य केंद्र"},
    "maint_subhd":      {"English": "AI diagnosis for pests, diseases, irrigation & nutrition — tailored to Maharashtra", "मराठी": "महाराष्ट्रासाठी कीड, रोग, सिंचन व पोषण यांचे AI निदान"},
    "tab_pest":         {"English": "🐛 Pest & Disease Diagnosis", "मराठी": "🐛 कीड व रोग निदान"},
    "tab_nutr":         {"English": "💧 Irrigation & Nutrition", "मराठी": "💧 सिंचन व पोषण"},
    "tab_hchat":        {"English": "💬 Health Chatbot", "मराठी": "💬 आरोग्य चॅटबॉट"},
    "crop_lbl":         {"English": "Crop", "मराठी": "पिक"},
    "growth_stage":     {"English": "Growth Stage", "मराठी": "वाढीचा टप्पा"},
    "symptom":          {"English": "Describe the problem", "मराठी": "समस्या सांगा"},
    "symptom_ph":       {"English": "E.g. leaves curling yellow, white powder on stem…", "मराठी": "उदा. पाने पिवळी होणे, खोडावर पांढरी पावडर…"},
    "diagnose_btn":     {"English": "🔬 Diagnose & Suggest Treatment", "मराठी": "🔬 निदान करा व उपचार सुचवा"},
    # Selling page
    "sell_header":      {"English": "💰 Market Intelligence Centre", "मराठी": "💰 बाजार माहिती केंद्र"},
    "sell_subhd":       {"English": "Live Agmarknet APMC prices · MSP 2024-25 · AI market forecasts", "मराठी": "Agmarknet APMC भाव · MSP 2024-25 · AI बाजार अंदाज"},
    "tab_price":        {"English": "📊 APMC Price Explorer", "मराठी": "📊 APMC भाव एक्सप्लोरर"},
    "tab_msp":          {"English": "🏛️ MSP Reference 2024-25", "मराठी": "🏛️ MSP संदर्भ 2024-25"},
    "tab_aimarket":     {"English": "🤖 AI Market Advisor", "मराठी": "🤖 AI बाजार सल्लागार"},
    "tab_sellchat":     {"English": "💬 Selling Chatbot", "मराठी": "💬 विक्री चॅटबॉट"},
    "best_market":      {"English": "Best Market", "मराठी": "सर्वोत्तम बाजार"},
    "avg_modal":        {"English": "Avg Modal Price", "मराठी": "सरासरी मोडल भाव"},
    "highest":          {"English": "Highest", "मराठी": "सर्वाधिक"},
    "lowest":           {"English": "Lowest", "मराठी": "सर्वात कमी"},
    "clear_chat":       {"English": "🗑️ Clear Chat", "मराठी": "🗑️ चॅट साफ करा"},
    "chat_input_grow":  {"English": "E.g. Best crop for black cotton soil pH 7.8 in Vidarbha?", "मराठी": "उदा. विदर्भातील काळी माती pH 7.8 साठी सर्वोत्तम पिक कोणते?"},
    "chat_input_maint": {"English": "Ask about pests, diseases, fertilisers, water…", "मराठी": "कीड, रोग, खते, पाणी याबद्दल विचारा…"},
    "chat_input_sell":  {"English": "Ask about mandi prices, MSP, storage, export…", "मराठी": "मंडी भाव, MSP, साठवण, निर्यात याबद्दल विचारा…"},
    "quick_q":          {"English": "Quick Questions — tap to ask:", "मराठी": "झटपट प्रश्न — दाबा:"},
    "ai_source":        {"English": "AI analysis grounded in ICAR soil profiles · Agmarknet price data · KVK Maharashtra", "मराठी": "ICAR माती प्रोफाइल · Agmarknet भाव डेटा · KVK महाराष्ट्र यावर आधारित AI विश्लेषण"},
    "region_lbl":       {"English": "Region", "मराठी": "प्रदेश"},
    "irr_type":         {"English": "Irrigation type", "मराठी": "सिंचन प्रकार"},
    "area_ha":          {"English": "Area (hectares)", "मराठी": "क्षेत्र (हेक्टर)"},
    "msp_season":       {"English": "Season", "मराठी": "हंगाम"},
    "qty_qtl":          {"English": "Quantity (quintals)", "मराठी": "प्रमाण (क्विंटल)"},
    "harvest_in":       {"English": "Harvest ready in", "मराठी": "काढणी कधी"},
    "storage_lbl":      {"English": "Storage available", "मराठी": "साठवण सुविधा"},
    "gen_schedule":     {"English": "Generate schedule", "मराठी": "वेळापत्रक तयार करा"},
    "market_strategy":  {"English": "Get market strategy", "मराठी": "बाजार धोरण मिळवा"},
    "ask_ai_exp":       {"English": "Ask AI about this section", "मराठी": "या विभागाबद्दल AI ला विचारा"},
    "your_question":    {"English": "Your question", "मराठी": "आपला प्रश्न"},
    "send":             {"English": "Send", "मराठी": "पाठवा"},
    "live_prices_note": {"English": "Prices merge live Maharashtra mandi rows when DATA_GOV_IN_API_KEY is set (refreshed daily). Otherwise embedded Agmarknet-style baselines apply.", "मराठी": "DATA_GOV_IN_API_KEY सेट केल्यास महाराष्ट्र मंडीचे थेट ओळ दर दररोज विलीन होतात; नसल्यास एम्बेड केलेले आधारभूत भाव लागू."},
    "env_key_hint":     {"English": "Or set GOOGLE_API_KEY in the environment.", "मराठी": "किंवा वातावरणात GOOGLE_API_KEY सेट करा."},
    "grow_nav_desc":    {"English": "Soil pH & type · Sowing · Crops & varieties", "मराठी": "माती pH व प्रकार · पेरणी · पिके व वाण"},
    "maint_nav_desc":   {"English": "Pests · Disease · Irrigation · Nutrition", "मराठी": "कीड · रोग · सिंचन · पोषण"},
    "sell_nav_desc":    {"English": "Maharashtra mandi prices · MSP · Sales tips", "मराठी": "महाराष्ट्र मंडी भाव · MSP · विक्री सल्ले"},
    "data_gov_api_hint": {"English": "Live mandi merge: set DATA_GOV_IN_API_KEY (data.gov.in).", "मराठी": "थेट मंडी: DATA_GOV_IN_API_KEY (data.gov.in) सेट करा."},
    "no_api_key":       {"English": "Set **GOOGLE_API_KEY** (Google AI Studio). Required for AI answers.", "मराठी": "**GOOGLE_API_KEY** सेट करा (Google AI Studio). AI साठी आवश्यक."},
    "api_deploy_note":  {"English": "Deployment: configure `GOOGLE_API_KEY` in your host (Streamlit Cloud secrets, Docker env, or `.env`).", "मराठी": "डिप्लॉयमेंट: होस्टवर `GOOGLE_API_KEY` कॉन्फिगर करा."},
    "chip_working":     {"English": "Preparing a detailed answer…", "मराठी": "तपशीलवार उत्तर तयार होत आहे…"},
    "clear_chip_ans":   {"English": "Clear answer", "मराठी": "उत्तर साफ करा"},
    "weather_toggle_on": {"English": "🌤 Live weather — tap to expand", "मराठी": "🌤 थेट हवामान — विस्तारासाठी दाबा"},
    "weather_collapse": {"English": "▲ Collapse", "मराठी": "▲ बंद करा"},
    "weather_head":     {"English": "Live weather", "मराठी": "थेट हवामान"},
    "weather_sub":      {"English": "Powered by Google Weather API · Maharashtra location", "मराठी": "Google Weather API · महाराष्ट्र स्थान"},
    "weather_district": {"English": "District (location)", "मराठी": "जिल्हा (स्थान)"},
    "weather_refresh":  {"English": "Refresh data", "मराठी": "डेटा रिफ्रेश करा"},
    "weather_no_key":   {"English": "Set **GOOGLE_WEATHER_API_KEY** or **GOOGLE_MAPS_API_KEY** in the environment with the Weather API enabled (Google Cloud Console) to load live forecasts.", "मराठी": "थेट अंदाजासाठी वातावरणात **GOOGLE_WEATHER_API_KEY** किंवा **GOOGLE_MAPS_API_KEY** सेट करा (Google Cloud Console मध्ये Weather API सुरू)."},
    "weather_err":      {"English": "Could not load weather", "मराठी": "हवामान मिळाले नाही"},
    "weather_hourly":   {"English": "Today — hourly", "मराठी": "आज — तासानुसार"},
    "weather_daily":    {"English": "Daily outlook (up to 10 days)", "मराठी": "दैनिक अंदाज (१० दिवसांपर्यंत)"},
    "weather_src":      {"English": "Source: Google Weather API", "मराठी": "स्रोत: Google Weather API"},
    "sell_intro":       {"English": "Mandi prices, MSP charts, and AI advice below — use the chart toolbar to zoom and pan.", "मराठी": "खाली मंडी भाव, MSP आलेख व AI सल्ला — झूम व पॅनसाठी आलेखाची साधने वापरा."},
}

def t(key):
    return T.get(key, {}).get(st.session_state.lang, T.get(key, {}).get("English", key))

# ══════════════════════════════════════════════════
# PROMPT CHIPS — bilingual
# ══════════════════════════════════════════════════
CHIPS = {
    "grow": {
        "English": [
            "Best crop for black cotton soil pH 7.8?",
            "Kharif sowing calendar for Vidarbha 2024?",
            "How to fix nitrogen deficiency in soybean?",
            "Fertiliser dose for cotton per hectare?",
            "Intercropping options after sugarcane?",
            "Organic farming certification in Maharashtra?",
        ],
        "मराठी": [
            "काळ्या मातीत pH 7.8 साठी सर्वोत्तम पिक?",
            "विदर्भातील खरीप पेरणी दिनदर्शिका 2024?",
            "सोयाबीनमधील नत्र कमतरता कशी दूर करावी?",
            "कापसाला प्रति हेक्टर खताचा डोस किती?",
            "उसानंतर आंतरपीक पर्याय कोणते?",
            "महाराष्ट्रात सेंद्रिय शेती प्रमाणपत्र कसे मिळवावे?",
        ],
    },
    "maintain": {
        "English": [
            "Yellow leaves on soybean — what's wrong?",
            "Bollworm attack on cotton — organic control?",
            "Fungicide spray schedule for grapes Nashik?",
            "Iron & zinc deficiency symptoms & treatment?",
            "Drip irrigation schedule for onion per stage?",
            "How to manage powdery mildew on grapes?",
        ],
        "मराठी": [
            "सोयाबीनची पाने पिवळी — काय चुकते आहे?",
            "कापसावर बोंड अळी — सेंद्रिय नियंत्रण?",
            "नाशिकमधील द्राक्षावर बुरशीनाशक वेळापत्रक?",
            "लोह व जस्त कमतरतेची लक्षणे व उपाय?",
            "कांद्यासाठी टप्प्यानुसार ठिबक सिंचन वेळापत्रक?",
            "द्राक्षावरील भुरी रोग कसा व्यवस्थापित करावा?",
        ],
    },
    "sell": {
        "English": [
            "Best time to sell onion in Nashik 2025?",
            "Cotton MSP 2024-25 vs current mandi rate?",
            "Which APMC gives best price for soybean?",
            "Grapes export procedure from Nashik?",
            "How to get better price than mandi rate?",
            "Warehouse receipt loan for cotton farmers?",
        ],
        "मराठी": [
            "नाशिकमध्ये कांदा विकण्याची सर्वोत्तम वेळ 2025?",
            "कापूस MSP 2024-25 विरुद्ध सध्याचा मंडी भाव?",
            "सोयाबीनला सर्वोत्तम भाव कोणत्या APMC मध्ये?",
            "नाशिकहून द्राक्ष निर्यात प्रक्रिया काय आहे?",
            "मंडी दरापेक्षा जास्त भाव कसा मिळवावा?",
            "कापूस शेतकऱ्यांसाठी गोदाम पावती कर्ज?",
        ],
    },
}

# Reference text for planting advice (RAG + prompts); supplement to ICAR/KVK — verify locally before sowing.
CROP_VARIETY_REFERENCE = """
Maharashtra crop & variety reference (indicative; confirm with KVK / SAU / seed dept):
Cereals: Paddy — Jaya, Indrayani, Karjat-3, MTU-1010, HMT, Pusa Basmati; Wheat — Lok-1, GW-496, HD-2189, DDW-47; Jowar — CSH-16, CSV-17, Phule Chitra, Phule Yashoda; Bajra — ICTP-8203, GHB-558, ICMV-221; Maize — PMH-10, DHM-117, HM-4, African tall.
Pulses: Tur — BDN-711, BDN-716, PKV TARA; Moong — BM-4, Kopergaon; Urad — TAU-1, PDU-105; Chickpea — JG-11, Vijay, Digvijay; Lentil — IPL-406; Field pea — AP-3.
Oilseeds: Soybean — JS-335, NRC-37, MAUS-71, Phule Kimaya, Kalitur; Groundnut — JL-24, TG-37A, ICGV-86590; Sunflower — KBSH-44, Phule Raviraj; Sesame — GT-10; Mustard — Pusa Bold, Varuna.
Fibres: Cotton — Bunny, H-4, H-6, Jayadhar, NHH-44, PKV Rajat, desi & Bt hybrids as per zone.
Cash / horticulture: Sugarcane — Co-86032, Co-0238, VSI-800; Onion — Bhima Kiran, Bhima Shakti, Agrifound Dark Red; Tomato — Pusa Ruby, Arka Rakshak; Grapes — Thompson Seedless, Sharad, Manik Chaman; Pomegranate — Bhagwa, Ruby, Ganesh; Banana — Grand Naine, Basrai; Turmeric — Salem; Mango — Kesar, Alphonso, Dashehari; Cashew — Vengurla-4.
Spices & others: Chili — Tejaswini, Byadgi; Coriander — Local improved; Garlic — Agrifound Parvati; Brinjal — Phule Arjun; Okra — Phule Utkarsh; Cucurbits — local hybrids; Coconut — WCT, COD.
For each district agro-climatic zone (Vidarbha, Marathwada, Western MH, North MH, Konkan) match Kharif/Rabi/Zaid calendars; soil pH 5.5–8.5 typical management: lime for acidity, gypsum for sodic, organic matter for sandy.
"""

# ══════════════════════════════════════════════════
# EMBEDDED DATA (Agmarknet / ICAR / CACP)
# ══════════════════════════════════════════════════
PRICE_CSV = """Crop,Crop_MR,District,Market,Min_Price,Modal_Price,Max_Price,Season,Arrival_MT
Onion,कांदा,Nashik,Lasalgaon,800,1100,1400,Kharif,12500
Onion,कांदा,Nashik,Pimpalgaon,750,1050,1350,Kharif,9800
Onion,कांदा,Nashik,Niphad,820,1080,1380,Kharif,7600
Onion,कांदा,Nashik,Satana,780,1020,1320,Kharif,5200
Onion,कांदा,Nashik,Malegaon,760,1000,1300,Kharif,4800
Onion,कांदा,Pune,Pune,900,1250,1600,Kharif,7200
Onion,कांदा,Pune,Junnar,870,1180,1520,Kharif,4100
Onion,कांदा,Solapur,Solapur,700,980,1250,Kharif,5400
Onion,कांदा,Solapur,Barshi,720,1000,1280,Kharif,3800
Onion,कांदा,Ahmednagar,Rahuri,820,1120,1450,Kharif,4300
Onion,कांदा,Ahmednagar,Shrirampur,800,1100,1420,Kharif,3600
Onion,कांदा,Dhule,Dhule,750,1020,1310,Kharif,3200
Onion,कांदा,Jalgaon,Jalgaon,760,1030,1320,Kharif,2900
Onion,कांदा,Aurangabad,Aurangabad,810,1080,1380,Kharif,3100
Onion,कांदा,Jalna,Jalna,790,1060,1360,Kharif,2700
Onion,कांदा,Nashik,Lasalgaon,1200,1850,2400,Rabi,18000
Onion,कांदा,Nashik,Pimpalgaon,1100,1720,2200,Rabi,14200
Onion,कांदा,Nashik,Niphad,1150,1780,2300,Rabi,11500
Onion,कांदा,Nashik,Satana,1080,1700,2200,Rabi,8900
Onion,कांदा,Nashik,Malegaon,1060,1680,2180,Rabi,7600
Onion,कांदा,Pune,Pune,1300,1950,2600,Rabi,9800
Onion,कांदा,Pune,Junnar,1250,1900,2500,Rabi,6200
Onion,कांदा,Solapur,Solapur,1000,1580,2100,Rabi,6500
Onion,कांदा,Solapur,Barshi,1020,1600,2120,Rabi,4800
Onion,कांदा,Ahmednagar,Rahuri,1150,1750,2350,Rabi,7100
Onion,कांदा,Ahmednagar,Shrirampur,1120,1720,2300,Rabi,5500
Onion,कांदा,Dhule,Dhule,1050,1650,2200,Rabi,4600
Onion,कांदा,Jalgaon,Jalgaon,1070,1670,2220,Rabi,4100
Onion,कांदा,Aurangabad,Aurangabad,1100,1700,2280,Rabi,4400
Onion,कांदा,Jalna,Jalna,1080,1680,2250,Rabi,3900
Onion,कांदा,Latur,Latur,980,1550,2080,Rabi,5200
Onion,कांदा,Nanded,Nanded,1000,1580,2100,Rabi,4300
Onion,कांदा,Osmanabad,Osmanabad,960,1530,2060,Rabi,3700
Onion,कांदा,Kolhapur,Kolhapur,1050,1620,2180,Rabi,3500
Onion,कांदा,Satara,Satara,1020,1600,2150,Rabi,3200
Onion,कांदा,Sangli,Sangli,1030,1610,2160,Rabi,2900
Cotton,कापूस,Nagpur,Nagpur,6000,6400,6800,Kharif,8200
Cotton,कापूस,Nagpur,Kamptee,5950,6350,6750,Kharif,5100
Cotton,कापूस,Amravati,Amravati,5800,6200,6600,Kharif,11500
Cotton,कापूस,Amravati,Daryapur,5750,6150,6550,Kharif,7800
Cotton,कापूस,Amravati,Achalpur,5780,6180,6580,Kharif,6200
Onion,कांदा,Washim,Washim,920,1420,1900,Rabi,2600
Cotton,कापूस,Yavatmal,Yavatmal,5900,6300,6700,Kharif,13200
Cotton,कापूस,Yavatmal,Wani,5850,6250,6650,Kharif,9800
Cotton,कापूस,Yavatmal,Pusad,5820,6220,6620,Kharif,7500
Cotton,कापूस,Wardha,Wardha,6100,6500,6900,Kharif,7800
Cotton,कापूस,Wardha,Hinganghat,6150,6550,6950,Kharif,9200
Cotton,कापूस,Akola,Akola,5750,6150,6550,Kharif,9600
Cotton,कापूस,Akola,Balapur,5700,6100,6500,Kharif,6800
Cotton,कापूस,Buldhana,Khamgaon,5900,6250,6600,Kharif,8100
Cotton,कापूस,Buldhana,Malkapur,5850,6200,6550,Kharif,6200
Cotton,कापूस,Washim,Washim,5800,6200,6600,Kharif,5400
Cotton,कापूस,Washim,Risod,5780,6180,6580,Kharif,4200
Cotton,कापूस,Hingoli,Hingoli,5820,6220,6620,Kharif,4800
Cotton,कापूस,Nanded,Nanded,5850,6250,6650,Kharif,5200
Cotton,कापूस,Osmanabad,Osmanabad,5780,6180,6580,Kharif,3900
Cotton,कापूस,Latur,Latur,5800,6200,6600,Kharif,4300
Cotton,कापूस,Aurangabad,Aurangabad,5900,6300,6700,Kharif,4100
Cotton,कापूस,Jalna,Jalna,5850,6250,6650,Kharif,3800
Cotton,कापूस,Beed,Beed,5820,6220,6620,Kharif,4600
Cotton,कापूस,Parbhani,Parbhani,5840,6240,6640,Kharif,5100
Cotton,कापूस,Dhule,Dhule,5750,6150,6550,Kharif,3200
Cotton,कापूस,Jalgaon,Jalgaon,5780,6180,6580,Kharif,3600
Soybean,सोयाबीन,Latur,Latur,3800,4200,4600,Kharif,14000
Soybean,सोयाबीन,Latur,Udgir,3780,4180,4580,Kharif,8200
Soybean,सोयाबीन,Osmanabad,Osmanabad,3700,4100,4500,Kharif,9300
Soybean,सोयाबीन,Osmanabad,Tuljapur,3720,4120,4520,Kharif,5800
Soybean,सोयाबीन,Nanded,Nanded,3750,4150,4550,Kharif,8700
Soybean,सोयाबीन,Nanded,Deglur,3730,4130,4530,Kharif,5400
Soybean,सोयाबीन,Jalna,Jalna,3900,4300,4700,Kharif,7200
Soybean,सोयाबीन,Aurangabad,Aurangabad,4000,4400,4800,Kharif,6100
Soybean,सोयाबीन,Beed,Beed,3850,4250,4650,Kharif,7800
Soybean,सोयाबीन,Parbhani,Parbhani,3820,4220,4620,Kharif,6900
Soybean,सोयाबीन,Hingoli,Hingoli,3800,4200,4600,Kharif,5600
Soybean,सोयाबीन,Washim,Washim,3780,4180,4580,Kharif,4900
Soybean,सोयाबीन,Akola,Akola,3850,4250,4650,Kharif,5200
Soybean,सोयाबीन,Amravati,Amravati,3870,4270,4670,Kharif,6100
Soybean,सोयाबीन,Yavatmal,Yavatmal,3860,4260,4660,Kharif,5800
Soybean,सोयाबीन,Nagpur,Nagpur,3900,4300,4700,Kharif,4200
Soybean,सोयाबीन,Ahmednagar,Ahmednagar,3950,4350,4750,Kharif,3900
Soybean,सोयाबीन,Pune,Pune,4000,4400,4800,Kharif,3600
Soybean,सोयाबीन,Solapur,Solapur,3780,4180,4580,Kharif,5100
Tur Dal,तूर डाळ,Latur,Latur,5500,6800,7500,Kharif,4200
Tur Dal,तूर डाळ,Latur,Udgir,5400,6700,7400,Kharif,3100
Tur Dal,तूर डाळ,Osmanabad,Osmanabad,5300,6600,7300,Kharif,3800
Tur Dal,तूर डाळ,Osmanabad,Tuljapur,5250,6550,7250,Kharif,2600
Tur Dal,तूर डाळ,Nanded,Nanded,5400,6700,7400,Kharif,2900
Tur Dal,तूर डाळ,Nanded,Deglur,5350,6650,7350,Kharif,2100
Tur Dal,तूर डाळ,Solapur,Solapur,5600,6900,7600,Kharif,3100
Tur Dal,तूर डाळ,Solapur,Barshi,5550,6850,7550,Kharif,2400
Tur Dal,तूर डाळ,Akola,Akola,5200,6400,7100,Kharif,2500
Tur Dal,तूर डाळ,Aurangabad,Aurangabad,5450,6750,7450,Kharif,2800
Tur Dal,तूर डाळ,Jalna,Jalna,5400,6700,7400,Kharif,2200
Tur Dal,तूर डाळ,Beed,Beed,5380,6680,7380,Kharif,2100
Tur Dal,तूर डाळ,Amravati,Amravati,5300,6600,7300,Kharif,1900
Tur Dal,तूर डाळ,Nagpur,Nagpur,5450,6750,7450,Kharif,2300
Tur Dal,तूर डाళ,Yavatmal,Yavatmal,5350,6650,7350,Kharif,2000
Tur Dal,తూర్ డాళ్,Parbhani,Parbhani,5320,6620,7320,Kharif,1800
Wheat,గహు,Pune,Pune,2100,2350,2600,Rabi,9200
Wheat,గహు,Pune,Baramati,2080,2330,2580,Rabi,6800
Wheat,గహు,Nashik,Nashik,2050,2300,2550,Rabi,8100
Wheat,గహు,Nashik,Yeola,2030,2280,2530,Rabi,5600
Wheat,గహు,Solapur,Solapur,2000,2250,2500,Rabi,7400
Wheat,గహు,Solapur,Barshi,1980,2230,2480,Rabi,5100
Wheat,గహు,Aurangabad,Aurangabad,2080,2320,2570,Rabi,6800
Wheat,గహు,Aurangabad,Paithan,2060,2300,2550,Rabi,4500
Wheat,గహు,Nagpur,Nagpur,2150,2400,2650,Rabi,5500
Wheat,గహు,Nagpur,Kamptee,2130,2380,2630,Rabi,3800
Wheat,గహు,Latur,Latur,2020,2270,2520,Rabi,6200
Wheat,గహు,Nanded,Nanded,2010,2260,2510,Rabi,5400
Wheat,గహు,Jalgaon,Jalgaon,2090,2340,2590,Rabi,7200
Wheat,గహు,Dhule,Dhule,2070,2320,2570,Rabi,6100
Wheat,గహు,Ahmednagar,Ahmednagar,2060,2310,2560,Rabi,5800
Wheat,గహు,Amravati,Amravati,2120,2370,2620,Rabi,4900
Wheat,గహు,Akola,Akola,2100,2350,2600,Rabi,4600
Wheat,గహు,Beed,Beed,2000,2250,2500,Rabi,4200
Wheat,గహు,Osmanabad,Osmanabad,1990,2240,2490,Rabi,3900
Wheat,గహు,Kolhapur,Kolhapur,2050,2300,2550,Rabi,3600
Wheat,గహు,Satara,Satara,2030,2280,2530,Rabi,3400
Wheat,గహు,Sangli,Sangli,2040,2290,2540,Rabi,3200
Wheat,గహు,Wardha,Wardha,2110,2360,2610,Rabi,3800
Wheat,గహు,Yavatmal,Yavatmal,2080,2330,2580,Rabi,3500
Wheat,గహు,Washim,Washim,2070,2320,2570,Rabi,2900
Wheat,గహు,Hingoli,Hingoli,2030,2280,2530,Rabi,2700
Wheat,గహు,Parbhani,Parbhani,2040,2290,2540,Rabi,3100
Wheat,గహు,Jalna,Jalna,2060,2310,2560,Rabi,3300
Sugarcane,ऊस,Kolhapur,Kolhapur,3400,3650,3900,Annual,25000
Sugarcane,ऊस,Kolhapur,Ichalkaranji,3380,3630,3880,Annual,18500
Sugarcane,ऊस,Satara,Satara,3300,3550,3800,Annual,21000
Sugarcane,ऊस,Satara,Karad,3280,3530,3780,Annual,15200
Sugarcane,ऊस,Sangli,Sangli,3350,3600,3850,Annual,18500
Sugarcane,ऊस,Sangli,Miraj,3330,3580,3830,Annual,14200
Sugarcane,ऊस,Pune,Pune,3200,3450,3700,Annual,15000
Sugarcane,ऊस,Pune,Baramati,3250,3500,3750,Annual,12000
Sugarcane,ऊस,Solapur,Solapur,3100,3350,3600,Annual,12000
Sugarcane,ऊस,Solapur,Akkalkot,3080,3330,3580,Annual,8500
Sugarcane,ऊस,Ahmednagar,Ahmednagar,3150,3400,3650,Annual,11000
Sugarcane,ऊस,Nashik,Nashik,3200,3450,3700,Annual,9500
Sugarcane,ऊस,Aurangabad,Aurangabad,3050,3300,3550,Annual,7800
Sugarcane,ऊस,Latur,Latur,3000,3250,3500,Annual,6500
Sugarcane,ऊस,Nanded,Nanded,3020,3270,3520,Annual,5800
Groundnut,भुईमूग,Solapur,Solapur,4500,5200,5900,Kharif,5800
Groundnut,भुईमूग,Solapur,Barshi,4450,5150,5850,Kharif,4100
Groundnut,भुईमूग,Latur,Latur,4400,5100,5800,Kharif,4200
Groundnut,भुईमूग,Osmanabad,Osmanabad,4300,5000,5700,Kharif,3600
Groundnut,भुईमूग,Ahmednagar,Ahmednagar,4600,5300,6000,Kharif,4900
Groundnut,भुईमूग,Ahmednagar,Rahuri,4550,5250,5950,Kharif,3800
Groundnut,भुईमूग,Pune,Pune,4700,5400,6100,Kharif,3500
Groundnut,భుईమూగ్,Nashik,Nashik,4650,5350,6050,Kharif,3200
Groundnut,భుఇమూగ్,Aurangabad,Aurangabad,4400,5100,5800,Kharif,2900
Groundnut,భుఇమూగ్,Jalna,Jalna,4380,5080,5780,Kharif,2600
Groundnut,భుఇమూగ్,Nagpur,Nagpur,4500,5200,5900,Kharif,2400
Groundnut,భుఇమూగ్,Amravati,Amravati,4420,5120,5820,Kharif,2700
Jowar,ज्वारी,Solapur,Solapur,2200,2550,2900,Rabi,6200
Jowar,ज्वारी,Solapur,Barshi,2150,2500,2850,Rabi,4500
Jowar,ज्वारी,Latur,Latur,2100,2450,2800,Rabi,5100
Jowar,ज्वारी,Osmanabad,Osmanabad,2050,2400,2750,Rabi,4300
Jowar,ज्वारी,Aurangabad,Aurangabad,2200,2550,2900,Rabi,3800
Jowar,ज्वारी,Nanded,Nanded,2120,2470,2820,Rabi,3600
Jowar,ज्वारी,Pune,Pune,2250,2600,2950,Rabi,3200
Jowar,ज्वारी,Nashik,Nashik,2180,2530,2880,Rabi,2900
Jowar,ज्वारी,Beed,Beed,2100,2450,2800,Rabi,2700
Jowar,ज्वारी,Kolhapur,Kolhapur,2050,2400,2750,Rabi,2200
Jowar,ज्वारी,Sangli,Sangli,2080,2430,2780,Rabi,2100
Jowar,ज्वारी,Satara,Satara,2060,2410,2760,Rabi,2000
Jowar,ज्वारी,Jalna,Jalna,2090,2440,2790,Rabi,2400
Jowar,ज्वारी,Ahmednagar,Ahmednagar,2200,2550,2900,Kharif,5800
Jowar,ज्वारी,Nagpur,Nagpur,2150,2500,2850,Kharif,4200
Bajra,बाजरी,Nashik,Nashik,1900,2200,2500,Kharif,4800
Bajra,बाजरी,Nashik,Nandurbar,1870,2170,2470,Kharif,3600
Bajra,बाजरी,Ahmednagar,Ahmednagar,1850,2150,2450,Kharif,3900
Bajra,बाजरी,Pune,Pune,1920,2220,2520,Kharif,3200
Bajra,बाजरी,Solapur,Solapur,1830,2130,2430,Kharif,2900
Bajra,बाजरी,Aurangabad,Aurangabad,1880,2180,2480,Kharif,2600
Bajra,बाजरी,Jalgaon,Jalgaon,1890,2190,2490,Kharif,2800
Bajra,बाजरी,Dhule,Dhule,1860,2160,2460,Kharif,2500
Bajra,बाजरी,Jalna,Jalna,1840,2140,2440,Kharif,2200
Bajra,बाजरी,Beed,Beed,1820,2120,2420,Kharif,2000
Bajra,बाजरी,Osmanabad,Osmanabad,1810,2110,2410,Kharif,1900
Bajra,बाजरी,Nanded,Nanded,1830,2130,2430,Kharif,2100
Bajra,बाजरी,Nagpur,Nagpur,1900,2200,2500,Kharif,1800
Tomato,टोमॅटो,Nashik,Lasalgaon,600,950,1600,Kharif,8500
Tomato,टोमॅटो,Nashik,Pimpalgaon,580,920,1580,Kharif,6200
Tomato,टोमॅटो,Nashik,Niphad,560,900,1550,Kharif,5100
Tomato,टोमॅटो,Pune,Pune,700,1100,1800,Kharif,6200
Tomato,टोमॅटो,Pune,Junnar,650,1050,1750,Kharif,4800
Tomato,टोमॅटो,Ahmednagar,Ahmednagar,680,1080,1780,Kharif,4200
Tomato,टोमॅटो,Solapur,Solapur,620,980,1620,Kharif,3800
Tomato,टोमॅटो,Aurangabad,Aurangabad,650,1020,1680,Kharif,3500
Tomato,टोमॅटो,Nagpur,Nagpur,700,1120,1800,Kharif,3100
Tomato,टोमॅटो,Latur,Latur,630,1000,1650,Kharif,2900
Tomato,टोमॅटो,Kolhapur,Kolhapur,720,1150,1850,Kharif,2700
Tomato,टोमॅटो,Satara,Satara,700,1100,1800,Kharif,2500
Tomato,टोमॅटो,Nashik,Lasalgaon,800,1200,2000,Rabi,9800
Tomato,टोमॅटो,Pune,Pune,900,1400,2200,Rabi,7500
Grapes,द्राक्षे,Nashik,Nashik,3000,4500,6500,Annual,15000
Grapes,द्राक्षे,Nashik,Niphad,2900,4400,6400,Annual,11500
Grapes,द्राक्षे,Nashik,Dindori,2800,4300,6200,Annual,9800
Grapes,द्राक्षे,Nashik,Malegaon,2700,4100,6000,Annual,7200
Grapes,द्राक्षे,Sangli,Sangli,2800,4200,6200,Annual,9800
Grapes,द्राक्षे,Sangli,Tasgaon,2750,4100,6100,Annual,7500
Grapes,द्राक्षे,Pune,Pune,3100,4700,6800,Annual,5200
Grapes,द्राक्षे,Solapur,Solapur,2600,3900,5800,Annual,3800
Grapes,द्राक्षे,Ahmednagar,Ahmednagar,2700,4000,5900,Annual,4200
Pomegranate,डाळिंब,Solapur,Solapur,4000,5800,7500,Annual,7200
Pomegranate,डाळिंब,Solapur,Barshi,3900,5700,7400,Annual,5400
Pomegranate,डाळिंब,Sangli,Sangli,3800,5500,7200,Annual,5400
Pomegranate,डाळिंब,Nashik,Nashik,4100,5900,7700,Annual,6200
Pomegranate,डाळिंब,Ahmednagar,Ahmednagar,4050,5850,7650,Annual,5800
Pomegranate,डाळिंब,Pune,Pune,4200,6000,7800,Annual,4900
Pomegranate,डाळिंब,Latur,Latur,3750,5400,7050,Annual,3600
Pomegranate,डाळिंब,Osmanabad,Osmanabad,3700,5350,7000,Annual,3200
Pomegranate,डाळिंब,Aurangabad,Aurangabad,3900,5600,7300,Annual,4100
Chickpea,हरभरा,Latur,Latur,4800,5200,5700,Rabi,8900
Chickpea,हरभरा,Latur,Udgir,4750,5150,5650,Rabi,6200
Chickpea,हरभरा,Osmanabad,Osmanabad,4700,5100,5600,Rabi,7200
Chickpea,हरभरा,Nanded,Nanded,4750,5150,5650,Rabi,6100
Chickpea,हरभरा,Solapur,Solapur,4820,5220,5720,Rabi,7800
Chickpea,हरभरा,Aurangabad,Aurangabad,4780,5180,5680,Rabi,5800
Chickpea,हरभरा,Jalna,Jalna,4760,5160,5660,Rabi,5200
Chickpea,हरभरा,Beed,Beed,4740,5140,5640,Rabi,4900
Chickpea,हरभरा,Ahmednagar,Ahmednagar,4800,5200,5700,Rabi,5500
Chickpea,हरभरा,Pune,Pune,4850,5250,5750,Rabi,4800
Chickpea,हरभरा,Nashik,Nashik,4830,5230,5730,Rabi,4500
Chickpea,हरभरा,Akola,Akola,4710,5110,5610,Rabi,4200
Chickpea,हरभरा,Amravati,Amravati,4720,5120,5620,Rabi,4000
Chickpea,हरभरा,Nagpur,Nagpur,4780,5180,5680,Rabi,3800
Chickpea,हरभरा,Wardha,Wardha,4760,5160,5660,Rabi,3600
Chickpea,हरभरा,Yavatmal,Yavatmal,4730,5130,5630,Rabi,3400
Rice,तांदूळ,Raigad,Pen,1800,2100,2400,Kharif,11000
Rice,तांदूळ,Raigad,Alibag,1820,2120,2420,Kharif,8500
Rice,तांदूळ,Ratnagiri,Ratnagiri,1900,2200,2500,Kharif,8500
Tur Dal,तूर डाळ,Buldhana,Khamgaon,5280,6580,7280,Kharif,1700
Rice,तांदूळ,Ratnagiri,Chiplun,1880,2180,2480,Kharif,6800
Rice,तांदूळ,Sindhudurg,Kudal,1850,2150,2450,Kharif,6200
Rice,तांदूळ,Sindhudurg,Sawantwadi,1830,2130,2430,Kharif,5100
Rice,तांदूळ,Thane,Bhiwandi,1750,2050,2350,Kharif,7800
Rice,तांदूळ,Thane,Shahapur,1730,2030,2330,Kharif,6200
Rice,तांदूळ,Palghar,Dahanu,1780,2080,2380,Kharif,5400
Rice,तांदूळ,Palghar,Vasai,1760,2060,2360,Kharif,4900
Rice,तांदूळ,Chandrapur,Chandrapur,1600,1950,2300,Kharif,9200
Rice,तांदूळ,Chandrapur,Ballarpur,1580,1930,2280,Kharif,7100
Rice,तांदूळ,Gadchiroli,Gadchiroli,1550,1880,2200,Kharif,4800
Rice,तांदूळ,Gadchiroli,Sironcha,1530,1860,2180,Kharif,3900
Rice,तांदूळ,Gondia,Gondia,1680,2000,2320,Kharif,7600
Rice,तांदूळ,Gondia,Tirora,1660,1980,2300,Kharif,5800
Rice,तांदूळ,Bhandara,Bhandara,1650,1980,2280,Kharif,6900
Rice,तांदूळ,Bhandara,Tumsar,1630,1960,2260,Kharif,5400
Rice,तांदूळ,Dhule,Dhule,1720,2040,2360,Kharif,7100
Rice,तांदूळ,Jalgaon,Jalgaon,1700,2020,2340,Kharif,8800
Rice,तांदूळ,Nandurbar,Nandurbar,1680,1990,2280,Kharif,5500
Rice,तांदूळ,Nandurbar,Shahada,1660,1970,2260,Kharif,4500
Rice,तांदूळ,Hingoli,Hingoli,1620,1940,2240,Kharif,4100
Rice,तांदूळ,Parbhani,Parbhani,1640,1960,2260,Kharif,5200
Rice,तांदूळ,Beed,Beed,1660,1980,2280,Kharif,4700
Rice,तांदूळ,Washim,Washim,1630,1950,2250,Kharif,3900
Maize,मका,Pune,Pune,1600,1880,2150,Kharif,6200
Maize,मका,Pune,Baramati,1580,1860,2130,Kharif,4800
Maize,मका,Nashik,Nashik,1580,1850,2120,Kharif,5800
Maize,मका,Nashik,Nandurbar,1560,1830,2100,Kharif,4500
Maize,मका,Dhule,Dhule,1550,1820,2080,Kharif,4900
Maize,मका,Jalna,Jalna,1570,1840,2100,Kharif,4400
Maize,मका,Aurangabad,Aurangabad,1590,1860,2130,Kharif,4100
Maize,मका,Latur,Latur,1560,1830,2100,Kharif,3800
Maize,मका,Solapur,Solapur,1540,1810,2080,Kharif,3600
Maize,मका,Nagpur,Nagpur,1600,1880,2150,Kharif,3400
Maize,मका,Amravati,Amravati,1610,1890,2160,Kharif,3200
Maize,मका,Kolhapur,Kolhapur,1650,1930,2200,Kharif,3000
Maize,मका,Satara,Satara,1630,1910,2180,Kharif,2800
Sunflower,सूर्यफूल,Latur,Latur,5200,5600,6000,Kharif,3200
Sunflower,सूर्यफूल,Osmanabad,Osmanabad,5100,5500,5900,Kharif,2800
Sunflower,सूर्यफूल,Solapur,Solapur,5250,5650,6050,Kharif,3500
Sunflower,सूर्यफूल,Ahmednagar,Ahmednagar,5300,5700,6100,Kharif,2900
Sunflower,सूर्यफूल,Nashik,Nashik,5200,5600,6000,Kharif,2500
Sunflower,सूर्यफूल,Aurangabad,Aurangabad,5180,5580,5980,Kharif,2300
Sesame,तीळ,Parbhani,Parbhani,8200,8800,9400,Kharif,2100
Sesame,तीळ,Nanded,Nanded,8100,8700,9300,Kharif,1800
Sesame,तीळ,Latur,Latur,8150,8750,9350,Kharif,1600
Sesame,तीळ,Osmanabad,Osmanabad,8050,8650,9250,Kharif,1500
Sesame,तीळ,Aurangabad,Aurangabad,8200,8800,9400,Kharif,1400
Moong,मूग,Latur,Latur,7800,8200,8600,Kharif,2600
Moong,मूग,Osmanabad,Osmanabad,7750,8150,8550,Kharif,2100
Moong,मूग,Nanded,Nanded,7780,8180,8580,Kharif,1900
Moong,मूग,Aurangabad,Aurangabad,7820,8220,8620,Kharif,1700
Moong,मूग,Solapur,Solapur,7800,8200,8600,Rabi,1500
Moong,मूग,Nashik,Nashik,7850,8250,8650,Kharif,1400
Moong,मूग,Pune,Pune,7900,8300,8700,Kharif,1300
Urad,उडीद,Osmanabad,Osmanabad,6800,7200,7600,Kharif,2200
Urad,उडीद,Latur,Latur,6820,7220,7620,Kharif,1900
Urad,उडीद,Nanded,Nanded,6800,7200,7600,Kharif,1700
Urad,उडीद,Aurangabad,Aurangabad,6850,7250,7650,Kharif,1500
Urad,उडीद,Solapur,Solapur,6780,7180,7580,Kharif,1400
Banana,केळी,Jalgaon,Jalgaon,800,1200,1800,Kharif,45000
Banana,केळी,Jalgaon,Raver,780,1180,1780,Kharif,38000
Banana,केळी,Jalgaon,Yawal,760,1160,1760,Kharif,32000
Banana,केळी,Solapur,Solapur,750,1100,1650,Kharif,28000
Banana,केळी,Pune,Pune,850,1250,1900,Kharif,22000
Banana,केळी,Nashik,Nashik,820,1220,1850,Kharif,18000
Banana,केळी,Kolhapur,Kolhapur,900,1300,1950,Kharif,15000
Banana,केळी,Satara,Satara,870,1270,1920,Kharif,12000
Banana,केळी,Sangli,Sangli,860,1260,1900,Kharif,11000
Banana,केळी,Nagpur,Nagpur,820,1220,1850,Kharif,9500
Banana,केळी,Amravati,Amravati,800,1200,1820,Kharif,8200
Turmeric,हळद,Sangli,Sangli,9000,10500,12000,Kharif,8500
Turmeric,हळद,Sangli,Tasgaon,8900,10400,11900,Kharif,6800
Turmeric,हळद,Chandrapur,Chandrapur,8800,10200,11500,Kharif,6200
Turmeric,हळद,Solapur,Solapur,9100,10600,12100,Kharif,5500
Turmeric,हळद,Nanded,Nanded,8950,10450,11950,Kharif,4800
Turmeric,हळद,Aurangabad,Aurangabad,9050,10550,12050,Kharif,4200
Turmeric,हळद,Latur,Latur,8900,10400,11900,Kharif,3900
Turmeric,हळद,Nashik,Nashik,9200,10700,12200,Kharif,3600
Chili,मिरची,Nashik,Nashik,12000,14500,17000,Kharif,4200
Chili,मिरची,Nashik,Pimpalgaon,11800,14300,16800,Kharif,3600
Chili,मिरची,Ahmednagar,Ahmednagar,11500,13800,16200,Kharif,3800
Chili,मिरची,Pune,Pune,12500,15000,17500,Kharif,3200
Chili,मिरची,Aurangabad,Aurangabad,11800,14200,16700,Kharif,2900
Chili,मिरची,Latur,Latur,11600,14000,16500,Kharif,2700
Chili,मिरची,Solapur,Solapur,11400,13800,16300,Kharif,2500
Chili,मिरची,Nagpur,Nagpur,12200,14700,17200,Kharif,2200
Potato,बटाटा,Pune,Pune,900,1250,1600,Rabi,15000
Potato,बटाटा,Pune,Baramati,880,1230,1580,Rabi,11500
Potato,बटाटा,Nashik,Nashik,880,1220,1580,Rabi,12000
Potato,बटाटा,Ahmednagar,Ahmednagar,870,1210,1570,Rabi,9800
Potato,बटाटा,Kolhapur,Kolhapur,920,1280,1640,Rabi,8500
Potato,बटाटा,Satara,Satara,900,1250,1600,Rabi,7200
Potato,बटाटा,Nagpur,Nagpur,950,1300,1650,Rabi,6500
Potato,बटाटा,Aurangabad,Aurangabad,890,1240,1590,Rabi,5800
Potato,बटाटा,Solapur,Solapur,860,1210,1560,Rabi,5500
Potato,बटाटा,Latur,Latur,850,1200,1550,Rabi,5000
Cabbage,कोबी,Pune,Pune,400,650,950,Kharif,9000
Cabbage,कोबी,Nashik,Nashik,380,620,920,Kharif,7200
Cabbage,कोबी,Satara,Satara,420,680,980,Kharif,5800
Cabbage,कोबी,Kolhapur,Kolhapur,430,690,990,Kharif,5100
Cabbage,कोबी,Nagpur,Nagpur,450,720,1020,Kharif,4500
Cauliflower,फुलकोबी,Pune,Pune,500,800,1100,Kharif,7500
Cauliflower,फुलकोबी,Nashik,Nashik,480,780,1080,Kharif,6200
Cauliflower,फुलकोबी,Satara,Satara,520,820,1120,Kharif,5000
Cauliflower,फुलकोबी,Nagpur,Nagpur,540,840,1140,Kharif,4200
Lentil,मसूर,Latur,Latur,5800,6200,6600,Rabi,3200
Lentil,मसूर,Osmanabad,Osmanabad,5750,6150,6550,Rabi,2800
Lentil,मसूर,Nanded,Nanded,5780,6180,6580,Rabi,2500
Lentil,मसूर,Solapur,Solapur,5820,6220,6620,Rabi,2900
Lentil,मसूर,Aurangabad,Aurangabad,5800,6200,6600,Rabi,2400
Mustard,मोहरी,Nashik,Nashik,5200,5500,5800,Rabi,2100
Mustard,मोहरी,Pune,Pune,5250,5550,5850,Rabi,1900
Mustard,मोहरी,Ahmednagar,Ahmednagar,5180,5480,5780,Rabi,1800
Barley,जव,Nashik,Nashik,1600,1800,2000,Rabi,2800
Barley,जव,Pune,Pune,1620,1820,2020,Rabi,2400
Barley,जव,Aurangabad,Aurangabad,1580,1780,1980,Rabi,2200
Mango,आंबा,Ratnagiri,Ratnagiri,5000,8000,12000,Annual,18500
Mango,आंबा,Sindhudurg,Kudal,4800,7800,11800,Annual,14200
Mango,आंबा,Raigad,Alibag,4600,7500,11500,Annual,12000
Mango,आंबा,Nashik,Nashik,3000,5000,8000,Annual,9500
Mango,आंबा,Pune,Pune,3200,5200,8200,Annual,8800
Mango,आंबा,Aurangabad,Aurangabad,2800,4800,7800,Annual,7200
Mango,आंबा,Nagpur,Nagpur,3000,5000,8000,Annual,6500
Cashew,काजू,Ratnagiri,Ratnagiri,8000,10000,12500,Annual,5800
Cashew,काजू,Sindhudurg,Kudal,7800,9800,12200,Annual,4900
Cashew,काजू,Raigad,Alibag,7500,9500,12000,Annual,4200
Coconut,नारळ,Ratnagiri,Ratnagiri,1800,2200,2800,Annual,8500
Coconut,नारळ,Sindhudurg,Kudal,1750,2150,2750,Annual,7200
Coconut,नारळ,Raigad,Alibag,1700,2100,2700,Annual,6500
Coconut,नारळ,Thane,Bhiwandi,1650,2050,2650,Annual,5800
Ginger,आले,Sangli,Sangli,4500,6500,9000,Kharif,3200
Ginger,आले,Satara,Satara,4300,6300,8800,Kharif,2800
Ginger,आले,Kolhapur,Kolhapur,4600,6600,9100,Kharif,2500
Garlic,लसूण,Nashik,Nashik,8000,12000,16000,Rabi,5500
Garlic,लसूण,Pune,Pune,8500,12500,16500,Rabi,4800
Garlic,लसूण,Ahmednagar,Ahmednagar,8200,12200,16200,Rabi,4200
Garlic,लसूण,Solapur,Solapur,7800,11800,15800,Rabi,3800
Garlic,लसूण,Satara,Satara,8100,12100,16100,Rabi,3500
Garlic,लसूण,Aurangabad,Aurangabad,8000,12000,16000,Rabi,3200
"""

SOIL_CSV = """Soil_Type,Soil_MR,Region,pH_Min,pH_Max,pH_Optimal,OC_pct,N_kg_ha,P_kg_ha,K_kg_ha,CEC,Fe_ppm,Zn_ppm,Texture,WHC,Drainage,Primary_Crops,Secondary_Crops,Deficiencies,Amendment
Black Cotton (Vertisol),काळी माती (वर्टिसोल),Vidarbha–Marathwada,7.5,8.5,7.8,0.30,180,12,450,45,18,0.6,Heavy Clay,Very High,Poor,Cotton–Soybean–Sorghum,Wheat–Chickpea–Linseed,Nitrogen–Zinc–Phosphorus,FYM 10t/ha + ZnSO4 25kg/ha + Urea 120kg/ha
Black Cotton (Vertisol),काळी माती (वर्टिसोल),Western Maharashtra,7.2,8.2,7.5,0.40,200,15,480,42,20,0.7,Clay,High,Moderate,Sugarcane–Cotton–Wheat,Bajra–Groundnut–Sunflower,Nitrogen–Zinc,FYM 8t/ha + vermicompost 3t/ha
Red Laterite,लाल लॅटेराइट माती,Konkan–Western Ghats,5.5,6.8,6.2,1.20,140,8,180,18,55,1.2,Sandy Loam,Low,High,Rice–Groundnut–Cashew,Mango–Coconut–Turmeric,Phosphorus–Potassium–Calcium,Lime 2t/ha + SSP 200kg/ha + MOP 60kg/ha
Alluvial,गाळाची माती,Konkan Coastal Belt,6.0,7.5,6.8,1.50,220,20,250,25,25,0.9,Silty Loam,Moderate,Moderate,Rice–Coconut–Vegetables,Sugarcane–Banana–Pulses,Nitrogen–Potassium,Urea + MOP + micronutrient mix
Sandy Red,वालुकामय लाल माती,Nashik–Ahmednagar,5.8,7.0,6.4,0.50,120,10,160,15,40,0.8,Sandy,Low,High,Grapes–Onion–Pomegranate,Millets–Pulses–Groundnut,All macronutrients,Drip + fertigation + FYM 12t/ha
Medium Black,मध्यम काळी माती,Marathwada,7.0,8.0,7.4,0.35,170,11,420,38,16,0.5,Clay Loam,High,Moderate,Soybean–Tur–Cotton,Chickpea–Jowar–Sunflower,Nitrogen–Sulfur–Zinc,PSB inoculant + ZnSO4 20kg/ha + gypsum 200kg/ha
Saline–Alkaline,क्षारयुक्त-अल्कधर्मी माती,Solapur–Osmanabad,8.2,9.5,8.8,0.20,100,8,280,30,10,0.3,Clay,High,Very Poor,Dhaincha–Barley–Saltbush,–,All nutrients severely deficient,Gypsum 5t/ha + drainage channels + S-fertiliser
Laterite Pune–Satara,लॅटेराइट पुणे-सातारा,Pune–Satara Plateau,5.5,6.5,6.0,0.80,150,9,200,20,48,1.0,Clay Loam,Moderate,Good,Sugarcane–Vegetables–Strawberry,Wheat–Grapes–Onion,Phosphorus–Zinc–Boron,Lime 1.5t/ha + SSP + borax 1kg/ha spray
Forest Loamy,जंगली चिकणमाती,Sahyadri Foothills,5.0,6.5,5.8,2.50,280,18,300,28,60,1.5,Loam,Moderate,Good,Cardamom–Coffee–Arecanut,Turmeric–Ginger–Bamboo,Potassium–Boron,Mulching + vermicompost 5t/ha
"""

MSP_CSV = """Crop,Crop_MR,MSP_2024_25,MSP_2023_24,Increase_pct,Category,Season
Common Paddy,सामान्य भात,2300,2183,5.4,Cereal,Kharif
Jowar (Hybrid),ज्वारी (हायब्रिड),3371,3180,6.0,Cereal,Kharif
Bajra,बाजरी,2625,2500,5.0,Cereal,Kharif
Maize,मका,2225,2090,6.5,Cereal,Kharif
Tur (Arhar),तूर (अरहर),7550,7000,7.9,Pulse,Kharif
Moong,मूग,8682,8558,1.4,Pulse,Kharif
Urad,उडीद,7400,6950,6.5,Pulse,Kharif
Groundnut,भुईमूग,6783,6377,6.4,Oilseed,Kharif
Sunflower Seed,सूर्यफूल,7280,6760,7.7,Oilseed,Kharif
Soybean,सोयाबीन,4892,4600,6.3,Oilseed,Kharif
Sesame,तीळ,9267,8635,7.3,Oilseed,Kharif
Cotton (Medium),कापूस (मध्यम),7121,6620,7.6,Fibre,Kharif
Cotton (Long),कापूस (लांब),7521,7020,7.1,Fibre,Kharif
Wheat,गहू,2275,2150,5.8,Cereal,Rabi
Barley,जव,1735,1635,6.1,Cereal,Rabi
Gram (Chickpea),हरभरा,5440,5440,0.0,Pulse,Rabi
Lentil (Masur),मसूर,6425,6000,7.1,Pulse,Rabi
Rapeseed–Mustard,मोहरी,5650,5450,3.7,Oilseed,Rabi
Safflower,करडई,5800,5800,0.0,Oilseed,Rabi
Sugarcane (FRP),ऊस (FRP),340,315,7.9,Sugarcane,Annual
"""


def _strip_html(html: str) -> str:
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_rag_corpus() -> str:
    """Daily-refresh cache: official Maharashtra / Agmarknet / state portal text for RAG grounding."""
    chunks = []
    headers = {"User-Agent": "KisanSakha/1.0 (educational; contact: local deployment)"}
    for url in _RAG_URLS:
        try:
            r = requests.get(url, timeout=18, headers=headers)
            r.raise_for_status()
            plain = _strip_html(r.text)[:12000]
            if plain:
                chunks.append(f"[SOURCE: {url}]\n{plain}")
        except Exception:
            continue
    return "\n\n".join(chunks) if chunks else ""


def retrieve_rag_snippets(query: str, corpus: str, max_chars: int = 3800) -> str:
    if not corpus or not query.strip():
        return ""
    q_terms = set(re.findall(r"[a-zA-Z\u0900-\u097F]{3,}", query.lower()))
    blocks = re.split(r"\[SOURCE:", corpus)
    scored = []
    for b in blocks:
        if not b.strip():
            continue
        low = b.lower()
        score = sum(1 for w in q_terms if w in low)
        scored.append((score, b[:6000]))
    scored.sort(key=lambda x: -x[0])
    out, n = [], 0
    for _, block in scored[:6]:
        if n + len(block) > max_chars:
            break
        out.append(("[SOURCE:" + block) if not block.lstrip().startswith("[") else block)
        n += len(block)
    return "\n\n".join(out)[:max_chars]


def _norm_price_row(rec: dict):
    """Map varied data.gov.in / Agmarknet-style keys to our schema."""
    keymap = {k.lower().replace(" ", "_"): v for k, v in rec.items()}
    def g(*names):
        for n in names:
            for k, v in keymap.items():
                if n in k and v not in (None, "", "NA"):
                    return v
        return None

    state = str(g("state") or "").lower()
    if state and "maharashtra" not in state:
        return None

    commodity = g("commodity", "crop", "variety")
    if not commodity:
        return None
    district = str(g("district") or "Unknown")
    market = str(g("market", "mandi", "apmc") or "APMC")
    try:
        mn = float(g("min_price", "min") or 0)
        mx = float(g("max_price", "max") or 0)
        md = float(g("modal_price", "modal", "avg") or (mn + mx) / 2 or 0)
    except (TypeError, ValueError):
        return None
    if md <= 0:
        return None
    crop_en = str(commodity).split("(")[0].strip()[:80]
    crop_mr = crop_en
    return {
        "Crop": crop_en,
        "Crop_MR": crop_mr,
        "District": district[:60],
        "Market": market[:80],
        "Min_Price": int(mn) if mn else int(md * 0.9),
        "Modal_Price": int(md),
        "Max_Price": int(mx) if mx else int(md * 1.1),
        "Season": "Kharif",
        "Arrival_MT": int(float(g("arrival", "quantity") or 100)),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_live_mandi_maharashtra() -> pd.DataFrame:
    if not _ENV_DATA_GOV_KEY:
        return pd.DataFrame()
    # National Data Sharing Platform — Current Daily Prices (commodities / mandi); resource ID from data.gov.in catalog.
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    rows = []
    try:
        offset = 0
        for _ in range(8):
            r = requests.get(
                url,
                params={
                    "api-key": _ENV_DATA_GOV_KEY,
                    "format": "json",
                    "limit": 500,
                    "offset": offset,
                    "filters[state]": "Maharashtra",
                },
                timeout=25,
                headers={"User-Agent": "KisanSakha/1.0"},
            )
            if r.status_code != 200:
                break
            data = r.json()
            recs = data.get("records") or []
            if not recs:
                break
            for rec in recs:
                m = _norm_price_row(rec if isinstance(rec, dict) else {})
                if m:
                    rows.append(m)
            offset += len(recs)
            if len(recs) < 500:
                break
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=86400)
def load_data():
    base = pd.read_csv(io.StringIO(PRICE_CSV))
    live = fetch_live_mandi_maharashtra()
    if live is not None and not live.empty:
        price = pd.concat([base, live], ignore_index=True)
        price = price.drop_duplicates(
            subset=["Crop", "District", "Market", "Season"], keep="last"
        )
    else:
        price = base
    return (
        price,
        pd.read_csv(io.StringIO(SOIL_CSV)),
        pd.read_csv(io.StringIO(MSP_CSV)),
    )


price_df, soil_df, msp_df = load_data()

# Approximate district centroids (Maharashtra) for Google Weather API lookups
MH_DISTRICT_LATLON = {
    "Ahmednagar": (19.0948, 74.7480),
    "Akola": (20.7002, 77.0082),
    "Amravati": (20.9374, 77.7796),
    "Aurangabad": (19.8762, 75.3433),
    "Beed": (18.9894, 75.7543),
    "Bhandara": (21.1667, 79.6501),
    "Buldhana": (20.5321, 76.1846),
    "Chandrapur": (19.9615, 79.2961),
    "Dhule": (20.9042, 74.7749),
    "Gadchiroli": (20.1687, 79.9827),
    "Gondia": (21.4529, 80.1920),
    "Hingoli": (19.7202, 77.1465),
    "Jalgaon": (21.0077, 75.5626),
    "Jalna": (19.8410, 75.8864),
    "Kolhapur": (16.7050, 74.2433),
    "Latur": (18.4088, 76.5604),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Nanded": (19.1383, 77.3210),
    "Nandurbar": (21.3667, 74.2399),
    "Nashik": (19.9975, 73.7898),
    "Osmanabad": (18.1810, 76.0389),
    "Palghar": (19.6967, 72.7654),
    "Parbhani": (19.2611, 76.7767),
    "Pune": (18.5204, 73.8567),
    "Raigad": (18.5158, 73.1352),
    "Ratnagiri": (16.9902, 73.3120),
    "Sangli": (16.8524, 74.5815),
    "Satara": (17.6805, 74.0183),
    "Sindhudurg": (16.1683, 73.5806),
    "Solapur": (17.6599, 75.9064),
    "Thane": (19.2183, 72.9781),
    "Wardha": (20.7453, 78.6022),
    "Washim": (20.1110, 77.1313),
    "Yavatmal": (20.3888, 78.1204),
}


def _latlon_for_district(name: str):
    return MH_DISTRICT_LATLON.get(name, (19.7515, 75.7139))


@st.cache_data(ttl=1800, show_spinner=False)
def _gemini_weather_rag(district: str, bump: int) -> dict:
    """
    Fetch weather for a Maharashtra district using Gemini + live RAG web search.
    Returns a structured dict with current conditions and a 5-day outlook.
    Uses only the GOOGLE_API_KEY — no separate Weather API key required.
    """
    if not _ENV_GOOGLE_KEY:
        return {"error": "no_key"}
    try:
        _model = genai.GenerativeModel(_GEMINI_MODEL)
    except Exception:
        return {"error": "model_init_failed"}

    prompt = (
        f"You are a weather assistant. Provide current weather conditions and a 5-day forecast "
        f"for {district} district, Maharashtra, India. "
        f"Search online for the latest weather data for {district}, Maharashtra right now. "
        f"Respond ONLY with a JSON object (no markdown, no backticks) with exactly this structure:\n"
        '{"temperature_c": <number>, "feels_like_c": <number>, "condition": "<short description>", '
        '"humidity_pct": <number>, "wind_kmh": <number>, "rain_chance_pct": <number>, '
        '"uv_index": <number>, '
        '"forecast": ['
        '{"day": "<Mon/Tue/etc>", "date": "<DD/MM>", "max_c": <number>, "min_c": <number>, '
        '"condition": "<short description>", "rain_chance_pct": <number>}, '
        '... 5 days total'
        '], '
        '"farming_advisory": "<1-2 sentence farming tip for current weather>"}'
        f"\nUse real current data for {district}, Maharashtra. All numbers must be realistic for Maharashtra climate."
    )
    try:
        resp = _model.generate_content(
            prompt,
            request_options={"timeout": 30},
        )
        raw = (resp.text or "").strip()
        # strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        import json
        data = json.loads(raw)
        return {"data": data, "error": None}
    except Exception as ex:
        return {"error": str(ex)}


# ══════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════
GREEN_PALETTE = ["#1a4731","#2d6a4f","#40916c","#52b788","#74c69d","#95d5b2","#b7e4c7","#d8f3dc"]
AMBER_PALETTE = ["#7f3f00","#b5570a","#d4722a","#f4a261","#f7b97a","#fad09e","#fce8cc"]

def plotly_bar(df, x, y, color=None, title="", labels={}, color_seq=None):
    fig = px.bar(
        df, x=x, y=y, color=color, title=title, labels=labels,
        color_discrete_sequence=color_seq or GREEN_PALETTE,
        text_auto=True,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif", size=12, color="#0d2b1a"),
        title_font=dict(size=15, color="#1a4731"),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(255,255,255,.7)", borderwidth=0),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor="rgba(210,240,220,.5)", zeroline=False),
    )
    fig.update_traces(marker_line_width=0, textfont_size=11)
    return fig

def plotly_grouped_bar(df, x, y_cols, labels, title=""):
    fig = go.Figure()
    colors = ["#2d6a4f","#f4a261","#52b788"]
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Bar(
            name=labels.get(col, col), x=df[x], y=df[col],
            marker_color=colors[i % len(colors)],
            text=df[col].apply(lambda v: f"₹{int(v):,}"),
            textposition="outside", textfont_size=10,
        ))
    fig.update_layout(
        barmode="group", title=title,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif", size=12, color="#0d2b1a"),
        title_font=dict(size=15, color="#1a4731"),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(255,255,255,.7)"),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor="rgba(210,240,220,.5)", title="₹/quintal"),
    )
    return fig

def plotly_scatter(df, x, y, size, color, title=""):
    fig = px.scatter(
        df, x=x, y=y, size=size, color=color,
        color_discrete_sequence=GREEN_PALETTE,
        title=title, size_max=50,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif", size=12, color="#0d2b1a"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig

# ══════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Tiro+Devanagari+Marathi:ital@0;1&display=swap');

:root{
  --g0:#0d2b1a;--g1:#1a4731;--g2:#2d6a4f;--g3:#52b788;--g4:#95d5b2;--g5:#d8f3dc;--g6:#f0faf3;
  --amber:#f4a261;--amber-d:#b5570a;--amber-pale:#fff8ee;
  --sky:#48cae4;--sky-pale:#e0f7fa;
  --text:#0d1f14;--muted:#3d5a45;--surface:#fff;--bg:#f2f7f4;
  --r:14px;--sh:0 4px 28px rgba(13,43,26,.10);
}
html,body,[class*="css"]{font-family:'Sora',sans-serif!important;color:var(--text);}
.stApp{background:var(--bg)!important;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--g0) 0%,#0a1f10 100%)!important;}
section[data-testid="stSidebar"] *{color:#c8e6d0!important;font-family:'Sora',sans-serif!important;}
section[data-testid="stSidebar"] input{background:rgba(255,255,255,.09)!important;border:1px solid rgba(255,255,255,.18)!important;border-radius:8px!important;color:#fff!important;}
section[data-testid="stSidebar"] .stRadio label{color:#b7e4c7!important;}
section[data-testid="stSidebar"] h2{font-size:1.45rem!important;color:#ffffff!important;letter-spacing:-.4px;}
/* Sidebar collapse: hide Material icon / "keyboard..." label; show → */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
  position:relative!important;min-height:2rem!important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] p,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
  font-size:0!important;line-height:0!important;color:transparent!important;width:0!important;overflow:hidden!important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]::after {
  content:"→"!important;font-size:1.25rem!important;color:#b7e4c7!important;display:block!important;
  position:absolute!important;left:50%!important;top:50%!important;transform:translate(-50%,-50%)!important;
  pointer-events:none!important;
}
/* Collapsed sidebar expand control (main strip) */
[data-testid="collapsedControl"]{position:relative!important;min-width:2rem!important;}
[data-testid="collapsedControl"] span[data-testid="stIconMaterial"],
[data-testid="collapsedControl"] p{font-size:0!important;color:transparent!important;}
[data-testid="collapsedControl"]::after{
  content:"→"!important;font-size:1.2rem!important;color:var(--g2,#2d6a4f)!important;
  position:absolute!important;left:50%!important;top:50%!important;transform:translate(-50%,-50%)!important;
  pointer-events:none!important;
}

/* ── Animations ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
@keyframes popIn{0%{opacity:0;transform:scale(.93)}60%{transform:scale(1.025)}100%{opacity:1;transform:scale(1)}}
@keyframes shimmer{0%{background-position:-600px 0}100%{background-position:600px 0}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
.fu{animation:fadeUp .55s cubic-bezier(.22,1,.36,1) both;}
.fu1{animation:fadeUp .55s .07s cubic-bezier(.22,1,.36,1) both;}
.fu2{animation:fadeUp .55s .14s cubic-bezier(.22,1,.36,1) both;}
.fu3{animation:fadeUp .55s .21s cubic-bezier(.22,1,.36,1) both;}
.pop{animation:popIn .4s cubic-bezier(.22,1,.36,1) both;}

/* ── Hero ── */
.hero{
  background:linear-gradient(135deg,var(--g0) 0%,var(--g2) 60%,#40916c 100%);
  border-radius:22px;padding:2.8rem 2.4rem 2.4rem;position:relative;overflow:hidden;
  margin-bottom:1.6rem;border:1px solid rgba(82,183,136,.2);
}
.hero::before{content:'';position:absolute;inset:0;
  background:radial-gradient(circle at 80% 50%,rgba(82,183,136,.18) 0%,transparent 60%);pointer-events:none;}
.hero::after{content:'🌾';position:absolute;right:2.4rem;top:50%;transform:translateY(-50%);
  font-size:6rem;opacity:.13;filter:drop-shadow(0 0 24px rgba(82,183,136,.4));}
.hero h1{font-size:2.4rem!important;color:#fff!important;margin:0 0 .3rem!important;font-weight:700!important;}
.hero .hero-sub{color:var(--g4)!important;font-size:1.05rem!important;margin:0!important;}
.hero .hero-badge{display:inline-block;background:rgba(82,183,136,.22);border:1px solid rgba(82,183,136,.4);
  color:var(--g4);border-radius:999px;padding:.2rem .85rem;font-size:.78rem;font-weight:600;margin-top:.7rem;}

/* ── Stats row ── */
.stat-card{background:var(--surface);border-radius:12px;padding:1rem 1.2rem;
  border:1px solid var(--g5);box-shadow:var(--sh);text-align:center;}
.stat-lbl{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.2rem;}
.stat-val{font-size:1.55rem;font-weight:700;color:var(--g1);}

/* ── Nav cards ── */
.nav-card{background:var(--surface);border-radius:20px;padding:2.2rem 1.8rem 1.8rem;
  border:1.5px solid var(--g5);box-shadow:var(--sh);text-align:center;height:100%;position:relative;
  transition:transform .3s cubic-bezier(.34,1.56,.64,1),box-shadow .3s ease,border-color .3s ease;}
.nav-card:hover{transform:translateY(-8px) scale(1.02);box-shadow:0 16px 44px rgba(13,43,26,.19);border-color:var(--g3);}
.nav-card::before{content:'';position:absolute;inset:0;border-radius:20px;
  background:linear-gradient(135deg,rgba(82,183,136,.06) 0%,transparent 60%);pointer-events:none;}
.nav-icon{font-size:3rem;margin-bottom:.7rem;display:block;}
.nav-en{font-weight:700;font-size:1.15rem;color:var(--g1);margin-bottom:.15rem;}
.nav-mr{font-family:'Tiro Devanagari Marathi',serif;font-size:.95rem;color:var(--g3);margin-bottom:.6rem;font-style:italic;}
.nav-desc{color:var(--muted);font-size:.83rem;line-height:1.45;}
.nav-card.nav-grow{border-top:4px solid #e85d04!important;border-color:rgba(232,93,4,.28)!important;}
.nav-card.nav-grow .nav-en{color:#c2410c!important;}
.nav-card.nav-grow .nav-mr{color:#ea580c!important;}
.nav-card.nav-maintain{border-top:4px solid #0077b6!important;border-color:rgba(0,119,182,.28)!important;}
.nav-card.nav-maintain .nav-en{color:#0077b6!important;}
.nav-card.nav-maintain .nav-mr{color:#0284c7!important;}
.nav-card.nav-sell{border-top:4px solid #15803d!important;border-color:rgba(21,128,61,.28)!important;}
.nav-card.nav-sell .nav-en{color:#15803d!important;}
.nav-card.nav-sell .nav-mr{color:#16a34a!important;}

/* ── Section header ── */
.sec-hd{font-size:1.9rem!important;color:var(--g0)!important;font-weight:700!important;margin-bottom:.2rem!important;}
.sec-sub{color:var(--muted)!important;font-size:.9rem!important;margin-bottom:1.1rem!important;}

/* ── Chips ── */
.chip-wrap{display:flex;flex-wrap:wrap;gap:.45rem;margin:.6rem 0 1rem;}
.chip-lbl{font-size:.73rem;color:var(--muted);margin-bottom:.3rem;font-weight:500;text-transform:uppercase;letter-spacing:.05em;}

/* ── Data/ICAR card ── */
.icar-card{background:var(--surface);border-radius:14px;padding:1.1rem 1.3rem;
  border-left:4px solid var(--g3);box-shadow:var(--sh);margin-bottom:.8rem;}
.icar-title{font-weight:700;color:var(--g1);font-size:.92rem;margin-bottom:.5rem;}
.badge{display:inline-block;background:var(--g5);color:var(--g1);border-radius:999px;
  padding:.18rem .7rem;font-size:.76rem;font-weight:600;margin:.12rem;}
.badge-a{background:var(--amber-pale);color:var(--amber-d);}
.badge-s{background:var(--sky-pale);color:#0a6e8a;}
.badge-r{background:#fce8e8;color:#8b0000;}

/* ── Response ── */
.resp-box{background:var(--surface);border-left:4px solid var(--g3);border-radius:14px;
  padding:1.3rem 1.5rem;box-shadow:var(--sh);font-size:.96rem;line-height:1.8;
  color:var(--text);animation:popIn .4s cubic-bezier(.22,1,.36,1) both;margin-top:.6rem;}
.src-tag{font-size:.72rem;color:var(--muted);margin-top:.5rem;}
.src-tag a{color:var(--g2);}

/* ── Metric cards ── */
.met-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1rem;}
.met-card{background:var(--surface);border-radius:12px;padding:.9rem 1rem;
  border:1px solid var(--g5);box-shadow:var(--sh);}
.met-lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.15rem;}
.met-val{font-size:1.3rem;font-weight:700;color:var(--g0);}
.met-sub{font-size:.72rem;color:var(--g2);margin-top:.1rem;}

/* ── Buttons ── */
.stButton>button{background:var(--g2)!important;color:#fff!important;border:none!important;
  border-radius:10px!important;font-weight:600!important;font-family:'Sora',sans-serif!important;
  padding:.55rem 1.3rem!important;letter-spacing:.01em!important;
  transition:background .2s,transform .2s cubic-bezier(.34,1.56,.64,1),box-shadow .2s!important;}
.stButton>button:hover{background:var(--g1)!important;transform:translateY(-2px)!important;
  box-shadow:0 7px 22px rgba(13,43,26,.28)!important;}
.stButton>button:active{transform:scale(.97)!important;}
.back-btn>button{background:transparent!important;color:var(--g2)!important;
  border:1.5px solid var(--g4)!important;font-weight:500!important;}
.back-btn>button:hover{background:var(--g5)!important;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"]{font-family:'Sora',sans-serif!important;font-weight:500!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;}

/* ── Dataframe ── */
.dataframe{font-size:.83rem!important;}
hr{border-color:var(--g5)!important;}
.stChatMessage{border-radius:12px!important;}

/* ── Price range bar ── */
.price-range-wrap{background:var(--g6);border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;}
.price-range-bar{height:8px;background:linear-gradient(90deg,var(--g4),var(--g2));border-radius:999px;margin:.3rem 0;}

/* ── Weather strip (Google Weather) ── */
.weather-strip-collapsed{
  background:linear-gradient(135deg,#0d3d2a 0%,#1a5c40 45%,#2d6a4f 100%);
  border-radius:14px;padding:.85rem 1.15rem;border:1px solid rgba(82,183,136,.38);
  box-shadow:0 4px 22px rgba(13,43,26,.14);margin-bottom:.35rem;
  transition:box-shadow .28s ease, transform .22s ease;
}
.weather-strip-collapsed:hover{box-shadow:0 10px 32px rgba(13,43,26,.2);transform:translateY(-1px);}
.weather-strip-title{color:#fff!important;font-weight:600;font-size:1rem;margin:0!important;line-height:1.25;}
.weather-strip-hint{color:rgba(216,243,220,.88)!important;font-size:.74rem;margin:.2rem 0 0!important;line-height:1.35;}
.weather-panel-expanded{
  background:linear-gradient(180deg,#ffffff 0%,#f2faf5 100%);
  border:1px solid rgba(45,106,79,.22);border-radius:16px;padding:1rem 1.2rem 1.25rem;
  box-shadow:0 10px 36px rgba(13,43,26,.1);
  animation:fadeUp .45s cubic-bezier(.22,1,.36,1) both;
  margin-bottom:.75rem;
  transition:opacity .35s ease;
}
.weather-panel-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.75rem;margin-bottom:.65rem;}
.weather-now-main{display:flex;flex-wrap:wrap;align-items:center;gap:1rem;margin:.5rem 0 1rem;}
.weather-now-temp{font-size:2.1rem;font-weight:700;color:var(--g0);line-height:1;}
.weather-now-meta{color:var(--muted);font-size:.88rem;max-width:28rem;line-height:1.45;}
.weather-daily-row{display:flex;gap:.55rem;overflow-x:auto;padding:.4rem 0 .2rem;scrollbar-width:thin;}
.weather-day-card{
  flex:0 0 auto;min-width:104px;background:#fff;border:1px solid var(--g5);border-radius:12px;
  padding:.55rem .5rem;text-align:center;box-shadow:0 2px 12px rgba(13,43,26,.06);
}
.weather-day-card .d1{font-weight:700;color:var(--g1);font-size:.78rem;}
.weather-day-card .d2{font-size:.72rem;color:var(--muted);margin-top:.2rem;}
.weather-day-card .d3{font-size:.85rem;color:var(--g0);margin-top:.35rem;font-weight:600;}

/* ── Tab panels (market & other pages) — contrast on app background ── */
.stTabs [data-baseweb="tab-panel"]{
  color:#0d1f14!important;
  background:linear-gradient(180deg,rgba(255,255,255,.95) 0%,rgba(242,247,244,.97) 100%)!important;
  border-radius:0 14px 14px 14px!important;
  border:1px solid rgba(45,106,79,.14)!important;
  border-top:none!important;
  padding:1rem 1.1rem 1.35rem!important;
  margin-top:0!important;
}
.stTabs [data-baseweb="tab-panel"] p, .stTabs [data-baseweb="tab-panel"] li,
.stTabs [data-baseweb="tab-panel"] .stMarkdown{color:#0d1f14!important;}
.sell-page-intro{
  background:#fff!important;border:1px solid rgba(45,106,79,.16)!important;border-radius:12px!important;
  padding:.7rem 1rem!important;color:#0d1f14!important;font-size:.88rem!important;line-height:1.45!important;
  margin:0 0 .85rem!important;box-shadow:0 2px 14px rgba(13,43,26,.06)!important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 Kisan Sakha · किसान सखा")
    st.caption("MMCOE · AI Farming Companion")
    st.caption(t("api_deploy_note"))
    st.divider()

    lang_choice = st.radio(t("lang_label"), ["English", "मराठी"], index=0 if st.session_state.lang == "English" else 1)
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    st.divider()
    st.markdown(f"**{t('data_sources')}**")
    st.caption("📊 [Agmarknet / data.gov.in](https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi)")
    st.caption("🌱 [ICAR Soil Survey](https://icar.org.in)")
    st.caption("💰 [CACP MSP 2024-25](https://cacp.dacnet.nic.in)")
    st.caption("🏛️ [Maharashtra Krishi Dept](https://krishi.maharashtra.gov.in)")
    st.caption(t("data_gov_api_hint"))
    st.divider()

IS_MR = st.session_state.lang == "मराठी"

# API key from environment only (deployment-ready)
_effective_api_key = _ENV_GOOGLE_KEY
if _effective_api_key:
    try:
        genai.configure(api_key=_effective_api_key)
        st.session_state.model = genai.GenerativeModel(_GEMINI_MODEL)
    except Exception as e:
        st.session_state.model = None
        st.sidebar.error(str(e))
else:
    st.session_state.model = None

if not _effective_api_key:
    st.warning(t("no_api_key"))

# ══════════════════════════════════════════════════
# GOOGLE AI (Generative Language API)
# ══════════════════════════════════════════════════
def expand_chip_question(question: str, domain: str) -> str:
    """Turn a short chip label into an instruction for a long, structured answer."""
    guides = {
        "grow": (
            "Domain: crop planning & soil for Maharashtra. "
            "Include soil pH interpretation, suitable crops/varieties, sowing windows, seed rate, NPK/micronutrients, "
            "organic options, water management, and relevant schemes (PM-KISAN, Soil Health Card, RKVY)."
        ),
        "maintain": (
            "Domain: crop health & maintenance in Maharashtra. "
            "Include likely diagnosis, IPM (cultural/bio/chemical with doses and PHI), irrigation/fertigation tweaks, "
            "prevention next season, and where to confirm (KVK, university advisory)."
        ),
        "sell": (
            "Domain: marketing & mandi economics in Maharashtra. "
            "Reference MSP vs mandi where relevant, APMC/e-NAM, timing, storage, grading, negotiation with adatiyas, "
            "schemes (PM-AASHA, etc.), and practical next steps."
        ),
    }
    g = guides.get(domain, guides["grow"])
    return (
        f"{g}\n\n"
        f"The farmer selected this suggested question from the app:\n« {question} »\n\n"
        "Answer in depth for a smallholder in Maharashtra. Structure your reply with clear headings and bullet points. "
        "Give concrete numbers, timings, and examples where possible. End with a short checklist and a line on "
        "verifying with the local agriculture office or KVK."
    )


def ask_gemini(prompt, context="", data_context="", use_rag=True, extra_knowledge=""):
    if not st.session_state.model:
        if IS_MR:
            return "⚠️ GOOGLE_API_KEY वातावरणात सेट करा (Google AI Studio)."
        return "⚠️ Set GOOGLE_API_KEY (from Google AI Studio)."
    lang = st.session_state.lang
    rag_block = ""
    if use_rag:
        try:
            corpus = fetch_rag_corpus()
            rag_block = retrieve_rag_snippets(f"{prompt} {context} {data_context}", corpus)
        except Exception:
            rag_block = ""
    rag_section = (
        f"\n\nOfficial website excerpts (RAG — prefer facts consistent with these; cite source URL if used):\n{rag_block}"
        if rag_block
        else ""
    )
    data_k = f"\n\nStructured / tabular context:\n{data_context}" if data_context else ""
    ref_k = f"\n\nCrop & variety reference (verify with local KVK):\n{extra_knowledge}" if extra_knowledge else ""
    system = (
        f"You are a senior agricultural scientist and extension officer with 25+ years of experience in Maharashtra, India. "
        f"You have deep expertise in ICAR soil science, Agmarknet mandi price dynamics, CACP MSP policies, "
        f"Krishi Vigyan Kendra (KVK) practices, Maharashtra government agricultural schemes (PM-KISAN, RKVY, Namo Shetkari), "
        f"and traditional Marathi farming methods across Vidarbha, Marathwada, Konkan, and Western Maharashtra. "
        f"{context} "
        f"{data_k}{ref_k}{rag_section} "
        f"CRITICAL: Respond ENTIRELY in {lang}. "
        f"{'Use fluent, correct Marathi with proper grammar and Devanagari script. Minimize unnecessary English.' if IS_MR else ''} "
        f"Be specific, data-driven, and farmer-friendly. Use bullet points, concrete numbers, and local context. "
        f"Mention relevant government schemes, APMC / e-NAM where relevant, and seasonal timing for Maharashtra. "
        f"If portal excerpts conflict with structured data, note the discrepancy and prefer official mandi/department figures when available."
    )
    full_prompt = f"{system}\n\nQuestion: {prompt}"

    def _call_model(model, text):
        try:
            return model.generate_content(text, request_options={"timeout": 120})
        except TypeError:
            return model.generate_content(text)

    try:
        response = _call_model(st.session_state.model, full_prompt)
        try:
            out = (response.text or "").strip()
        except ValueError:
            fb = getattr(response, "prompt_feedback", None)
            out = (
                (
                    "मॉडेलने मजकूर दिला नाही (सुरक्षा/फिल्टर). प्रश्न सोपा करून पुन्हा प्रयत्न करा."
                    if IS_MR
                    else "No text returned (safety filter or empty candidates). Try rephrasing."
                )
                if not fb
                else str(fb)
            )
        return out or ("कोणताही प्रतिसाद मिळाला नाही." if IS_MR else "No response generated.")
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower() or "is not found" in err.lower():
            try:
                genai.configure(api_key=_effective_api_key)
                st.session_state.model = genai.GenerativeModel("gemini-2.0-flash")
                response = _call_model(st.session_state.model, full_prompt)
                try:
                    out = (response.text or "").strip()
                except ValueError:
                    out = ""
                return out or ("कोणताही प्रतिसाद मिळाला नाही." if IS_MR else "No response generated.")
            except Exception as e2:
                return f"❌ Error: {e2}"
        return f"❌ Error: {err}"

# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════
def go(page):
    st.session_state.page = page
    st.rerun()

def back_btn():
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button(t("back")):
        go("home")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

def chip_row(chips_list, prefix):
    """Suggestion buttons; selection is stored and must be consumed with finish_chip_qa (or chat-tab pop)."""
    st.markdown(f'<div class="chip-lbl">{t("quick_q")}</div>', unsafe_allow_html=True)
    if not chips_list:
        return
    cols = st.columns(len(chips_list))
    for i, chip in enumerate(chips_list):
        with cols[i]:
            if st.button(chip, key=f"{prefix}_{i}", use_container_width=True):
                st.session_state[f"{prefix}_pending"] = chip


def finish_chip_qa(prefix, domain, context, data_context="", extra_knowledge=""):
    """After widgets define context: run Gemini on pending chip and persist answer until cleared."""
    pend = st.session_state.pop(f"{prefix}_pending", None)
    res_key = f"{prefix}_result"
    if pend:
        expanded = expand_chip_question(pend, domain)
        with st.spinner(t("chip_working")):
            try:
                ans = ask_gemini(
                    expanded,
                    context=context,
                    data_context=data_context,
                    extra_knowledge=extra_knowledge,
                )
            except Exception as ex:
                ans = f"❌ {ex}"
        st.session_state[res_key] = {"q": pend, "a": ans}
    block = st.session_state.get(res_key)
    if block:
        ql = "प्रश्न" if IS_MR else "Question"
        st.markdown(f"**{ql}:** {block['q']}")
        st.markdown(block["a"])
        if st.button(t("clear_chip_ans"), key=f"{prefix}_clr_res"):
            del st.session_state[res_key]
            st.rerun()

def get_chips(section):
    return CHIPS[section][st.session_state.lang]

def tab_ai_ask(exp_key, ai_context, data_context="", extra_knowledge=""):
    """Lightweight AI bar inside a tab (expander + text input) — same model + RAG as main chat."""
    with st.expander(f"💬 {t('ask_ai_exp')}", expanded=False):
        q = st.text_input(t("your_question"), key=f"{exp_key}_q", label_visibility="visible")
        if st.button(t("send"), key=f"{exp_key}_go") and q.strip():
            with st.spinner(t("chip_working")):
                try:
                    r = ask_gemini(
                        q.strip(),
                        context=ai_context,
                        data_context=data_context,
                        extra_knowledge=extra_knowledge,
                    )
                except Exception as ex:
                    r = f"❌ {ex}"
            st.session_state[f"{exp_key}_ans"] = r
        ans = st.session_state.get(f"{exp_key}_ans")
        if ans:
            st.markdown(f'<div class="resp-box">{ans}</div>', unsafe_allow_html=True)

def crop_label(crop_en):
    row = price_df[price_df["Crop"] == crop_en]
    if not row.empty and IS_MR:
        return row.iloc[0]["Crop_MR"]
    return crop_en

PLOTLY_INTERACTIVE = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _sell_chart_layout(fig):
    """Solid surfaces so charts stay readable on Streamlit light/dark themes."""
    fig.update_layout(
        paper_bgcolor="#f4faf7",
        plot_bgcolor="#e8f2ec",
        font=dict(family="Sora, sans-serif", size=12, color="#0d2b1a"),
        title_font=dict(size=15, color="#1a4731"),
        hovermode="x unified",
    )
    return fig


def render_weather_panel():
    """Top bar: expand for AI-powered weather (Gemini RAG — uses GOOGLE_API_KEY only, no separate weather key)."""
    if "weather_bump" not in st.session_state:
        st.session_state.weather_bump = 0

    if not st.session_state.weather_open:
        st.markdown(
            '<div class="weather-strip-collapsed">'
            f'<p class="weather-strip-title">{t("weather_head")}</p>'
            f'<p class="weather-strip-hint">{"AI-powered · Google Gemini RAG · No extra API key needed" if not IS_MR else "AI-आधारित हवामान · Gemini RAG · वेगळी API की लागत नाही"}</p></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            t("weather_toggle_on"),
            key="wx_open",
            use_container_width=True,
        ):
            st.session_state.weather_open = True
            st.rerun()
        return

    st.markdown('<div class="weather-panel-expanded">', unsafe_allow_html=True)
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            f'<p class="weather-strip-title" style="color:#0d2b1a!important;">{t("weather_head")}</p>'
            f'<p class="weather-strip-hint" style="color:#3d5a45!important;">{"Powered by Google Gemini AI · Live search · No extra API key" if not IS_MR else "Google Gemini AI · थेट माहिती · वेगळी की नाही"}</p>',
            unsafe_allow_html=True,
        )
    with h2:
        if st.button(t("weather_collapse"), key="wx_close", use_container_width=True):
            st.session_state.weather_open = False
            st.rerun()

    dist_list = sorted(price_df["District"].dropna().unique().tolist())
    if st.session_state.weather_district not in dist_list:
        st.session_state.weather_district = dist_list[0] if dist_list else "Pune"

    cwa, cwb, cwc = st.columns([2, 2, 1])
    with cwa:
        sel = st.selectbox(
            t("weather_district"),
            dist_list,
            index=dist_list.index(st.session_state.weather_district) if st.session_state.weather_district in dist_list else 0,
            key="wx_district_sel",
        )
        st.session_state.weather_district = sel
    with cwb:
        st.caption("Source: Google Gemini AI (live RAG)" if not IS_MR else "स्रोत: Google Gemini AI (RAG)")
    with cwc:
        if st.button(t("weather_refresh"), key="wx_refresh"):
            st.session_state.weather_bump += 1
            st.rerun()

    if not _ENV_GOOGLE_KEY:
        st.info(
            "Set **GOOGLE_API_KEY** to enable AI-powered weather." if not IS_MR
            else "AI हवामानासाठी **GOOGLE_API_KEY** सेट करा."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    spin_msg = "AI हवामान माहिती मिळवत आहे…" if IS_MR else "Fetching AI weather data…"
    with st.spinner(spin_msg):
        result = _gemini_weather_rag(sel, int(st.session_state.weather_bump))

    if result.get("error"):
        st.warning(
            f'{"हवामान मिळाले नाही" if IS_MR else "Could not load weather"}: {result["error"]}'
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    wx = result.get("data", {})
    if not wx:
        st.warning(t("weather_err"))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Current conditions ──
    temp = wx.get("temperature_c")
    feels = wx.get("feels_like_c")
    cond_txt = wx.get("condition", "—")
    hum = wx.get("humidity_pct")
    wind = wx.get("wind_kmh")
    uv = wx.get("uv_index")
    precip = wx.get("rain_chance_pct")
    advisory = wx.get("farming_advisory", "")

    st.markdown('<div class="weather-now-main">', unsafe_allow_html=True)
    col_ic, col_tx = st.columns([1, 5])
    with col_ic:
        # Pick emoji icon based on condition text
        cond_lower = cond_txt.lower()
        if any(w in cond_lower for w in ["rain", "shower", "drizzle", "thunder", "storm"]):
            wx_icon = "🌧️"
        elif any(w in cond_lower for w in ["cloud", "overcast", "fog", "mist", "haze"]):
            wx_icon = "⛅"
        elif any(w in cond_lower for w in ["clear", "sunny", "fair"]):
            wx_icon = "☀️"
        elif any(w in cond_lower for w in ["snow", "sleet", "hail"]):
            wx_icon = "🌨️"
        elif any(w in cond_lower for w in ["wind", "breezy", "gust"]):
            wx_icon = "💨"
        else:
            wx_icon = "🌤️"
        st.markdown(f'<div style="font-size:3.5rem;text-align:center;">{wx_icon}</div>', unsafe_allow_html=True)
    with col_tx:
        unit = "°C"
        tline = f"{temp:.1f}{unit}" if temp is not None else "—"
        if feels is not None:
            tline += f" · {'अनुभव' if IS_MR else 'feels'} {feels:.1f}{unit}"
        st.markdown(f'<div class="weather-now-temp">{tline}</div>', unsafe_allow_html=True)
        meta = f"**{cond_txt}**"
        if hum is not None:
            meta += f" · {'आर्द्रता' if IS_MR else 'Humidity'} {hum}%"
        if wind is not None:
            meta += f" · {'वारा' if IS_MR else 'Wind'} {wind} km/h"
        if uv is not None:
            meta += f" · UV {uv}"
        if precip is not None:
            meta += f" · {'पावसाची शक्यता' if IS_MR else 'Rain chance'} {precip}%"
        st.markdown(f'<div class="weather-now-meta">{meta}</div>', unsafe_allow_html=True)
        if advisory:
            st.markdown(
                f'<div style="margin-top:.5rem;padding:.45rem .7rem;background:#f0faf3;border-left:3px solid #52b788;'
                f'border-radius:8px;font-size:.82rem;color:#1a4731;">🌾 {advisory}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 5-day forecast cards ──
    forecast = wx.get("forecast", [])
    if forecast:
        st.markdown(f"**{'५ दिवसांचा अंदाज' if IS_MR else '5-Day Outlook'}**")
        cards_html = ['<div class="weather-daily-row">']
        for d in forecast[:5]:
            day_lbl = d.get("day", "")
            date_lbl = d.get("date", "")
            max_t = d.get("max_c")
            min_t = d.get("min_c")
            dtxt = d.get("condition", "—")[:28]
            rain_pct = d.get("rain_chance_pct")
            rng = ""
            if max_t is not None and min_t is not None:
                rng = f"{max_t:.0f}° / {min_t:.0f}°"
            elif max_t is not None:
                rng = f"↑{max_t:.0f}°"
            rain_info = f"🌧 {rain_pct}%" if rain_pct is not None else ""
            cards_html.append(
                f'<div class="weather-day-card">'
                f'<div class="d1">{day_lbl} {date_lbl}</div>'
                f'<div class="d2">{dtxt}</div>'
                f'<div class="d3">{rng}</div>'
                f'<div class="d2" style="margin-top:.2rem;color:#0077b6;">{rain_info}</div>'
                f'</div>'
            )
        cards_html.append("</div>")
        st.markdown("".join(cards_html), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════
render_weather_panel()

if st.session_state.page == "home":
    badge_txt = "महाराष्ट्रातील शेतकऱ्यांसाठी" if IS_MR else "Built for Maharashtra Farmers"
    st.markdown(f"""
    <div class="hero fu">
        <h1>{t('app_title')}</h1>
        <p class="hero-sub">{t('app_subtitle')}</p>
        <span class="hero-badge">{badge_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    _nc = int(price_df["Crop"].nunique())
    _nd = int(price_df["District"].nunique())
    _nm = int(price_df["Market"].nunique())
    stat_labels = {
        "English": [
            (t("crops_covered"), str(_nc)),
            (t("districts"), str(_nd)),
            (t("markets"), str(_nm)),
            (t("data_src_lbl"), "Agmarknet"),
        ],
        "मराठी": [
            (t("crops_covered"), str(_nc)),
            (t("districts"), str(_nd)),
            (t("markets"), str(_nm)),
            (t("data_src_lbl"), "Agmarknet"),
        ],
    }
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (lbl, val) in zip([sc1, sc2, sc3, sc4], stat_labels[st.session_state.lang]):
        with col:
            st.markdown(
                f'<div class="stat-card fu1"><div class="stat-lbl">{lbl}</div><div class="stat-val">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2, c3 = st.columns(3, gap="large")
    nav_data = [
        ("growing", "🌱", "Crop Growth", "पिक वाढ", t("grow_nav_desc"), "fu1 nav-grow"),
        ("maintaining", "🩺", "Crop Maintenance", "पिक देखभाल", t("maint_nav_desc"), "fu2 nav-maintain"),
        ("selling", "💰", "Crop Sales", "पिक विक्री", t("sell_nav_desc"), "fu3 nav-sell"),
    ]
    for col, (pg, icon, en, mr, desc, anim) in zip([c1, c2, c3], nav_data):
        with col:
            display_title = f"{en} · {mr}" if not IS_MR else f"{mr} · {en}"
            st.markdown(f"""
            <div class="nav-card {anim}">
                <span class="nav-icon">{icon}</span>
                <div class="nav-en">{en}</div>
                <div class="nav-mr">{mr}</div>
                <div class="nav-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(display_title, key=pg, use_container_width=True):
                go(pg)

    st.caption(
        "Agmarknet · ICAR · CACP · [krishi.maharashtra.gov.in](https://krishi.maharashtra.gov.in)"
    )

# ══════════════════════════════════════════════════
# GROWING
# ══════════════════════════════════════════════════
elif st.session_state.page == "growing":
    back_btn()
    st.markdown(f'<p class="sec-hd fu">{t("grow_header")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-sub">{t("grow_subhd")}</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([t("tab_soil"), t("tab_soildb"), t("tab_aichat")])

    soil_names_en = soil_df["Soil_Type"].tolist()
    soil_names_mr = soil_df["Soil_MR"].tolist()
    soil_display  = soil_names_mr if IS_MR else soil_names_en

    with tab1:
        chip_row(get_chips("grow")[:5], "g_soil")

        col_a, col_b = st.columns(2)
        with col_a:
            soil_sel = st.selectbox(t("soil_type"), soil_display, key="g_soil")
            soil_idx = soil_display.index(soil_sel)
            soil_en  = soil_names_en[soil_idx]

            seasons = {
                "English": ["Kharif (Jun–Oct)","Rabi (Nov–Mar)","Zaid (Apr–Jun)","Annual"],
                "मराठी":   ["खरीप (जून–ऑक्टोबर)","रब्बी (नोव्हेंबर–मार्च)","उन्हाळी (एप्रिल–जून)","वार्षिक"],
            }
            season = st.selectbox(t("season"), seasons[st.session_state.lang])
        with col_b:
            ph = st.slider(t("soil_ph"), 4.0, 9.5, 7.0, step=0.1)
            water_opts = {
                "English": ["Rain-fed","Canal / River","Borewell","Drip / Micro-irrigation"],
                "मराठी":   ["पावसावर अवलंबून","कालवा / नदी","बोअरवेल","ठिबक / सूक्ष्म सिंचन"],
            }
            water = st.selectbox(t("water_src"), water_opts[st.session_state.lang])
            district = st.selectbox(t("district"), sorted(price_df["District"].unique()))

        soil_row = soil_df[soil_df["Soil_Type"] == soil_en].iloc[0]
        soil_mr_name = soil_row["Soil_MR"]

        st.markdown(f"""
        <div class="icar-card pop">
            <div class="icar-title">📊 ICAR {"माती प्रोफाइल" if IS_MR else "Soil Profile"} — {soil_mr_name if IS_MR else soil_en}</div>
            <span class="badge">pH: {soil_row['pH_Min']}–{soil_row['pH_Max']}</span>
            <span class="badge badge-a">OC: {soil_row['OC_pct']}%</span>
            <span class="badge">N: {soil_row['N_kg_ha']} kg/ha</span>
            <span class="badge">P: {soil_row['P_kg_ha']} kg/ha</span>
            <span class="badge">K: {soil_row['K_kg_ha']} kg/ha</span>
            <span class="badge badge-s">Fe: {soil_row['Fe_ppm']} ppm</span>
            <span class="badge badge-s">Zn: {soil_row['Zn_ppm']} ppm</span>
            <span class="badge badge-a">{"पोत" if IS_MR else "Texture"}: {soil_row['Texture']}</span>
            <span class="badge">{"निचरा" if IS_MR else "Drainage"}: {soil_row['Drainage']}</span><br>
            <small style="color:var(--muted);margin-top:.4rem;display:block;">
            ⚠️ {"कमतरता" if IS_MR else "Deficiencies"}: {soil_row['Deficiencies']}<br>
            🌿 {"शिफारस" if IS_MR else "Amendment"}: {soil_row['Amendment']}
            </small>
        </div>
        """, unsafe_allow_html=True)

        # pH suitability chart
        ph_data = soil_df[["Soil_MR" if IS_MR else "Soil_Type","pH_Min","pH_Optimal","pH_Max"]].copy()
        ph_data.columns = ["Soil","pH Min","pH Optimal","pH Max"]
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Bar(name="pH Min", x=ph_data["Soil"], y=ph_data["pH Min"], marker_color="#95d5b2"))
        fig_ph.add_trace(go.Bar(name="pH Optimal", x=ph_data["Soil"], y=ph_data["pH Optimal"], marker_color="#2d6a4f"))
        fig_ph.add_trace(go.Bar(name="pH Max", x=ph_data["Soil"], y=ph_data["pH Max"], marker_color="#f4a261"))
        fig_ph.add_hline(y=ph, line_dash="dash", line_color="#b5570a",
                         annotation_text=f"{'तुमचा' if IS_MR else 'Your'} pH={ph}", annotation_position="top right")
        fig_ph.update_layout(
            barmode="group",
            title="pH Range by Soil Type — Maharashtra (ICAR)" if not IS_MR else "महाराष्ट्रातील मातींचे pH श्रेणी (ICAR)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Sora, sans-serif", size=11, color="#0d2b1a"),
            xaxis=dict(showgrid=False, tickangle=-35),
            yaxis=dict(gridcolor="rgba(210,240,220,.5)", title="pH"),
            margin=dict(l=10,r=10,t=45,b=10), height=340,
            legend=dict(bgcolor="rgba(255,255,255,.7)"),
        )
        st.plotly_chart(fig_ph, use_container_width=True)

        if st.button(t("get_rec"), use_container_width=True, key="grow_btn"):
            data_ctx = (f"ICAR soil data: pH {soil_row['pH_Min']}–{soil_row['pH_Max']}, "
                        f"OC {soil_row['OC_pct']}%, N {soil_row['N_kg_ha']} kg/ha, "
                        f"P {soil_row['P_kg_ha']} kg/ha, K {soil_row['K_kg_ha']} kg/ha, "
                        f"Fe {soil_row['Fe_ppm']} ppm, Zn {soil_row['Zn_ppm']} ppm. "
                        f"Primary crops: {soil_row['Primary_Crops']}. Deficiencies: {soil_row['Deficiencies']}. "
                        f"Standard amendment: {soil_row['Amendment']}.")
            prompt = (f"For {soil_en} soil at measured pH {ph} in {district}, Maharashtra, season {season}, water: {water}. "
                      f"Answer as a complete planting guide: (1) soil pH interpretation and correction if needed; "
                      f"(2) soil texture & drainage implications; (3) exhaustive list of suitable crops for this zone with named varieties "
                      f"(cereals, pulses, oilseeds, fibre, cash crops, vegetables, spices — as many as realistically grown in Maharashtra); "
                      f"(4) top 5 priority crops for this farmer with seed rate, sowing window, spacing; "
                      f"(5) full NPK + micronutrient plan; (6) organic options; (7) schemes: PM-KISAN, Soil Health Card, RKVY, state advisories.")
            with st.spinner("Analysing with ICAR soil data…" if not IS_MR else "ICAR माती डेटासह विश्लेषण होत आहे…"):
                result = ask_gemini(
                    prompt,
                    context="You specialise in agronomy and soil–crop matching for Maharashtra.",
                    data_context=data_ctx,
                    extra_knowledge=CROP_VARIETY_REFERENCE,
                )
            st.markdown(f'<div class="resp-box">{result}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="src-tag">{t("ai_source")}</p>', unsafe_allow_html=True)

        _grow_ctx = f"Soil type: {soil_en}. pH: {ph}. District: {district}. Season: {season}. Water: {water}."
        finish_chip_qa(
            "g_soil",
            "grow",
            "You specialise in crop planning and soil science for Maharashtra.",
            data_context=_grow_ctx,
            extra_knowledge=CROP_VARIETY_REFERENCE,
        )

        tab_ai_ask(
            "grow_t1",
            "You specialise in crop planting: soil pH, soil type, varieties, and calendars for Maharashtra.",
            data_context=f"Soil: {soil_en}, pH {ph}, District: {district}, Season: {season}, Water: {water}",
            extra_knowledge=CROP_VARIETY_REFERENCE,
        )

    with tab2:
        st.markdown(f"**{'महाराष्ट्र माती रसायन संदर्भ' if IS_MR else 'Maharashtra Soil Chemistry Reference'}** *(ICAR-NBSS&LUP / KVK)*")
        show_cols = (["Soil_MR","Region","pH_Min","pH_Max","OC_pct","N_kg_ha","P_kg_ha","K_kg_ha","WHC","Drainage","Primary_Crops","Deficiencies","Amendment"]
                     if IS_MR else
                     ["Soil_Type","Region","pH_Min","pH_Max","OC_pct","N_kg_ha","P_kg_ha","K_kg_ha","WHC","Drainage","Primary_Crops","Deficiencies","Amendment"])
        rename = {
            "Soil_Type":"Soil","Soil_MR":"माती प्रकार","pH_Min":"pH Min","pH_Max":"pH Max",
            "OC_pct":"OC(%)","N_kg_ha":"N kg/ha","P_kg_ha":"P kg/ha","K_kg_ha":"K kg/ha",
            "WHC":"जलधारण" if IS_MR else "WHC","Drainage":"निचरा" if IS_MR else "Drainage",
            "Primary_Crops":"मुख्य पिके" if IS_MR else "Primary Crops",
            "Deficiencies":"कमतरता" if IS_MR else "Deficiencies","Amendment":"सुधारणा" if IS_MR else "Amendment",
        }
        st.dataframe(soil_df[show_cols].rename(columns=rename), use_container_width=True, hide_index=True, height=380)

        # Nutrient chart
        fig_nut = go.Figure()
        soil_x = soil_df["Soil_MR"].tolist() if IS_MR else soil_df["Soil_Type"].tolist()
        for col, name, color in [("N_kg_ha","N (kg/ha)","#2d6a4f"),("P_kg_ha","P (kg/ha)","#f4a261"),("K_kg_ha","K (kg/ha)","#48cae4")]:
            fig_nut.add_trace(go.Bar(name=name, x=soil_x, y=soil_df[col], marker_color=color))
        fig_nut.update_layout(
            barmode="group",
            title="NPK Availability by Soil Type (kg/ha) — ICAR Data" if not IS_MR else "मातीनुसार NPK उपलब्धता (kg/ha) — ICAR डेटा",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Sora, sans-serif", size=11, color="#0d2b1a"),
            xaxis=dict(showgrid=False, tickangle=-35),
            yaxis=dict(gridcolor="rgba(210,240,220,.5)", title="kg/ha"),
            margin=dict(l=10,r=10,t=45,b=10), height=350,
        )
        st.plotly_chart(fig_nut, use_container_width=True)
        st.caption("Source: ICAR-NBSS&LUP · Krishi Vigyan Kendra Maharashtra · Maharashtra State Soil Survey")
        tab_ai_ask(
            "grow_t2",
            "You explain soil chemistry tables and nutrient planning for Maharashtra farmers.",
            data_context="Use the soil chemistry dataframe shown in this tab (ICAR-style reference).",
            extra_knowledge=CROP_VARIETY_REFERENCE,
        )

    with tab3:
        st.markdown(f"**{'AI कृषितज्ञाला विचारा — माती, पिके, शेती पद्धती:' if IS_MR else 'Ask the AI Agronomist — soil, crops, farming practices:'}**")
        chip_row(get_chips("grow")[3:], "g2_chip")
        _g2_pending = st.session_state.pop("g2_chip_pending", None)
        if _g2_pending:
            with st.spinner(t("chip_working")):
                try:
                    _r = ask_gemini(
                        expand_chip_question(_g2_pending, "grow"),
                        context="You specialise in crop science and soil chemistry for Maharashtra.",
                        extra_knowledge=CROP_VARIETY_REFERENCE,
                    )
                except Exception as _ex:
                    _r = f"❌ {_ex}"
            st.session_state.grow_msgs.append({"role": "user", "content": _g2_pending})
            st.session_state.grow_msgs.append({"role": "assistant", "content": _r})
        for msg in st.session_state.grow_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_in = st.chat_input(t("chat_input_grow"), key="grow_chat")
        if user_in:
            st.session_state.grow_msgs.append({"role": "user", "content": user_in})
            with st.spinner(t("chip_working")):
                try:
                    r = ask_gemini(
                        user_in,
                        context="You specialise in crop science and soil chemistry for Maharashtra.",
                        extra_knowledge=CROP_VARIETY_REFERENCE,
                    )
                except Exception as _ex:
                    r = f"❌ {_ex}"
            st.session_state.grow_msgs.append({"role": "assistant", "content": r})
            st.rerun()
        if st.session_state.grow_msgs:
            if st.button(t("clear_chat"), key="clr_g"):
                st.session_state.grow_msgs = []
                st.rerun()

# ══════════════════════════════════════════════════
# MAINTAINING
# ══════════════════════════════════════════════════
elif st.session_state.page == "maintaining":
    back_btn()
    st.markdown(f'<p class="sec-hd fu">{t("maint_header")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-sub">{t("maint_subhd")}</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([t("tab_pest"), t("tab_nutr"), t("tab_hchat")])

    crops_list = ["Cotton","Soybean","Onion","Grapes","Tomato","Wheat","Sugarcane","Tur Dal","Pomegranate","Jowar","Chickpea","Groundnut"]
    crops_mr   = ["कापूस","सोयाबीन","कांदा","द्राक्षे","टोमॅटो","गहू","ऊस","तूर डाळ","डाळिंब","ज्वारी","हरभरा","भुईमूग"]
    crop_display = crops_mr if IS_MR else crops_list

    with tab1:
        chip_row(get_chips("maintain")[:4], "m_pest")
        col_l, col_r = st.columns(2)
        with col_l:
            crop_sel = st.selectbox(t("crop_lbl"), crop_display, key="m_crop")
            crop_en  = crops_list[crop_display.index(crop_sel)]
            stages = {
                "English": ["Germination","Seedling (0–30 days)","Vegetative (30–60 days)","Flowering","Pod / Fruit Fill","Maturity / Harvest"],
                "मराठी":   ["उगवण","रोपे (0–30 दिवस)","वाढीचा (30–60 दिवस)","फुलोरा","शेंगा / फळ भरण","परिपक्वता / काढणी"],
            }
            growth = st.selectbox(t("growth_stage"), stages[st.session_state.lang])
        with col_r:
            symptom = st.text_area(t("symptom"), placeholder=t("symptom_ph"), height=110)
            region_opts = {
                "English": ["Vidarbha","Marathwada","Western Maharashtra","Konkan","North Maharashtra"],
                "मराठी":   ["विदर्भ","मराठवाडा","पश्चिम महाराष्ट्र","कोकण","उत्तर महाराष्ट्र"],
            }
            region = st.selectbox(t("region_lbl"), region_opts[st.session_state.lang])

        if st.button(t("diagnose_btn"), use_container_width=True):
            prompt = (f"Full crop maintenance advisory for {crop_en} at {growth} in {region}, Maharashtra. "
                      f"Symptoms: {symptom or 'general health check'}. Cover: "
                      f"(1) diagnosis with possible pests/diseases/nutrient issues; "
                      f"(2) IPM: cultural, biological, organic sprays with doses; "
                      f"(3) chemical options (molecule, dose/ha, PHI, resistance management); "
                      f"(4) irrigation & fertigation adjustments; "
                      f"(5) Maharashtra schemes / university/KVK contacts pattern; "
                      f"(6) preventive package for next season.")
            with st.spinner("Diagnosing…" if not IS_MR else "निदान होत आहे…"):
                res = ask_gemini(
                    prompt,
                    context="You are a plant pathologist and crop physiologist for Maharashtra.",
                )
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

        finish_chip_qa(
            "m_pest",
            "maintain",
            "You are a plant pathologist and crop maintenance expert for Maharashtra.",
            data_context=f"Crop: {crop_en}, Stage: {growth}, Region: {region}, Symptoms note: {(symptom or '')[:500]}",
        )

        tab_ai_ask(
            "maint_t1",
            "You answer questions on pest, disease, and field symptoms for Maharashtra crops.",
            data_context=f"Crop: {crop_en}, Stage: {growth}, Region: {region}",
        )

    with tab2:
        chip_row(get_chips("maintain")[3:], "m_nutr")
        col_a,col_b = st.columns(2)
        with col_a:
            crop_n = st.selectbox(t("crop_lbl"), crop_display, key="n_crop")
            crop_n_en = crops_list[crop_display.index(crop_n)]
            soil_n = st.selectbox(t("soil_type"), soil_df["Soil_MR"].tolist() if IS_MR else soil_df["Soil_Type"].tolist(), key="n_soil")
            soil_n_en = soil_df.iloc[(soil_df["Soil_MR"].tolist() if IS_MR else soil_df["Soil_Type"].tolist()).index(soil_n)]["Soil_Type"]
        with col_b:
            irr_opts = {
                "English": ["Rain-fed","Drip","Sprinkler","Furrow","Flood"],
                "मराठी":   ["पावसावर अवलंबून","ठिबक","तुषार","सरी","पूर सिंचन"],
            }
            irr = st.selectbox(t("irr_type"), irr_opts[st.session_state.lang])
            area = st.number_input(t("area_ha"), 0.5, 50.0, 1.0, step=0.5)

        if st.button(f"📋 {t('gen_schedule')}", use_container_width=True):
            sr = soil_df[soil_df["Soil_Type"] == soil_n_en].iloc[0]
            data_ctx = f"ICAR: N {sr['N_kg_ha']} kg/ha, P {sr['P_kg_ha']} kg/ha, K {sr['K_kg_ha']} kg/ha, OC {sr['OC_pct']}%"
            prompt = (f"Create a complete fertiliser and {irr} irrigation schedule for {crop_n_en} on {soil_n_en} soil "
                      f"over {area} hectares in Maharashtra. Include: "
                      f"1) Stage-wise NPK doses (basal, top-dress 1, top-dress 2) in kg/ha, "
                      f"2) Micronutrient schedule (ZnSO4, Borax, FeSO4 if needed), "
                      f"3) Irrigation frequency and water quantity per stage in litres/plant or mm/ha, "
                      f"4) Soil moisture target per stage, "
                      f"5) Bio-fertiliser recommendations (Rhizobium, PSB, Azotobacter), "
                      f"6) Total input cost estimate per hectare, "
                      f"7) Expected ROI.")
            with st.spinner("Generating…" if not IS_MR else "तयार होत आहे…"):
                res = ask_gemini(prompt, data_context=data_ctx)
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)

        try:
            _srn = soil_df[soil_df["Soil_Type"] == soil_n_en].iloc[0]
            _nk_hint = f"N {_srn['N_kg_ha']} kg/ha (ICAR-style ref)"
        except Exception:
            _nk_hint = "soil row N/A"
        finish_chip_qa(
            "m_nutr",
            "maintain",
            "You advise on irrigation and nutrition schedules for Maharashtra crops.",
            data_context=(
                f"Crop: {crop_n_en}, Soil: {soil_n_en}, Irrigation: {irr}, Area (ha): {area}. {_nk_hint}."
            ),
        )

        tab_ai_ask(
            "maint_t2",
            "You advise on irrigation scheduling, NPK, and micronutrients for Maharashtra crops.",
            data_context=f"Crop: {crop_n_en}, Soil profile: {soil_n_en}, Irrigation: {irr}, Area ha: {area}",
        )

    with tab3:
        chip_row(get_chips("maintain")[4:], "m3_chip")
        _m3 = st.session_state.pop("m3_chip_pending", None)
        if _m3:
            with st.spinner(t("chip_working")):
                try:
                    _rm = ask_gemini(
                        expand_chip_question(_m3, "maintain"),
                        context="You are a crop health expert for Maharashtra.",
                    )
                except Exception as _ex:
                    _rm = f"❌ {_ex}"
            st.session_state.maintain_msgs.append({"role": "user", "content": _m3})
            st.session_state.maintain_msgs.append({"role": "assistant", "content": _rm})
        for msg in st.session_state.maintain_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_in = st.chat_input(t("chat_input_maint"), key="m_chat")
        if user_in:
            st.session_state.maintain_msgs.append({"role": "user", "content": user_in})
            with st.spinner(t("chip_working")):
                try:
                    r = ask_gemini(user_in, context="You are a crop health expert for Maharashtra.")
                except Exception as _ex:
                    r = f"❌ {_ex}"
            st.session_state.maintain_msgs.append({"role": "assistant", "content": r})
            st.rerun()
        if st.session_state.maintain_msgs:
            if st.button(t("clear_chat"), key="clr_m"):
                st.session_state.maintain_msgs = []
                st.rerun()

# ══════════════════════════════════════════════════
# SELLING
# ══════════════════════════════════════════════════
elif st.session_state.page == "selling":
    back_btn()
    st.markdown(f'<p class="sec-hd fu">{t("sell_header")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-sub">{t("sell_subhd")}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="sell-page-intro">{t("sell_intro")}</div>', unsafe_allow_html=True)

    crops_avail = sorted(price_df["Crop"].unique())
    crops_mr_avail = [price_df[price_df["Crop"] == c].iloc[0]["Crop_MR"] for c in crops_avail]
    crop_disp = crops_mr_avail if IS_MR else crops_avail

    tab1, tab2, tab3, tab4 = st.tabs(
        [t("tab_price"), t("tab_msp"), t("tab_aimarket"), t("tab_sellchat")]
    )

    with tab1:
        c_f1,c_f2,c_f3 = st.columns(3)
        with c_f1:
            crop_sel_d = st.selectbox(t("crop_lbl"), crop_disp, key="s_crop")
            crop_sel = crops_avail[crop_disp.index(crop_sel_d)]
        with c_f2:
            season_opts = ["All / सर्व"] + list(price_df["Season"].unique()) if IS_MR else ["All"] + list(price_df["Season"].unique())
            season_sel = st.selectbox(t("season"), season_opts)
        with c_f3:
            dist_opts = (["सर्व जिल्हे"] if IS_MR else ["All"]) + sorted(price_df["District"].unique())
            dist_sel = st.selectbox(t("district"), dist_opts)

        st.caption(t("live_prices_note"))

        filt = price_df[price_df["Crop"] == crop_sel].copy()
        if season_sel not in ["All","All / सर्व"]:
            filt = filt[filt["Season"] == season_sel]
        if dist_sel not in ["All","सर्व जिल्हे"]:
            filt = filt[filt["District"] == dist_sel]

        if not filt.empty:
            best = filt.loc[filt["Modal_Price"].idxmax()]
            avg_m = int(filt["Modal_Price"].mean())
            max_m = int(filt["Modal_Price"].max())
            min_m = int(filt["Modal_Price"].min())
            tot_a = int(filt["Arrival_MT"].sum())

            m1,m2,m3,m4 = st.columns(4)
            m1.metric(t("best_market"), best["Market"])
            m2.metric(t("avg_modal"), f"₹{avg_m:,}/qtl")
            m3.metric(t("highest"), f"₹{max_m:,}/qtl")
            m4.metric(t("lowest"), f"₹{min_m:,}/qtl")

            # Grouped bar: Min / Modal / Max per market
            chart_filt = filt.sort_values("Modal_Price", ascending=False).head(10)
            lbl_map = {
                "Min_Price":  "Min ₹/qtl" if not IS_MR else "किमान ₹/qtl",
                "Modal_Price":"Modal ₹/qtl" if not IS_MR else "मोडल ₹/qtl",
                "Max_Price":  "Max ₹/qtl" if not IS_MR else "कमाल ₹/qtl",
            }
            fig_price = plotly_grouped_bar(
                chart_filt, "Market", ["Min_Price","Modal_Price","Max_Price"], lbl_map,
                title=f"{'APMC भाव तुलना' if IS_MR else 'APMC Price Comparison'} — {crop_sel_d} (₹/qtl)"
            )
            for tr in fig_price.data:
                tr.hovertemplate = "<b>%{x}</b><br>%{fullData.name}: ₹%{y:,.0f}/qtl<extra></extra>"
            _sell_chart_layout(fig_price)
            st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_INTERACTIVE)

            # Arrival bubble chart (sanitized dtypes for Plotly Express)
            try:
                _fd = filt.copy()
                _fd["District"] = _fd["District"].astype(str)
                _fd["Market"] = _fd["Market"].astype(str)
                _fd["Arrival_MT"] = pd.to_numeric(_fd["Arrival_MT"], errors="coerce").fillna(0).clip(lower=1)
                _fd["Modal_Price"] = pd.to_numeric(_fd["Modal_Price"], errors="coerce")
                _fd = _fd.dropna(subset=["Modal_Price"])
                if _fd.empty:
                    fig_arr = go.Figure()
                else:
                    fig_arr = px.scatter(
                        _fd,
                        x="Market",
                        y="Modal_Price",
                        size="Arrival_MT",
                        color="District",
                        size_max=55,
                        title=f"{'मोडल भाव व आवक' if IS_MR else 'Modal Price vs Arrival Volume'} — {crop_sel_d}",
                        labels={"Modal_Price": "₹/qtl", "Arrival_MT": "Arrival (MT)"},
                        color_discrete_sequence=GREEN_PALETTE,
                    )
                    fig_arr.update_layout(
                        margin=dict(l=10, r=10, t=45, b=10), height=360,
                        xaxis=dict(showgrid=False, tickangle=-35),
                        yaxis=dict(gridcolor="rgba(45,106,79,.2)", title="₹/qtl"),
                    )
                    fig_arr.update_traces(
                        marker=dict(line=dict(width=0.6, color="#fff"), opacity=0.9),
                        hovertemplate="<b>%{fullData.name}</b><br>Market: %{x}<br>₹%{y}/qtl<br>Size ∝ arrival<extra></extra>",
                    )
                    _sell_chart_layout(fig_arr)
            except Exception as _sc_ex:
                st.caption(f"Scatter chart skipped: {_sc_ex}" if not IS_MR else f"स्कॅटर आलेख: {_sc_ex}")
                fig_arr = go.Figure()
            if fig_arr.data:
                st.plotly_chart(fig_arr, use_container_width=True, config=PLOTLY_INTERACTIVE)

            # Table
            lbl = "मराठी नाव" if IS_MR else "Crop"
            disp_cols = ["District","Market","Season","Min_Price","Modal_Price","Max_Price","Arrival_MT"]
            st.dataframe(
                filt[disp_cols].rename(columns={
                    "District":"जिल्हा" if IS_MR else "District",
                    "Market":"बाजार" if IS_MR else "Market",
                    "Season":"हंगाम" if IS_MR else "Season",
                    "Min_Price":"किमान ₹/qtl","Modal_Price":"मोडल ₹/qtl","Max_Price":"कमाल ₹/qtl",
                    "Arrival_MT":"आवक (MT)" if IS_MR else "Arrival (MT)",
                }).sort_values("मोडल ₹/qtl" if IS_MR else "Modal_Price", ascending=False),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"{'स्रोत' if IS_MR else 'Source'}: DMI · [Agmarknet](https://agmarknet.gov.in) · data.gov.in"
            )
        else:
            st.info("No data for selected filters." if not IS_MR else "निवडलेल्या फिल्टरसाठी डेटा नाही.")

        _price_ctx = ""
        if not filt.empty:
            _price_ctx = (
                f"Crop {crop_sel}: modal ₹{int(filt['Modal_Price'].min())}–₹{int(filt['Modal_Price'].max())}/qtl across "
                f"{filt['District'].nunique()} districts in dataset."
            )
        chip_row(get_chips("sell")[:3], "s_price")
        _sell_dctx = _price_ctx or f"Crop filter: {crop_sel}. No rows for current filters."
        if not filt.empty:
            _sell_dctx = (
                f"{_sell_dctx} Modal ₹{int(filt['Modal_Price'].min())}–₹{int(filt['Modal_Price'].max())}/qtl, "
                f"markets: {filt['Market'].nunique()}, districts: {filt['District'].nunique()}."
            )
        finish_chip_qa(
            "s_price",
            "sell",
            "You advise Maharashtra farmers on crop marketing, mandi prices, and timing.",
            data_context=_sell_dctx,
        )

        tab_ai_ask(
            "sell_t1",
            "You interpret mandi prices, arrivals, and timing for selling in Maharashtra.",
            data_context=_price_ctx,
        )

    with tab2:
        st.markdown(f"**{'किमान आधारभूत किंमत (MSP) 2024-25' if IS_MR else 'Minimum Support Price (MSP) 2024-25'}** *(CACP — Ministry of Agriculture, GoI)*")

        msp_cat = st.radio(
            t("msp_season"),
            (["All", "Kharif", "Rabi", "Annual"] if not IS_MR else ["सर्व", "खरीप", "रब्बी", "वार्षिक"]),
            horizontal=True,
        )
        cat_map = {"सर्व":"All","खरीप":"Kharif","रब्बी":"Rabi","वार्षिक":"Annual"}
        msp_key = cat_map.get(msp_cat, msp_cat)
        msp_filt = msp_df if msp_key == "All" else msp_df[msp_df["Season"] == msp_key]

        # MSP bar chart (melted long form — avoids Plotly Express wide-y errors)
        try:
            if msp_filt.empty:
                raise ValueError("No MSP rows for this filter")
            cx = "Crop_MR" if IS_MR else "Crop"
            _cols = [cx, "MSP_2023_24", "MSP_2024_25"]
            plot_m = msp_filt.sort_values("MSP_2024_25", ascending=False)[_cols].copy()
            plot_m = plot_m.rename(columns={cx: "__crop"})
            melted = plot_m.melt(id_vars=["__crop"], var_name="Year", value_name="MSP")
            melted["Year"] = melted["Year"].map(
                {"MSP_2023_24": "2023-24", "MSP_2024_25": "2024-25"}
            ).fillna(melted["Year"])
            fig_msp = px.bar(
                melted,
                x="__crop",
                y="MSP",
                color="Year",
                barmode="group",
                title="MSP 2023-24 vs 2024-25 (₹/qtl)" if not IS_MR else "MSP 2023-24 विरुद्ध 2024-25 (₹/qtl)",
                labels={"__crop": ("पिक" if IS_MR else "Crop"), "MSP": "₹/qtl"},
                color_discrete_sequence=["#95d5b2", "#1a4731"],
                text_auto=".0f",
            )
            fig_msp.update_layout(
                xaxis=dict(showgrid=False, tickangle=-40),
                yaxis=dict(gridcolor="rgba(45,106,79,.2)", title="₹/quintal"),
                margin=dict(l=10, r=10, t=45, b=10), height=400,
            )
            _sell_chart_layout(fig_msp)
            fig_msp.update_traces(
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: ₹%{y:,.0f}/qtl<extra></extra>"
            )
        except Exception as _msp_ex:
            st.warning(f"MSP chart: {_msp_ex}" if not IS_MR else f"MSP आलेख: {_msp_ex}")
            fig_msp = go.Figure()
        if fig_msp.data:
            st.plotly_chart(fig_msp, use_container_width=True, config=PLOTLY_INTERACTIVE)

        # Hike % chart
        try:
            msp_h = msp_filt.copy()
            msp_h["Increase_pct"] = pd.to_numeric(msp_h["Increase_pct"], errors="coerce").fillna(0)
            cx2 = "Crop_MR" if IS_MR else "Crop"
            fig_hike = px.bar(
                msp_h.sort_values("Increase_pct", ascending=False),
                x=cx2,
                y="Increase_pct",
                title="MSP Hike % (2024-25 over 2023-24)" if not IS_MR else "MSP वाढ % (2024-25, 2023-24 पेक्षा)",
                color="Increase_pct",
                color_continuous_scale=["#d8f3dc", "#2d6a4f"],
                text_auto=".1f",
            )
            fig_hike.update_layout(
                xaxis=dict(showgrid=False, tickangle=-40),
                yaxis=dict(gridcolor="rgba(45,106,79,.2)", title="%"),
                margin=dict(l=10, r=10, t=45, b=10), height=340,
                coloraxis_showscale=False,
            )
            _sell_chart_layout(fig_hike)
            fig_hike.update_traces(
                hovertemplate="<b>%{x}</b><br>"
                + ("वाढ %{y:.1f}%" if IS_MR else "Hike %{y:.1f}%")
                + "<extra></extra>"
            )
        except Exception as _h_ex:
            st.warning(f"Hike chart: {_h_ex}" if not IS_MR else f"वाढ आलेख: {_h_ex}")
            fig_hike = go.Figure()
        if fig_hike.data:
            st.plotly_chart(fig_hike, use_container_width=True, config=PLOTLY_INTERACTIVE)

        # Table
        rename_msp = {
            "Crop":"Crop","Crop_MR":"पिक",
            "MSP_2024_25":"MSP 2024-25 (₹/qtl)","MSP_2023_24":"MSP 2023-24",
            "Increase_pct":"वाढ (%)" if IS_MR else "Hike (%)","Category":"Category","Season":"Season",
        }
        display_msp_cols = ["Crop_MR","MSP_2024_25","MSP_2023_24","Increase_pct","Category","Season"] if IS_MR else ["Crop","MSP_2024_25","MSP_2023_24","Increase_pct","Category","Season"]
        st.dataframe(msp_filt[display_msp_cols].rename(columns=rename_msp), use_container_width=True, hide_index=True)
        st.caption(f"{'स्रोत' if IS_MR else 'Source'}: [CACP — Commission for Agricultural Costs & Prices](https://cacp.dacnet.nic.in) · Ministry of Agriculture & Farmers Welfare, GoI")
        tab_ai_ask(
            "sell_t2",
            "You explain MSP, procurement, and how it relates to mandi prices in Maharashtra.",
            data_context="MSP table from CACP reference in app (verify current year on official site).",
        )

    with tab3:
        chip_row(get_chips("sell")[2:5], "s_ai")
        col_a2,col_b2 = st.columns(2)
        with col_a2:
            crop_ai_d = st.selectbox(t("crop_lbl"), crop_disp, key="ai_crop")
            crop_ai   = crops_avail[crop_disp.index(crop_ai_d)]
            qty = st.number_input(t("qty_qtl"), 10, 5000, 100, step=10)
        with col_b2:
            harvest_opts = {
                "English": ["Now / This week","2–4 weeks","1–2 months","3+ months"],
                "मराठी":   ["आता / या आठवड्यात","2–4 आठवड्यात","1–2 महिन्यांत","3+ महिन्यांत"],
            }
            harvest_in = st.selectbox(t("harvest_in"), harvest_opts[st.session_state.lang])
            storage_opts = {
                "English": ["None","Cold storage","Gunny bags / Warehouse","FPO / Co-op storage"],
                "मराठी":   ["नाही","शीतगृह","गोदाम / पोती","FPO / सहकारी साठवण"],
            }
            storage = st.selectbox(t("storage_lbl"), storage_opts[st.session_state.lang])

        if st.button(f"📈 {t('market_strategy')}", use_container_width=True):
            crop_data = price_df[price_df["Crop"] == crop_ai]
            if not crop_data.empty:
                best_mkt = crop_data.loc[crop_data["Modal_Price"].idxmax()]
                data_ctx = (f"Agmarknet 2024-25: Best market for {crop_ai} is {best_mkt['Market']}, {best_mkt['District']} "
                            f"at ₹{best_mkt['Modal_Price']}/qtl modal. Range: ₹{int(crop_data['Modal_Price'].min())}–₹{int(crop_data['Modal_Price'].max())}/qtl. "
                            f"Total market arrival: {int(crop_data['Arrival_MT'].sum())} MT.")
                msp_match = msp_df[msp_df["Crop"].str.contains(crop_ai.split()[0], case=False, na=False)]
                if not msp_match.empty:
                    data_ctx += f" MSP 2024-25: ₹{int(msp_match.iloc[0]['MSP_2024_25'])}/qtl."
            else:
                data_ctx = ""
            prompt = (f"Farmer has {qty} quintals of {crop_ai}, harvest ready {harvest_in}, storage: {storage}. "
                      f"Advise: 1) Best 3 APMCs to sell with expected price, "
                      f"2) Optimal timing (sell now vs store), "
                      f"3) Grading and value-addition options, "
                      f"4) Negotiation tips with commission agents (adatiyas), "
                      f"5) e-NAM platform usage steps, "
                      f"6) Relevant Maharashtra/central govt schemes (PM-AASHA, Price Stabilisation Fund), "
                      f"7) Export potential if applicable.")
            with st.spinner("Analysing…" if not IS_MR else "विश्लेषण होत आहे…"):
                res = ask_gemini(
                    prompt,
                    context="You are a commodity marketing advisor for Maharashtra.",
                    data_context=data_ctx,
                )
            st.markdown(f'<div class="resp-box">{res}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="src-tag">{t("ai_source")}</p>', unsafe_allow_html=True)

        _sell_ai_ctx = f"Crop: {crop_ai}, qty {qty} qtl, harvest: {harvest_in}, storage: {storage}."
        cd = price_df[price_df["Crop"] == crop_ai]
        if not cd.empty:
            _sell_ai_ctx += (
                f" Dataset modal range ₹{int(cd['Modal_Price'].min())}–₹{int(cd['Modal_Price'].max())}/qtl."
            )
        finish_chip_qa(
            "s_ai",
            "sell",
            "You advise on crop sales, APMC choice, and market strategy in Maharashtra.",
            data_context=_sell_ai_ctx,
        )

        tab_ai_ask(
            "sell_t3",
            "You advise on when/where to sell, APMC, e-NAM, and storage for Maharashtra farmers.",
            data_context=f"Crop: {crop_ai}, qty: {qty} qtl, harvest: {harvest_in}, storage: {storage}",
        )

    with tab4:
        chip_row(get_chips("sell")[4:], "s3_chip")
        _s4 = st.session_state.pop("s3_chip_pending", None)
        if _s4:
            with st.spinner(t("chip_working")):
                try:
                    _rs = ask_gemini(
                        expand_chip_question(_s4, "sell"),
                        context="You are a commodity market expert for Maharashtra farmers.",
                    )
                except Exception as _ex:
                    _rs = f"❌ {_ex}"
            st.session_state.sell_msgs.append({"role": "user", "content": _s4})
            st.session_state.sell_msgs.append({"role": "assistant", "content": _rs})
        for msg in st.session_state.sell_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_in = st.chat_input(t("chat_input_sell"), key="s_chat")
        if user_in:
            st.session_state.sell_msgs.append({"role": "user", "content": user_in})
            with st.spinner(t("chip_working")):
                try:
                    r = ask_gemini(
                        user_in,
                        context="You are a commodity market expert for Maharashtra farmers.",
                    )
                except Exception as _ex:
                    r = f"❌ {_ex}"
            st.session_state.sell_msgs.append({"role": "assistant", "content": r})
            st.rerun()
        if st.session_state.sell_msgs:
            if st.button(t("clear_chat"), key="clr_s"):
                st.session_state.sell_msgs = []
                st.rerun()
