"""
MMCOE Kisan Sakha — deployment-ready Streamlit app.
Set GOOGLE_API_KEY in the environment (from Google AI Studio). Optional: DATA_GOV_IN_API_KEY for live mandi merge.
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
Onion,कांदा,Pune,Pune,900,1250,1600,Kharif,7200
Onion,कांदा,Solapur,Solapur,700,980,1250,Kharif,5400
Onion,कांदा,Ahmednagar,Rahuri,820,1120,1450,Kharif,4300
Onion,कांदा,Nashik,Lasalgaon,1200,1850,2400,Rabi,18000
Onion,कांदा,Nashik,Pimpalgaon,1100,1720,2200,Rabi,14200
Onion,कांदा,Pune,Pune,1300,1950,2600,Rabi,9800
Onion,कांदा,Solapur,Solapur,1000,1580,2100,Rabi,6500
Onion,कांदा,Ahmednagar,Rahuri,1150,1750,2350,Rabi,7100
Cotton,कापूस,Nagpur,Nagpur,6000,6400,6800,Kharif,8200
Cotton,कापूस,Amravati,Amravati,5800,6200,6600,Kharif,11500
Cotton,कापूस,Yavatmal,Yavatmal,5900,6300,6700,Kharif,13200
Cotton,कापूस,Wardha,Wardha,6100,6500,6900,Kharif,7800
Cotton,कापूस,Akola,Akola,5750,6150,6550,Kharif,9600
Cotton,कापूस,Buldhana,Khamgaon,5900,6250,6600,Kharif,8100
Soybean,सोयाबीन,Latur,Latur,3800,4200,4600,Kharif,14000
Soybean,सोयाबीन,Osmanabad,Osmanabad,3700,4100,4500,Kharif,9300
Soybean,सोयाबीन,Nanded,Nanded,3750,4150,4550,Kharif,8700
Soybean,सोयाबीन,Jalna,Jalna,3900,4300,4700,Kharif,7200
Soybean,सोयाबीन,Aurangabad,Aurangabad,4000,4400,4800,Kharif,6100
Tur Dal,तूर डाळ,Latur,Latur,5500,6800,7500,Kharif,4200
Tur Dal,तूर डाळ,Osmanabad,Osmanabad,5300,6600,7300,Kharif,3800
Tur Dal,तूर डाळ,Nanded,Nanded,5400,6700,7400,Kharif,2900
Tur Dal,तूर डाळ,Solapur,Solapur,5600,6900,7600,Kharif,3100
Tur Dal,तूर डाळ,Akola,Akola,5200,6400,7100,Kharif,2500
Wheat,गहू,Pune,Pune,2100,2350,2600,Rabi,9200
Wheat,गहू,Nashik,Nashik,2050,2300,2550,Rabi,8100
Wheat,गहू,Solapur,Solapur,2000,2250,2500,Rabi,7400
Wheat,गहू,Aurangabad,Aurangabad,2080,2320,2570,Rabi,6800
Wheat,गहू,Nagpur,Nagpur,2150,2400,2650,Rabi,5500
Sugarcane,ऊस,Kolhapur,Kolhapur,3400,3650,3900,Annual,25000
Sugarcane,ऊस,Satara,Satara,3300,3550,3800,Annual,21000
Sugarcane,ऊस,Sangli,Sangli,3350,3600,3850,Annual,18500
Sugarcane,ऊस,Pune,Pune,3200,3450,3700,Annual,15000
Sugarcane,ऊस,Solapur,Solapur,3100,3350,3600,Annual,12000
Groundnut,भुईमूग,Solapur,Solapur,4500,5200,5900,Kharif,5800
Groundnut,भुईमूग,Latur,Latur,4400,5100,5800,Kharif,4200
Groundnut,भुईमूग,Osmanabad,Osmanabad,4300,5000,5700,Kharif,3600
Groundnut,भुईमूग,Ahmednagar,Ahmednagar,4600,5300,6000,Kharif,4900
Jowar,ज्वारी,Solapur,Solapur,2200,2550,2900,Rabi,6200
Jowar,ज्वारी,Latur,Latur,2100,2450,2800,Rabi,5100
Jowar,ज्वारी,Osmanabad,Osmanabad,2050,2400,2750,Rabi,4300
Bajra,बाजरी,Nashik,Nashik,1900,2200,2500,Kharif,4800
Bajra,बाजरी,Ahmednagar,Ahmednagar,1850,2150,2450,Kharif,3900
Tomato,टोमॅटो,Nashik,Lasalgaon,600,950,1600,Kharif,8500
Tomato,टोमॅटो,Pune,Pune,700,1100,1800,Kharif,6200
Grapes,द्राक्षे,Nashik,Nashik,3000,4500,6500,Annual,15000
Grapes,द्राक्षे,Sangli,Sangli,2800,4200,6200,Annual,9800
Pomegranate,डाळिंब,Solapur,Solapur,4000,5800,7500,Annual,7200
Pomegranate,डाळिंब,Sangli,Sangli,3800,5500,7200,Annual,5400
Chickpea,हरभरा,Latur,Latur,4800,5200,5700,Rabi,8900
Chickpea,हरभरा,Osmanabad,Osmanabad,4700,5100,5600,Rabi,7200
Chickpea,हरभरा,Nanded,Nanded,4750,5150,5650,Rabi,6100
Rice,तांदूळ,Raigad,Pen,1800,2100,2400,Kharif,11000
Rice,तांदूळ,Ratnagiri,Ratnagiri,1900,2200,2500,Kharif,8500
Rice,तांदूळ,Sindhudurg,Kudal,1850,2150,2450,Kharif,6200
Rice,तांदूळ,Thane,Bhiwandi,1750,2050,2350,Kharif,7800
Rice,तांदूळ,Palghar,Dahanu,1780,2080,2380,Kharif,5400
Rice,तांदूळ,Chandrapur,Chandrapur,1600,1950,2300,Kharif,9200
Rice,तांदूळ,Gadchiroli,Gadchiroli,1550,1880,2200,Kharif,4800
Rice,तांदूळ,Gondia,Gondia,1680,2000,2320,Kharif,7600
Rice,तांदूळ,Bhandara,Bhandara,1650,1980,2280,Kharif,6900
Rice,तांदूळ,Dhule,Dhule,1720,2040,2360,Kharif,7100
Rice,तांदूळ,Jalgaon,Jalgaon,1700,2020,2340,Kharif,8800
Rice,तांदूळ,Nandurbar,Nandurbar,1680,1990,2280,Kharif,5500
Rice,तांदूळ,Hingoli,Hingoli,1620,1940,2240,Kharif,4100
Rice,तांदूळ,Parbhani,Parbhani,1640,1960,2260,Kharif,5200
Rice,तांदूळ,Beed,Beed,1660,1980,2280,Kharif,4700
Rice,तांदूळ,Washim,Washim,1630,1950,2250,Kharif,3900
Maize,मका,Pune,Pune,1600,1880,2150,Kharif,6200
Maize,मका,Nashik,Nashik,1580,1850,2120,Kharif,5800
Maize,मका,Dhule,Dhule,1550,1820,2080,Kharif,4900
Maize,मका,Jalna,Jalna,1570,1840,2100,Kharif,4400
Sunflower,सूर्यफूल,Latur,Latur,5200,5600,6000,Kharif,3200
Sunflower,सूर्यफूल,Osmanabad,Osmanabad,5100,5500,5900,Kharif,2800
Sesame,तीळ,Parbhani,Parbhani,8200,8800,9400,Kharif,2100
Moong,मूग,Latur,Latur,7800,8200,8600,Kharif,2600
Urad,उडीद,Osmanabad,Osmanabad,6800,7200,7600,Kharif,2200
Banana,केळी,Jalgaon,Jalgaon,800,1200,1800,Kharif,45000
Banana,केळी,Solapur,Solapur,750,1100,1650,Kharif,28000
Turmeric,हळद,Sangli,Sangli,9000,10500,12000,Kharif,8500
Turmeric,हळद,Chandrapur,Chandrapur,8800,10200,11500,Kharif,6200
Chili,मिरची,Nashik,Nashik,12000,14500,17000,Kharif,4200
Chili,मिरची,Ahmednagar,Ahmednagar,11500,13800,16200,Kharif,3800
Potato,बटाटा,Pune,Pune,900,1250,1600,Rabi,15000
Potato,बटाटा,Nashik,Nashik,880,1220,1580,Rabi,12000
Cabbage,कोबी,Pune,Pune,400,650,950,Kharif,9000
Cauliflower,फुलकोबी,Pune,Pune,500,800,1100,Kharif,7500
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

# ══════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════
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

    tab1,tab2,tab3,tab4 = st.tabs([t("tab_price"),t("tab_msp"),t("tab_aimarket"),t("tab_sellchat")])

    with tab1:
        crops_avail = sorted(price_df["Crop"].unique())
        crops_mr_avail = [price_df[price_df["Crop"]==c].iloc[0]["Crop_MR"] for c in crops_avail]
        crop_disp = crops_mr_avail if IS_MR else crops_avail

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
            st.plotly_chart(fig_price, use_container_width=True)

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
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Sora, sans-serif", size=11, color="#0d2b1a"),
                        margin=dict(l=10, r=10, t=45, b=10), height=340,
                        xaxis=dict(showgrid=False, tickangle=-35),
                        yaxis=dict(gridcolor="rgba(210,240,220,.5)"),
                    )
            except Exception as _sc_ex:
                st.caption(f"Scatter chart skipped: {_sc_ex}" if not IS_MR else f"स्कॅटर आलेख: {_sc_ex}")
                fig_arr = go.Figure()
            st.plotly_chart(fig_arr, use_container_width=True)

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
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Sora, sans-serif", size=11, color="#0d2b1a"),
                xaxis=dict(showgrid=False, tickangle=-40),
                yaxis=dict(gridcolor="rgba(210,240,220,.5)", title="₹/quintal"),
                margin=dict(l=10, r=10, t=45, b=10), height=370,
            )
        except Exception as _msp_ex:
            st.warning(f"MSP chart: {_msp_ex}" if not IS_MR else f"MSP आलेख: {_msp_ex}")
            fig_msp = go.Figure()
        st.plotly_chart(fig_msp, use_container_width=True)

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
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Sora, sans-serif", size=11, color="#0d2b1a"),
                xaxis=dict(showgrid=False, tickangle=-40),
                yaxis=dict(gridcolor="rgba(210,240,220,.5)", title="%"),
                margin=dict(l=10, r=10, t=45, b=10), height=320,
                coloraxis_showscale=False,
            )
        except Exception as _h_ex:
            st.warning(f"Hike chart: {_h_ex}" if not IS_MR else f"वाढ आलेख: {_h_ex}")
            fig_hike = go.Figure()
        st.plotly_chart(fig_hike, use_container_width=True)

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
