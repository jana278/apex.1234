import os
import re
import json
import base64
import unicodedata
from pathlib import Path
from PIL import Image
import io
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
import urllib.parse
from urllib.parse import urljoin, urlparse

# 1. Streamlit Initialization
import streamlit as st

st.set_page_config(
    page_title="Apex Motors | Smart Car Market",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. PyTorch & Transformers (LOCAL Vision Engine - No APIs)
try:
    import torch
    # Optimize threads to prevent Streamlit RAM crash
    torch.set_num_threads(1)
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# 3. CatBoost (Valuation Engine)
class ApexProductionValuationEngine:
    def __init__(self, model, num_cols, cat_cols, medians):
        self.model = model
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians = medians

import sys
sys.modules['__main__'].ApexProductionValuationEngine = ApexProductionValuationEngine

@st.cache_resource(show_spinner=False)
def load_catboost():
    try:
        from catboost import Pool
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apex_catboost_valuation.joblib")
        if not os.path.exists(model_path): model_path = "apex_catboost_valuation.joblib"
        if os.path.exists(model_path):
            return joblib.load(model_path), Pool
    except Exception: pass
    return None, None

full_pricing_pipeline, CB_Pool = load_catboost()

# ─── UI & CSS Setup ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_image_data(image_name="mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg", mime="image/jpeg"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, image_name),
        os.path.join(base_dir, "public", image_name),
        image_name
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    return ""

BG_IMAGE = get_image_data()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --blue: #38bdf8; --blue-dim: rgba(56,189,248,.16); --blue-glow: rgba(56,189,248,.42);
  --bg: #030712; --card-bg: rgba(10,13,20,.88);
  --text: #f1f5f9; --muted: #94a3b8; --border: rgba(56,189,248,.22);
  --rad: 16px; --shadow: 0 12px 32px rgba(0,0,0,.55);
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', sans-serif !important;
}}

.background-car {{
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: url("{BG_IMAGE}"); background-size: cover; background-position: center; opacity: .82;
}}
.background-car:before {{
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg,rgba(3,7,18,.88) 0%,rgba(3,7,18,.42) 48%,rgba(3,7,18,.88) 100%),
                linear-gradient(180deg,rgba(3,7,18,.35) 0%,rgba(3,7,18,.15) 46%,rgba(3,7,18,.92) 100%);
}}

.main .block-container {{ position: relative; z-index: 2; max-width: 1240px; padding: 2.5rem 1.25rem 5rem; }}
#MainMenu, header, footer {{visibility: hidden !important; display: none !important;}}

div[data-testid="stTextInput"] label {{ display: none; }}
div[data-testid="stTextInput"] div[data-baseweb="input"] {{
    background: linear-gradient(180deg,rgba(15,23,42,.93),rgba(3,7,18,.97));
    border: 1.2px solid rgba(56,189,248,.42);
    border-radius: 999px; padding: 6px 16px;
    box-shadow: 0 16px 48px rgba(0,0,0,.75), inset 0 1px 0 rgba(56,189,248,.18);
}}
div[data-testid="stTextInput"] input {{ color: #fff; font-size: 1.05rem; }}
div[data-testid="stFormSubmitButton"] button {{
    background: rgba(56,189,248,.16); border: 1px solid #38bdf8; color: #fff;
    border-radius: 999px; font-weight: 700; transition: all .2s; margin-top: 10px;
}}
div[data-testid="stFormSubmitButton"] button:hover {{ background: rgba(56,189,248,.42); box-shadow: 0 0 14px rgba(56,189,248,.42); color:#fff; }}
div[data-testid="stFileUploader"] label {{ display: none; }}

.hero {{ text-align: center; margin-bottom: 1.5rem; }}
.hero-pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; margin-bottom: 14px; border: 1px solid rgba(56,189,248,.28); border-radius: 999px; background: rgba(15,23,42,.72); color: #e2e8f0; font-size: .72rem; font-weight: 800; letter-spacing: 2px; backdrop-filter: blur(10px); }}
.hero h1 {{ color: #fff; font-size: clamp(2.6rem,5vw,4.2rem); font-weight: 800; letter-spacing: -2px; line-height: 1.1; margin: 0; }}
.hero h1 span {{ color: var(--blue); text-shadow: 0 0 30px rgba(56,189,248,.5); }}
.hero-line {{ width: 54px; height: 3px; background: var(--blue); border-radius: 99px; margin: 14px auto 0; box-shadow: 0 0 20px var(--blue-glow); }}
.hero-sub {{ color: #d1d5db; font-size: .95rem; max-width: 620px; margin: 12px auto 0; line-height: 1.6; }}

.feat-row {{ display: grid; grid-template-columns: repeat(4,1fr); max-width: 760px; margin: 18px auto 24px; }}
.feat-item {{ text-align: center; padding: 5px 16px; border-right: 1px solid rgba(255,255,255,.12); }}
.feat-item:last-child {{ border-right: none; }}
.feat-icon {{ color: var(--blue); font-size: 1rem; margin-bottom: 3px; }}
.feat-title {{ color: #fff; font-size: .76rem; font-weight: 700; }}
.feat-desc {{ color: #9ca3af; font-size: .64rem; margin-top: 2px; }}

.grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; max-width: 1200px; margin: 20px auto; }}
@media(max-width:992px){{ .grid {{ grid-template-columns: repeat(2,1fr); }} }}
@media(max-width:640px){{ .grid {{ grid-template-columns: 1fr; }} }}

.card {{ background: var(--card-bg); border: 1.2px solid var(--border); border-radius: var(--rad); overflow: hidden; display: flex; flex-direction: column; box-shadow: var(--shadow); backdrop-filter: blur(14px); transition: transform .22s, border-color .22s; }}
.card:hover {{ transform: translateY(-3px); border-color: rgba(56,189,248,.5); box-shadow: 0 18px 44px rgba(56,189,248,.14); }}
.gallery {{ position: relative; width: 100%; height: 176px; background: #080c14; overflow: hidden; flex-shrink: 0; }}
.gallery-img {{ width: 100%; height: 100%; object-fit: cover; transition: transform .3s
