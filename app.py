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

try:
    from catboost import Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    Pool = None

@st.cache_resource(show_spinner=False)
def load_catboost():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apex_catboost_valuation.joblib")
    if not os.path.exists(model_path): model_path = "apex_catboost_valuation.joblib"
    if os.path.exists(model_path):
        try: return joblib.load(model_path)
        except Exception: pass
    return None

full_pricing_pipeline = load_catboost()

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
.gallery-img {{ width: 100%; height: 100%; object-fit: cover; transition: transform .3s ease; }}
.card:hover .gallery-img {{ transform: scale(1.04); }}
.no-photo {{ width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:#0b111e; color:#475569; font-size:.72rem; }}

.card-body {{ padding: 14px 16px 16px; display: flex; flex-direction: column; flex: 1; gap: 9px; }}
.card-title-row {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }}
.card-title {{ font-size: 1rem; font-weight: 700; color: #fff; line-height: 1.3; margin: 0; padding: 0; }}
.badge {{ flex-shrink: 0; font-weight: 700; padding: 3px 9px; border-radius: 20px; font-size: .71rem; white-space: nowrap; }}
.badge-great {{ background: rgba(34,197,94,.14); border: 1px solid rgba(34,197,94,.6); color: #86efac; }}
.badge-over {{ background: rgba(239,68,68,.14); border: 1px solid rgba(239,68,68,.6); color: #fca5a5; }}
.badge-fair {{ background: rgba(56,189,248,.14); border: 1px solid rgba(56,189,248,.6); color: #bae6fd; }}
.badge-none {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.15); color: #94a3b8; }}

.specs {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: .78rem; }}
.price-area {{ display: flex; justify-content: space-between; align-items: flex-end; padding-top: 10px; border-top: 1px solid rgba(255,255,255,.07); margin-top: auto; }}
.price-label {{ color: var(--muted); font-size: .71rem; display: block; margin-bottom: 2px; }}
.price-val {{ color: #fff; font-size: 1.1rem; font-weight: 800; }}
.est-val {{ color: var(--blue); font-size: .98rem; font-weight: 700; }}

.view-btn {{ display: block; width: 100%; text-align: center; background: var(--blue-dim); border: 1px solid var(--blue); color: #fff; padding: 8px 14px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: .83rem; margin-top: 10px; }}
.view-btn:hover {{ background: rgba(56,189,248,.28); box-shadow: 0 0 14px rgba(56,189,248,.32); color: #fff; text-decoration: none; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="background-car"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero">
    <div class="hero-pill">✦ SMART CAR MARKET</div>
    <h1>Apex <span>Motors</span></h1>
    <div class="hero-line"></div>
    <p class="hero-sub">Find the right car, get expert insights, and make smarter decisions with the power of AI.</p>
</div>
<div class="feat-row">
    <div class="feat-item"><div class="feat-icon">⌁</div><div class="feat-title">Analysis</div><div class="feat-desc">Understand your needs</div></div>
    <div class="feat-item"><div class="feat-icon">▧</div><div class="feat-title">Image Detection</div><div class="feat-desc">Identify car details</div></div>
    <div class="feat-item"><div class="feat-icon">◇</div><div class="feat-title">Price Insights</div><div class="feat-desc">Fair market estimates</div></div>
    <div class="feat-item"><div class="feat-icon">▥</div><div class="feat-title">Smart Results</div><div class="feat-desc">Best matches for you</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Load Local PyTorch Vision Engine ────────────────────────────────────────
VISION_MODEL = "dima806/car_models_image_detection"

@st.cache_resource(show_spinner=False)
def load_vision_model():
    if not HAS_TORCH: return None, None
    try:
        proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
        mod = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
        mod.eval()
        return proc, mod
    except Exception as e:
        print("PyTorch Load Error:", e)
        return None, None

img_processor, car_vision_model = load_vision_model()

# ─── NLP Dictionary & Utilities ──────────────────────────────────────────────
ARABIC_TO_ENG_BRAND = {
    "كيا": "kia", "kia": "kia", "هيونداي": "hyundai", "hyundai": "hyundai", "هيوانداي": "hyundai",
    "نيسان": "nissan", "nissan": "nissan", "مرسيدس": "mercedes", "mercedes": "mercedes", "mercedes-benz": "mercedes", "مرسيدس-بنز": "mercedes",
    "تويوتا": "toyota", "toyota": "toyota", "بي ام دبليو": "bmw", "bmw": "bmw", "بي ام": "bmw", "بي إم": "bmw",
    "ام جي": "mg", "mg": "mg", "إم جي": "mg", "سوبارو": "subaru", "subaru": "subaru",
    "بورش": "porsche", "بورشه": "porsche", "porsche": "porsche", "أوبل": "opel", "اوبل": "opel", "opel": "opel",
    "رينو": "renault", "renault": "renault", "أودي": "audi", "اودي": "audi", "audi": "audi",
    "شيفروليه": "chevrolet", "شفروليه": "chevrolet", "chevrolet": "chevrolet", "بيجو": "peugeot", "peugeot": "peugeot",
    "شيري": "chery", "chery": "chery", "سكودا": "skoda", "skoda": "skoda", "فولكس": "volkswagen", "فولكس فاجن": "volkswagen", "vw": "volkswagen", "volkswagen": "volkswagen",
    "فيات": "fiat", "fiat": "fiat", "سيات": "seat", "seat": "seat", "فورد": "ford", "ford": "ford",
    "سوزوكي": "suzuki", "suzuki": "suzuki", "جيب": "jeep", "jeep": "jeep", "بي واي دي": "byd", "byd": "byd",
    "هوندا": "honda", "honda": "honda", "مازدا": "mazda", "mazda": "mazda",
    "ستروين": "citroen", "سيتروين": "citroen", "citroen": "citroen", "هافال": "haval", "haval": "haval",
    "شانجان": "changan", "changan": "changan", "جيلي": "geely", "geely": "geely"
}

MODEL_ARABIC_MAP = {
    "سبورتاج": "sportage", "sportage": "sportage", "النترا": "elantra", "elantra": "elantra", "hd": "elantra", "cn7": "elantra", "ad": "elantra",
    "صني": "sunny", "sunny": "sunny", "توسان": "tucson", "tucson": "tucson", "كورولا": "corolla", "corolla": "corolla",
    "سي ال ايه": "cla", "cla": "cla", "سي 180": "c180", "c180": "c180", "c 180": "c180", "c-class": "c180",
    "بي ار زد": "brz", "brz": "brz", "911": "911", "e200": "e200", "e250": "e250", "سيراتو": "cerato", "cerato": "cerato", "k3": "cerato",
    "اكسنت": "accent", "أكسنت": "accent", "accent": "accent", "rb": "accent", "ياريس": "yaris", "يارس": "yaris", "yaris": "yaris",
    "فورتشنر": "fortuner", "fortuner": "fortuner", "أوكتافيا": "octavia", "اوكتافيا": "octavia", "octavia": "octavia",
    "باست": "passat", "باسات": "passat", "passat": "passat", "جولف": "golf", "golf": "golf", "تيجوان": "tiguan", "tiguan": "tiguan",
    "ميجان": "megane", "megane": "megane", "لوجان": "logan", "logan": "logan", "داستر": "duster", "duster": "duster",
    "سينترا": "sentra", "سنترا": "sentra", "sentra": "sentra", "قشقاي": "qashqai", "qashqai": "qashqai",
    "لانسر": "lancer", "lancer": "lancer", "أوبترا": "optra", "اوبترا": "optra", "optra": "optra", "أفيو": "aveo", "افيو": "aveo", "aveo": "aveo",
    "كروز": "cruze", "cruze": "cruze", "301": "301", "3008": "3008", "2008": "2008", "5008": "5008", "zs": "zs", "mg6": "mg6", "mg5": "mg5", "rx5": "rx5",
    "أريزو": "arrizo", "اريزو": "arrizo", "arrizo": "arrizo", "tiggo": "tiggo", "تيجو": "tiggo"
}

LOCATION_MAP = {
    "تجمع": "New Cairo", "new cairo": "New Cairo", "tagamo3": "New Cairo", "القاهرة الجديدة": "New Cairo",
    "القاهرة": "Cairo", "cairo": "Cairo",
    "مدينة نصر": "Nasr City", "nasr": "Nasr City", "nasr city": "Nasr City",
    "مصر الجديدة": "Heliopolis", "heliopolis": "Heliopolis",
    "المعادي": "Maadi", "maadi": "Maadi",
    "زايد": "Sheikh Zayed", "الشيخ زايد": "Sheikh Zayed", "zayed": "Sheikh Zayed",
    "اكتوبر": "6th of October", "october": "6th of October", "oct": "6th of October", "6 اكتوبر": "6th of October",
    "اسكندرية": "Alexandria", "الإسكندرية": "Alexandria", "alexandria": "Alexandria", "alex": "Alexandria",
    "الجيزة": "Giza", "giza": "Giza",
    "المهندسين": "Mohandessin", "mohandessin": "Mohandessin"
}

def normalize_arabic(text: str) -> str:
    if not text: return ""
    t = unicodedata.normalize("NFKC", text.lower().strip())
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ة\b", "ه", t)
    t = re.sub(r"ى\b", "ي", t)
    t = t.replace("ونص", ".5").replace("وربع", ".25").replace("وتلت", ".33")
    t = t.replace("باكو", " الف").replace("ارنب", " مليون").replace("أرنب", " مليون")
    return t

def extract_budget(text: str):
    def scale(val, unit):
        if not unit: return val if val >= 10000 else val * 1000000
        u = unit.lower()
        if u in ("m", "million", "مليون"): return val * 1000000
        if u in ("k", "thousand", "الف", "ألف"): return val * 1000
        return val
    m = re.search(r'(?:من|from)?\s*(\d+(?:\.\d+)?)\s*(m|million|k|thousand|مليون|الف)?\s*(?:-|to|حتى|لحد|الى|لـ)\s*(\d+(?:\.\d+)?)\s*(m|million|k|thousand|مليون|الف)?', text)
    if m:
        u_fin = m.group(4) or m.group(2)
        v1, v2 = scale(float(m.group(1)), u_fin), scale(float(m.group(3)), u_fin)
        return min(v1, v2), max(v1, v2)
    m = re.search(r'(?:تحت|اقل من|under|below|<)\s*(\d+(?:\.\d+)?)\s*(m|million|k|thousand|مليون|الف)?', text)
    if m: return None, scale(float(m.group(1)), m.group(2))
    m = re.search(r'(?:بـ|ب|معايا|around|for)\s*(\d+(?:\.\d+)?)\s*(m|million|k|thousand|مليون|الف)', text)
    if m:
        v = scale(float(m.group(1)), m.group(2))
        return v * 0.85, v * 1.15
    return None, None

def parse_search_query(user_query: str):
    q_clean = normalize_arabic(user_query)
    detected_brand = None
    detected_model = None
    detected_location = None
    
    for ar, en in ARABIC_TO_ENG_BRAND.items():
        if re.search(r'\b' + re.escape(ar) + r'\b', q_clean) or ar in q_clean:
            detected_brand = en; break
    for ar, en in MODEL_ARABIC_MAP.items():
        if re.search(r'\b' + re.escape(ar) + r'\b', q_clean) or ar in q_clean:
            detected_model = en; break
    for loc_ar, loc_en in LOCATION_MAP.items():
        if loc_ar in q_clean:
            detected_location = loc_en; break

    min_p, max_p = extract_budget(q_clean)
    words = [w for w in re.findall(r'[a-zA-Z0-9]+', q_clean) if not w.isdigit()]
    if not detected_brand and words: detected_brand = words[0]
    if not detected_model and len(words) > 1 and words[0] == detected_brand: detected_model = words[1]

    return detected_brand, detected_model, detected_location, min_p, max_p

# ─── Pure Local PyTorch Engine ──────────────────────────────────────────────
KNOWN_MULTIWORD_BRANDS = ["alfa romeo", "land rover", "aston martin", "mercedes benz", "rolls royce"]
BRAND_RENAME = {"mercedes-benz": "mercedes", "vw": "volkswagen", "chevy": "chevrolet", "alfa-romeo": "alfa romeo"}

def classify_car_image(image_bytes) -> str:
    if not HAS_TORCH or car_vision_model is None or img_processor is None:
        return ""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = img_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            logits = car_vision_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
        top_probs, top_indices = torch.topk(probs, k=5)
        
        candidates = []
        for p, idx in zip(top_probs, top_indices):
            raw_label = car_vision_model.config.id2label[idx.item()].replace("_", " ").strip().lower()
            candidates.append(raw_label)
            
        if candidates:
            sel_model = candidates[0]
            if any(k in " ".join(candidates) for k in ["subaru brz", "toyota gr86", "toyota gt86", "toyota 86"]):
                for c in candidates:
                    if any(k in c for k in ["subaru brz", "toyota gr86", "toyota gt86", "toyota 86"]):
                        sel_model = c; break
            
            clean_model = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", sel_model, flags=re.IGNORECASE).strip()
            
            for mb in KNOWN_MULTIWORD_BRANDS:
                if clean_model.startswith(mb):
                    b = BRAND_RENAME.get(mb, mb)
                    m = clean_model[len(mb):].strip()
                    return f"{b} {m}".strip()
            
            tokens = clean_model.split()
            brand = BRAND_RENAME.get(tokens[0], tokens[0])
            model = " ".join(tokens[1:]) if len(tokens) > 1 else ""
            return f"{brand} {model}".strip()
    except Exception as e:
        print("Local PyTorch Vision Error:", e)
    return ""

# ─── Robust Scraper Engine (With Proxy Bypass for Streamlit Cloud) ─────────
class MarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        }
        self.h_base = "https://eg.hatla2ee.com"

    def fetch_html(self, url):
        try:
            r = self.session.get(url, headers=self.headers, timeout=8)
            if r.status_code == 200 and "cloudflare" not in r.text.lower() and "attention required" not in r.text.lower():
                return r.text
        except: pass
        
        try:
            proxy_url = f"https://api.allorigins.win/get?url={urllib.parse.quote(url)}"
            r = requests.get(proxy_url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if data.get("contents") and "cloudflare" not in data["contents"].lower():
                    return data["contents"]
        except: pass
        return ""

    def scrape_hatla2ee(self, brand, model=None):
        records = []
        if not brand: return records
        b = brand.lower().strip()
        m = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model or "", flags=re.IGNORECASE).strip().replace(' ', '-')
        
        urls_to_scrape = [
            f"{self.h_base}/ar/car/{b}/{m}" if m else f"{self.h_base}/ar/car/{b}",
            f"{self.h_base}/ar/car/{b}/{m}/page/2" if m else f"{self.h_base}/ar/car/{b}/page/2"
        ]
        
        seen_urls = set()
        for url in urls_to_scrape:
            try:
                html = self.fetch_html(url)
                if not html: continue
                soup = BeautifulSoup(html, "html.parser")
                
                cards = soup.select(".listing-item, .listing-body, .car-list-item, .CarItem, .bg-card")
                if not cards:
                    cards = soup.find_all("div", class_=lambda c: c and any(x in c.lower() for x in ["card", "item", "unit", "listing"]))
                    
                for card in cards:
                    try:
                        detail_a = None
                        for a in card.find_all("a", href=True):
                            href = a["href"].strip()
                            if re.search(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', href, re.IGNORECASE) and "teraz/" not in href.lower():
                                detail_a = a
                                title_text = a.get_text(strip=True)
                                if title_text and not any(k in title_text.lower() for k in ["slide","previous","next","عرض الكل"]):
                                    break
                        if not detail_a: continue
                        
                        full_link = f"https://eg.hatla2ee.com{detail_a['href']}" if detail_a['href'].startswith('/') else detail_a['href']
                        if full_link in seen_urls: continue
                        seen_urls.add(full_link)
                        
                        image_url = ""
                        for img in card.find_all("img"):
                            src = img.get("src") or img.get("data-src") or ""
                            if src and not any(x in src for x in ["logo","icon","agency"]):
                                image_url = src; break
                                
                        card_text_space = (card.get_text(" ", strip=True) or "").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
                        
                        price = None
                        p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E|egp)', card_text_space, re.IGNORECASE)
                        if p_match:
                            try:
                                p_val = float(p_match.group(1).replace(",","").replace(" ",""))
                                if p_val > 10000: price = p_val
                            except ValueError: pass
                        
                        year = None
                        y_match = re.search(r"\b(19\d{2}|20\d{2})\b", card_text_space)
                        if y_match: year = int(y_match.group(1))
                        
                        mileage = None
                        km_match = re.search(r"([\d,]{1,8})\s*(?:کم|كم|km|كيلومتر|كيلو)", card_text_space, re.IGNORECASE)
                        if km_match:
                            try: mileage = float(km_match.group(1).replace(",","").strip())
                            except ValueError: pass
                        elif re.search(r"\b0\s*(?:کم|كم|km)\b", card_text_space, re.IGNORECASE):
                            mileage = 0.0
                            
                        loc_name = "Cairo"
                        for k, v in LOCATION_MAP.items():
                            if k in card_text_space.lower():
                                loc_name = v; break
                                
                        records.append({
                            "name": title_text or f"{brand.title()} {model.title() if model else ''} {year or ''}".strip(), 
                            "brand": brand.title(), "model": model.title() if model else "Model",
                            "price": price, "year": year if year else 2024, "mileage": mileage, 
                            "location": loc_name, 
                            "transmission": "Manual" if any(t in card_text_space for t in ["يدوي","مانيوال","Manual"]) else "Automatic",
                            "condition_tag": "Factory Paint" if any(k in card_text_space for k in ["فابريكا", "فبريكة", "زيرو", "factory paint"]) else "Normal", 
                            "trim_tier": "Topline" if any(k in card_text_space.lower() for k in ["اعلى فئة", "توب لاين", "topline", "بانوراما"]) else "Standard",
                            "source": "Hatla2ee", "item_url": full_link, "image_url": image_url
                        })
                    except Exception: pass
            except Exception as e: print("Hatla2ee Loop Exception:", e)
        return records

    def scrape_dubizzle(self, brand, model=None):
        records = []
        if not brand: return records
        b = brand.lower().strip()
        m = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", model or "", flags=re.IGNORECASE).strip()
        query_str = f"{b} {m}".strip() if m else b
        search_url = f"https://www.dubizzle.com.eg/vehicles/cars-for-sale/?q={urllib.parse.quote(query_str)}"
        try:
            html = self.fetch_html(search_url)
            if not html: return records
            soup = BeautifulSoup(html, "html.parser")
            next_data = soup.select_one("script#__NEXT_DATA__")
            if next_data and next_data.string:
                payload = json.loads(next_data.string)
                raw_ads = payload.get("props", {}).get("pageProps", {}).get("initialState", {}).get("feed", {}).get("data", [])
                for item in raw_ads:
                    title = (item.get("title") or item.get("name") or "").strip()
                    if b not in title.lower(): continue
                    price_val = item.get("price", {}).get("value")
                    if not price_val: continue
                    loc_raw = item.get("location", {}).get("name", "").lower()
                    
                    loc_name = "Cairo"
                    for k, v in LOCATION_MAP.items():
                        if k in loc_raw: loc_name = v; break
                            
                    ad_id = item.get("id", "")
                    ad_url = f"https://www.dubizzle.com.eg/ad/{ad_id}" if ad_id else search_url
                    y_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                    year = int(y_match.group(1)) if y_match else 2024
                    
                    img_url = ""
                    images = item.get("images", [])
                    if images: img_url = images[0].get("url", "")

                    records.append({
                        "name": title, "brand": brand.title(), "model": model.title() if model else "Model",
                        "price": float(price_val), "year": year, "mileage": 35000.0,
                        "location": loc_name, "transmission": "Automatic",
                        "condition_tag": "Factory Paint" if any(k in title for k in ["فابريك", "فبريك", "وكالة", "زيرو", "factory paint"]) else "Normal",
                        "trim_tier": "Standard", "source": "Dubizzle", "item_url": ad_url, "image_url": img_url
                    })
        except Exception: pass
        return records

scraper = MarketScraper()

def calculate_match_score(brand, model, loc_filter, price_filter, row):
    base = 96.0
    if loc_filter and loc_filter.lower() in str(row.get("location", "")).lower():
        base += 3.0
    if price_filter:
        diff_ratio = abs(row["price"] - price_filter) / price_filter
        if diff_ratio < 0.15: base += 2.0
    return min(round(base + np.random.uniform(0.1, 0.9), 1), 99.5)

def run_hybrid_search(user_query, uploaded_file):
    detected_car = ""
    if uploaded_file:
        try:
            detected_car = classify_car_image(uploaded_file.getvalue())
        except Exception as e: print("Vision Failed:", e)
    
    combined_query = f"{detected_car} {user_query}".strip()
    brand, model, location, min_p, max_p = parse_search_query(combined_query)
    
    if not brand: return pd.DataFrame(), combined_query
    
    h_ads = scraper.scrape_hatla2ee(brand, model)
    o_ads = scraper.scrape_dubizzle(brand, model)
    combined = h_ads + o_ads
    
    if not combined: return pd.DataFrame(), combined_query
    
    df = pd.DataFrame(combined)
    
    # ─── Strict NLP Filtering ────────────────────────────────────────────────
    if location:
        df = df[df["location"].astype(str).str.lower().str.contains(location.lower())]
    if min_p is not None:
        df = df[df["price"].notna() & (df["price"] >= min_p)]
    if max_p is not None:
        df = df[df["price"].notna() & (df["price"] <= max_p)]
    if any(k in combined_query.lower() for k in ["فابريكا", "فبريكة", "زيرو", "factory paint"]):
        df = df[df["condition_tag"] == "Factory Paint"]
    year_match = re.search(r"\b(20[012]\d|19\d{2})\b", combined_query)
    if year_match:
        df = df[df["year"] == int(year_match.group(1))]
        
    if df.empty: return df, combined_query
    # ─────────────────────────────────────────────────────────────────────────

    target_price = min_p if min_p else (max_p if max_p else None)
    if target_price:
        df["price_dist"] = (df["price"] - target_price).abs()
        df = df.sort_values("price_dist", ascending=True)
        
    predicted = []
    if full_pricing_pipeline is not None and HAS_CATBOOST and Pool is not None:
        try:
            eval_df = df.copy()
            num_cols = ["year", "mileage", "car_age", "km_per_year"]
            cat_cols = ["brand", "model", "location", "transmission", "fuel_type", "car_condition", "condition_tag", "trim_tier"]
            
            eval_df['year'] = pd.to_numeric(eval_df.get('year'), errors='coerce').fillna(2016.0)
            eval_df['mileage'] = pd.to_numeric(eval_df.get('mileage'), errors='coerce').fillna(122000.0)
            eval_df['car_age'] = (2026 - eval_df['year']).clip(lower=0)
            eval_df['km_per_year'] = np.where(eval_df['car_age'] > 0, eval_df['mileage'] / eval_df['car_age'].replace(0, 1), eval_df['mileage'])
            for c in cat_cols: eval_df[c] = eval_df.get(c, "Missing").fillna("Missing").astype(str).str.title()
            
            pool = Pool(eval_df[num_cols + cat_cols], cat_features=cat_cols)
            preds_log = full_pricing_pipeline.model.predict(pool)
            preds_egp = np.expm1(preds_log)
            predicted = [float(np.round(p, 0)) if p > 0 else None for p in preds_egp]
        except Exception as e: 
            print("Valuation error:", e)
            predicted = [None] * len(df)
    else:
        for _, r in df.iterrows():
            base = {"mercedes":3000000,"bmw":2900000,"audi":2800000,"kia":1800000,"hyundai":1700000,"toyota":1750000}.get(str(r.get("brand")).lower(), 1500000)
            age = max(0, 2026 - int(r.get("year", 2024)))
            predicted.append(float(round(base * (0.925 ** age), 0)))
            
    df["predicted_fair_price"] = predicted
    
    deal_labels = []
    for _, r in df.iterrows():
        p, f = r.get("price"), r.get("predicted_fair_price")
        if p and f:
            pct = (p - f) / f
            if pct <= -0.05: deal_labels.append("Great Deal 🔥")
            elif pct >= 0.08: deal_labels.append("Overpriced ⚠️")
            else: deal_labels.append("Fair Price ⚖")
        else: deal_labels.append("N/A")
    df["deal_label"] = deal_labels
    
    scores = []
    for _, r in df.iterrows():
        scores.append(calculate_match_score(brand, model, location, target_price, r))
    df["match_score"] = scores
    
    return df.head(12), combined_query

# ─── Streamlit UI Forms & Execution ──────────────────────────────────────────
with st.form("search_form", clear_on_submit=False):
    c_in, c_up = st.columns([0.85, 0.15])
    with c_in:
        user_query = st.text_input("Search", placeholder="e.g. Mercedes CLA 2022 factory paint in Zayed under 3 million", label_visibility="collapsed")
    with c_up:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    submitted = st.form_submit_button("Search Market", use_container_width=True)

if submitted or user_query or uploaded_file:
    with st.spinner("Analyzing and fetching market data..."):
        df_res, final_q = run_hybrid_search(user_query, uploaded_file)
        
    st.markdown(f'<div style="color:#fff; font-size:1.15rem; font-weight:700; margin:35px auto 15px; max-width:1200px;">🎯 Live Market Results for: "{final_q}"</div>', unsafe_allow_html=True)
    
    if df_res.empty:
        st.info("No matching vehicle listings found for your search criteria. (If using strict filters like year or location, try removing them)")
    else:
        html_cards = ""
        for _, r in df_res.iterrows():
            deal = r.get('deal_label', 'N/A')
            if "Great Deal" in deal:
                badge_class = "badge-great"
                badge_text = "🟢 Great Deal"
            elif "Overpriced" in deal:
                badge_class = "badge-over"
                badge_text = "🔴 Overpriced"
            elif "Fair" in deal:
                badge_class = "badge-fair"
                badge_text = "⚖️ Fair Price"
            else:
                badge_class = "badge-none"
                badge_text = "Valuation N/A"

            km_val = r.get('mileage')
            km_text = f"{float(km_val):,.0f} km" if km_val else "Not provided"
            price_val = r.get('price')
            price_text = f"{float(price_val):,.0f} EGP" if price_val else "On request"
            fair_val = r.get('predicted_fair_price')
            fair_text = f"{float(fair_val):,.0f} EGP" if fair_val else "N/A"
            img_html = f'<img class="gallery-img" src="{r.get("image_url")}" onerror="this.style.display=\'none\'">' if r.get("image_url") else '<div class="no-photo">Photo unavailable</div>'
            link = r.get("item_url", "#")
            
            card_html = f"""<div class="card">
<div class="gallery">{img_html}</div>
<div class="card-body">
<div class="card-title-row">
<h2 class="card-title">{r['name']}</h2>
<span class="badge {badge_class}">{badge_text}</span>
</div>
<div class="specs">
<span>📅 {r.get('year', '')}</span>
<span>⚙ {r.get('transmission', 'Auto')}</span>
<span>🛣️ {km_text}</span>
<span>📍 {r.get('location', '')}</span>
<span>⚡ Match: {r.get('match_score', '')}%</span>
<span>🏷️ {r.get('condition_tag', '')}</span>
</div>
<div class="price-area">
<div>
<span class="price-label">Listed Price</span>
<span class="price-val">{price_text}</span>
</div>
<div style="text-align:right">
<span class="price-label">Fair Price (AI Est.)</span>
<span class="est-val">{fair_text}</span>
</div>
</div>
<a href="{link}" target="_blank" rel="noopener noreferrer" class="view-btn">View on {r.get('source', 'Marketplace')} ↗</a>
</div>
</div>"""
            html_cards += card_html
            
        st.markdown(f'<div class="grid">{html_cards}</div>', unsafe_allow_html=True)
