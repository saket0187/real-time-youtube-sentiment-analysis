from dotenv import load_dotenv
import os
import time
import requests
import streamlit as st
from googleapiclient.discovery import build
from google.cloud import storage
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from io import BytesIO
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from streamlit_autorefresh import st_autorefresh

# ─── Load .env & configure Gemini ─────────────────────────────────────────────
load_dotenv()
gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not gemini_key:
    st.error("🔑 Gemini API key missing. Set GEMINI_API_KEY in .env or Streamlit secrets.")
    st.stop()
genai.configure(api_key=gemini_key)

# ─── Streamlit page setup ─────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Sentiment Dashboard", page_icon="🎬", layout="wide")

# ─── Enhanced Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  
  /* Global Styles */
  html, body, [class*="st-"] { 
    font-family: 'Inter', sans-serif !important; 
  }
  
  .main > div {
    padding-top: 2rem;
  }
  
  /* Remove white bars/containers */
  .main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
  }
  
  /* Hide default streamlit header/footer */
  header[data-testid="stHeader"] {
    display: none !important;
  }
  
  .stApp > header {
    display: none !important;
  }
  
  /* Remove default streamlit margins */
  .main .block-container {
    max-width: 100%;
    padding-left: 2rem;
    padding-right: 2rem;
  }
  
  /* Background */
  .stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
  }
  
/* Header Styles */
.main-header { 
    background: linear-gradient(135deg, 
        rgba(15,15,35,0.95) 0%, 
        rgba(25,25,55,0.98) 25%,
        rgba(35,15,45,0.95) 50%,
        rgba(20,20,40,0.92) 100%);
    backdrop-filter: blur(25px);
    border-radius: 25px;
    padding: 40px;
    margin-bottom: 30px;
    box-shadow: 
        0 25px 50px rgba(0,0,0,0.3),
        0 0 0 1px rgba(100,200,255,0.3),
        inset 0 1px 0 rgba(255,255,255,0.2),
        0 0 60px rgba(0,150,255,0.15);
    border: 2px solid rgba(100,200,255,0.4);
    display: flex;
    align-items: center;
    gap: 30px;
    animation: slideInDown 0.8s ease-out, headerPulse 3s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

/* Animated background with AI-themed colors */
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: 
        radial-gradient(circle at 25% 25%, rgba(0,200,255,0.08) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(150,0,255,0.06) 0%, transparent 50%),
        radial-gradient(circle at 50% 10%, rgba(255,0,150,0.04) 0%, transparent 60%);
    animation: aiParticles 15s linear infinite;
    pointer-events: none;
    z-index: 1;
}

/* Floating neural network effect */
.main-header::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(0,255,200,0.1) 2px, transparent 2px),
        radial-gradient(circle at 80% 20%, rgba(255,0,200,0.1) 1px, transparent 1px),
        radial-gradient(circle at 60% 80%, rgba(100,200,255,0.1) 1.5px, transparent 1.5px),
        radial-gradient(circle at 30% 70%, rgba(200,100,255,0.1) 1px, transparent 1px);
    background-size: 100px 100px, 80px 80px, 120px 120px, 90px 90px;
    animation: neuralNetwork 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 2;
}

/* Enhanced title styling */
.main-header h1 {
    position: relative;
    z-index: 3;
    background: linear-gradient(45deg, 
        #00d4ff 0%, 
        #ff0080 25%, 
        #8000ff 50%, 
        #00ff80 75%, 
        #ff4000 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease-in-out infinite;
    font-weight: 700;
    text-shadow: 0 0 30px rgba(0,200,255,0.3);
}

/* AI Powered subtitle with enhanced effects */
.ai-powered-text {
    position: relative;
    z-index: 3;
    font-size: 1.2em;
    font-weight: 600;
    background: linear-gradient(90deg, 
        #00ff88 0%,
        #0088ff 25%,
        #8800ff 50%,
        #ff0088 75%,
        #ff8800 100%);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: aiTextFlow 3s linear infinite;
    text-transform: uppercase;
    letter-spacing: 2px;
    position: relative;
}

/* Glowing AI chip icon effect */
.ai-powered-text::before {
    content: '🧠';
    position: absolute;
    left: -30px;
    top: 50%;
    transform: translateY(-50%);
    animation: brainPulse 2s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(0,255,150,0.6));
}

/* Animated underline for AI text */
.ai-powered-text::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, 
        transparent 0%,
        #00ff88 20%,
        #0088ff 40%,
        #8800ff 60%,
        #ff0088 80%,
        transparent 100%);
    animation: underlineGlow 2s ease-in-out infinite;
}

/* Keyframe animations */
@keyframes headerPulse {
    0%, 100% { 
        box-shadow: 
            0 25px 50px rgba(0,0,0,0.3),
            0 0 0 1px rgba(100,200,255,0.3),
            inset 0 1px 0 rgba(255,255,255,0.2),
            0 0 60px rgba(0,150,255,0.15);
    }
    50% { 
        box-shadow: 
            0 30px 60px rgba(0,0,0,0.4),
            0 0 0 1px rgba(100,200,255,0.5),
            inset 0 1px 0 rgba(255,255,255,0.3),
            0 0 80px rgba(0,150,255,0.25);
    }
}

@keyframes aiParticles {
    0% { transform: rotate(0deg) scale(1); opacity: 0.8; }
    33% { transform: rotate(120deg) scale(1.1); opacity: 1; }
    66% { transform: rotate(240deg) scale(0.9); opacity: 0.6; }
    100% { transform: rotate(360deg) scale(1); opacity: 0.8; }
}

@keyframes neuralNetwork {
    0%, 100% { 
        background-position: 0% 0%, 100% 100%, 50% 50%, 25% 75%; 
        opacity: 0.3;
    }
    50% { 
        background-position: 100% 100%, 0% 0%, 75% 25%, 50% 50%; 
        opacity: 0.6;
    }
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes aiTextFlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

@keyframes brainPulse {
    0%, 100% { 
        transform: translateY(-50%) scale(1); 
        filter: drop-shadow(0 0 10px rgba(0,255,150,0.6));
    }
    50% { 
        transform: translateY(-50%) scale(1.2); 
        filter: drop-shadow(0 0 20px rgba(0,255,150,0.9));
    }
}

@keyframes underlineGlow {
    0%, 100% { opacity: 0.6; transform: scaleX(1); }
    50% { opacity: 1; transform: scaleX(1.05); }
}

@keyframes slideInDown {
    from {
        opacity: 0;
        transform: translate3d(0, -100%, 0);
    }
    to {
        opacity: 1;
        transform: translate3d(0, 0, 0);
    }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main-header {
        padding: 25px;
        gap: 20px;
        flex-direction: column;
        text-align: center;
    }
    
    .ai-powered-text::before {
        position: static;
        display: block;
        margin-bottom: 10px;
    }
}

  .main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 3s infinite;
  }

  @keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
  }

  .youtube-logo { 
    height: 90px; 
    width: auto; 
    filter: drop-shadow(0 8px 16px rgba(255,0,0,0.3));
    transition: transform 0.3s ease;
  }

  .youtube-logo:hover {
    transform: scale(1.05) rotate(2deg);
  }

  .project-title { 
    font-size: 3.5em; 
    font-weight: 900; 
    background: linear-gradient(135deg, #FF0000 0%, #FF4500 25%, #FF6B6B 50%, #CC0000 75%, #8B0000 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 4px 8px rgba(255,0,0,0.2);
    position: relative;
    animation: titleGlow 2s ease-in-out infinite alternate;
  }

  @keyframes titleGlow {
    from { filter: drop-shadow(0 0 5px rgba(255,0,0,0.3)); }
    to { filter: drop-shadow(0 0 20px rgba(255,0,0,0.6)); }
  }

  .subtitle {
    font-size: 1.3em;
    color: #555;
    font-weight: 500;
    margin-top: 15px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  @keyframes slideInDown {
    from { transform: translateY(-30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  .youtube-logo { 
    height: 80px; 
    width: auto; 
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
  }
  
  .project-title { 
    font-size: 3.2em; 
    font-weight: 800; 
    background: linear-gradient(135deg, #FF0000, #CC0000, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .subtitle {
    font-size: 1.2em;
    color: #666;
    font-weight: 400;
    margin-top: 10px;
  }
  
  /* Container Styles */
  .glass-container { 
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    animation: fadeInUp 0.6s ease-out;
  }
  
  @keyframes fadeInUp {
    from { transform: translateY(30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  /* Search Styles */
  .search-header {
    font-size: 1.8em;
    font-weight: 700;
    color: #333;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  .stTextInput > div > div > input {
    border-radius: 15px !important;
    border: 2px solid #e0e0e0 !important;
    padding: 15px 20px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
  }
  
  .stTextInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
  }
  
  /* Video Card Styles */
  .video-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.3);
  }
  
  .video-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    background: rgba(255, 255, 255, 1);
  }
  
  .video-title {
    font-size: 1.3em;
    font-weight: 700;
    color: #000000;
    margin-bottom: 8px;
  }

  /* For dashboard video title - FIXED */
  .dashboard-video-title {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 1.4em !important;
  }
  
  /* Fix for markdown links in dashboard */
  .dashboard-video-title a {
    color: #000000 !important;
    text-decoration: none !important;
  }
  
  .dashboard-video-title a:hover {
    color: #333333 !important;
    text-decoration: underline !important;
  }
  
  .video-meta {
    color: #444;
    font-size: 1em;
    margin-bottom: 15px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
  }

  .meta-item {
    display: flex;
    align-items: center;
    gap: 5px;
    background: rgba(102, 126, 234, 0.1);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.95em;
    font-weight: 600;
    color: #333;
    border: 1px solid rgba(102, 126, 234, 0.2);
    transition: all 0.3s ease;
  }

  .meta-item:hover {
    background: rgba(102, 126, 234, 0.2);
    transform: translateY(-1px);
  }

  .video-description {
    color: #555;
    font-size: 0.95em;
    line-height: 1.5;
    margin-top: 10px;
    font-weight: 400;
    background: rgba(0,0,0,0.03);
    padding: 12px 15px;
    border-radius: 10px;
    border-left: 3px solid #667eea;
  }
  
  /* Metric Cards */
  .metric-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    transition: transform 0.3s ease;
  }
  
  .metric-card:hover {
    transform: translateY(-3px);
  }
  
  .metric-value {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 5px;
  }
  
  .metric-label {
    font-size: 0.9em;
    opacity: 0.9;
  }
  
  /* Buttons */
  .stButton > button { 
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
  }
  
  .stButton > button:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    background: linear-gradient(135deg, #5a67d8, #6b46c1) !important;
  }
  
  /* Status Messages */
  .status-success {
    background: linear-gradient(135deg, #48bb78, #38a169);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
  }
  
  .status-processing {
    background: linear-gradient(135deg, #ed8936, #dd6b20);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
  }
  
  .status-error {
    background: linear-gradient(135deg, #f56565, #e53e3e);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(245, 101, 101, 0.3);
  }
  
  /* Loading Animations */
  .loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px;
  }
  
  .spinner {
    width: 60px;
    height: 60px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 20px;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  .loading-text {
    font-size: 1.2em;
    color: #667eea;
    font-weight: 600;
    text-align: center;
  }
  
  .loading-dots::after {
    content: '';
    animation: dots 1.5s steps(5, end) infinite;
  }
  
  @keyframes dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
  }
  
  /* Insights Container */
  .insights-container {
    background: linear-gradient(135deg, rgba(168, 237, 234, 0.2), rgba(254, 214, 227, 0.2));
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    border: 1px solid rgba(168, 237, 234, 0.3);
    backdrop-filter: blur(10px);
  }
  
  /* Download Section */
  .download-section {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    border: 1px solid rgba(102, 126, 234, 0.2);
  }
  
  /* Progress Bar */
  .progress-container {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
  }
  
  .progress-bar {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
  }
  
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 4px;
    animation: progress 2s ease-in-out infinite;
  }
  
  @keyframes progress {
    0% { width: 30%; }
    50% { width: 70%; }
    100% { width: 30%; }
  }
  
  /* Charts Container */
  .chart-container {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
  }
  
  /* Footer */
  .footer {
    text-align: center;
    color: rgba(255,255,255,0.8);
    padding: 30px;
    font-size: 1.1em;
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    margin-top: 40px;
    backdrop-filter: blur(10px);
  }
  
  /* Responsive adjustments */
  @media (max-width: 768px) {
    .project-title { font-size: 2.2em; }
    .glass-container { padding: 20px; }
    .main-header { padding: 20px; }
  }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
def init_session_state():
    """Centralized session state initialization - ADD THIS AT TOP OF YOUR APP"""
    defaults = {
        "search_results": [],
        "selected_video": None,
        "raw_summary": None,
        "ai_insights": None,
        "analysis_status": "idle",
        "dashboard_mode": False,
        "processing_stage": "",
        "analysis_start_time": None,
        "last_check_time": None,
        "check_count": 0,
        "last_processed_blob": None,
        "error_count": 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "raw_summary" not in st.session_state:
    st.session_state.raw_summary = None
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = None
if "analysis_status" not in st.session_state:
    st.session_state.analysis_status = "idle"  # idle, processing, complete, error
if "dashboard_mode" not in st.session_state:
    st.session_state.dashboard_mode = False
if "processing_stage" not in st.session_state:
    st.session_state.processing_stage = ""
if "analysis_start_time" not in st.session_state:
    st.session_state.analysis_start_time = None
if "auto_check_count" not in st.session_state:
    st.session_state.auto_check_count = 0
if "last_processed_blob" not in st.session_state:
    st.session_state.last_processed_blob = None

# ─── Enhanced Loading Animation ──────────────────────────────────────────────
def show_loading_animation(phase, estimate, show_time=False):
    """FIXED: Simple loading animation without complex state changes"""
    
    if show_time and st.session_state.analysis_start_time:
        elapsed = time.time() - st.session_state.analysis_start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f" ({minutes}m {seconds}s elapsed)"
    else:
        time_str = ""
    
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(255,255,255,0.1));
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 20px 0;
    ">
        <div style="font-size: 1.2em; font-weight: bold; color: #333; margin-bottom: 10px;">
            🔄 {phase}{time_str}
        </div>
        <div style="color: #666; font-size: 0.9em;">
            {estimate}
        </div>
        <div style="margin-top: 15px;">
            <div style="display: inline-block; animation: spin 1s linear infinite;">⚙️</div>
        </div>
    </div>
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)
    

# ─── Enhanced Header ──────────────────────────────────────────────────────────
def show_header():
    youtube_logo_url = "https://cdn-icons-png.flaticon.com/512/1384/1384060.png"
    st.markdown(f"""
    <div class="main-header">
      <img src="{youtube_logo_url}" class="youtube-logo" alt="YouTube Logo">
      <div>
        <div class="project-title">YouTube Sentiment Dashboard</div>
        <div class="subtitle">AI-Powered Comment Analysis & Insights</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Enhanced Search Interface ───────────────────────────────────────────────
def search_interface():
    st.markdown('''
    <div class="glass-container">
        <div class="search-header">🔍 Search YouTube Videos</div>
        <!-- Content will go here -->
    </div>
    ''', unsafe_allow_html=True)
    
    # Search form
    # Add alignment CSS
    st.markdown("""
    <style>
    .search-row {
        display: flex;
        align-items: end;
        gap: 15px;
        margin-bottom: 20px;
    }
    .search-input {
        flex: 4;
    }
    .search-select {
        flex: 1;
    }
    .search-button {
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        query = st.text_input("", key="search_query", placeholder="Enter keywords to search YouTube videos...")

    with col2:
        max_results = st.selectbox("Results", [10, 25, 50], key="search_max")

    with col3:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)  # Add spacing
        search_clicked = st.button("🔍 Search", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if search_clicked:
        if not query.strip():
            st.markdown('<div class="status-error">⚠️ Please enter a search query.</div>', unsafe_allow_html=True)
        else:
            perform_search(query, max_results)
    
    display_search_results()

def perform_search(query, max_results):
    """Enhanced search with better error handling"""
    placeholder = st.empty()
    with placeholder.container():
        show_loading_animation("Searching YouTube videos", "Connecting to YouTube API...")
    
    yt_key = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY"))
    if not yt_key:
        placeholder.markdown('<div class="status-error">❌ YouTube API key missing.</div>', unsafe_allow_html=True)
        return
    
    try:
        yt = build("youtube", "v3", developerKey=yt_key)
        resp = yt.search().list(
            q=query, 
            part="snippet", 
            type="video", 
            maxResults=max_results
        ).execute()
        
        st.session_state.search_results = [
            {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "published": item["snippet"]["publishedAt"][:10],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "description": item["snippet"]["description"]
            }
            for item in resp["items"]
        ]
        
        placeholder.markdown(f'<div class="status-success">✅ Found {len(st.session_state.search_results)} videos!</div>', unsafe_allow_html=True)
        
    except Exception as e:
        placeholder.markdown(f'<div class="status-error">❌ Search failed: {str(e)}</div>', unsafe_allow_html=True)

def display_search_results():
    """Enhanced search results display"""
    if st.session_state.search_results:
        # Display count outside the container with better styling
        st.markdown(f'''
        <div style="
            font-size: 1.4em; 
            font-weight: 700; 
            color: white; 
            margin: 20px 0 15px 0; 
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            background: rgba(255,255,255,0.1);
            padding: 12px 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        ">
            📺 Found {len(st.session_state.search_results)} Videos
        </div>
        ''', unsafe_allow_html=True)
        
        for i, video in enumerate(st.session_state.search_results):
            st.markdown('<div class="video-card">', unsafe_allow_html=True)
            
            cols = st.columns([1, 4, 1])
            
            with cols[0]:
                st.image(video["thumbnail"], width=150)
            
            with cols[1]:
                st.markdown(f'<div class="video-title">{video["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="video-meta">📺 {video["channel"]} • 📅 {video["published"]}</div>', unsafe_allow_html=True)
                description = video.get("description", "")
                if description:
                    st.markdown(f'''
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(255,255,255,0.9));
                        border-left: 4px solid #667eea;
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                        color: #333;
                        line-height: 1.5;
                    ">
                        📝 <strong>Description:</strong><br>
                        {description[:250] + ('...' if len(description) > 250 else '')}
                    </div>
                    ''', unsafe_allow_html=True)
            
            with cols[2]:
                if st.button("🚀 Analyze", key=f"select_{i}", use_container_width=True):
                    st.session_state.selected_video = video
                    st.session_state.search_results = []
                    st.session_state.dashboard_mode = True
                    st.session_state.raw_summary = None
                    st.session_state.ai_insights = None
                    st.session_state.analysis_status = "idle"
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ─── Enhanced Dashboard Interface (FIXED) ────────────────────────────────────────────
def display_search_results():
    """Enhanced search results display"""
    if st.session_state.search_results:
        # Display count outside the container with better styling
        st.markdown(f'''
        <div style="
            font-size: 1.4em; 
            font-weight: 700; 
            color: white; 
            margin: 20px 0 15px 0; 
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            background: rgba(255,255,255,0.1);
            padding: 12px 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        ">
            📺 Found {len(st.session_state.search_results)} Videos
        </div>
        ''', unsafe_allow_html=True)
        
        for i, video in enumerate(st.session_state.search_results):
            st.markdown('<div class="video-card">', unsafe_allow_html=True)
            
            cols = st.columns([1, 4, 1])
            
            with cols[0]:
                st.image(video["thumbnail"], width=150)
            
            with cols[1]:
                st.markdown(f'<div class="video-title">{video["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="video-meta">📺 {video["channel"]} • 📅 {video["published"]}</div>', unsafe_allow_html=True)
                description = video.get("description", "")
                if description:
                    st.markdown(f'''
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(255,255,255,0.9));
                        border-left: 4px solid #667eea;
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                        color: #333;
                        line-height: 1.5;
                    ">
                        📝 <strong>Description:</strong><br>
                        {description[:250] + ('...' if len(description) > 250 else '')}
                    </div>
                    ''', unsafe_allow_html=True)
            
            with cols[2]:
                if st.button("🚀 Analyze", key=f"select_{i}", use_container_width=True):
                    st.session_state.selected_video = video
                    st.session_state.search_results = []
                    st.session_state.dashboard_mode = True
                    reset_analysis_state()
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ─── STREAMLIT CLOUD SAFE Dashboard Interface ───────────────────────────────
def dashboard_interface():
    """REPLACED: Cloud-safe dashboard interface"""
    
    # Initialize session state
    init_session_state()
    
    video = st.session_state.selected_video
    
    # Back button
    if st.button("← Back to Search", key="back_button"):
        st.session_state.dashboard_mode = False
        st.session_state.selected_video = None
        reset_analysis_state()
        st.rerun()
    
    st.markdown("---")
    
    # Video info section
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 🎬 Selected Video")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(video["thumbnail"], width=250)
    
    with col2:
        # Fixed video title display with proper styling
        st.markdown(f"""
        <div class="dashboard-video-title">
            <a href="https://youtu.be/{video['video_id']}" target="_blank" style="color: #000000 !important; text-decoration: none;">
                {video['title']}
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"📺 **Channel:** {video['channel']}")
        st.write(f"📅 **Published:** {video['published']}")
        st.write(f"🔗 **Video ID:** `{video['video_id']}`")
        
        # Analysis button
        if st.session_state.analysis_status == "idle":
            if st.button("🚀 Start Sentiment Analysis", use_container_width=True, key="start_analysis"):
                success = trigger_sentiment_analysis(video['video_id'])
                if success:
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis status and results
    show_enhanced_analysis_status()
    show_analysis_results()

# Also add this debug version of the button handler
def show_enhanced_analysis_status():
    """DEBUG VERSION with detailed button feedback"""
    
    if st.session_state.analysis_status == "processing":
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        
        # Initialize time tracking safely
        if st.session_state.analysis_start_time is None:
            st.session_state.analysis_start_time = time.time()
            st.session_state.check_count = 0
        
        elapsed_time = time.time() - st.session_state.analysis_start_time
        
        # Show current status with time
        if elapsed_time < 60:
            phase = "Fetching comments from YouTube"
            estimate = "Usually takes 1-2 minutes"
        elif elapsed_time < 180:
            phase = "Analyzing sentiment with AI"
            estimate = "Processing comments..."
        else:
            phase = "Finalizing results"
            estimate = "Almost done!"
        
        show_loading_animation(phase, estimate, show_time=True)
        
        # DEBUG: Show current state
        st.write("**DEBUG INFO:**")
        st.write(f"- Analysis Status: `{st.session_state.analysis_status}`")
        st.write(f"- Selected Video: `{st.session_state.selected_video['video_id'] if st.session_state.selected_video else 'None'}`")
        st.write(f"- Last Processed Blob: `{st.session_state.last_processed_blob}`")
        st.write(f"- Elapsed Time: `{elapsed_time:.1f} seconds`")
        
        # Manual controls with debug feedback
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Check Results Now", key="manual_check", use_container_width=True):
                st.write("🔄 **Button clicked! Checking results...**")
                with st.spinner("Checking for results..."):
                    result = check_for_results()
                if result:
                    st.write("✅ **Results found! Rerunning app...**")
                    st.rerun()
                else:
                    st.write("❌ **No results yet. Try again in 1-2 minutes.**")
        
        with col2:
            if st.button("🔄 Refresh Status", key="refresh_status", use_container_width=True):
                st.write("🔄 **Refreshing page...**")
                st.rerun()
        
        with col3:
            if st.button("❌ Cancel Analysis", key="cancel_analysis", use_container_width=True):
                st.write("❌ **Cancelling analysis...**")
                reset_analysis_state()
                st.rerun()
        
        # Progress indicator
        progress_value = min(0.9, elapsed_time / 300)
        st.progress(progress_value, text=f"Analysis Progress: {int(progress_value * 100)}%")
        
        # Timeout warning
        if elapsed_time > 600:  # 10 minutes
            st.warning("⚠️ Analysis is taking longer than expected. You may want to try again.")
            if st.button("🔄 Start Fresh Analysis", key="restart_analysis"):
                reset_analysis_state()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.analysis_status == "complete":
        st.success("✅ Analysis Complete! Results are ready below.")
    
    elif st.session_state.analysis_status == "error":
        st.error("❌ Analysis failed. Please try again.")
        if st.button("🔄 Try Again", key="retry_analysis"):
            reset_analysis_state()
            st.rerun()

def reset_analysis_state():
    """REPLACED: Clean reset of all analysis state"""
    st.session_state.analysis_status = "idle"
    st.session_state.raw_summary = None
    st.session_state.ai_insights = None
    st.session_state.analysis_start_time = None
    st.session_state.last_check_time = None
    st.session_state.check_count = 0
    st.session_state.last_processed_blob = None
    st.session_state.error_count = 0
    st.session_state.processing_stage = ""

def trigger_sentiment_analysis(video_id):
    """FIXED: Better error handling and user feedback"""
    func_url = st.secrets.get("COMMENTS_FUNC_URL", os.getenv("COMMENTS_FUNC_URL"))
    bucket_name = st.secrets.get("RESULTS_BUCKET", os.getenv("RESULTS_BUCKET"))
    
    if not func_url or not bucket_name:
        st.error("❌ Missing configuration: COMMENTS_FUNC_URL or RESULTS_BUCKET")
        return False
    
    # Reset state before starting
    reset_analysis_state()
    
    # Create placeholders for dynamic updates
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # Show initial progress
        with progress_container.container():
            progress_bar = st.progress(0, text="Initializing analysis...")
        
        with status_container.container():
            st.info("🚀 Starting sentiment analysis...")
        
        # Update progress
        progress_bar.progress(25, text="Sending request to cloud function...")
        
        # Make request with appropriate timeout for Streamlit Cloud
        response = requests.post(
            func_url,
            json={"video_url": f"https://www.youtube.com/watch?v={video_id}"},
            timeout=30,  # Shorter timeout
            headers={'Content-Type': 'application/json'}
        )
        
        progress_bar.progress(75, text="Request sent, checking response...")
        
        if response.status_code == 200:
            progress_bar.progress(100, text="Analysis started successfully!")
            
            # Set processing state
            st.session_state.analysis_status = "processing"
            st.session_state.analysis_start_time = time.time()
            st.session_state.check_count = 0
            st.session_state.error_count = 0
            
            # Show success and clear progress
            with status_container.container():
                st.success("✅ Analysis started! Use 'Check Results Now' button to monitor progress.")
            
            # Clear progress after short delay
            time.sleep(2)
            progress_container.empty()
            
            return True
            
        else:
            # Handle HTTP errors
            error_msg = f"Cloud function returned error: {response.status_code}"
            try:
                error_detail = response.json().get('error', 'Unknown error')
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - Response: {response.text[:200]}"
            
            progress_container.empty()
            with status_container.container():
                st.error(f"❌ {error_msg}")
            
            st.session_state.analysis_status = "error"
            return False
    
    except requests.exceptions.Timeout:
        # Handle timeout gracefully
        progress_container.empty()
        with status_container.container():
            st.warning("⚠️ Request timed out, but analysis may still be running in the background. Try checking for results.")
        
        # Assume it might be processing
        st.session_state.analysis_status = "processing"
        st.session_state.analysis_start_time = time.time()
        st.session_state.check_count = 0
        
        return True
    
    except requests.exceptions.ConnectionError:
        progress_container.empty()
        with status_container.container():
            st.error("❌ Could not connect to the analysis service. Please check your internet connection.")
        
        st.session_state.analysis_status = "error"
        return False
    
    except Exception as e:
        progress_container.empty()
        with status_container.container():
            st.error(f"❌ Unexpected error: {str(e)}")
        
        st.session_state.analysis_status = "error"
        return False
        
def check_for_results():
    """FIXED VERSION: Check results with proper GCS authentication"""
    
    # Add debug info
    st.write("🔍 **DEBUG: Starting check_for_results()**")
    
    if not st.session_state.selected_video:
        st.error("❌ DEBUG: No video selected")
        return False
    
    video_id = st.session_state.selected_video['video_id']
    st.write(f"📹 **DEBUG: Checking for video_id:** `{video_id}`")
    
    # Check configuration
    bucket_name = st.secrets.get("RESULTS_BUCKET", os.getenv("RESULTS_BUCKET"))
    project_id = st.secrets.get("GOOGLE_CLOUD_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT"))
    
    if not bucket_name:
        st.error("❌ DEBUG: RESULTS_BUCKET configuration missing")
        return False
    
    if not project_id:
        st.error("❌ DEBUG: GOOGLE_CLOUD_PROJECT configuration missing")
        st.write("**Available secrets:**", list(st.secrets.keys()) if hasattr(st, 'secrets') else "No secrets found")
        return False
    
    st.write(f"🪣 **DEBUG: Using bucket:** `{bucket_name}`")
    st.write(f"🏗️ **DEBUG: Using project:** `{project_id}`")
    
    try:
        st.write("🔄 **DEBUG: Initializing Google Cloud Storage client...**")
        
        # METHOD 1: Initialize with explicit project ID
        try:
            client = storage.Client(project=project_id)
            st.write("✅ **DEBUG: Client initialized with explicit project ID**")
        except Exception as client_error:
            st.error(f"❌ **DEBUG: Failed to create client with project ID:** {str(client_error)}")
            
            # METHOD 2: Try with service account key
            service_account_key = st.secrets.get("GOOGLE_CLOUD_SERVICE_ACCOUNT_KEY")
            if service_account_key:
                st.write("🔄 **DEBUG: Trying with service account key...**")
                try:
                    import json
                    from google.oauth2 import service_account
                    
                    # Parse service account key
                    if isinstance(service_account_key, str):
                        credentials_info = json.loads(service_account_key)
                    else:
                        credentials_info = service_account_key
                    
                    credentials = service_account.Credentials.from_service_account_info(credentials_info)
                    client = storage.Client(project=project_id, credentials=credentials)
                    st.write("✅ **DEBUG: Client initialized with service account key**")
                    
                except Exception as sa_error:
                    st.error(f"❌ **DEBUG: Service account method failed:** {str(sa_error)}")
                    return False
            else:
                st.error("❌ **DEBUG: No service account key found in secrets**")
                return False
        
        bucket = client.bucket(bucket_name)
        
        # List blobs with debug info
        st.write(f"🔍 **DEBUG: Looking for files with prefix:** `{video_id}`")
        
        try:
            blobs = list(bucket.list_blobs(prefix=video_id, timeout=15))
            st.write(f"📁 **DEBUG: Found {len(blobs)} files**")
            
            # Show all found files
            if blobs:
                for i, blob in enumerate(blobs):
                    st.write(f"   - File {i+1}: `{blob.name}` (size: {blob.size} bytes, created: {blob.time_created})")
            else:
                st.write("   - No files found yet")
                
        except Exception as list_error:
            st.error(f"❌ **DEBUG: Error listing files:** {str(list_error)}")
            return False
        
        if not blobs:
            st.info("⏳ **DEBUG: No results found yet. Analysis is still running...**")
            return False
        
        # Get the most recent blob
        latest_blob = max(blobs, key=lambda b: b.time_created)
        blob_name = latest_blob.name
        st.write(f"📄 **DEBUG: Latest file:** `{blob_name}`")
        
        # Check if this is a new result
        if st.session_state.last_processed_blob == blob_name:
            st.info(f"📄 **DEBUG: Same result file found (`{blob_name}`). Waiting for updates...**")
            return False
        
        st.write(f"📥 **DEBUG: Downloading file:** `{blob_name}`")
        
        # Download with timeout and size check
        try:
            # Check blob size first
            if latest_blob.size is None or latest_blob.size == 0:
                st.info(f"⏳ **DEBUG: Result file is empty (size: {latest_blob.size}). Still being written...**")
                return False
            
            st.write(f"📊 **DEBUG: File size:** {latest_blob.size} bytes")
            
            # Download with timeout
            content = latest_blob.download_as_text(timeout=15)
            
            st.write(f"📝 **DEBUG: Downloaded {len(content)} characters**")
            
            # Show first 200 characters
            st.write(f"📖 **DEBUG: Content preview:** `{content[:200]}...`")
            
            # Validate content
            if not content or len(content.strip()) < 50:
                st.info(f"⏳ **DEBUG: Content too short ({len(content)} chars). File appears incomplete...**")
                return False
            
            # Success!
            st.success(f"✅ **DEBUG: Successfully processed file:** `{blob_name}`")
            st.session_state.raw_summary = content
            st.session_state.analysis_status = "complete"
            st.session_state.last_processed_blob = blob_name
            
            return True
            
        except Exception as download_error:
            st.error(f"❌ **DEBUG: Download error:** {str(download_error)}")
            return False
    
    except Exception as e:
        st.error(f"❌ **DEBUG: General error:** {str(e)}")
        st.write(f"**Error type:** {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        return False


def show_analysis_results():
    """Enhanced results display with better error handling"""
    if not st.session_state.raw_summary:
        return
    
    raw_summary = st.session_state.raw_summary
    
    # Parse metrics with better error handling
    try:
        lines = [line.strip() for line in raw_summary.splitlines() if line.strip()]
        data = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        
        # Extract metrics with defaults
        total_comments = 0
        avg_sentiment = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        try:
            if "Total comments" in data:
                total_comments = int(re.search(r'\d+', data["Total comments"]).group())
            
            if "Avg sentiment score" in data:
                avg_sentiment = float(re.search(r'-?\d+\.?\d*', data["Avg sentiment score"]).group())
            
            if "Positive comments" in data:
                sentiment_text = data["Positive comments"]
                numbers = re.findall(r'\d+', sentiment_text)
                if len(numbers) >= 3:
                    positive_count = int(numbers[0])
                    negative_count = int(numbers[1])
                    neutral_count = int(numbers[2])
        except (AttributeError, ValueError, IndexError) as e:
            st.warning(f"⚠️ Could not parse some metrics: {e}")
        
        # Display metrics dashboard
        show_metrics_dashboard(total_comments, avg_sentiment, positive_count, negative_count, neutral_count)
        
        # Visualizations (only if we have data)
        if total_comments > 0:
            show_enhanced_visualizations(positive_count, negative_count, neutral_count, avg_sentiment)
        
        # AI Insights
        show_enhanced_ai_insights(raw_summary)
        
        # Raw data and downloads
        show_enhanced_downloads(raw_summary)
        
    except Exception as e:
        st.markdown(f'<div class="status-error">❌ Could not parse analysis results: {str(e)}</div>', unsafe_allow_html=True)
        
        # Show raw data as fallback
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Raw Analysis Data")
        st.text_area("Raw Results", raw_summary, height=300, key="fallback_raw_data")
        st.markdown('</div>', unsafe_allow_html=True)

def show_metrics_dashboard(total_comments, avg_sentiment, positive_count, negative_count, neutral_count):
    """Enhanced metrics display"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Sentiment Analysis Overview")
    
    # Create metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_comments}</div>
            <div class="metric-label">Total Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sentiment_color = "#48bb78" if avg_sentiment > 0 else "#f56565" if avg_sentiment < 0 else "#ed8936"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {sentiment_color}, {sentiment_color}aa);">
            <div class="metric-value">{avg_sentiment:.2f}</div>
            <div class="metric-label">Avg Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #48bb78, #38a169);">
            <div class="metric-value">{positive_count}</div>
            <div class="metric-label">😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f56565, #e53e3e);">
            <div class="metric-value">{negative_count}</div>
            <div class="metric-label">😞 Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ed8936, #dd6b20);">
            <div class="metric-value">{neutral_count}</div>
            <div class="metric-label">😐 Neutral</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_enhanced_visualizations(positive_count, negative_count, neutral_count, avg_sentiment):
    """Enhanced visualizations with multiple chart types"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Sentiment Visualizations")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for sentiment distribution
        labels = ['Positive', 'Negative', 'Neutral']
        values = [positive_count, negative_count, neutral_count]
        colors = ['#48bb78', '#f56565', '#ed8936']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig_pie.update_layout(
            title="Sentiment Distribution",
            font=dict(size=14),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for sentiment counts
        fig_bar = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto',
            )
        ])
        
        fig_bar.update_layout(
            title="Sentiment Counts",
            xaxis_title="Sentiment Type",
            yaxis_title="Number of Comments",
            font=dict(size=14),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sentiment score visualization
    if avg_sentiment != 0:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Average Sentiment Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "#f56565"},
                    {'range': [-0.5, 0], 'color': "#fc8181"},
                    {'range': [0, 0.5], 'color': "#68d391"},
                    {'range': [0.5, 1], 'color': "#48bb78"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_enhanced_ai_insights(raw_summary):
    """Enhanced AI insights generation"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Generated Insights")
    
    if not st.session_state.ai_insights:
        if st.button("🧠 Generate AI Insights", use_container_width=True):
            generate_ai_insights(raw_summary)
    else:
        st.markdown('<div class="insights-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.ai_insights)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Regenerate Insights", use_container_width=True):
            st.session_state.ai_insights = None
            generate_ai_insights(raw_summary)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_ai_insights(raw_summary):
    """Generate AI insights using Gemini"""
    placeholder = st.empty()
    with placeholder.container():
        show_loading_animation("Generating AI Insights", "Analyzing patterns and trends...")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        Analyze this YouTube video sentiment analysis data and provide insightful observations:
        
        {raw_summary}
        
        Please provide:
        1. **Key Findings**: What are the main sentiment patterns?
        2. **Audience Engagement**: What does this tell us about viewer engagement?
        3. **Content Performance**: How is the content being received?
        4. **Recommendations**: What actionable insights can you provide?
        5. **Notable Patterns**: Any interesting trends or outliers?
        
        Format your response in markdown with clear sections and bullet points.
        Keep it concise but insightful (max 500 words).
        """
        
        response = model.generate_content(prompt)
        st.session_state.ai_insights = response.text
        
        placeholder.markdown('<div class="status-success">✅ AI insights generated successfully!</div>', unsafe_allow_html=True)
        time.sleep(1)
        placeholder.empty()
        st.rerun()
        
    except Exception as e:
        placeholder.markdown(f'<div class="status-error">❌ Failed to generate insights: {str(e)}</div>', unsafe_allow_html=True)

def show_enhanced_downloads(raw_summary):
    """Enhanced download section with multiple formats"""
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Text file download
        st.download_button(
            label="📄 Download as TXT",
            data=raw_summary,
            file_name=f"sentiment_analysis_{st.session_state.selected_video['video_id']}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # JSON file download
        try:
            # Convert raw summary to structured JSON
            json_data = {
                "video_id": st.session_state.selected_video['video_id'],
                "video_title": st.session_state.selected_video['title'],
                "analysis_timestamp": datetime.now().isoformat(),
                "raw_analysis": raw_summary,
                "ai_insights": st.session_state.ai_insights or "Not generated"
            }
            
            st.download_button(
                label="📊 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"sentiment_analysis_{st.session_state.selected_video['video_id']}.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"JSON generation failed: {e}")
    
    with col3:
        # PDF report download
        if st.button("📑 Generate PDF Report", use_container_width=True):
            generate_pdf_report(raw_summary)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_pdf_report(raw_summary):
    """Generate and download PDF report"""
    placeholder = st.empty()
    with placeholder.container():
        show_loading_animation("Generating PDF Report", "Creating formatted document...")
    
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        
        # Title
        pdf.cell(0, 10, "YouTube Sentiment Analysis Report", ln=True, align="C")
        pdf.ln(10)
        
        # Video info
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Video: {st.session_state.selected_video['title']}", ln=True)
        pdf.cell(0, 8, f"Channel: {st.session_state.selected_video['channel']}", ln=True)
        pdf.cell(0, 8, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)
        
        # Analysis results
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Analysis Results:", ln=True)
        pdf.set_font("Arial", size=10)
        
        # Split raw summary into lines and add to PDF
        for line in raw_summary.split('\n'):
            if line.strip():
                # Handle long lines by wrapping
                if len(line) > 80:
                    words = line.split(' ')
                    current_line = ""
                    for word in words:
                        if len(current_line + word) < 80:
                            current_line += word + " "
                        else:
                            pdf.cell(0, 6, current_line.strip(), ln=True)
                            current_line = word + " "
                    if current_line.strip():
                        pdf.cell(0, 6, current_line.strip(), ln=True)
                else:
                    pdf.cell(0, 6, line, ln=True)
        
        # AI Insights section
        if st.session_state.ai_insights:
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "AI Insights:", ln=True)
            pdf.set_font("Arial", size=10)
            
            # Clean up markdown formatting for PDF
            insights_text = st.session_state.ai_insights.replace('**', '').replace('*', '').replace('#', '')
            for line in insights_text.split('\n'):
                if line.strip():
                    if len(line) > 80:
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 80:
                                current_line += word + " "
                            else:
                                pdf.cell(0, 6, current_line.strip(), ln=True)
                                current_line = word + " "
                        if current_line.strip():
                            pdf.cell(0, 6, current_line.strip(), ln=True)
                    else:
                        pdf.cell(0, 6, line, ln=True)
        
        # Generate PDF bytes
        pdf_bytes = BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')
        pdf_bytes.write(pdf_output)
        pdf_bytes.seek(0)
        
        placeholder.empty()
        
        # Download button for PDF
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes.getvalue(),
            file_name=f"sentiment_report_{st.session_state.selected_video['video_id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    except Exception as e:
        placeholder.markdown(f'<div class="status-error">❌ PDF generation failed: {str(e)}</div>', unsafe_allow_html=True)

def show_footer():
    """Enhanced footer"""
    st.markdown("""
    <div class="footer">
        <div style="font-size: 1.3em; font-weight: 600; margin-bottom: 10px;">
            🎬 YouTube Sentiment Dashboard
        </div>
        <div>
            Powered by AI • Built with Streamlit • Enhanced Analytics
        </div>
        <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
            Analyze • Visualize • Understand
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Main App Logic ───────────────────────────────────────────────────────────
def main():
    """Main application logic"""
    show_header()
    
    if not st.session_state.dashboard_mode:
        search_interface()
    else:
        dashboard_interface()
    
    show_footer()

# ─── Run the app ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
