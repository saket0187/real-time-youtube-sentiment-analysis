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
import base64
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from google.oauth2.service_account import Credentials

# ─── Load .env & configure Gemini ─────────────────────────────────────────────
load_dotenv()
gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not gemini_key:
    st.error("🔑 Gemini API key missing. Set GEMINI_API_KEY in .env or Streamlit secrets.")
    st.stop()
genai.configure(api_key=gemini_key)

# Google Cloud Storage configuration
creds_json = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

if not creds_json:
    st.error("❌ GOOGLE_APPLICATION_CREDENTIALS missing in Streamlit secrets or environment variable.")
    st.stop()

# Parse the JSON string into a dictionary
try:
    creds_dict = creds_json if isinstance(creds_json, dict) else json.loads(creds_json)
except Exception as e:
    st.error(f"❌ Failed to parse GOOGLE_APPLICATION_CREDENTIALS as JSON: {e}")
    st.stop()

# Create credentials and client explicitly by passing the credentials object
try:
    google_creds = Credentials.from_service_account_info(creds_dict)
    google_project = creds_dict["project_id"]
    st.session_state['google_creds'] = google_creds
    st.session_state['google_project'] = google_project
except Exception as e:
    st.error(f"❌ Failed to create Google Cloud Storage client: {e}")
    st.stop()

# ─── Streamlit page setup ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Sentiment Dashboard", 
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/32px-YouTube_full-color_icon_%282017%29.svg.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── ENHANCED CSS with MASSIVE Floating Elements and Perfect Alignment ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Remove all Streamlit default styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    
    /* Force dark background on everything */
    html, body, .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #0A0B1A !important;
        color: white !important;
    }
    
    /* Main app background with MASSIVE floating shapes */
    .stApp {
        background: linear-gradient(135deg, #0A0B1A 0%, #111225 50%, #1a1a2e 100%) !important;
        min-height: 100vh !important;
        position: relative !important;
        overflow-x: hidden !important;
    }
    
    /* MASSIVE floating background shapes - EXACTLY like reference image */
    .stApp::before {
        content: '';
        position: fixed;
        top: -400px;
        right: -400px;
        width: 800px;
        height: 800px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, rgba(139, 92, 246, 0.12) 40%, rgba(139, 92, 246, 0.05) 70%, transparent 85%);
        border-radius: 50%;
        z-index: 0;
        animation: float-massive-1 25s ease-in-out infinite;
        pointer-events: none !important;
    }
    
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -500px;
        left: -400px;
        width: 900px;
        height: 900px;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.22) 0%, rgba(236, 72, 153, 0.1) 50%, rgba(236, 72, 153, 0.04) 75%, transparent 90%);
        border-radius: 50%;
        z-index: 0;
        animation: float-massive-2 30s ease-in-out infinite;
        pointer-events: none !important;
    }
    
    /* Additional HUGE floating elements */
    body::before {
        content: '';
        position: fixed;
        top: 20%;
        left: -300px;
        width: 700px;
        height: 700px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.18) 0%, rgba(59, 130, 246, 0.08) 60%, transparent 80%);
        border-radius: 50%;
        z-index: 0;
        animation: float-massive-3 35s ease-in-out infinite;
        pointer-events: none !important;
    }
    
    body::after {
        content: '';
        position: fixed;
        top: 60%;
        right: -250px;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.06) 65%, transparent 85%);
        border-radius: 50%;
        z-index: 0;
        animation: float-massive-4 28s ease-in-out infinite;
        pointer-events: none !important;
    }
    
    /* More medium floating elements */
    .main::before {
        content: '';
        position: fixed;
        top: 10%;
        left: 30%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.04) 70%, transparent 85%);
        border-radius: 50%;
        z-index: 0;
        animation: float-medium-1 22s ease-in-out infinite;
        pointer-events: none !important;
    }
    
    .main::after {
        content: '';
        position: fixed;
        bottom: 15%;
        right: 25%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.14) 0%, rgba(168, 85, 247, 0.05) 65%, transparent 80%);
        border-radius: 50%;
        z-index: 0;
        animation: float-medium-2 26s ease-in-out infinite;
        pointer-events: none !important;
    }
    
    @keyframes float-massive-1 {
        0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.9; }
        25% { transform: translate(-100px, 80px) scale(1.1) rotate(90deg); opacity: 0.7; }
        50% { transform: translate(-50px, -60px) scale(0.95) rotate(180deg); opacity: 1; }
        75% { transform: translate(80px, 40px) scale(1.05) rotate(270deg); opacity: 0.8; }
    }
    
    @keyframes float-massive-2 {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.8; }
        33% { transform: translate(120px, -80px) scale(1.15); opacity: 0.9; }
        66% { transform: translate(-80px, 60px) scale(0.9); opacity: 0.7; }
    }
    
    @keyframes float-massive-3 {
        0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: 0.7; }
        50% { transform: translate(100px, 50px) rotate(180deg) scale(1.2); opacity: 0.9; }
    }
    
    @keyframes float-massive-4 {
        0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.6; }
        30% { transform: translate(-60px, -40px) scale(1.1) rotate(120deg); opacity: 0.8; }
        70% { transform: translate(40px, 80px) scale(0.95) rotate(240deg); opacity: 0.7; }
    }
    
    @keyframes float-medium-1 {
        0%, 100% { transform: translate(0, 0) rotate(0deg); opacity: 0.6; }
        50% { transform: translate(60px, -40px) rotate(180deg); opacity: 0.8; }
    }
    
    @keyframes float-medium-2 {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.5; }
        33% { transform: translate(-40px, 30px) scale(1.1); opacity: 0.7; }
        66% { transform: translate(30px, -50px) scale(0.9); opacity: 0.6; }
    }
    
    /* Main container with proper z-index */
    .main .block-container {
        padding: 2rem !important;
        max-width: none !important;
        background: transparent !important;
        position: relative !important;
        z-index: 100 !important;
    }
    
    /* Header styling */
    .header-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 3rem 2rem !important;
        margin-bottom: 3rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
        z-index: 100 !important;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .youtube-logo {
        width: 80px;
        height: 60px;
        background: linear-gradient(45deg, #FF0000, #FF6B6B);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(255, 0, 0, 0.4);
        transition: transform 0.3s ease;
    }
    
    .youtube-logo:hover {
        transform: rotateY(15deg) rotateX(5deg);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.7);
        margin: 1rem 0 0 0;
    }
    
    /* Search section with better alignment */
    .search-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 3rem 2rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        z-index: 100 !important;
    }
    
    .search-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .search-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* PERFECT ALIGNMENT: Same height for all form elements */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 1rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        color: #121212 !important;
        transition: all 0.1s ease !important;
        height: 52px !important;
        min-height: 52px !important;
        box-sizing: border-box !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.2) !important;
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        height: 52px !important;
        min-height: 52px !important;
        box-sizing: border-box !important;
    }
    
    /* PERFECT ALIGNMENT & CURSOR FIX: Button same height as inputs */
    .stButton > button {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0rem 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.1s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        height: 52px !important;
        min-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        user-select: none !important;
        position: relative !important;
        z-index: 999 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.5) !important;
        transition: all 0.1s ease !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) scale(1) !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.05s ease !important;
    }
    
    /* Fix button containers that might block clicks */
    .stButton,
    .stDownloadButton {
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 999 !important;
        position: relative !important;
    }

    .stButton > div,
    .stDownloadButton > div {
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 999 !important;
    }
    
    /* DOWNLOAD BUTTON FIX */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        height: 52px !important;
        min-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.1s ease !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        user-select: none !important;
        position: relative !important;
        z-index: 999 !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.5) !important;
    }
    
    .stDownloadButton > button:active {
        transform: translateY(0px) scale(1) !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.05s ease !important;
    }
    
    /* Ensure download button text is visible */
    .stDownloadButton > button *,
    .stDownloadButton > button span,
    .stDownloadButton > button div {
        color: #ffffff !important;
        opacity: 1 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        pointer-events: none !important;
    }
    
    /* Video card styling with z-index */
    .video-card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        z-index: 100 !important;
        transition: all 0.4s ease !important;
        overflow: hidden !important;
    }
    
    .video-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .video-card:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(139, 92, 246, 0.5) !important;
        transform: translateY(-6px) !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4) !important;
    }
    
    .video-card:hover::before {
        opacity: 1;
    }
    
    .video-thumbnail {
        border-radius: 16px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .video-thumbnail:hover {
        transform: scale(1.03);
    }
    
    .video-thumbnail img {
        width: 100%;
        height: auto;
        display: block;
    }
    
    .video-thumbnail::after {
        content: '▶';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 3rem;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: all 0.3s ease;
        border: 3px solid #8B5CF6;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
    }
    
    .video-thumbnail:hover::after {
        opacity: 1;
        transform: translate(-50%, -50%) scale(1.1);
    }
    
    .video-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: white;
        margin: 1.5rem 0 1rem 0;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .video-meta {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }
    
    .video-channel {
        color: #EC4899;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 1rem;
    }
    
    .video-date {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-family: 'Inter', sans-serif;
    }
    
    .video-description {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
        background: rgba(255, 255, 255, 0.03);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #8B5CF6;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .results-counter {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 0;
    }
    
    .results-icon {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Dashboard sections with z-index */
    .dashboard-section {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 3rem 2rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        z-index: 100 !important;
        transition: all 0.3s ease !important;
    }
    
    .dashboard-section:hover {
        border-color: rgba(139, 92, 246, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 2.5rem 2rem !important;
        text-align: center !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.4s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        opacity: 0.8;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(139, 92, 246, 0.5) !important;
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3) !important;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    /* Loading animation */
    .loading-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 4rem 3rem !important;
        text-align: center !important;
        margin: 3rem 0 !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        z-index: 100 !important;
        position: relative !important;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid #8B5CF6;
        border-right: 4px solid #EC4899;
        border-radius: 50%;
        animation: spin 1.2s linear infinite;
        margin-bottom: 2rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-family: 'Space Grotesk', sans-serif;
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .loading-stage {
        font-family: 'Inter', sans-serif;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
    }
    
    /* Status messages */
    .status-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.05)) !important;
        backdrop-filter: blur(15px) !important;
        color: white !important;
        padding: 1.5rem 2rem !important;
        border-radius: 16px !important;
        margin: 1.5rem 0 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        z-index: 100 !important;
        position: relative !important;
    }
    
    .status-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.05)) !important;
        backdrop-filter: blur(15px) !important;
        color: white !important;
        padding: 1.5rem 2rem !important;
        border-radius: 16px !important;
        margin: 1.5rem 0 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        z-index: 100 !important;
        position: relative !important;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.05)) !important;
        backdrop-filter: blur(15px) !important;
        color: white !important;
        padding: 1.5rem 2rem !important;
        border-radius: 16px !important;
        margin: 1.5rem 0 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        z-index: 100 !important;
        position: relative !important;
    }
    
    /* AI insights */
    .ai-insights {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.1)) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        border-radius: 24px !important;
        padding: 2.5rem !important;
        margin: 2rem 0 !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        z-index: 100 !important;
        position: relative !important;
    }
    
    /* Footer */
    .footer-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        color: white !important;
        text-align: center !important;
        padding: 3rem !important;
        margin-top: 4rem !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        z-index: 100 !important;
        position: relative !important;
    }
    
    .footer-container h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .footer-container p {
        font-family: 'Inter', sans-serif;
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            text-align: center;
            gap: 2rem;
        }
        
        .main-title {
            font-size: 2.5rem;
        }
        
        .main-subtitle {
            font-size: 1.1rem;
        }
        
        .youtube-logo {
            width: 70px;
            height: 50px;
            font-size: 24px;
        }
        
        .search-container, .video-card, .dashboard-section {
            padding: 1.5rem !important;
        }
        
        .metric-card {
            padding: 2rem 1.5rem !important;
        }
        
        .metric-value {
            font-size: 2.2rem;
        }
        
        .search-title, .section-title {
            font-size: 1.6rem;
        }
        
        /* PERFECT ALIGNMENT: Same height and baseline for all form elements */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stButton > button,
        .stDownloadButton > button {
            height: 48px !important;
            min-height: 48px !important;
            max-height: 48px !important;
            box-sizing: border-box !important;
            border-radius: 16px !important;
            font-size: 1rem !important;
            margin: 0 !important;
            padding: 0 1rem !important;
            display: flex !important;
            align-items: center !important;
            vertical-align: top !important;
        }

        /* Fix button container alignment for mobile */
        .stButton > div,
        .stDownloadButton > div {
            display: flex !important;
            align-items: center !important;
            height: 48px !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            padding: 0 1.5rem !important;
            margin: 0 !important;
            justify-content: center !important;
            line-height: 1 !important;
            vertical-align: baseline !important;
        }
            
        /* Ensure all column containers align properly */
        div[data-testid="column"] > div {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
        }

        /* Fix any wrapper divs that might cause misalignment */
        .stTextInput > div,
        .stSelectbox > div,
        .stButton > div,
        .stDownloadButton > div {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "raw_summary" not in st.session_state:
    st.session_state.raw_summary = None
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = None
if "analysis_status" not in st.session_state:
    st.session_state.analysis_status = "idle"
if "dashboard_mode" not in st.session_state:
    st.session_state.dashboard_mode = False
if "processing_stage" not in st.session_state:
    st.session_state.processing_stage = ""
if "analysis_start_time" not in st.session_state:
    st.session_state.analysis_start_time = None

# ─── Loading Animation ──────────────────────────────────────────────
def show_loading_animation(text="Processing", stage=""):
    """Enhanced loading animation"""
    loading_html = f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">{text}</div>
        {f'<div class="loading-stage">{stage}</div>' if stage else ''}
    </div>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────
def show_header():
    """Enhanced header design"""
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <div class="youtube-logo">
                ▶
            </div>
            <div>
                <h1 class="main-title">YouTube Sentiment Dashboard</h1>
                <p class="main-subtitle">AI-Powered Comment Analysis & Insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Search Interface ───────────────────────────────────────────
def search_interface():
    """Enhanced search interface design"""
    st.markdown('''
    <div class="search-container">
        <div class="search-title">
            <span class="search-icon">🔍</span>
            Search YouTube Videos
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Search form with PERFECT alignment - all elements same height (52px)
    col1, col2, col3 = st.columns([6, 1, 1.5])

    with col1:
        query = st.text_input(
            "Search Query", 
            key="search_query", 
            placeholder="Enter keywords to search YouTube videos...",
            label_visibility="collapsed"
        )

    with col2:
        max_results = st.selectbox(
            "Max Results", 
            [10, 25, 50], 
            key="search_max",
            label_visibility="collapsed"
        )

    with col3:
        search_clicked = st.button("🔍 Search", use_container_width=True)
    
    if search_clicked:
        if not query.strip():
            st.markdown('<div class="status-warning">⚠️ Please enter a search query.</div>', unsafe_allow_html=True)
        else:
            perform_search(query, max_results)
    
    display_search_results()

def perform_search(query, max_results):
    """Enhanced search with modern styling"""
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
        time.sleep(1)
        placeholder.empty()
        
    except Exception as e:
        placeholder.markdown(f'<div class="status-error">❌ Search failed: {str(e)}</div>', unsafe_allow_html=True)

def display_search_results():
    """Display search results with enhanced card styling"""
    if st.session_state.search_results:
        st.markdown(f'''
        <div class="results-counter">
            <span class="results-icon">📺</span>
            Found {len(st.session_state.search_results)} Videos
        </div>
        ''', unsafe_allow_html=True)
        
        for i, video in enumerate(st.session_state.search_results):
            
            
            cols = st.columns([1, 4, 1])
            
            with cols[0]:
                st.markdown(f'''
                <div class="video-thumbnail">
                    <img src="{video["thumbnail"]}" alt="Video thumbnail" />
                </div>
                ''', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f'''
                <div class="video-title">{video["title"]}</div>
                <div class="video-meta">
                    <div class="video-channel">📺 {video["channel"]}</div>
                    <div class="video-date">📅 {video["published"]}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                description = video.get("description", "")
                if description:
                    st.markdown(f'''
                    <div class="video-description">
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

# ─── Dashboard Interface ────────────────────────────────────────────
def dashboard_interface():
    video = st.session_state.selected_video
    
    # Back button
    if st.button("← Back to Search", key="back_button"):
        st.session_state.dashboard_mode = False
        st.session_state.selected_video = None
        st.session_state.raw_summary = None
        st.session_state.ai_insights = None
        st.session_state.analysis_status = "idle"
        st.rerun()
    
    st.markdown("---")
    
    # Video info section
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎬 Selected Video</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(video["thumbnail"], width=250)
    
    with col2:
        st.markdown(f"""
        <div class="video-title">{video['title']}</div>
        """, unsafe_allow_html=True)
        st.write(f"📺 **Channel:** {video['channel']}")
        st.write(f"📅 **Published:** {video['published']}")
        st.write(f"🔗 **Video ID:** `{video['video_id']}`")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    if st.session_state.analysis_status == "idle":
        if st.button("🚀 Start Sentiment Analysis", use_container_width=True, key="start_analysis"):
            trigger_sentiment_analysis(video['video_id'])
    
    # Analysis status and results
    show_analysis_status()
    show_analysis_results()

@st.fragment
def show_analysis_status():
    """Analysis status with modern theming"""
    
    if st.session_state.analysis_status == "processing":
        # Initialize required session state variables
        if not hasattr(st.session_state, 'analysis_start_time') or st.session_state.analysis_start_time is None:
            st.session_state.analysis_start_time = time.time()
        
        if not hasattr(st.session_state, 'last_check_time'):
            st.session_state.last_check_time = 0
        
        if not hasattr(st.session_state, 'auto_check_count'):
            st.session_state.auto_check_count = 0
        
        elapsed_time = time.time() - st.session_state.analysis_start_time
        
        # Progressive checking intervals
        check_intervals = [30, 60, 90, 150, 210, 270, 330, 420, 510, 600]
        
        auto_check_triggered = False
        
        # Check if we should trigger auto-check
        for i, interval in enumerate(check_intervals):
            if elapsed_time >= interval and st.session_state.auto_check_count <= i:
                st.markdown('<div class="status-warning">⏰ Auto-checking results...</div>', unsafe_allow_html=True)
                st.session_state.auto_check_count = i + 1
                st.session_state.last_check_time = interval
                
                # Trigger check and break to avoid infinite loop
                check_result = check_for_results()
                auto_check_triggered = True
                
                # If results found, don't continue processing
                if st.session_state.analysis_status == "complete":
                    break
                
                time.sleep(2)
                break
        
        # Display current status
        if st.session_state.analysis_status == "processing":
            # Find next check interval for display
            next_check = None
            for interval in check_intervals:
                if elapsed_time < interval:
                    next_check = interval
                    break
            
            if next_check:
                remaining = max(0, int(next_check - elapsed_time))
                minutes = remaining // 60
                seconds = remaining % 60
                
                # Determine current phase based on elapsed time
                if elapsed_time < 60:
                    phase = "Fetching comments"
                    estimated = "1-2 minutes remaining"
                elif elapsed_time < 120:
                    phase = "Analyzing sentiment"
                    estimated = "2-3 minutes remaining"
                elif elapsed_time < 240:
                    phase = "Generating insights"
                    estimated = "1-2 minutes remaining"
                else:
                    phase = "Finalizing results"
                    estimated = "Almost done..."
                
                if minutes > 0:
                    next_check_text = f"Next auto-check in {minutes}m {seconds}s"
                else:
                    next_check_text = f"Next auto-check in {seconds}s"
                
                show_loading_animation(phase, f"{estimated} • {next_check_text}")
            else:
                show_loading_animation("Still Processing", f"Running for {int(elapsed_time//60)}m {int(elapsed_time%60)}s...")
            
            # Progress simulation
            st.markdown("""
            <div class="dashboard-section">
                <h4>Processing stages:</h4>
                <p>✅ Fetching comments</p>
                <p>🔄 Analyzing sentiment...</p>
                <p>⏳ Generating insights...</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Check Results Now", key="check_results", use_container_width=True):
                    check_for_results()
            
            with col2:
                if st.button("🔄 Reset Analysis", key="reset_analysis", use_container_width=True):
                    reset_analysis_state()
                    st.rerun()
        
        # Auto-refresh fragment
        if not auto_check_triggered and st.session_state.analysis_status == "processing":
            time.sleep(1)
            st.rerun()
    
    elif st.session_state.analysis_status == "complete":
        st.markdown('<div class="status-success">✅ Analysis Complete! Results are ready below.</div>', unsafe_allow_html=True)
    
    elif st.session_state.analysis_status == "error":
        st.markdown('<div class="status-error">❌ Analysis failed. Please try again or check your configuration.</div>', unsafe_allow_html=True)

def reset_analysis_state():
    """Helper function to reset all analysis-related state"""
    st.session_state.analysis_status = "idle"
    st.session_state.raw_summary = None
    st.session_state.ai_insights = None
    st.session_state.analysis_start_time = None
    st.session_state.last_check_time = 0
    st.session_state.auto_check_count = 0
    if 'refresh_placeholder' in st.session_state:
        del st.session_state.refresh_placeholder

def trigger_sentiment_analysis(video_id):
    """Enhanced analysis trigger with modern theming"""
    func_url = st.secrets.get("COMMENTS_FUNC_URL", os.getenv("COMMENTS_FUNC_URL"))
    bucket_name = st.secrets.get("RESULTS_BUCKET", os.getenv("RESULTS_BUCKET"))
    
    if not func_url or not bucket_name:
        st.markdown('<div class="status-error">❌ COMMENTS_FUNC_URL or RESULTS_BUCKET missing in configuration.</div>', unsafe_allow_html=True)
        return
    
    # Reset state before starting new analysis
    reset_analysis_state()
    
    placeholder = st.empty()
    with placeholder.container():
        show_loading_animation("Triggering Analysis", "Sending request to cloud function...")
    
    try:
        response = requests.post(
            func_url, 
            json={"video_url": f"https://www.youtube.com/watch?v={video_id}"},
            timeout=30
        )
        
        if response.status_code == 200:
            st.session_state.analysis_status = "processing"
            st.session_state.analysis_start_time = time.time()
            st.session_state.auto_check_count = 0
            placeholder.markdown('<div class="status-success">✅ Analysis started successfully!</div>', unsafe_allow_html=True)
            time.sleep(2)
            placeholder.empty()
        
        else:
            st.session_state.analysis_status = "error"
            placeholder.markdown(f'<div class="status-error">❌ Function call failed with status: {response.status_code}<br>Response: {response.text}</div>', unsafe_allow_html=True)
        
    except requests.exceptions.Timeout:
        st.session_state.analysis_status = "processing"
        st.session_state.analysis_start_time = time.time()
        st.session_state.auto_check_count = 0
        placeholder.markdown('<div class="status-warning">⏳ Function call timed out, but analysis may still be running. Will check for results automatically.</div>', unsafe_allow_html=True)
        time.sleep(2)
        placeholder.empty()
        
    except Exception as e:
        st.session_state.analysis_status = "error"
        placeholder.markdown(f'<div class="status-error">❌ Function call failed: {str(e)}</div>', unsafe_allow_html=True)

def check_for_results():
    """Results checking with modern theming"""
    video_id = st.session_state.selected_video['video_id']
    bucket_name = st.secrets.get("RESULTS_BUCKET", os.getenv("RESULTS_BUCKET"))
    
    if not bucket_name:
        st.markdown('<div class="status-error">❌ RESULTS_BUCKET missing in configuration.</div>', unsafe_allow_html=True)
        return False
    
    try:
        client = storage.Client(credentials=st.session_state['google_creds'], project=st.session_state['google_project'])
        bucket = client.bucket(bucket_name)
        
        # List all blobs with video_id prefix
        blobs = list(bucket.list_blobs(prefix=video_id))
        
        if blobs:
            # Get the most recent blob
            latest_blob = max(blobs, key=lambda b: b.time_created)
            
            # Check if this is a new result (not already processed)
            blob_name = latest_blob.name
            if hasattr(st.session_state, 'last_processed_blob') and st.session_state.last_processed_blob == blob_name:
                return False  # Already processed this result
            
            # Download the content
            content = latest_blob.download_as_text()
            
            # Validate content is not empty or error
            if content and len(content.strip()) > 50:  # Basic validation
                # Store in session state
                st.session_state.raw_summary = content
                st.session_state.analysis_status = "complete"
                st.session_state.last_processed_blob = blob_name
                
                # Show success message briefly
                success_placeholder = st.empty()
                success_placeholder.markdown(f'<div class="status-success">✅ Results found! File: {latest_blob.name}</div>', unsafe_allow_html=True)
                time.sleep(2)
                success_placeholder.empty()
                
                return True
            else:
                st.warning("⚠️ Found result file but content appears incomplete. Continuing to wait...")
                return False
        else:
            # No results found yet
            return False
        
    except Exception as e:
        error_placeholder = st.empty()
        error_placeholder.markdown(f'<div class="status-error">❌ Error checking results: {str(e)}</div>', unsafe_allow_html=True)
        time.sleep(3)
        error_placeholder.empty()
        return False

@st.fragment
def show_analysis_results():
    """Enhanced results display with modern styling and FIXED Plotly compatibility"""
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
            show_visualizations(positive_count, negative_count, neutral_count, avg_sentiment)
        
        # AI Insights
        show_ai_insights(raw_summary)
        
        # Raw data and downloads
        show_downloads(raw_summary)
        
    except Exception as e:
        st.markdown(f'<div class="status-error">❌ Could not parse analysis results: {str(e)}</div>', unsafe_allow_html=True)
        
        # Show raw data as fallback
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📄 Raw Analysis Data</div>', unsafe_allow_html=True)
        st.text_area("Raw Results", raw_summary, height=300, key="fallback_raw_data")
        st.markdown('</div>', unsafe_allow_html=True)

def show_metrics_dashboard(total_comments, avg_sentiment, positive_count, negative_count, neutral_count):
    """Enhanced metrics display with modern cards"""
    st.markdown('<div class="section-title">📊 Sentiment Analysis Overview</div>', unsafe_allow_html=True)
    
    # Create metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_comments:,}</div>
            <div class="metric-label">Total Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sentiment_color = "#10B981" if avg_sentiment > 0 else "#EF4444" if avg_sentiment < 0 else "#F59E0B"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {sentiment_color};">{avg_sentiment:.2f}</div>
            <div class="metric-label">Avg Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10B981;">{positive_count:,}</div>
            <div class="metric-label">😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #EF4444;">{negative_count:,}</div>
            <div class="metric-label">😞 Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #F59E0B;">{neutral_count:,}</div>
            <div class="metric-label">😐 Neutral</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_visualizations(positive_count, negative_count, neutral_count, avg_sentiment):
    """FIXED: Enhanced visualizations with modern styling and proper Plotly syntax"""
    st.markdown('<div class="section-title">📈 Sentiment Visualizations</div>', unsafe_allow_html=True)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        values = [positive_count, negative_count, neutral_count]
        colors = ['#10B981', '#EF4444', '#F59E0B']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.4,
            marker=dict(
                colors=colors,
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title=dict(
                text="<b>Sentiment Distribution</b>",
                font=dict(size=18, color='white', family='Space Grotesk')
            ),
            font=dict(size=14, color='white'),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # FIXED: Bar chart with correct title_font syntax
        fig_bar = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                text=values,
                textposition='auto',
                textfont=dict(color='white', size=14),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig_bar.update_layout(
            title=dict(
                text="<b>Sentiment Counts</b>",
                font=dict(size=18, color='white', family='Space Grotesk')
            ),
            xaxis=dict(
                title="Sentiment Type",
                title_font=dict(color='white'),  # FIXED: was titlefont
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title="Number of Comments",
                title_font=dict(color='white'),  # FIXED: was titlefont
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            font=dict(size=14, color='white'),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gauge chart
    if avg_sentiment != 0:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_sentiment,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Average Sentiment Score</b>", 
                   'font': {'size': 20, 'color': 'white', 'family': 'Space Grotesk'}},
            delta={'reference': 0, 'font': {'color': 'white'}},
            number={'font': {'color': 'white', 'size': 24}},
            gauge={
                'axis': {
                    'range': [-1, 1],
                    'tickwidth': 2,
                    'tickcolor': "white",
                    'tickfont': {'color': 'white'}
                },
                'bar': {'color': "#8B5CF6", 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0.3)",
                'borderwidth': 3,
                'bordercolor': "rgba(139, 92, 246, 0.5)",
                'steps': [
                    {'range': [-1, -0.5], 'color': "rgba(239, 68, 68, 0.3)"},
                    {'range': [-0.5, 0], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [0, 0.5], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [0.5, 1], 'color': "rgba(16, 185, 129, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "#EC4899", 'width': 4},
                    'thickness': 0.8,
                    'value': 0
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            margin=dict(t=50, b=50, l=50, r=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_ai_insights(raw_summary):
    """Enhanced AI insights with modern theming"""
    st.markdown('<div class="section-title">🤖 AI-Generated Insights</div>', unsafe_allow_html=True)
    
    if not st.session_state.ai_insights:
        if st.button("🧠 Generate AI Insights", use_container_width=True):
            generate_ai_insights(raw_summary)
    else:
        st.markdown('<div class="ai-insights">', unsafe_allow_html=True)
        st.markdown(st.session_state.ai_insights)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Regenerate Insights", use_container_width=True):
            st.session_state.ai_insights = None
            generate_ai_insights(raw_summary)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_ai_insights(raw_summary):
    """Generate AI insights with modern theming"""
    placeholder = st.empty()
    with placeholder.container():
        show_loading_animation("Generating AI Insights", "Analyzing patterns and trends...")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""Here is the data from their latest video:
        {raw_summary}
        Now, craft a creative and deeply analytical report that goes beyond the numbers. Structure your response using the following creative headers in Markdown:

        🎭 The Emotional Pulse: What's the Story?
        Instead of just listing stats, tell the sentiment story. Is the overall feeling celebratory, critical, or divided? Are there specific emotional undercurrents (e.g., excitement, confusion, gratitude)? Paint a vivid picture of the audience's collective mood.

        💬 Decoding the Dialogue: Beyond Likes and Dislikes
        Analyze the nature of the engagement. Are viewers just leaving one-word comments, or are they having detailed discussions? Are there recurring questions, suggestions, or debates? What does the quality of the conversation tell you about the community's health and investment in the content?

        🔬 Creator's Report Card: What Worked and What Didn't?
        Pinpoint the video's strengths and weaknesses based on the comments. What specific topics, moments, or editing choices are viewers praising? Conversely, what elements are drawing criticism or causing confusion? Be specific if possible.

        🚀 Strategic Growth Blueprint: Your Next Moves
        Provide 3–5 concrete, actionable recommendations based on your analysis. For each recommendation, use this format:

        The Insight: (e.g., "Viewers are repeatedly asking for a follow-up on topic X.")

        The Action: (e.g., "Create a dedicated video addressing topic X and pin a comment linking to it.")

        The Expected Outcome: (e.g., "Increased viewer satisfaction and higher engagement on a subsequent video.")

        ✨ Hidden Gems & Red Flags: The Signals in the Noise
        Uncover any surprising or outlier findings. Is there an unexpected feature request that could be a goldmine? A single, highly-upvoted critical comment that represents a silent majority? A niche topic that sparked an unusual amount of passion? Highlight these opportunities and potential pitfalls.

        Your final output should be an inspiring, data-driven narrative that empowers the creator to understand their audience better and make smarter content decisions. Keep it professional, yet engaging and creative.
        """
        
        response = model.generate_content(prompt)
        st.session_state.ai_insights = response.text
        
        placeholder.markdown('<div class="status-success">✅ AI insights generated successfully!</div>', unsafe_allow_html=True)
        time.sleep(1)
        placeholder.empty()
        st.rerun()
        
    except Exception as e:
        placeholder.markdown(f'<div class="status-error">❌ Failed to generate insights: {str(e)}</div>', unsafe_allow_html=True)

def show_downloads(raw_summary):
    """Enhanced download section with modern styling"""
    st.markdown('<div class="section-title">📥 Download Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Text file download
        st.download_button(
            label="📄 Download as TXT",
            data=raw_summary,
            file_name=f"sentiment_analysis_{st.session_state.selected_video['video_id']}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_txt"
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
                use_container_width=True,
                key="download_json"
            )
        except Exception as e:
            st.error(f"JSON generation failed: {e}")
    
    with col3:
        # PDF report download
        if st.button("📑 Generate PDF Report", use_container_width=True):
            generate_pdf_report(raw_summary)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_pdf_report(raw_summary):
    """Generate PDF report with modern theming"""
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

def get_image_as_base64(url):
    """Fetches an image from a URL and returns it as a Base64 encoded string."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Read image content into a BytesIO object
            img_content = BytesIO(response.content)
            # Encode to base64
            base64_encoded = base64.b64encode(img_content.read()).decode()
            return f"data:image/png;base64,{base64_encoded}"
        return None
    except Exception as e:
        print(f"Error fetching or encoding image: {e}")
        return None
    
def show_footer():
    """Enhanced footer with modern theming"""
    st.markdown("""
    <div class="footer-container" style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/32px-YouTube_full-color_icon_%282017%29.svg.png" alt="YouTube Logo" style="width:32px; height:32px; margin-bottom:0.7rem;">
        <h3 style="margin-top:0.4rem;">YouTube Sentiment Dashboard</h3>
        <p>Powered by AI • Built with Streamlit • Enhanced Analytics</p>
        <p>Analyze • Visualize • Understand</p>
    </div>
    """, unsafe_allow_html=True)




# ─── Main App Logic ───────────────────────────────────────────────────────────
def main():
    """Main application logic with modern theming"""
    show_header()
    
    if not st.session_state.dashboard_mode:
        search_interface()
    else:
        dashboard_interface()
    
    show_footer()

# ─── Run the app ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
