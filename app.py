from dotenv import load_dotenv
import os
import time
import requests
import streamlit as st
from googleapiclient.discovery import build
from google.cloud import storage
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Configure Gemini API ---
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not gemini_api_key:
    st.error("Gemini API Key not found. Please set 'GEMINI_API_KEY'.")
    st.stop()
try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

# --- Page config ---
st.set_page_config(
    page_title="YouTube Sentiment Dashboard",
    page_icon="🎬",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

  html, body, [class*="st-"] {
    font-family: 'Roboto', sans-serif !important;
  }

  /* Transparent header container */
  .main-header {
    display: flex !important;
    align-items: center !important;
    gap: 20px !important;
    padding: 15px 30px !important;
    background: none !important;
    margin-bottom: 10px !important;
  }

  /* Logo always visible */
  .youtube-logo {
    height: 60px !important;
    width: auto !important;
    object-fit: contain !important;
  }

  /* Big red title with text-shadow for contrast */
  .project-title {
    font-size: 36px !important;
    font-weight: 700 !important;
    color: #FF0000 !important;
    margin: 0 !important;
    line-height: 1 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
  }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
def show_header():
    youtube_logo_url = "https://cdn-icons-png.flaticon.com/512/1384/1384060.png" # Or the .ico link
    st.markdown(f"""
    <div class="main-header">
      <img src="{youtube_logo_url}" class="youtube-logo" alt="YouTube Logo" style="height: 60px;">
      <div class="project-title">Sentiment Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

# --- Session State Init ---
def init_state():
    for k in ("search_results","selected_video","raw_summary","ai_insights"):
        if k not in st.session_state:
            st.session_state[k] = None

# --- Search UI ---
def show_search():
    st.subheader("🔍 Search YouTube Videos")
    c1,c2 = st.columns([4,1])
    with c1:
        q = st.text_input("Search", placeholder="e.g., AI tutorials")
    with c2:
        m = st.selectbox("Max Results", [10,25,50])
    if st.button("Search"):
        if not q:
            st.warning("Please enter search terms.")
            return
        key = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY"))
        if not key:
            st.error("YOUTUBE_API_KEY missing.")
            return
        try:
            yt = build("youtube","v3",developerKey=key)
            resp = yt.search().list(part="snippet",q=q,type="video",maxResults=m).execute()
            st.session_state.search_results = [
                {
                  "video_id": i["id"]["videoId"],
                  "title": i["snippet"]["title"],
                  "channel": i["snippet"]["channelTitle"],
                  "published": i["snippet"]["publishedAt"][:10],
                  "thumbnail": i["snippet"]["thumbnails"]["medium"]["url"],
                  "description": i["snippet"]["description"]
                } for i in resp["items"]
            ]
        except Exception as e:
            st.error(f"YouTube API error: {e}")

# --- Show Results ---
def show_results():
    vids = st.session_state.search_results
    if not vids: return
    st.subheader("📹 Search Results")
    for idx,v in enumerate(vids):
        c1,c2,c3 = st.columns([1,3,1])
        with c1:
            st.image(v["thumbnail"],width=120)
        with c2:
            st.markdown(f"**[{v['title']}](https://www.youtube.com/watch?v={v['video_id']})**")
            st.caption(f"{v['channel']} • {v['published']}")
            st.write(v["description"][:100]+"…")
        with c3:
            if st.button("Select",key=f"sel_{idx}"):
                st.session_state.selected_video=v
                st.session_state.raw_summary=None
                st.session_state.ai_insights=None
                st.rerun()

# --- Selected & Trigger ---
def show_selected():
    v = st.session_state.selected_video
    if not v: return
    st.subheader("🎯 Selected Video")
    st.image(v["thumbnail"],width=300)
    st.markdown(f"**{v['title']}**")
    st.caption(f"{v['channel']} • {v['published']}")
    if st.button("🚀 Analyze Comments"):
        trigger_sentiment_analysis(v["video_id"])

def trigger_sentiment_analysis(video_id):
    func = st.secrets.get("COMMENTS_FUNC_URL", os.getenv("COMMENTS_FUNC_URL"))
    bucket = st.secrets.get("RESULTS_BUCKET", os.getenv("RESULTS_BUCKET"))
    if not func or not bucket:
        st.error("COMMENTS_FUNC_URL or RESULTS_BUCKET missing.")
        return
    try:
        requests.post(func,json={"video_url":f"https://www.youtube.com/watch?v={video_id}"})
        st.success("Extraction started.")
    except Exception as e:
        st.error(f"Function call failed: {e}")
        return
    try:
        client = storage.Client()
        b = client.bucket(bucket)
        blob = None
        for _ in range(20):
            items = list(b.list_blobs(prefix=video_id))
            if items:
                blob = max(items, key=lambda bb: bb.time_created)
                break
            time.sleep(3)
        if not blob:
            st.error("Summary not ready—try again later.")
            return
        st.session_state.raw_summary = blob.download_as_text()
        st.rerun()
    except Exception as e:
        st.error(f"GCS error: {e}")

# --- Summary & Insights ---
def show_summary_and_insights():
    txt = st.session_state.raw_summary
    if not txt: return
    st.subheader("📄 Raw Summary")
    st.text_area("", txt, height=300)
    st.subheader("🤖 Gemini AI Insights")
    if not st.session_state.ai_insights:
        with st.spinner("Generating…"):
            try:
                model = genai.GenerativeModel("gemini-1.5-pro")
                prompt = f"Analyze this summary:\n\n{txt}\n\nProvide sentiment, themes, and recommendations."
                resp = model.generate_content(prompt)
                st.session_state.ai_insights = resp.text
            except Exception as e:
                st.error(f"Gemini error: {e}")
                st.session_state.ai_insights = "Insight generation failed."
    st.markdown(st.session_state.ai_insights)

# --- Main ---
def main():
    init_state()
    show_header()
    t1,t2 = st.tabs(["🔍 Search Videos","📊 Analyze Sentiment"])
    with t1:
        show_search()
        show_results()
    with t2:
        show_selected()
        show_summary_and_insights()
    st.caption("Built with YouTube Data API & Google Gemini AI")

if __name__=="__main__":
    main()
