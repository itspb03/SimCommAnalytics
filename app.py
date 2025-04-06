import os
import streamlit as st
import yt_dlp
import whisper
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment, silence
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#---------------------------------------Data Handling-------------------------------------------#

# Load models
whisper_model = whisper.load_model("base")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment", weights_only=True
)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Directories
VIDEO_DIR = "videos/"
AUDIO_DIR = "audio/"
OUTPUT_DIR = "output/"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utilities
def download_video(video_url):
    """Downloads a video from a given URL."""
    ydl_opts = {
        'outtmpl': os.path.join(VIDEO_DIR, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        return ydl.prepare_filename(info_dict)

def extract_video_links(folder_url):
    """Extracts video links from a folder URL (web page)."""
    response = requests.get(folder_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(('.mp4', '.mkv', '.webm'))]
    return [folder_url + link if not link.startswith('http') else link for link in links]

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file."""
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

def segment_audio(audio_path, method="fixed"):
    """Splits audio into chunks."""
    audio = AudioSegment.from_wav(audio_path).low_pass_filter(3000)
    if method == "silence":
        chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    else:
        chunk_length = 5000
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    return chunks

def transcribe_audio(chunks):
    """Transcribes each audio chunk and returns results with timestamps."""
    results = []
    elapsed_time = 0
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        result = whisper_model.transcribe(chunk_path, beam_size=5, language="en")
        text = result["text"].strip()
        if text:
            results.append({"timestamp": round(elapsed_time, 2), "transcription": text})
        elapsed_time += len(chunk) / 1000
        os.remove(chunk_path)
    return results

def analyze_sentiment(text):
    """Performs sentiment analysis and categorizes as Positive, Negative, or Neutral."""
    if not text.strip():
        return "Neutral"
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = sentiment_model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=-1)
    sentiment_index = torch.argmax(scores).item()
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[sentiment_index]

def process_video(video_source, chunking_method="fixed"):
    """Processes a video file or downloads it from a URL."""
    if video_source.startswith("http"):
        video_path = download_video(video_source)
    else:
        video_path = video_source
    audio_path = os.path.join(AUDIO_DIR, os.path.basename(video_path).replace(".mp4", ".wav"))
    extract_audio(video_path, audio_path)
    chunks = segment_audio(audio_path, method=chunking_method)
    transcription_data = transcribe_audio(chunks)

    data = []
    for entry in transcription_data:
        sentiment = analyze_sentiment(entry["transcription"])
        data.append([entry["timestamp"], entry["transcription"], sentiment])

    df = pd.DataFrame(data, columns=["Timestamp", "Transcription", "Sentiment"])
    csv_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(video_path))[0] + ".csv")
    df.to_csv(csv_path, index=False)

    os.remove(video_path)
    os.remove(audio_path)

    return df, csv_path

#--------------------------------------------Data Plots------------------------------------------------------#

def generate_histogram(df):
    """Reads a CSV transcription file and creates a histogram of word counts per 5s bucket, colored by sentiment."""
    df.columns = df.columns.str.strip().str.lower()
    df['start_seconds'] = df['timestamp']
    max_time = df['start_seconds'].max()
    buckets = np.arange(0, max_time + 5, 5)
    colors = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'lime'}
    word_counts = {sentiment: np.zeros(len(buckets) - 1) for sentiment in colors}

    for _, row in df.iterrows():
        words = str(row['transcription']).split()
        timestamp = row['start_seconds']
        sentiment = row['sentiment']
        bucket_index = np.searchsorted(buckets, timestamp, side='right') - 1
        if 0 <= bucket_index < len(buckets) - 1:
            word_counts[sentiment][bucket_index] += len(words)

    fig, ax = plt.subplots(figsize=(12, 6))
    for sentiment, counts in word_counts.items():
        ax.bar(buckets[:-1], counts, width=5, align='edge', color=colors[sentiment], label=sentiment, edgecolor='black')

    ax.set_xticks(np.arange(0, max_time + 20, 20))
    ax.set_xlabel("Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Word Count", fontsize=14, fontweight="bold")
    ax.set_title("Word Count per 5-Second Time Bucket", fontsize=16, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    return fig

def plot_sentiment_pie_chart(df):
    """Creates a pie chart to show the percentage of each sentiment in the dataset, with the legend outside."""
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'Neutral': 'orange', 'Positive': 'lime', 'Negative': 'red'}
    sentiment_labels = ['Neutral', 'Positive', 'Negative']
    sentiment_values = [sentiment_counts.get(label, 0) for label in sentiment_labels]
    total_lines = len(df)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%',
           colors=[colors[label] for label in sentiment_labels], startangle=140,
           textprops={'fontsize': 12})

    legend_labels = [
        f"Total Transcribed Lines: {total_lines}",
        f"Neutral Lines: {sentiment_counts.get('Neutral', 0)}",
        f"Positive Lines: {sentiment_counts.get('Positive', 0)}",
        f"Negative Lines: {sentiment_counts.get('Negative', 0)}"
    ]

    legend_patches = [mpatches.Patch(color="white", label=legend_labels[0])] + \
                     [mpatches.Patch(color=colors[label], label=legend_labels[i + 1]) for i, label in enumerate(sentiment_labels)]

    plt.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    ax.set_title("Sentiment Classification", fontsize=16, fontweight="bold")
    return fig

def plot_sentiment_trend_line(df):
    sentiment_to_score = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df = df[df['sentiment'].isin(sentiment_to_score.keys())]
    df['sentiment_score'] = df['sentiment'].map(sentiment_to_score)
    df = df.sort_values('timestamp')
    df['rolling_score'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'], df['rolling_score'], marker='o', linestyle='-', color='blue', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sentiment Trend", fontsize=13, fontweight="bold")
    ax.set_title("Sentiment Trend Over Time", fontsize=15, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


# -------------------------------- STREAMLIT UI -------------------------------------- #


st.title("🎙️ SimCommAnalytics")
st.markdown(
    "_Analyzing communication dynamics in group simulations using AI-powered transcription and sentiment analysis._",
    unsafe_allow_html=True
)

upload_mode = st.sidebar.selectbox("Select Upload Method", [
    "URL to a video",
    "URL to a folder of videos",
    "Upload a single video file",
    "Upload a folder of videos"
])

chunking_method = st.sidebar.radio("Select Chunking Method", ["fixed", "silence"])

input_sources = []

if upload_mode == "URL to a video":
    video_url = st.text_input("Enter Video URL")
    if video_url:
        input_sources = [video_url]

elif upload_mode == "URL to a folder of videos":
    folder_url = st.text_input("Enter Folder URL")
    if folder_url:
        input_sources = extract_video_links(folder_url)

elif upload_mode == "Upload a single video file":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mkv", "webm"])
    if uploaded_file:
        file_path = os.path.join(VIDEO_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        input_sources = [file_path]

elif upload_mode == "Upload a folder of videos":
    uploaded_files = st.file_uploader("Upload Multiple Videos", type=["mp4", "mkv", "webm"], accept_multiple_files=True)
    if uploaded_files:
        input_sources = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(VIDEO_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            input_sources.append(file_path)

if st.button("Start Processing") and input_sources:
    with st.spinner("⏳ Processing... this may take a while depending on video length."):
        for i, src in enumerate(input_sources):
            st.info(f"Processing video {i+1}/{len(input_sources)}...")
            df, csv_file = process_video(src, chunking_method=chunking_method)
            st.success(f"Done! Download your CSV below.")
            st.download_button(
                label="📥 Download CSV",
                data=open(csv_file, "rb"),
                file_name=os.path.basename(csv_file),
                mime="text/csv"
            )

            # Display plots
            st.subheader("📊 Visualizations")

            # Reload DataFrame (to be safe)
            df_plot = pd.read_csv(csv_file)
            df_plot.columns = df_plot.columns.str.strip().str.lower()

            with st.expander("🔹 Word Count Histogram"):
                fig1 = generate_histogram(df_plot)
                st.pyplot(fig1)

            with st.expander("🔹 Sentiment Pie Chart"):
                fig2 = plot_sentiment_pie_chart(df_plot)
                st.pyplot(fig2)

            with st.expander("🔹 Sentiment Trend Line"):
                fig3 = plot_sentiment_trend_line(df_plot)
                st.pyplot(fig3)


