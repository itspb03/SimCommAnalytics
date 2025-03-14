import os
import yt_dlp
import whisper
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
from transformers import AutoModelForSequenceClassification, AutoTokenizer



# Loading Whisper model for speech recognition
whisper_model = whisper.load_model("base") 

# Directory paths
VIDEO_DIR = "videos/"  # Folder containing video files
OUTPUT_DIR = "output/"  # Folder to store CSV outputs
AUDIO_DIR = "audio/"    # Temporary folder for extracted audio

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_video(video_url):
    """Downloads a video from a given URL."""
    ydl_opts = {
        'outtmpl': os.path.join(VIDEO_DIR, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        return ydl.prepare_filename(info_dict)  # Returns the downloaded file path

def extract_video_links(folder_url):
    """Extracts video links from a folder URL (web page)."""
    response = requests.get(folder_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(('.mp4', '.mkv', '.webm'))]
    return [folder_url + link if not link.startswith('http') else link for link in links]   # Returns extracted video link from folder link

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file."""
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

def segment_audio(audio_path):
    """Splits audio into chunks (≤5 seconds)."""
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.low_pass_filter(3000)  # Removes high-frequency noise

    chunk_length = 5000  # 5 seconds in milliseconds
    chunks = [audio[i : i + chunk_length] for i in range(0, len(audio), chunk_length)]
    return chunks

def transcribe_audio(chunks):
    """Transcribes each audio chunk and returns results with timestamps."""
    results = []
    elapsed_time = 0  # Tracks cumulative time for timestamps

    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        # Transcribe with Whisper
        result = whisper_model.transcribe(chunk_path, beam_size=5,  language="en")  # TranscribeS chunked audio using Whisper
        text = result["text"].strip()

        if text:
            results.append({
                "timestamp": round(elapsed_time, 2),
                "transcription": text
            })

        elapsed_time += len(chunk) / 1000  # Converts ms to seconds
        os.remove(chunk_path)  # Cleans up temp file

    return results

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment",
     weights_only=True
)  # Prevents loading arbitrary Python objects
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


def analyze_sentiment(text):
    """Performs sentiment analysis and categorizes as Positive, Negative, or Neutral."""
    if not text.strip():
        return "Neutral"

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = sentiment_model(**tokens)
    
    scores = torch.nn.functional.softmax(output.logits, dim=-1)
    sentiment_index = torch.argmax(scores).item()

    # Map model output to sentiment labels
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    return sentiment_mapping[sentiment_index]

def process_video(video_source):
    """Processes a video file or downloads it from a URL."""
    if video_source.startswith("http"):
        video_path = download_video(video_source)
    else:
        video_path = video_source  # Local file path
    
    audio_path = os.path.join(AUDIO_DIR, os.path.basename(video_path).replace(".mp4", ".wav"))
    
    extract_audio(video_path, audio_path)
    chunks = segment_audio(audio_path)
    transcription_data = transcribe_audio(chunks)

    # Perform sentiment analysis and save results
    data = []
    for entry in transcription_data:
        sentiment = analyze_sentiment(entry["transcription"])
        data.append([entry["timestamp"], entry["transcription"], sentiment])

    df = pd.DataFrame(data, columns=["Timestamp", "Transcription", "Sentiment"])
    csv_filename = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(video_path))[0] + ".csv")
    df.to_csv(csv_filename, index=False)

    print(f"Processed: {video_path} -> {csv_filename}")

    # Deleting video and audio files after processing
    os.remove(video_path)  # Delete the video file
    os.remove(audio_path)  # Delete the extracted audio file

def main(input_source):
    """Main function to handle multiple input types."""
    if isinstance(input_source, list):  # List of URLs or file paths
        video_sources = input_source
    elif os.path.isdir(input_source):  # Folder containing local videos
        video_sources = [os.path.join(input_source, f) for f in os.listdir(input_source) if f.endswith(('.mp4', '.mkv', '.webm'))]
    elif input_source.startswith("http"):  # Single URL (file or folder page)
        if input_source.endswith(('.mp4', '.mkv', '.webm')):
            video_sources = [input_source]  # Single direct video link
        else:
            video_sources = extract_video_links(input_source)  # Extract multiple videos from a folder page
    else:
        raise ValueError("Invalid input source. Provide a file path, list of URLs, or a URL to a video folder.")
    
    for video in video_sources:
        process_video(video)

if __name__ == "__main__":
    input_source = [
        "https://alabama.app.box.com/s/vnwa6b4qi42ts39exv8jej6vzabdu78k"
    ]
    main(input_source)
