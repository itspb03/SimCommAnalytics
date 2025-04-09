## **SimCommAnalytics**  
Analysis of communication data in a group simulation setting to understand group dynamics. This project implements advanced data analytics and AI modeling to analyze human-to-human communication within simulated environments.    

---

## **Table of Contents**  
- [Overview](#overview)  
- [Requirements](#requirements)  
- [Setup and Usage Instructions](#setup-and-usage-instructions)  
- [Overall Functionality of the Application](#overall-functionality-of-the-application)    
- [Testing](#testing)
- [Output of Sample Data](#output-of-sample-data)
- [References](#references)
- [Future Improvements](#future-improvements)  

---

## **Overview**
This project processes video files, extracts audio, segments it into **fixed-length** or **silence-based chunks**, transcribes speech using **Whisper**, and performs **sentiment analysis** using a transformer model. The results are saved as CSV files and can be visualized using histograms and pie charts.

The entire functionality is encapsulated in a single main script:
 **`app.py`** 
 
 This script integrates all components of the pipeline and is divided into clearly labeled sections through comments:
 
  a) **Data Handling** – Manages video/audio input, chunking, transcription, and sentiment analysis.

  b) **Data Plots** – Generates sentiment-based histograms, pie charts, and line plots.
 
  c) **Streamlit UI** – Provides an interactive web interface for uploading videos, selecting analysis options, and visualizing results.



---

## **Requirements**  
Ensure the following libraries are installed before running the scripts:  
```bash
pip install torch transformers openai-whisper yt-dlp pandas requests bs4 pydub matplotlib numpy streamlit
```

### **Required Libraries**  
| Library | Purpose |
|---------|---------|
| `torch` | Used for running deep learning models, including sentiment analysis. |
| `transformers` | Provides the pre-trained RoBERTa model for sentiment analysis. |
| `openai-whisper` | Utilized for automatic speech recognition (ASR). |
| `yt-dlp` | Downloads video files from a given URL. |
| `pandas` | Handles tabular data and stores transcriptions in CSV format. |
| `requests` | Fetches HTML pages to extract video links. |
| `bs4` (BeautifulSoup) | Parses HTML to extract video links from webpages. |
| `pydub` | Handles audio file conversion and segmentation. |
| `matplotlib` | Generates plots for visualization. |
| `numpy` | Supports numerical computations for histogram binning. |
| `streamlit` | Provides the interactive web interface for user interaction. |


---



## **Setup and Usage Instructions**

Follow the steps below to run the Communication Analysis Tool:

### **1. Clone the Repository**
```bash
git clone https://github.com/itspb03/SimCommAnalytics.git
cd SimCommAnalytics
```

### **2. Install Dependencies**
Ensure Python 3.8+ is installed, then install all required libraries using:
```bash
pip install -r requirements.txt
```

> **Note:** If Whisper or PyTorch installation fails, refer to [Whisper’s installation guide](https://github.com/openai/whisper) or [PyTorch’s official site](https://pytorch.org/get-started/locally/).

---

### **3. Run the Application**
Start the Streamlit interface using:
```bash
streamlit run app.py
```

### **4. Using the Tool**
Once the app launches in your browser:

- **Select the uploading method:**
  
  ![Screenshot 2025-04-06 211738](https://github.com/user-attachments/assets/7d75287e-ab49-4fc1-932b-6600eab4b4be)

- **Upload a video file/folder** or **provide a YouTube link or video url** (single video or playlist).
  
- **Choose chunking method**:
  - `Fixed`: Segments audio into equal 5-second parts.  
  - `Silence-based`: Detects pauses to segment naturally.

    
  ![Screenshot 2025-04-06 213412](https://github.com/user-attachments/assets/087f5458-da9e-4500-a650-361e0b6f4d28)


- **Download CSV** with transcriptions and sentiment scores.

- **Visualize**:
  - Histogram of word count per chunk with sentiment color-coding.
  - Pie chart showing overall sentiment distribution.
  - Line plot of sentiment trend over time.


### **5. Output UI**


  <img src="https://github.com/user-attachments/assets/3f49c872-4334-4599-9d47-26c8338750cf" width="1000"/>



- Processed data is saved as a `.csv` file.
- Plots are displayed directly in the Streamlit interface.
- Temporary audio chunks are auto-deleted after processing.

## **Demo Video :** [Link](https://drive.google.com/file/d/1_hUoyB47qfiCUrbaAiwDTVAWdLBExtHk/view?usp=sharing) 


---


## **Overall Functionality of the Application**

This Communication Analysis Tool is a fully integrated pipeline that allows users to analyze spoken content from videos in terms of sentiment. It handles everything from video ingestion to insightful visualizations. Here’s what the application does in detail:

1. **Video Ingestion**  
   - Accepts a video file **uploaded manually** or **fetched from a URL** (supports individual YouTube videos or full playlists via `yt-dlp`).
   - Automatically downloads and prepares the video for audio processing.

2. **Audio Extraction & Segmentation**  
   - Converts the video to audio using `pydub`.
   - Allows the user to choose between:
     - **Fixed-length chunking**: Divides the audio into uniform 5-second segments.
     - **Silence-based chunking**: Dynamically segments audio based on detected pauses or silence intervals for more natural splits.

3. **Transcription (Speech-to-Text)**  
   - Each audio chunk is transcribed using **OpenAI’s Whisper model**, which is known for high-accuracy speech recognition.
   - Transcriptions are timestamped to reflect the original audio timeline.

4. **Sentiment Analysis**  
   - Each transcribed chunk is analyzed using the **`cardiffnlp/twitter-roberta-base-sentiment`** model.
   - Supports **3-class sentiment categorization**: **Negative**, **Neutral**, and **Positive**.

5. **Result Generation**  
   - The complete output is saved as a `.csv` file inside the `output/` folder.
   - Each row in the CSV includes:  
     - Chunk timestamp  
     - Transcribed text  
     - Sentiment label  
   - The CSV can be **downloaded directly through the Streamlit app interface**.
     
     #### **Example CSV Output Format**
      | Timestamp | Transcription | Sentiment |
      |-----------|-------------|-----------|
      | 0.00      | Hello world! | Neutral   |
      | 5.00      | This is amazing. | Positive |
     

6. **Data Visualization**  
   - Once processing is complete, the app generates insightful visual summaries:
     
     - **Histogram**: Word count per audio chunk, color-coded by sentiment.
     - **Pie Chart**: Overall sentiment distribution.
     - **Line Plot**: Sentiment trend over time (based on chunk timeline and rolling average).



---

## **Testing**
Each component was tested using sample video files and URLs.

### **Unit Tests**
1. **Video Downloading** – Verified using multiple video URLs.
2. **Audio Extraction & Segmentation** – Tested on different formats (`mp4`, `mkv`, `webm`).
3. **Transcription Accuracy** – Compared Whisper output with expected transcriptions.
4. **Sentiment Analysis** – Manually verified sentiment labels for correctness.
5. **Visualization Outputs** – Checked histograms and pie charts for proper sentiment distribution.

### **Edge Cases**
- Empty transcriptions default to `Neutral`.
- Non-English speech might produce incorrect transcriptions.
- Poor audio quality affects transcription accuracy.

---

## **Output of Sample Data**  

Below is an example of the output generated from a sample video processed through our pipeline.  

### **Sample Video**  
[Click here to view the sample video](https://alabama.app.box.com/s/vnwa6b4qi42ts39exv8jej6vzabdu78k)

### **Generated Visualizations**  

#### **1. Histogram: Word Count Per 5-Second Interval**  
*(This histogram represents the frequency of words spoken in each 5-second chunk, color-coded based on sentiment classification.)*  
![image](https://github.com/user-attachments/assets/b7f8a797-4665-4e65-b82b-b7a3c84499f5)   

#### **2. Pie Chart: Sentiment Distribution**  
*(This pie chart visualizes the overall sentiment distribution across the transcription.)*  
![image](https://github.com/user-attachments/assets/d07e9ec2-cd7f-48d3-9191-22bd2d2d10f3) 

#### **3. Line Plot: Sentiment Trend Over Time**
*(This line plot displays the progression of sentiment across the audio timeline. Each point represents the sentiment of a chunk, and a rolling average is applied to smooth out short-term fluctuations and highlight the overall trend in speaker emotion.)*
![image](https://github.com/user-attachments/assets/ce3dc49d-825f-43c9-8d70-95400ab96da1)


---





## **References**  

### **1. Whisper Model (Automatic Speech Recognition)**  
- **GitHub**: [OpenAI Whisper](https://github.com/openai/whisper)  
- **Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)  

### **2. Sentiment Analysis Model**  
- **Model**: [Cardiff NLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
- **Paper**: [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/abs/2010.12421)  

### **3. YouTube-DLP for Video Downloading**  
- **Repository**: [yt-dlp](https://github.com/yt-dlp/yt-dlp)  

### **4. Pydub for Audio Processing**  
- **Documentation**: [Pydub](https://github.com/jiaaro/pydub)

### **5. Hugging Face Transformers**  
- **Library**: [Transformers](https://github.com/huggingface/transformers)  
- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) *(underlying transformer architecture)*  

### **6. Streamlit for UI**  
- **Library**: [Streamlit](https://github.com/streamlit/streamlit)  
- **Documentation**: [streamlit.io](https://docs.streamlit.io/)  

### **7. BeautifulSoup (bs4) for Web Scraping**  
- **Library**: [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)  
- **Documentation**: [BeautifulSoup Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)  

### **8. Matplotlib for Plotting**  
- **Library**: [Matplotlib](https://matplotlib.org/stable/index.html)  
- **Paper**: [Matplotlib: A 2D Graphics Environment](https://ieeexplore.ieee.org/document/4160265)  


---




## **Future Improvements**
- Add support for multiple languages.
- Improve segmentation logic for more accurate timestamps.
- More fine-grained sentiment ranking.
- Enhance UI for better visualization.

---



















