## **SimCommAnalytics**  
Analysis of communication data in a group simulation setting to understand group dynamics. This project implements advanced data analytics and AI modeling to analyze human-to-human communication within simulated environments.    

---

## **Table of Contents**  
- [Overview](#overview)  
- [Requirements](#requirements)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
  - [Processing Videos](#1-processing-videos)  
  - [Generating Visualizations](#2-generating-visualizations)  
- [Testing](#testing)  
- [References](#references)  

---

## **Overview**  
This project processes video files, extracts audio, segments it into **5-second chunks**, transcribes speech using **Whisper**, and performs **sentiment analysis** using a transformer model. The results are saved as CSV files and can be visualized using histograms and pie charts.

The project consists of two main scripts:  

- **`datahandling.py`** – Handles video downloading, audio extraction, transcription, and sentiment analysis.  
- **`data_plots.ipynb`** – Reads the processed CSV files and generates visualizations.  

---

## **Requirements**  
Ensure the following libraries are installed before running the scripts:  
```bash
pip install torch transformers whisper yt-dlp pandas requests bs4 pydub matplotlib numpy 
```

### **Required Libraries**  
| Library | Purpose |
|---------|---------|
| `torch` | Used for running deep learning models, including sentiment analysis. |
| `transformers` | Provides the pre-trained RoBERTa model for sentiment analysis. |
| `whisper` | Utilized for automatic speech recognition (ASR). |
| `yt-dlp` | Downloads video files from a given URL. |
| `pandas` | Handles tabular data and stores transcriptions in CSV format. |
| `requests` | Fetches HTML pages to extract video links. |
| `bs4` (BeautifulSoup) | Parses HTML to extract video links from webpages. |
| `pydub` | Handles audio file conversion and segmentation. |
| `matplotlib` | Generates plots for visualization. |
| `numpy` | Supports numerical computations for histogram binning. |

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/your_username/your_project.git
cd your_project
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Set Up Required Directories**  
The scripts automatically create the necessary directories (`videos/`, `audio/`, `output/`). Ensure that:  
- **Video files** are downloaded or placed in the `videos/` folder.  
- The **`output/` folder** stores processed CSV results.  

---

## **Usage**  

### **1. Processing Videos**  
Run `datahandling.py` to process a video file or URL.  

#### **Run the Script**  
```bash
python datahandling.py
```

#### **Example: Processing a Single Video**  
```python
main("path/to/local/video.mp4")
```

#### **Example: Processing a List of URLs**  
```python
main([
    "https://example.com/video1.mp4",
    "https://example.com/video2.mp4"
])
```

#### **Example: Processing All Videos in a Folder**  
```python
main("videos/")
```

---

### **2. Generating Visualizations**  
Run `data_plots.ipynb` to visualize sentiment classification and word frequency histograms.  

#### **Run the Script**  
```bash
python data_plots.py
```

#### **Expected Outputs:**  
- **Histogram**: Displays word count per 5-second time interval, color-coded by sentiment.  
- **Pie Chart**: Shows the distribution of sentiments in the transcription.  

---

## **Testing**  

The code was tested using the following methods:  

✅ **Manual Testing**  
- Processed different video files and URLs to verify that the pipeline works as expected.  

✅ **Edge Case Handling**  
- Verified behavior for **silent audio chunks**.  
- Checked **sentiment analysis results** for empty transcriptions.  

✅ **Output Verification**  
- Manually reviewed transcriptions in **CSV files**.  
- Checked **sentiment classifications** for accuracy.  
- Validated **visualizations** using sample data.  

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

---

This should be **directly copy-pastable** into your `README.md` file. Let me know if you need any modifications! 🚀
